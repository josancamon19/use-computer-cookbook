"""macOS-side helpers for MminiEnvironment: exec routing, path remap, transfer."""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.environments.base import ExecResult

if TYPE_CHECKING:
    from runner.environments.mmini import MminiEnvironment


# macOS SIP blocks writes under /. Harbor's expected paths get remapped to
# /tmp/harbor (writable through cua-server's launchd-domain chain).
PATH_PREFIX = "/tmp/harbor"
WORKSPACE_DIR = "/Users/lume"


def remap_str(s: str) -> str:
    """Rewrite harbor's `/logs`, `/tests`, `/solution`, `/app`, `/workspace` roots."""
    return (
        s.replace("/logs/", f"{PATH_PREFIX}/logs/")
        .replace("/logs\"", f"{PATH_PREFIX}/logs\"")
        .replace("/logs'", f"{PATH_PREFIX}/logs'")
        .replace("/tests/", f"{PATH_PREFIX}/tests/")
        .replace("/solution/", f"{PATH_PREFIX}/solution/")
        .replace("/installed-agent", f"{PATH_PREFIX}/installed-agent")
        .replace("/app/", f"{WORKSPACE_DIR}/")
        .replace("/workspace/", f"{WORKSPACE_DIR}/")
    )


def remap(path: str) -> str:
    """Remap a single absolute path."""
    if path.startswith(("/logs", "/tests", "/solution", "/installed-agent")):
        return PATH_PREFIX + path
    if path.startswith(("/app", "/workspace")):
        return WORKSPACE_DIR
    return path


def wrap_with_timeout(cmd: str, timeout: int) -> str:
    """Wrap with `perl alarm` to kill hung processes inside the VM."""
    kill_after = max(timeout - 2, 5)
    escaped = cmd.replace("'", "'\\''")
    return f"perl -e 'alarm {kill_after}; exec @ARGV' -- bash -c '{escaped}'"


async def exec_on_vm(
    env: MminiEnvironment,
    command: str,
    cwd: str | None,
    extra_env: dict[str, str] | None,
    timeout_sec: int | None,
    user: str | int | None,
) -> ExecResult:
    """Run `command` on the macOS VM via SSH (or AX exec for verifier scripts)."""
    remapped_cmd = remap_str(command)
    parts: list[str] = []
    if user is not None:
        parts.append(f"sudo -u {shlex.quote(str(user))}")
    if extra_env:
        remapped_env = {k: remap_str(v) for k, v in extra_env.items()}
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in remapped_env.items())
        parts.append(f"export {exports};")
    if cwd:
        parts.append(f"cd {shlex.quote(remap(cwd))} &&")
    parts.append(remapped_cmd)
    full_cmd = wrap_with_timeout(" ".join(parts), timeout_sec or 30)
    timeout = timeout_sec or 30

    # Verifier scripts (test.sh) need cua-server's responsibility chain so AX
    # TCC grants apply — exec_ssh's sshd-keygen-wrapper denies them.
    t0 = time.monotonic()
    if "test.sh" in full_cmd:
        result = await env.sandbox.exec_ax(full_cmd, timeout=timeout)  # type: ignore[union-attr]
    else:
        result = await env.sandbox.exec_ssh(full_cmd, timeout=timeout)  # type: ignore[union-attr]
    elapsed = time.monotonic() - t0

    is_verifier = "test.sh" in full_cmd or "_ax_" in full_cmd
    if is_verifier and result.return_code == -14:
        await _diagnose_alarm_kill(env, full_cmd, elapsed, timeout)
    elif is_verifier:
        env.logger.info(
            f"verifier exec rc={result.return_code} ({elapsed:.1f}s) "
            f"cmd={full_cmd[:100]}"
        )
    else:
        env.logger.debug(
            f"exec rc={result.return_code} ({elapsed:.1f}s) cmd={full_cmd[:100]}"
        )
    return ExecResult(stdout=result.stdout, stderr=None, return_code=result.return_code)


async def _diagnose_alarm_kill(
    env: MminiEnvironment, full_cmd: str, elapsed: float, timeout: int
) -> None:
    """Best-effort triage when our perl-alarm wrapper kills a verifier mid-run."""
    env.logger.warning(
        f"verifier ALARM-KILLED after {elapsed:.1f}s (timeout={timeout}s) — "
        f"reward=0.0 is a verifier hang, NOT a task failure. cmd={full_cmd[:120]}"
    )
    try:
        await env.sandbox.exec_ssh(  # type: ignore[union-attr]
            "touch /tmp/harbor/logs/verifier/alarm-killed.marker", timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        env.logger.debug(f"alarm-killed marker write failed: {exc}")
    try:
        r = await env.sandbox.exec_ssh(  # type: ignore[union-attr]
            "cat /tmp/harbor/logs/verifier/test-stdout.txt 2>/dev/null || echo '<empty>'",
            timeout=5,
        )
        stdout_text = (r.stdout or "").strip()
        env.logger.warning(f"verifier test-stdout (partial): {stdout_text[:500]!r}")
        (env.trial_paths.trial_dir / "verifier-stdout.txt").write_text(stdout_text)
    except Exception as exc:  # noqa: BLE001
        env.logger.debug(f"alarm-killed stdout fetch failed: {exc}")
    try:
        r = await env.sandbox.exec_ssh(  # type: ignore[union-attr]
            "curl -s -m 3 -o /dev/null -w '%{http_code}' "
            "http://127.0.0.1:8000/health 2>/dev/null || echo 'no-response'",
            timeout=6,
        )
        env.logger.warning(
            f"verifier post-alarm computer-server probe: {(r.stdout or '').strip()!r}"
        )
    except Exception as exc:  # noqa: BLE001
        env.logger.debug(f"alarm-killed server probe failed: {exc}")
    try:
        r = await env.sandbox.exec_ssh(  # type: ignore[union-attr]
            "ps aux | grep -E 'ax_helper|cua.server|python|curl' | grep -v grep || true",
            timeout=5,
        )
        procs = (r.stdout or "").strip()
        if procs:
            env.logger.warning(f"verifier post-alarm processes:\n{procs[:800]}")
    except Exception as exc:  # noqa: BLE001
        env.logger.debug(f"alarm-killed process list failed: {exc}")


async def fire_in_process(env: MminiEnvironment, step: int) -> None:
    """Pop the per-task in_process dialog if the current step matches."""
    if env._in_process_cmd and step == env._in_process_step:
        env.logger.info(f"in_process: firing dialog at step {step}")
        await env.sandbox.exec_ssh(  # type: ignore[union-attr]
            f"nohup {env._in_process_cmd} > /dev/null 2>&1 &"
        )


# --- transfer helpers ---

async def upload_file(env: MminiEnvironment, source: Path | str, target: str) -> None:
    await env.sandbox.upload(str(source), remap(target))  # type: ignore[union-attr]


async def upload_dir(env: MminiEnvironment, source: Path | str, target: str) -> None:
    await env.sandbox.upload_dir(str(source), remap(target))  # type: ignore[union-attr]


async def download_file(env: MminiEnvironment, source: str, target: Path | str) -> None:
    await env.sandbox.download_file(remap(source), str(target))


async def download_dir(env: MminiEnvironment, source: str, target: Path | str) -> None:
    await env.sandbox.download_dir(remap(source), str(target))  # type: ignore[union-attr]
