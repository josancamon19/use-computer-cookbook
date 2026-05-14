"""iOS-side helpers for UseComputerEnvironment: device pin, local exec for the verifier."""

from __future__ import annotations

import asyncio
import os
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.environments.base import ExecResult

if TYPE_CHECKING:
    from runner.environments.use_computer import UseComputerEnvironment


def expand_ios_id(short: str, kind: str) -> str:
    """Re-prepend `com.apple.CoreSimulator.<kind>.` if missing."""
    if not short or short.startswith("com.apple."):
        return short
    return f"com.apple.CoreSimulator.{kind}.{short}"


def read_ios_pin(task_dir: Path) -> tuple[str, str]:
    """Pull (device_type, runtime) from task.toml's [ios] block, fully expanded."""
    toml_path = task_dir / "task.toml"
    data = tomllib.loads(toml_path.read_text())
    device_type = data["ios"]['device_type'].strip()
    runtime = data['ios']['runtime'].strip()
    return (
        expand_ios_id(device_type, "SimDeviceType"),
        expand_ios_id(runtime, "SimRuntime"),
    )


async def exec_local(
    env: UseComputerEnvironment,
    command: str,
    extra_env: dict[str, str] | None,
    timeout_sec: int | None,
) -> ExecResult:
    """Run a verifier command on the runner host (no remote VM for iOS).

    Harbor uses absolute paths assuming a remote linux env (`/tests/`,
    `/logs/verifier/`). We rewrite them to the local task_dir + trial_dir so
    bash can find/write them, then parse Score: X out of the captured stdout
    and write reward.txt where harbor reads it.
    """
    rewritten = command
    local_tests = env._task_dir / "tests"
    if local_tests.exists():
        rewritten = rewritten.replace("/tests/", f"{local_tests}/")
    verifier_dir = env.trial_paths.verifier_dir
    verifier_dir.mkdir(parents=True, exist_ok=True)
    rewritten = rewritten.replace("/logs/verifier/", f"{verifier_dir}/")
    trial_dir = verifier_dir.parent
    for sub in ("agent", "artifacts"):
        rewritten = rewritten.replace(f"/logs/{sub}/", f"{trial_dir / sub}/")

    full_env = os.environ.copy()
    if extra_env:
        full_env.update(extra_env)
    full_env["GATEWAY_URL"] = env._gateway_url
    full_env["SANDBOX_ID"] = env._sandbox_id or ""
    full_env["USE_COMPUTER_API_KEY"] = env._api_key

    timeout = timeout_sec or 60
    proc = None
    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", rewritten,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        rc = proc.returncode or 0
    except asyncio.TimeoutError:
        if proc is not None and proc.returncode is None:
            proc.kill()
        env.logger.warning(f"ios exec timeout after {timeout}s: {rewritten[:120]}")
        return ExecResult(stdout="", stderr=None, return_code=124)

    elapsed = time.monotonic() - t0
    stdout = stdout_b.decode("utf-8", "replace")
    stderr = stderr_b.decode("utf-8", "replace")

    if "test.sh" in command:
        captured = ""
        stdout_file = verifier_dir / "test-stdout.txt"
        if stdout_file.exists():
            try:
                captured = stdout_file.read_text()
            except OSError:
                captured = ""
        if not captured:
            captured = stdout
        score = "1" if "Score: 1" in captured else "0"
        (verifier_dir / "reward.txt").write_text(f"{score}\n")
        env.logger.info(
            f"ios verifier rc={rc} ({elapsed:.1f}s) score={score} "
            f"stdout={captured.strip()[:200]!r}"
        )
        if stderr.strip():
            env.logger.info(f"ios verifier stderr: {stderr.strip()[:300]}")
    else:
        env.logger.debug(f"ios exec rc={rc} ({elapsed:.1f}s) cmd={rewritten[:120]}")

    return ExecResult(stdout=stdout, stderr=None, return_code=rc)
