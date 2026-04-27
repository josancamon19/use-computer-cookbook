"""macOS pre-command setup: transpile + benchmark uploads + per-line exec."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.environments.base import ExecResult
from mmini.ax_transpile import PRE_COMMAND_OSASCRIPT_TIMEOUT_S, needs_exec_ax, transpile

from runner.environments.macos_runtime import remap_str, wrap_with_timeout

if TYPE_CHECKING:
    from runner.environments.mmini import MminiEnvironment


# macOSWorld benchmark assets — referenced from pre_commands as "Benchmark_Backup/<name>".
BENCHMARK_ASSETS_DIR = Path(__file__).resolve().parents[3] / "macosworld" / "files"
BENCHMARK_VM_DIR = "/Users/lume/Benchmark_Backup"


async def run_macos_setup(env: MminiEnvironment) -> None:
    """Upload test.sh, run pre_command.sh line-by-line, load in_process config."""
    setup_dir = env._task_dir / "tests" / "setup"

    test_script = env._task_dir / "tests" / "test.sh"
    if test_script.exists():
        env.logger.info("uploading test.sh")
        await env.upload_file(test_script, "/tests/test.sh")

    pre_cmd_path = setup_dir / "pre_command.sh"
    if pre_cmd_path.exists():
        await _run_pre_command(env, pre_cmd_path.read_text().strip())

    # in_process config — pop a dialog mid-run on certain tasks.
    env._in_process_cmd = None
    env._in_process_step = None
    config_path = setup_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        ip = cfg.get("in_process")
        if ip and isinstance(ip, list) and len(ip) >= 2:
            env._in_process_cmd = ip[0]
            env._in_process_step = ip[1]

    env.logger.info("task setup complete")


async def _run_pre_command(env: MminiEnvironment, raw: str) -> None:
    await _upload_benchmark_files(env, raw)
    lines = [
        ln.strip()
        for ln in raw.split("\n")
        if ln.strip() and not ln.startswith("#!") and not ln.strip().startswith("#")
    ]
    for i, line in enumerate(lines):
        # Transpile osascript AX patterns (keystroke, etc.) so AX-needing calls
        # route through cua-server's exec_ax (regular exec_ssh hits TCC denials).
        transpiled, n = transpile(line, fallback_timeout_s=PRE_COMMAND_OSASCRIPT_TIMEOUT_S)
        result = await _exec_pre_line(env, line, transpiled, n)
        if result.return_code == 0:
            out = ((result.stdout or "") + (result.stderr or "")).strip()
            env.logger.info(
                f"pre_command [{i+1}/{len(lines)}] ok"
                + (f" (output: {out[:120]})" if out else "")
            )
            continue

        combined = (result.stdout or "") + (result.stderr or "")
        # -14 = our perl-alarm SIGALRM. -1712/-1728 = AppleEvent app cold-start.
        if result.return_code == -14:
            env.logger.warning(
                f"pre_command [{i+1}/{len(lines)}] ALARM-KILLED. cmd={line[:80]}"
            )
        if any(e in combined for e in ["-1712", "-1728"]):
            env.logger.warning(
                f"pre_command [{i+1}/{len(lines)}] AE error, retrying in 15s..."
            )
            await asyncio.sleep(15)
            result = await _exec_pre_line(env, line, transpiled, n)
        if result.return_code != 0:
            raise RuntimeError(
                f"pre_command [{i+1}/{len(lines)}] failed (rc={result.return_code}):\n"
                f"cmd: {line}\nstdout: {result.stdout or ''}\nstderr: {result.stderr or ''}"
            )


async def _exec_pre_line(
    env: MminiEnvironment, raw: str, transpiled: str, n_replacements: int
) -> ExecResult:
    if n_replacements > 0:
        if needs_exec_ax(transpiled):
            return await exec_ax(env, transpiled)
        return await env.exec(transpiled, timeout_sec=PRE_COMMAND_OSASCRIPT_TIMEOUT_S + 10)
    return await env.exec(raw, timeout_sec=PRE_COMMAND_OSASCRIPT_TIMEOUT_S + 10)


async def _upload_benchmark_files(env: MminiEnvironment, pre_command_text: str) -> None:
    """Find Benchmark_Backup/<name> refs in the script and upload each one."""
    refs: set[str] = set()
    for m in re.finditer(r'"[^"]*Benchmark_Backup/([^"]+)"', pre_command_text):
        refs.add(m.group(1).split("/")[0].rstrip())
    for m in re.finditer(r'(?<!")Benchmark_Backup/(\S+)', pre_command_text):
        refs.add(m.group(1).split("/")[0].rstrip(";"))
    if not refs:
        return

    await env.sandbox.exec_ssh(f"mkdir -p {BENCHMARK_VM_DIR}")
    for name in sorted(refs):
        local = BENCHMARK_ASSETS_DIR / name
        remote = f"{BENCHMARK_VM_DIR}/{name}"
        if not local.exists():
            env.logger.warning(f"benchmark asset not found: {local}")
            continue
        if local.is_dir():
            env.logger.info(f"uploading {name}/ -> {remote}")
            await env.upload_dir(local, remote)
        else:
            env.logger.info(f"uploading {name} -> {remote}")
            await env.upload_file(local, remote)
    result = await env.sandbox.exec_ssh(
        f"find {BENCHMARK_VM_DIR} -maxdepth 2 -type f | head -30 && "
        f"echo '---' && du -sh {BENCHMARK_VM_DIR}/*"
    )
    env.logger.info(f"Benchmark_Backup contents:\n{result.stdout}")


async def exec_ax(env: MminiEnvironment, command: str, timeout_sec: int = 30) -> ExecResult:
    """Run via cua-server's run_command — for commands that need TCC Accessibility."""
    full_cmd = wrap_with_timeout(remap_str(command), timeout_sec)
    result = await env.sandbox.exec_ax(full_cmd, timeout=timeout_sec)  # type: ignore[union-attr]
    out = (result.stdout or "").strip()
    env.logger.info(
        f"exec_ax rc={result.return_code} cmd={full_cmd[:100]}"
        + (f" output={out[:120]!r}" if out else "")
    )
    return ExecResult(stdout=result.stdout, stderr=None, return_code=result.return_code)
