"""
Minimal HTTP sidecar that wraps run.py.

  POST /run   — body JSON: {task: {instruction, pre_command, grading_command},
                             model, agent?, max_steps?, gateway_url?}
                response: streaming NDJSON events, one per line

Events mirror run.py --events output (sandbox_created / pre_command_* /
agent_step / agent_done / grading_done / done / error).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from aiohttp import web

RUNNER_DIR = Path(__file__).resolve().parent


async def handle_run(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    task = body.get("task") or {}
    model = body.get("model", "claude-sonnet-4-6")
    agent = body.get("agent", "anthropic")
    max_steps = int(body.get("max_steps", 30))
    gateway_url = body.get("gateway_url") or os.environ.get(
        "GATEWAY_URL", "http://localhost:8080"
    )

    instruction = task.get("instruction", "")
    if not instruction:
        return web.json_response({"error": "task.instruction required"}, status=400)

    task_dir = Path(tempfile.mkdtemp(prefix="run_"))
    (task_dir / "instruction.md").write_text(instruction)
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    config = {
        "pre_command": task.get("pre_command", ""),
        "before_action_delay_seconds": task.get("before_action_delay_seconds", 5),
        "grading_command": task.get("grading_command", []),
        "before_grading_delay_seconds": task.get("before_grading_delay_seconds", 2),
    }
    (tests_dir / "task_config.json").write_text(json.dumps(config))

    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "application/x-ndjson"},
    )
    await resp.prepare(request)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(RUNNER_DIR / "run.py"),
        "--task-dir", str(task_dir),
        "--agent", agent,
        "--model", model,
        "--max-steps", str(max_steps),
        "--gateway", gateway_url,
        "--events",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(RUNNER_DIR),
    )
    assert proc.stdout is not None

    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            # Only forward lines that parse as JSON — human-readable prose
            # from the prints in run.py gets dropped.
            try:
                json.loads(line)
            except ValueError:
                continue
            await resp.write(line)
        await proc.wait()
        if proc.returncode and proc.returncode != 0:
            err = ""
            if proc.stderr is not None:
                try:
                    err = (await proc.stderr.read()).decode(errors="replace")
                except Exception:
                    pass
            await resp.write(
                (json.dumps({"event": "error", "message": f"runner exited {proc.returncode}: {err[:400]}"}) + "\n").encode()
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    return resp


async def handle_health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/run", handle_run)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8090"))
    web.run_app(make_app(), host="0.0.0.0", port=port)
