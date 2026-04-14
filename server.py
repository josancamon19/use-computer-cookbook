"""
Minimal HTTP sidecar: POST /run drives a single task through `harbor run`,
streaming NDJSON events back. No custom agent loop — harbor loads
AnthropicCUAAgent or GenericCUAAgent from the runner package based on model.

  POST /run   body: {task: {instruction, pre_command, grading_command},
                     model, gateway_url, gateway_api_key, max_steps?}
              response: application/x-ndjson stream of events
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import yaml
from aiohttp import web

REPO_ROOT = Path("/repo")
RUNNER_DIR = REPO_ROOT / "runner"


def agent_spec_for(model: str, max_steps: int = 30) -> dict:
    """Pick a CUA agent class based on model name."""
    if "kimi" in model or "moonshot" in model or model.startswith("openai/"):
        return {
            "import_path": "runner.agents.generic_cua:GenericCUAAgent",
            "model_name": model if "/" in model else f"openai/{model}",
            "kwargs": {
                "max_steps": max_steps,
                "max_tokens": 4096,
                "api_base": "https://api.fireworks.ai/inference/v1",
            },
        }
    # default: Anthropic CUA (Sonnet 4.6, Opus 4.6, etc.)
    return {
        "import_path": "runner.agents.anthropic_cua:AnthropicCUAAgent",
        "model_name": model if "/" in model else f"anthropic/{model}",
        "kwargs": {"max_steps": max_steps},
    }


def materialize_task_dir(base: Path, task: dict) -> Path:
    """Write a collected task as a harbor task layout that `harbor run -p` can load."""
    task_dir = base / "task"
    (task_dir / "tests" / "setup").mkdir(parents=True, exist_ok=True)
    (task_dir / "environment").mkdir(exist_ok=True)

    (task_dir / "instruction.md").write_text(task.get("instruction", ""))
    (task_dir / "task.toml").write_text(
        "[metadata]\n"
        "author_name = \"mmini-collect\"\n"
        "difficulty = \"unknown\"\n"
        "\n[verifier]\ntimeout_sec = 1800\n"
        "\n[agent]\ntimeout_sec = 1800\n"
        "\n[environment]\ncpus = 4\nmemory_mb = 8192\nallow_internet = true\n"
    )
    # Grader script — mirrors the shape in datasets/macosworld_ready/*/tests/test.sh
    graders = task.get("grading_command", [])
    grader_sh = "#!/bin/bash\n# auto-generated from collected task\nset +e\n"
    grader_sh += 'PREFIX=""\n[ -d "/tmp/harbor/logs" ] && PREFIX="/tmp/harbor"\n'
    grader_sh += 'REWARD="${PREFIX}/logs/verifier/reward.txt"\n'
    if not graders:
        # No grader = assume pass (harbor requires something)
        grader_sh += 'echo "1" > "$REWARD"; echo "Score: 1"; exit 0\n'
    else:
        # Each entry is [cmd, weight]; treat as AND of all being True
        for cmd, _weight in graders:
            grader_sh += f"if ! bash -c {json.dumps(cmd)} 2>/dev/null | grep -qi 'true'; then echo '0' > \"$REWARD\"; echo 'Score: 0'; exit 0; fi\n"
        grader_sh += 'echo "1" > "$REWARD"; echo "Score: 1"; exit 0\n'
    (task_dir / "tests" / "test.sh").write_text(grader_sh)
    (task_dir / "tests" / "test.sh").chmod(0o755)

    # pre_command.sh + config.json under tests/setup/ — harbor executes these in-VM
    pre = task.get("pre_command", "") or ""
    if pre:
        (task_dir / "tests" / "setup" / "pre_command.sh").write_text(
            "#!/bin/bash\n" + pre + "\n"
        )
        (task_dir / "tests" / "setup" / "pre_command.sh").chmod(0o755)
    (task_dir / "tests" / "setup" / "config.json").write_text(json.dumps({
        "before_action_delay_seconds": task.get("before_action_delay_seconds", 5),
        "before_grading_delay_seconds": task.get("before_grading_delay_seconds", 5),
    }))
    return task_dir


def build_job_yaml(base: Path, agent_spec: dict, gateway_url: str) -> Path:
    job_path = base / "job.yaml"
    job = {
        "jobs_dir": str(base / "jobs"),
        "n_attempts": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {
            "import_path": "runner.environments.mmini:MminiEnvironment",
            "kwargs": {"gateway_url": gateway_url},
            "delete": True,
        },
        "agents": [agent_spec],
    }
    job_path.write_text(yaml.safe_dump(job))
    return job_path


SANDBOX_RE = re.compile(r"(sb-[0-9a-f]{16,})")
STEP_RE = re.compile(r"step (\d+)/(\d+):")
STEP_DONE_RE = re.compile(r"step (\d+): API responded in [\d.]+s,\s*in=(\d+)\s+out=(\d+)")
REWARD_RE = re.compile(r"Results written to (\S+)")


async def handle_run(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    task = body.get("task") or {}
    model = body.get("model", "claude-sonnet-4-6")
    max_steps = int(body.get("max_steps", 30))
    gateway_url = body.get("gateway_url") or os.environ.get("GATEWAY_URL", "http://localhost:8080")
    gateway_api_key = body.get("gateway_api_key") or os.environ.get("MMINI_API_KEY", "")
    if not task.get("instruction"):
        return web.json_response({"error": "task.instruction required"}, status=400)

    work = Path(tempfile.mkdtemp(prefix="modelrun_"))
    task_dir = materialize_task_dir(work, task)
    job_yaml = build_job_yaml(work, agent_spec_for(model, max_steps), gateway_url)

    resp = web.StreamResponse(status=200, headers={"Content-Type": "application/x-ndjson"})
    await resp.prepare(request)

    async def emit(event: str, **fields):
        await resp.write((json.dumps({"event": event, **fields}) + "\n").encode())

    await emit("agent_start", model=model)

    env = os.environ.copy()
    if gateway_api_key:
        env["MMINI_API_KEY"] = gateway_api_key
    env["MMINI_GATEWAY_URL"] = gateway_url

    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "harbor", "run",
        "-c", str(job_yaml),
        "-p", str(task_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(RUNNER_DIR),
        env=env,
    )

    assert proc.stdout is not None
    seen_sandbox = set()
    tail_lines: list[str] = []
    result_path: Path | None = None
    tokens_in_total = 0
    tokens_out_total = 0
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip("\n")
            tail_lines.append(line)
            if len(tail_lines) > 50:
                tail_lines = tail_lines[-50:]

            m = SANDBOX_RE.search(line)
            if m and m.group(1) not in seen_sandbox:
                seen_sandbox.add(m.group(1))
                await emit("sandbox_created", sandbox_id=m.group(1))

            m = STEP_RE.search(line)
            if m:
                await emit("agent_step", idx=int(m.group(1)), max_steps=int(m.group(2)))

            m = STEP_DONE_RE.search(line)
            if m:
                tokens_in_total += int(m.group(2))
                tokens_out_total += int(m.group(3))

            m = REWARD_RE.search(line)
            if m:
                result_path = Path(m.group(1))
        await proc.wait()
    finally:
        # Find the reward: first try the exact result.json harbor printed,
        # then glob the jobs dir as a fallback.
        reward = None
        candidate_paths = [result_path] if result_path else []
        candidate_paths += list((work / "jobs").rglob("result.json"))
        for p in candidate_paths:
            if not p or not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            # Harbor writes different shapes across versions; probe a few.
            if isinstance(data, dict):
                if "reward" in data and isinstance(data["reward"], (int, float)):
                    reward = float(data["reward"]); break
                trials = data.get("trials") or []
                rewards = [t.get("reward") for t in trials if isinstance(t.get("reward"), (int, float))]
                if rewards:
                    reward = float(rewards[0]); break
                # Look for nested per-agent summaries: data["agents"][0]["reward"]
                for agent in (data.get("agents") or []):
                    if isinstance(agent.get("reward"), (int, float)):
                        reward = float(agent["reward"]); break
                if reward is not None:
                    break
        if proc.returncode == 0:
            await emit(
                "agent_done",
                tokens_in=tokens_in_total,
                tokens_out=tokens_out_total,
            )
            await emit("done", reward=reward if reward is not None else -1)
        else:
            # Surface harbor's tail so a failed run isn't a black box
            await emit(
                "error",
                message=f"harbor exited {proc.returncode}",
                tail="\n".join(tail_lines[-30:]),
            )
        shutil.rmtree(work, ignore_errors=True)

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
