"""
Minimal HTTP sidecar for "Run with model". No log scraping, no event streams —
just process lifecycle + reading files harbor writes.

  POST /run          start a harbor run in the background; returns {job_id}
  GET  /jobs/{id}    current status pulled from disk; terminal state has reward
  GET  /health

Jobs are written under /data/jobs (shared with harbor-viewer) so once a run
completes the UI can deep-link straight to `harbor view` for inspection.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/data/jobs"))
REPO_ROOT = Path("/repo")
RUNNER_DIR = REPO_ROOT / "runner"
SANDBOX_RE = re.compile(r"(sb-[0-9a-f]{16,})")


@dataclass
class JobRec:
    job_id: str
    work_dir: Path
    task: asyncio.Task
    created_at: float = field(default_factory=time.time)
    returncode: int | None = None
    error: str | None = None


JOBS: dict[str, JobRec] = {}


def agent_spec_for(model: str, max_steps: int = 30) -> dict:
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
    return {
        "import_path": "runner.agents.anthropic_cua:AnthropicCUAAgent",
        "model_name": model if "/" in model else f"anthropic/{model}",
        "kwargs": {"max_steps": max_steps},
    }


def write_task_dir(task_dir: Path, task: dict) -> None:
    (task_dir / "tests" / "setup").mkdir(parents=True, exist_ok=True)
    (task_dir / "environment").mkdir(exist_ok=True)

    (task_dir / "instruction.md").write_text(task.get("instruction", ""))
    (task_dir / "task.toml").write_text(
        "[metadata]\nauthor_name = \"mmini-collect\"\ndifficulty = \"unknown\"\n"
        "[verifier]\ntimeout_sec = 1800\n"
        "[agent]\ntimeout_sec = 1800\n"
        "[environment]\ncpus = 4\nmemory_mb = 8192\nallow_internet = true\n"
    )

    graders = task.get("grading_command") or []
    grader_sh = [
        "#!/bin/bash",
        'PREFIX=""',
        '[ -d "/tmp/harbor/logs" ] && PREFIX="/tmp/harbor"',
        'REWARD="${PREFIX}/logs/verifier/reward.txt"',
    ]
    if not graders:
        grader_sh += ['echo "1" > "$REWARD"', 'echo "Score: 1"', 'exit 0']
    else:
        for cmd, _weight in graders:
            grader_sh += [
                f"if ! bash -c {json.dumps(cmd)} 2>/dev/null | grep -qi 'true'; then",
                '  echo "0" > "$REWARD"; echo "Score: 0"; exit 0',
                "fi",
            ]
        grader_sh += ['echo "1" > "$REWARD"', 'echo "Score: 1"', 'exit 0']
    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text("\n".join(grader_sh) + "\n")
    test_sh.chmod(0o755)

    pre = (task.get("pre_command") or "").strip()
    if pre:
        pc = task_dir / "tests" / "setup" / "pre_command.sh"
        pc.write_text("#!/bin/bash\n" + pre + "\n")
        pc.chmod(0o755)
    (task_dir / "tests" / "setup" / "config.json").write_text(json.dumps({
        "before_action_delay_seconds": task.get("before_action_delay_seconds", 5),
        "before_grading_delay_seconds": task.get("before_grading_delay_seconds", 5),
    }))


def write_job_yaml(job_yaml: Path, jobs_dir: Path, agent_spec: dict, gateway_url: str) -> None:
    job_yaml.write_text(yaml.safe_dump({
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {
            "import_path": "runner.environments.mmini:MminiEnvironment",
            "kwargs": {"gateway_url": gateway_url},
            "delete": True,
        },
        "agents": [agent_spec],
    }))


async def run_harbor(rec: JobRec, task_dir: Path, job_yaml: Path, env: dict) -> None:
    log = rec.work_dir / "harbor.log"
    with log.open("w") as lf:
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "harbor", "run",
            "-c", str(job_yaml),
            "-p", str(task_dir),
            stdout=lf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(RUNNER_DIR),
            env=env,
        )
        rc = await proc.wait()
        rec.returncode = rc


def find_trial_dir(jobs_dir: Path) -> Path | None:
    """The first subdir of the first harbor-job dir, if any."""
    for job in sorted(jobs_dir.iterdir()):
        if not job.is_dir():
            continue
        for trial in sorted(job.iterdir()):
            if trial.is_dir():
                return trial
    return None


def peek_sandbox_id(trial_dir: Path) -> str | None:
    log = trial_dir / "trial.log"
    if not log.exists():
        return None
    try:
        head = log.read_text(errors="replace")[:4000]
    except Exception:
        return None
    m = SANDBOX_RE.search(head)
    return m.group(1) if m else None


def read_reward(jobs_dir: Path) -> tuple[float | None, dict | None]:
    """Return (reward, raw_result_json). Harbor writes per-trial result.json
    with verifier_result.rewards.reward. We glob over the whole jobs dir so
    this works regardless of the harbor-job subdir name."""
    for rj in sorted(jobs_dir.rglob("result.json")):
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        # Trial-level shape: verifier_result.rewards.reward
        vr = data.get("verifier_result") or {}
        rewards = vr.get("rewards") or {}
        r = rewards.get("reward")
        if isinstance(r, (int, float)):
            return float(r), data
        # Job-level shape: trial_results[].reward (older harbor versions)
        for t in (data.get("trial_results") or []):
            r = t.get("reward")
            if isinstance(r, (int, float)):
                return float(r), data
    return None, None


def read_step_counts(trial_dir: Path | None) -> tuple[int | None, int | None]:
    """Return (n_steps, n_actions). n_steps is agent turns (excludes the
    initial user message); n_actions is turns that contain at least one
    tool_call. A completed run with n_actions=0 means the agent refused to
    touch the UI — the usual signature of 'answered in text then quit'."""
    if trial_dir is None:
        return None, None
    traj = trial_dir / "agent" / "trajectory.json"
    if not traj.exists():
        return None, None
    try:
        data = json.loads(traj.read_text())
    except Exception:
        return None, None
    steps = data.get("steps") or []
    agent_steps = [s for s in steps if s.get("source") == "agent"]
    actions = sum(1 for s in agent_steps if s.get("tool_calls"))
    return len(agent_steps), actions


async def handle_run(request: web.Request) -> web.Response:
    body = await request.json()
    task = body.get("task") or {}
    if not task.get("instruction"):
        return web.json_response({"error": "task.instruction required"}, status=400)
    model = body.get("model", "claude-sonnet-4-6")
    max_steps = int(body.get("max_steps", 30))
    gateway_url = body.get("gateway_url") or os.environ.get("GATEWAY_URL", "http://localhost:8080")
    gateway_api_key = body.get("gateway_api_key") or os.environ.get("MMINI_API_KEY", "")

    task_id = (body.get("task_id") or "task")[:16]
    job_id = f"modelrun-{task_id}-{int(time.time())}"
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    task_dir = work_dir / "task"
    task_dir.mkdir(exist_ok=True)
    write_task_dir(task_dir, task)
    job_yaml = work_dir / "job.yaml"
    write_job_yaml(job_yaml, work_dir / "jobs", agent_spec_for(model, max_steps), gateway_url)
    (work_dir / "jobs").mkdir(exist_ok=True)

    env = os.environ.copy()
    if gateway_api_key:
        env["MMINI_API_KEY"] = gateway_api_key
    env["MMINI_GATEWAY_URL"] = gateway_url

    rec = JobRec(
        job_id=job_id,
        work_dir=work_dir,
        task=None,  # type: ignore[arg-type]
    )
    rec.task = asyncio.ensure_future(run_harbor(rec, task_dir, job_yaml, env))
    JOBS[job_id] = rec
    return web.json_response({"job_id": job_id})


async def handle_get_job(request: web.Request) -> web.Response:
    job_id = request.match_info["job_id"]
    rec = JOBS.get(job_id)
    if rec is None:
        return web.json_response({"error": "job not found"}, status=404)

    jobs_dir = rec.work_dir / "jobs"
    trial_dir = find_trial_dir(jobs_dir) if jobs_dir.exists() else None
    sandbox_id = peek_sandbox_id(trial_dir) if trial_dir else None

    # Build the harbor-viewer URL as a RELATIVE path from the viewer's jobs root.
    # harbor-viewer serves /data/jobs at https://jobs.api.use.computer/ so the
    # trial's URL is /<job_id>/jobs/<harbor-job-subdir>/<trial-subdir> (or just
    # /<job_id>/ which lists everything).
    trial_rel_path = None
    if trial_dir:
        try:
            trial_rel_path = str(trial_dir.relative_to(JOBS_DIR))
        except ValueError:
            pass

    n_steps, n_actions = read_step_counts(trial_dir)

    done = rec.task.done()
    if not done:
        return web.json_response({
            "job_id": job_id,
            "status": "running",
            "sandbox_id": sandbox_id,
            "trial_path": trial_rel_path,
            "n_steps": n_steps,
            "n_actions": n_actions,
        })

    # Harbor subprocess finished — decide completed vs failed.
    if rec.task.cancelled() or (rec.task.exception() is not None):
        err = rec.error or (str(rec.task.exception()) if rec.task.exception() else "cancelled")
        return web.json_response({
            "job_id": job_id,
            "status": "failed",
            "sandbox_id": sandbox_id,
            "trial_path": trial_rel_path,
            "error": err,
            "n_steps": n_steps,
            "n_actions": n_actions,
        })

    reward, result_json = read_reward(jobs_dir)
    status = "completed" if rec.returncode == 0 else "failed"
    return web.json_response({
        "job_id": job_id,
        "status": status,
        "sandbox_id": sandbox_id,
        "trial_path": trial_rel_path,
        "reward": reward,
        "returncode": rec.returncode,
        "n_steps": n_steps,
        "n_actions": n_actions,
        # The caller may want to render a few result fields without fetching.
        "stats": (result_json or {}).get("stats"),
    })


async def handle_health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/run", handle_run)
    app.router.add_get("/jobs/{job_id}", handle_get_job)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "8090"))
    web.run_app(make_app(), host="0.0.0.0", port=port)
