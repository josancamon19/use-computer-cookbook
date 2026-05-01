"""HTTP handlers for the runner sidecar."""
from __future__ import annotations

import asyncio
import os
import time

from aiohttp import web

from runner.server.agent import agent_spec_for
from runner.server.config import JOBS_DIR
from runner.server.jobs import (
    JOBS,
    JobRec,
    find_trial_dir,
    flatten_trial_dir,
    peek_sandbox_id,
    run_harbor,
)
from runner.server.results import read_reward, read_step_counts
from runner.server.task_dir import write_job_yaml, write_task_dir


async def handle_run(request: web.Request) -> web.Response:
    body = await request.json()
    task = body.get("task") or {}
    if not task.get("instruction"):
        return web.json_response({"error": "task.instruction required"}, status=400)
    model = body.get("model", "claude-sonnet-4-6")
    max_steps = int(body.get("max_steps", 30))
    platform = (body.get("platform") or task.get("platform") or "macos").lower()
    if platform not in ("macos", "ios"):
        return web.json_response({"error": f"unknown platform: {platform}"}, status=400)
    gateway_url = body.get("gateway_url") or os.environ.get("GATEWAY_URL", "http://localhost:8080")
    gateway_api_key = body.get("gateway_api_key") or os.environ.get("MMINI_API_KEY", "")

    task_id = (body.get("task_id") or "task")[:16]
    # Prefer the gateway-supplied job_id (a layman label like
    # "replay-37baaa19-7604387" or "adhoc-open-calculator-7604387"). Falls
    # back to a generic stamp if missing for direct callers of /run.
    job_id = body.get("job_id") or f"run-{task_id}-{int(time.time())}"
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    task_dir = work_dir / "task"
    task_dir.mkdir(exist_ok=True)
    write_task_dir(task_dir, task, platform=platform)
    job_yaml = work_dir / "job.yaml"
    write_job_yaml(
        job_yaml,
        work_dir / "jobs",
        agent_spec_for(model, max_steps, platform=platform),
        gateway_url,
        platform=platform,
    )
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
    # If harbor finished, hoist the inner <jobs/<ts>/<trial>> dir up to
    # <work_dir>/<trial> so harbor-viewer's --jobs mode can render it.
    flat_trial = None
    if rec.task.done():
        flat_trial = flatten_trial_dir(rec.work_dir)
    trial_dir = flat_trial or (find_trial_dir(jobs_dir) if jobs_dir.exists() else None)
    sandbox_id = peek_sandbox_id(trial_dir) if trial_dir else None

    # Build the harbor-viewer URL as a RELATIVE path from JOBS_DIR. The
    # trial path becomes /<job_id>/<trial> after flatten.
    trial_rel_path = None
    if trial_dir:
        try:
            trial_rel_path = str(trial_dir.relative_to(JOBS_DIR))
        except ValueError:
            pass

    n_steps, n_actions = read_step_counts(trial_dir)

    if not rec.task.done():
        return web.json_response({
            "job_id": job_id,
            "status": "running",
            "sandbox_id": sandbox_id,
            "trial_path": trial_rel_path,
            "n_steps": n_steps,
            "n_actions": n_actions,
        })

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

    # rglob from work_dir picks up result.json whether it's still nested in
    # jobs/<ts>/<trial>/ or already flattened to <trial>/.
    reward, result_json = read_reward(rec.work_dir)
    status = "completed" if rec.returncode == 0 else "failed"

    # Surface uploaded-file artifacts (originals + post-run snapshots) so
    # direct callers can fetch them without scraping the trial dir.
    artifacts = None
    if trial_dir:
        manifest_path = trial_dir / "artifacts" / "uploaded_files.json"
        if manifest_path.exists():
            try:
                import json as _json
                artifacts = _json.loads(manifest_path.read_text())
            except Exception:
                artifacts = None

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
        "artifacts": artifacts,
    })


async def handle_health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


def make_app() -> web.Application:
    # /run carries file uploads as base64 in the manifest, so the default
    # 1 MB body cap rejects PowerPoints/screenshots. Bump to 1 GB —
    # runner is a same-host sidecar, no external traffic.
    app = web.Application(client_max_size=1024 * 1024 * 1024)
    app.router.add_post("/run", handle_run)
    app.router.add_get("/jobs/{job_id}", handle_get_job)
    app.router.add_get("/health", handle_health)
    return app
