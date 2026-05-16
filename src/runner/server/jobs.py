"""Background job lifecycle: run harbor, flatten output, peek sandbox id."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from runner.server.config import RUNNER_DIR, SANDBOX_RE


@dataclass
class JobRec:
    job_id: str
    work_dir: Path
    task: asyncio.Task
    env: dict[str, str] = field(default_factory=dict)
    analyze: bool = False
    analysis_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    returncode: int | None = None
    error: str | None = None


# Process-global registry — adequate while the sidecar is single-process.
JOBS: dict[str, JobRec] = {}


async def run_harbor(rec: JobRec, task_dir: Path, job_yaml: Path, env: dict) -> None:
    """Spawn `harbor run -c <yaml> -p <task_dir>` and wait."""
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


def flatten_trial_dir(work_dir: Path) -> Path | None:
    """Harbor writes <work_dir>/jobs/<ts>/<trial>/. harbor-viewer's --jobs mode
    expects <JOBS_DIR>/<job-id>/<trial>/ + a job-level result.json at the
    same root. Hoist both up so the viewer recognises the run as a job and
    the trial as its child.

    Returns the new trial dir, or None if nothing to flatten."""
    flat = find_flat_trial_dir(work_dir)
    if flat is not None:
        return flat

    inner_jobs = work_dir / "jobs"
    if not inner_jobs.exists():
        return None
    moved_trial: Path | None = None
    for ts_dir in sorted(inner_jobs.iterdir()):
        if not ts_dir.is_dir():
            continue
        # Hoist the job-level result.json so harbor-viewer lists this run.
        job_result = ts_dir / "result.json"
        if job_result.exists() and not (work_dir / "result.json").exists():
            try:
                (work_dir / "result.json").write_bytes(job_result.read_bytes())
            except Exception:
                pass
        for trial in sorted(ts_dir.iterdir()):
            if not trial.is_dir():
                continue
            dest = work_dir / trial.name
            if dest.exists():
                # Already moved on a prior call (idempotent).
                return dest
            trial.rename(dest)
            moved_trial = dest
            break
        break
    return moved_trial


def find_flat_trial_dir(work_dir: Path) -> Path | None:
    for entry in sorted(work_dir.iterdir()):
        if not entry.is_dir() or entry.name in {"task", "jobs", "artifacts"}:
            continue
        if (entry / "result.json").exists():
            return entry
    return None


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
    """Scan the start of trial.log for the first sb-<hex> match."""
    log = trial_dir / "trial.log"
    if not log.exists():
        return None
    try:
        head = log.read_text(errors="replace")[:4000]
    except Exception:
        return None
    m = SANDBOX_RE.search(head)
    return m.group(1) if m else None
