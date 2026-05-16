"""Harbor analyze lifecycle for runner jobs."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runner.server.config import RUNNER_DIR

ANALYSIS_MODEL = "haiku"
ANALYSIS_STATUS = "analysis.status.json"
ANALYSIS_OUTPUT = "analysis.json"
ANALYSIS_LOG = "analysis.log"
ANALYSIS_PROMPT = Path(__file__).with_name("analyze_prompt.txt")
TERMINAL = {"completed", "failed", "skipped"}


def analysis_status_from_disk(trial_dir: Path | None) -> dict[str, Any]:
    if trial_dir is None:
        return {"status": "pending", "model": ANALYSIS_MODEL}

    status = _read_status(trial_dir) or {"status": "pending", "model": ANALYSIS_MODEL}
    output = trial_dir / ANALYSIS_OUTPUT
    if output.exists():
        try:
            data = json.loads(output.read_text())
        except Exception:
            data = {}
        status.update({
            "status": "completed",
            "model": status.get("model") or ANALYSIS_MODEL,
            "trial_name": data.get("trial_name") or trial_dir.name,
            "summary": data.get("summary") or "",
            "checks": data.get("checks") or {},
        })
    return status


def analysis_terminal(status: dict[str, Any] | None) -> bool:
    return bool(status and status.get("status") in TERMINAL)


def ensure_analysis_started(rec: Any, trial_dir: Path) -> None:
    status = analysis_status_from_disk(trial_dir)
    if analysis_terminal(status):
        return
    task = getattr(rec, "analysis_task", None)
    if task is not None and not task.done():
        return
    rec.analysis_task = asyncio.create_task(run_harbor_analyze(trial_dir, rec.env))


async def run_harbor_analyze(trial_dir: Path, env: dict[str, str]) -> None:
    started_at = _now()
    _write_status(trial_dir, {
        "status": "running",
        "model": ANALYSIS_MODEL,
        "started_at": started_at,
    })
    log_path = trial_dir / ANALYSIS_LOG
    output_path = trial_dir / ANALYSIS_OUTPUT
    with log_path.open("w") as lf:
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "harbor", "analyze",
            str(trial_dir),
            "-m", ANALYSIS_MODEL,
            "-p", str(ANALYSIS_PROMPT),
            "-o", str(output_path),
            stdout=lf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(RUNNER_DIR),
            env=env,
        )
        rc = await proc.wait()

    if rc == 0 and output_path.exists():
        status = analysis_status_from_disk(trial_dir)
        status.update({
            "status": "completed",
            "model": ANALYSIS_MODEL,
            "started_at": started_at,
            "finished_at": _now(),
        })
        _write_status(trial_dir, status)
        return

    _write_status(trial_dir, {
        "status": "failed",
        "model": ANALYSIS_MODEL,
        "started_at": started_at,
        "finished_at": _now(),
        "error": _tail(log_path),
    })


def _read_status(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / ANALYSIS_STATUS
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_status(trial_dir: Path, status: dict[str, Any]) -> None:
    status.setdefault("model", ANALYSIS_MODEL)
    (trial_dir / ANALYSIS_STATUS).write_text(json.dumps(status, indent=2))


def _tail(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        return str(e)
    return text[-max_chars:]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
