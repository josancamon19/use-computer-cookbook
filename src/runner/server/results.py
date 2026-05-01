"""Read reward + step counts off harbor's on-disk output."""
from __future__ import annotations

import json
from pathlib import Path


def read_reward(jobs_dir: Path) -> tuple[float | None, dict | None]:
    """Return (reward, raw_result_json). Harbor writes per-trial result.json
    with verifier_result.rewards.reward. Globs the whole jobs dir so this
    works regardless of the harbor-job subdir name."""
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
