"""Export collected tasks from the gateway as Harbor task directories.

Output: runner/datasets/collected/<platform>/<task-name>/ — same layout as
macosworld_ready (instruction.md, task.toml, tests/test.sh, tests/setup/
pre_command.sh) so harbor can run them with no extra config.

    # one task by id (any platform)
    uv run python collected/export.py --task col-...

    # all collected tasks of a platform
    uv run python collected/export.py --all --platform macos
    uv run python collected/export.py --all --platform ios
    uv run python collected/export.py --all                 # both

Env: MMINI_API_KEY / MMINI_GATEWAY_URL (or pass --api-key / --gateway).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx

# server.py lives one level up — reuse its task-dir helper verbatim so the
# export can never drift from what the production HTTP sidecar produces.
RUNNER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNNER_DIR))
from server import write_task_dir  # noqa: E402

COLLECTED_ROOT = RUNNER_DIR / "datasets" / "collected"
CONFIG_REL = "src/runner/configs/job-collected.yaml"


def _safe_name(name: str, fallback: str) -> str:
    """Filesystem-safe slug for the export directory name."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "").strip()).strip("-")
    return cleaned or fallback


async def fetch_task_list(
    http: httpx.AsyncClient, gateway: str, platform: str | None
) -> list[dict]:
    """Pull /admin/tasks, optionally filtered by platform."""
    params = {}
    if platform and platform != "all":
        params["platform"] = platform
    r = await http.get(f"{gateway.rstrip('/')}/admin/tasks", params=params)
    r.raise_for_status()
    rows = r.json()
    # Older gateways may ignore ?platform — filter client-side as a backstop.
    if platform and platform != "all":
        rows = [t for t in rows if (t.get("platform") or "macos") == platform]
    return rows


async def fetch_task_detail(
    http: httpx.AsyncClient, gateway: str, task_id: str
) -> dict:
    r = await http.get(f"{gateway.rstrip('/')}/admin/tasks/{task_id}")
    r.raise_for_status()
    return r.json()


async def fetch_trajectory(
    http: httpx.AsyncClient, gateway: str, task_id: str
) -> dict | None:
    """Fetch the full trajectory (every recorded step's tool_calls + screenshot
    refs). Used by debug_cua's replay mode. Returns None on 404 / failure so a
    missing trajectory doesn't block the export."""
    r = await http.get(f"{gateway.rstrip('/')}/admin/tasks/{task_id}/trajectory")
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def trajectory_to_actions(traj: dict) -> list[dict]:
    """Flatten ATIF trajectory steps into a plain list of {function, args}
    dicts the debug agent can replay verbatim. Drops screenshots and the
    initial user message — only the agent-source actions matter for replay."""
    out: list[dict] = []
    for step in traj.get("steps") or []:
        if step.get("source") == "user":
            continue
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function") or ""
            args = tc.get("args") or {}
            if fn:
                out.append({"function": fn, "args": args})
    return out


def task_to_runner_dict(t: dict) -> dict:
    """Translate gateway /admin/tasks/<id> JSON into write_task_dir's input shape."""
    meta = t.get("task_meta") or {}
    instruction = t.get("instruction") or meta.get("instruction") or ""
    grader = (t.get("grader") or "").strip()
    platform = t.get("platform") or meta.get("platform") or "macos"
    # iOS sims have no shell — pre_commands are unrunnable. macOS keeps them.
    pre_command = "" if platform == "ios" else (t.get("pre_command") or "").strip()
    return {
        "instruction": instruction,
        "pre_command": pre_command,
        "grading_command": [[grader, 100]] if grader else [],
        "platform": platform,
        # iOS-only: the device pin lands in task.toml's [ios] block.
        "device_type": t.get("device_type") or "",
        "runtime": t.get("runtime") or "",
    }


def export_one(
    task: dict, out_root: Path, trajectory: dict | None = None
) -> tuple[Path, bool]:
    """Materialize one task. Returns (out_dir, has_grader)."""
    task_id = task["id"]
    runner_task = task_to_runner_dict(task)
    platform = runner_task["platform"]
    if platform not in ("macos", "ios"):
        raise ValueError(f"{task_id}: unsupported platform {platform!r}")

    # `task_name` is flat on the list endpoint, nested under task_meta on
    # the detail endpoint. Try both.
    name_hint = (
        task.get("task_name")
        or (task.get("task_meta") or {}).get("name")
        or ""
    )
    name = _safe_name(name_hint, fallback=task_id[:24])
    out_dir = out_root / platform / name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_task_dir(out_dir, runner_task, platform=platform)

    # Bundle the recorded action list so debug_cua's replay mode can walk it
    # without re-fetching. iOS-only consumer today; harmless on macOS.
    if trajectory is not None:
        actions = trajectory_to_actions(trajectory)
        if actions:
            (out_dir / "actions.json").write_text(
                json.dumps({"steps": actions}, indent=2)
            )

    return out_dir, bool(runner_task["grading_command"])


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", "-t", help="Single task id (col-...)")
    src.add_argument("--all", action="store_true", help="Export every matching task")
    ap.add_argument(
        "--platform",
        choices=["macos", "ios", "all"],
        default="all",
        help="With --all: filter by platform (default: all)",
    )
    ap.add_argument(
        "--gateway",
        default=os.environ.get("MMINI_GATEWAY_URL", "http://10.10.10.2:8081"),
        help="Gateway base URL (default: dev)",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("MMINI_API_KEY", ""),
        help="Bearer token (default: $MMINI_API_KEY)",
    )
    ap.add_argument(
        "--out",
        default=str(COLLECTED_ROOT),
        help=f"Dataset root (default: {COLLECTED_ROOT})",
    )
    args = ap.parse_args()

    if not args.api_key:
        print("error: no $MMINI_API_KEY and no --api-key given", file=sys.stderr)
        return 2

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient(headers=headers, timeout=30) as http:
        if args.task:
            print(f"fetching {args.task} from {args.gateway} ...")
            tasks = [await fetch_task_detail(http, args.gateway, args.task)]
        else:
            print(f"listing {args.platform} tasks from {args.gateway} ...")
            summaries = await fetch_task_list(http, args.gateway, args.platform)
            print(f"  {len(summaries)} matching")
            tasks = []
            for s in summaries:
                tasks.append(await fetch_task_detail(http, args.gateway, s["id"]))
        trajectories: dict[str, dict | None] = {}
        for t in tasks:
            trajectories[t["id"]] = await fetch_trajectory(
                http, args.gateway, t["id"]
            )

    written: list[Path] = []
    no_grader: list[str] = []
    skipped: list[tuple[str, str]] = []
    for t in tasks:
        try:
            out, has_grader = export_one(t, out_root, trajectory=trajectories.get(t["id"]))
            written.append(out)
            try:
                rel = out.relative_to(RUNNER_DIR)
            except ValueError:
                rel = out
            print(f"  wrote {rel}  grader={'yes' if has_grader else 'NO'}")
            if not has_grader:
                no_grader.append(out.name)
        except Exception as e:  # noqa: BLE001
            skipped.append((t.get("id", "?"), str(e)))
            print(f"  skipped {t.get('id', '?')}: {e}", file=sys.stderr)

    if not written:
        return 1

    print()
    print(f"exported {len(written)} task(s) to {out_root}")
    if no_grader:
        print()
        print(
            f"  warning: {len(no_grader)} task(s) have no grader yet — "
            "their reward will default to 0 at run time:"
        )
        for n in no_grader:
            print(f"    - {n}")
    if skipped:
        print(f"  ({len(skipped)} skipped)")
    print()
    print("Run any of them with:")
    sample = written[0]
    try:
        rel = sample.relative_to(RUNNER_DIR)
    except ValueError:
        rel = sample
    print(f"    cd {RUNNER_DIR}")
    print(f"    uv run harbor run -c {CONFIG_REL} -p {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
