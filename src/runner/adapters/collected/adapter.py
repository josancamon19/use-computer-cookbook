"""Gateway collected tasks → Harbor format.

The SDK's `tasks.export_harbor` does all the heavy lifting: writes
`instruction.md`, `task.toml`, `tests/test.sh`, `tests/setup/pre_command.sh`,
`tests/setup/files/...`, and `actions.json` (for debug-agent replay).

Library use:

    with CollectedTasksAdapter(gateway_url, api_key) as a:
        a.export_task("col-...", "datasets/collected/macos")

CLI (writes under datasets/collected/<platform>/):

    uv run python -m runner.adapters.collected.adapter --task col-...
    uv run python -m runner.adapters.collected.adapter --all --platform macos
    uv run python -m runner.adapters.collected.adapter --all   # both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from mmini import Mmini, TaskSummary

RUNNER_DIR = Path(__file__).resolve().parents[4]
COLLECTED_ROOT = RUNNER_DIR / "datasets" / "collected"
CONFIG_REL = "src/runner/configs/job-collected.yaml"


class CollectedTasksAdapter:
    def __init__(
        self,
        gateway_url: str = "https://api.dev.use.computer",
        api_key: str = "",
    ):
        self._client = Mmini(api_key=api_key or None, base_url=gateway_url)

    def list_tasks(
        self, limit: int = 200, platform: Optional[str] = None
    ) -> List[TaskSummary]:
        tasks = self._client.tasks.list(limit=limit)
        if platform:
            tasks = [t for t in tasks if t.platform == platform]
        return tasks

    def list_runnable(self, limit: int = 200) -> List[TaskSummary]:
        return [t for t in self.list_tasks(limit=limit) if t.runnable]

    def export_task(
        self, task_id: str, output_dir: str | Path, *, overwrite: bool = False
    ) -> Path:
        return self._client.tasks.export_harbor(
            task_id, output_dir, overwrite=overwrite
        )

    def export_all(
        self,
        output_dir: str | Path,
        *,
        platform: Optional[str] = None,
        runnable_only: bool = False,
        overwrite: bool = False,
        limit: int = 200,
    ) -> tuple[list[Path], list[tuple[str, str]]]:
        """Returns (success_paths, [(task_id, error_msg), ...])."""
        tasks = self.list_tasks(limit=limit, platform=platform)
        if runnable_only:
            tasks = [t for t in tasks if t.runnable]

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        successes: list[Path] = []
        failures: list[tuple[str, str]] = []

        for i, summary in enumerate(tasks, 1):
            try:
                path = self.export_task(summary.id, output, overwrite=overwrite)
                status = "runnable" if summary.runnable else "NO GRADER"
                print(f"[{i}/{len(tasks)}] {status:>10}  {summary.id}  →  {path.name}")
                successes.append(path)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(tasks)}]      FAIL  {summary.id}: {e}")
                failures.append((summary.id, str(e)))

        print(f"\nExported {len(successes)}/{len(tasks)} tasks to {output}")
        not_runnable = sum(1 for t in tasks if not t.runnable)
        if not_runnable:
            print(f"  {not_runnable} tasks have no grader (reward will default to 0)")
        if failures:
            print(f"  {len(failures)} failures")

        return successes, failures

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main() -> int:
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
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task dirs instead of erroring",
    )
    args = ap.parse_args()

    if not args.api_key:
        print("error: no $MMINI_API_KEY and no --api-key given", file=sys.stderr)
        return 2

    out_root = Path(args.out).resolve()

    with CollectedTasksAdapter(args.gateway, args.api_key) as a:
        if args.task:
            # Look up the task to learn its platform so we can route to the
            # right datasets/collected/<platform>/ subdir.
            task = a._client.tasks.get(args.task)
            platform_dir = out_root / task.platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            path = a.export_task(args.task, platform_dir, overwrite=args.overwrite)
            print(f"wrote {path}")
            return 0

        platforms = ["macos", "ios"] if args.platform == "all" else [args.platform]
        total = 0
        for p in platforms:
            print(f"\n=== {p} ===")
            p_out = out_root / p
            p_out.mkdir(parents=True, exist_ok=True)
            successes, _ = a.export_all(
                p_out, platform=p, overwrite=args.overwrite
            )
            total += len(successes)

        if total == 0:
            return 1

        print(f"\nRun any of them with:")
        print(f"    cd {RUNNER_DIR}")
        print(f"    uv run harbor run -c {CONFIG_REL} -p datasets/collected/<platform>/<task>")
        return 0


if __name__ == "__main__":
    sys.exit(main())
