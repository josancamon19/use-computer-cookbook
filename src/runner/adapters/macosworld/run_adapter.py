from __future__ import annotations

import argparse
from pathlib import Path

from runner.adapters.macosworld.adapter import MacOSWorldToHarbor


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert macOSWorld tasks to Harbor format"
    )

    ap.add_argument(
        "--macosworld-root",
        type=Path,
        required=True,
        help="Path to macOSWorld repo root",
    )
    ap.add_argument(
        "--task-dir", type=Path, required=True, help="Output Harbor tasks directory"
    )
    ap.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to a category (e.g. sys_apps)",
    )
    ap.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Single task UUID (requires --category)",
    )
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="Only tasks that don't need App Store apps",
    )
    ap.add_argument(
        "--ready-only",
        action="store_true",
        default=True,
        help="Only tasks runnable without extra data or apps (default)",
    )
    ap.add_argument(
        "--all", action="store_true", help="Export all tasks (overrides --ready-only)"
    )
    ap.add_argument(
        "--timeout", type=float, default=1800.0, help="Agent timeout seconds"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing task dirs"
    )
    ap.add_argument("--limit", type=int, default=None, help="Max tasks to convert")

    args = ap.parse_args()

    conv = MacOSWorldToHarbor(
        macosworld_root=args.macosworld_root,
        harbor_tasks_root=args.task_dir,
        max_timeout_sec=args.timeout,
    )

    if args.task_id:
        if not args.category:
            ap.error("--task-id requires --category")
        out = conv.generate_task(args.category, args.task_id, overwrite=args.overwrite)
        print(f"Harbor task created at: {out}")
        return

    ready_only = args.ready_only and not getattr(args, "all", False)
    ids = conv.get_all_ids(base_only=args.base_only, ready_only=ready_only)
    if args.category:
        ids = [(c, t) for c, t in ids if c == args.category]
    if args.limit:
        ids = ids[: args.limit]

    print(f"Converting {len(ids)} macOSWorld tasks into {args.task_dir} ...")
    ok, bad = conv.generate_many(ids, overwrite=args.overwrite)
    print(f"Done. Success: {len(ok)}  Failures: {len(bad)}")


if __name__ == "__main__":
    main()
