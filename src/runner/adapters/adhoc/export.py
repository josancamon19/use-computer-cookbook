from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from runner.server.task_dir import write_task_dir

RUNNER_DIR = Path(__file__).resolve().parents[4]
ADHOC_BASE = RUNNER_DIR / "datasets" / "adhoc"
DEFAULT_JSON = Path(__file__).resolve().parent / "tasks" / "ios.json"
DEFAULT_PLATFORM = "ios"
DEFAULT_DEVICE_TYPE = "iPhone-17-Pro"
DEFAULT_RUNTIME = "iOS-26-4"


def _slug(text: str, fallback: str, max_words: int = 5) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text or "")
    s = "-".join(words[:max_words]).lower()
    return s or fallback


def _normalize(spec: object) -> dict:
    """Accept either a bare list or an object — return {device_type, runtime, tasks}."""
    if isinstance(spec, list):
        return {"tasks": spec}
    if isinstance(spec, dict):
        return spec
    raise ValueError("input must be a JSON list or object")


def materialize(spec: dict, base_root: Path) -> list[Path]:
    s = _normalize(spec)
    platform = (s.get("platform") or DEFAULT_PLATFORM).strip().lower()
    if platform not in ("ios", "macos"):
        raise ValueError(f"unknown platform: {platform!r} (want 'ios' or 'macos')")
    # iOS pins a specific simulator. macOS picks any warm VM at run time, no
    # device_type/runtime needed.
    default_device = (s.get("device_type") or DEFAULT_DEVICE_TYPE).strip()
    default_runtime = (s.get("runtime") or DEFAULT_RUNTIME).strip()
    tasks = s.get("tasks") or []
    if not tasks:
        raise ValueError("no tasks provided")

    out_root = base_root / platform
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    used_names: set[str] = set()
    for i, raw in enumerate(tasks):
        if isinstance(raw, str):
            t = {"instruction": raw}
        elif isinstance(raw, dict):
            t = raw
        else:
            raise ValueError(
                f"task #{i}: must be a string or object, got {type(raw).__name__}"
            )

        instruction = (t.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"task #{i}: missing instruction")

        name = _slug(t.get("name") or instruction, fallback=f"task-{i:03d}")
        # Disambiguate name collisions deterministically.
        base, n = name, 2
        while name in used_names:
            name = f"{base}-{n}"
            n += 1
        used_names.add(name)

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        task_payload: dict = {
            "instruction": instruction,
            "pre_command": "",
            "grading_command": [],  # → "Score: 0" stub, no verifier
        }
        if platform == "ios":
            task_payload["device_type"] = (t.get("device_type") or default_device).strip()
            task_payload["runtime"] = (t.get("runtime") or default_runtime).strip()
        write_task_dir(out_dir, task_payload, platform=platform)
        written.append(out_dir)
        print(f"  wrote {out_dir.relative_to(RUNNER_DIR)}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export adhoc tasks (iOS or macOS) from a JSON list"
    )
    ap.add_argument(
        "json",
        nargs="?",
        default=str(DEFAULT_JSON),
        help=f"Path to tasks JSON (default: {DEFAULT_JSON.relative_to(RUNNER_DIR)})",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the platform's datasets/adhoc/<platform>/ dir first",
    )
    args = ap.parse_args()

    spec = json.loads(Path(args.json).read_text())
    platform = (spec.get("platform") if isinstance(spec, dict) else None) or DEFAULT_PLATFORM
    plat_root = ADHOC_BASE / platform.strip().lower()

    if args.clean and plat_root.exists():
        print(f"wiping {plat_root.relative_to(RUNNER_DIR)}")
        shutil.rmtree(plat_root)

    written = materialize(spec, ADHOC_BASE)
    print(f"\nexported {len(written)} task(s) to {plat_root.relative_to(RUNNER_DIR)}")
    print("\nrun with:")
    cfg = (
        "src/runner/configs/job-collected-ios.yaml"
        if platform.strip().lower() == "ios"
        else "src/runner/configs/job-collected.yaml"
    )
    print(f"  uv run harbor run -c {cfg} -p {plat_root.relative_to(RUNNER_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
