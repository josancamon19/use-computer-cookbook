from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNNER_DIR))
from server import write_task_dir  # noqa: E402

ADHOC_ROOT = RUNNER_DIR / "datasets" / "adhoc" / "ios"
DEFAULT_JSON = Path(__file__).resolve().parent / "tasks.json"
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


def materialize(spec: dict, out_root: Path) -> list[Path]:
    s = _normalize(spec)
    default_device = (s.get("device_type") or DEFAULT_DEVICE_TYPE).strip()
    default_runtime = (s.get("runtime") or DEFAULT_RUNTIME).strip()
    tasks = s.get("tasks") or []
    if not tasks:
        raise ValueError("no tasks provided")

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
        write_task_dir(
            out_dir,
            {
                "instruction": instruction,
                "pre_command": "",
                "grading_command": [],  # → "Score: 0" stub, no verifier
                "device_type": (t.get("device_type") or default_device).strip(),
                "runtime": (t.get("runtime") or default_runtime).strip(),
            },
            platform="ios",
        )
        written.append(out_dir)
        print(f"  wrote {out_dir.relative_to(RUNNER_DIR)}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Export adhoc iOS tasks from a JSON list")
    ap.add_argument(
        "json",
        nargs="?",
        default=str(DEFAULT_JSON),
        help=f"Path to tasks JSON (default: {DEFAULT_JSON.relative_to(RUNNER_DIR)})",
    )
    ap.add_argument(
        "--clean", action="store_true", help="Wipe datasets/adhoc/ios first"
    )
    args = ap.parse_args()

    spec = json.loads(Path(args.json).read_text())

    if args.clean and ADHOC_ROOT.exists():
        print(f"wiping {ADHOC_ROOT.relative_to(RUNNER_DIR)}")
        shutil.rmtree(ADHOC_ROOT)

    written = materialize(spec, ADHOC_ROOT)
    print(f"\nexported {len(written)} task(s) to {ADHOC_ROOT.relative_to(RUNNER_DIR)}")
    print("\nrun with:")
    print(
        f"  uv run harbor run -c src/runner/configs/job-collected.yaml -p {ADHOC_ROOT.relative_to(RUNNER_DIR)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
