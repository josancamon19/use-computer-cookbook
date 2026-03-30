"""
Convert macOSWorld benchmark tasks into Harbor task directories.

Reads task JSON files from the macOSWorld repo's tasks/ directory
and produces one Harbor task directory per task.

When a task references Benchmark_Backup files, the needed files are
copied into the task's tests/setup/files/ directory and the pre_command
is rewritten to use a VM-local staging path (/tmp/harbor/task_files/).

Known issues with specific tasks on our VMs:
- Contacts tasks: osascript data queries hang (iCloud sync blocks on fresh VMs).
  Grading fails, tasks themselves are runnable. Affected tasks:
    sys_apps/89f88c8c, sys_apps/b071a2dc, sys_apps/64824755,
    sys_apps/48cf0af3, safety/9d73393d
- Reminders tasks: same iCloud sync hang on osascript data queries.
    sys_apps/e8bc0bd1, sys_apps/2f72dba5, sys_apps/b431b94b,
    safety/e7491382, sys_apps/a204124e, sys_apps/4a89fe83
- Keystroke task: osascript not allowed to send keystrokes (accessibility TCC).
    sys_and_interface/1df5d9f4
- Backtick task: literal backtick in macOSWorld pre_command data, bad source data.
    sys_and_interface/17fed363
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Benchmark_Backup items that a task can reference (keyword → actual filename)
_BACKUP_ITEMS = {
    "benchmark_files": "benchmark_files",
    "BenchmarkApp": "BenchmarkApp",
    "com.apple.dock.plist": "com.apple.dock.plist",
    "iMovie": "iMovie Library.imovielibrary",
}


@dataclass
class MacOSWorldTask:
    task_id: str
    category: str
    instruction: str
    pre_command: str | dict = ""
    grading_command: list = field(default_factory=list)
    before_action_delay_seconds: int = 10
    before_grading_delay_seconds: int = 30
    # Parsed but unused by us — kept for filtering
    snapshot: dict = field(default_factory=dict)
    in_process: list | None = None

    @classmethod
    def from_json(cls, path: Path, category: str) -> "MacOSWorldTask":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            task_id=data["id"],
            category=category,
            instruction=data.get("task", {}).get("en", ""),
            pre_command=data.get("pre_command", ""),
            grading_command=data.get("grading_command", []),
            before_action_delay_seconds=data.get("before_action_delay_seconds", 10),
            before_grading_delay_seconds=data.get("before_grading_delay_seconds", 30),
            snapshot=data.get("snapshot", {}),
            in_process=data.get("in_process"),
        )

    @property
    def needs_apps(self) -> bool:
        return any("usedApps" in v for v in self.snapshot.values())

    @property
    def needs_benchmark_backup(self) -> bool:
        return "Benchmark_Backup" in json.dumps(self.pre_command)


class MacOSWorldLoader:
    """Load macOSWorld tasks from tasks/ directory."""

    CATEGORIES = [
        "advanced",
        "file_management",
        "media",
        "multi_apps",
        "productivity",
        "safety",
        "sys_and_interface",
        "sys_apps",
    ]

    def __init__(self, macosworld_root: Path) -> None:
        self.root = Path(macosworld_root)
        self.tasks_dir = self.root / "tasks"
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"tasks/ not found at {self.tasks_dir}")

    def all_task_ids(
        self,
        category: Optional[str] = None,
        base_only: bool = False,
        ready_only: bool = False,
    ) -> List[Tuple[str, str]]:
        pairs = []
        for cat_dir in sorted(self.tasks_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            if category and cat_dir.name != category:
                continue
            for json_file in sorted(cat_dir.glob("*.json")):
                task = MacOSWorldTask.from_json(json_file, cat_dir.name)
                if (base_only or ready_only) and task.needs_apps:
                    continue
                if task.task_id in _EXCLUDED_TASK_IDS:
                    continue
                pairs.append((cat_dir.name, json_file.stem))
        return pairs

    def load_task(self, category: str, task_id: str) -> MacOSWorldTask:
        path = self.tasks_dir / category / f"{task_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Task not found: {path}")
        return MacOSWorldTask.from_json(path, category)

    def task_json_path(self, category: str, task_id: str) -> Path:
        return self.tasks_dir / category / f"{task_id}.json"

    def total_tasks(self, base_only: bool = False) -> int:
        return len(self.all_task_ids(base_only=base_only))


def _build_test_sh(grading_cmds: list) -> str:
    """Generate test.sh with grading commands inlined."""
    lines = [
        "#!/bin/bash",
        "# Auto-generated grading script",
        "",
        '# Detect path prefix (macOS SIP blocks /tests, so mmini uses /tmp/harbor)',
        'PREFIX=""',
        '[ -d "/tmp/harbor/logs" ] && PREFIX="/tmp/harbor"',
        'REWARD="${PREFIX}/logs/verifier/reward.txt"',
        "",
    ]

    # before_grading_delay intentionally skipped — the runner handles delays externally

    if not grading_cmds:
        lines.append('echo "0" > "$REWARD"')
        lines.append('echo "Score: 0"')
    else:
        # Only check commands with score 100 (full pass)
        checks = [cmd for cmd, score in grading_cmds if score == 100]
        if checks:
            for i, cmd in enumerate(checks):
                escaped = cmd.replace("'", "'\\''")
                lines.append(f"# Check {i + 1}")
                lines.append(f"if bash -c '{escaped}' 2>/dev/null | grep -qi 'true'; then")
                lines.append('  echo "1" > "$REWARD"')
                lines.append('  echo "Score: 1"')
                lines.append("  exit 0")
                lines.append("fi")
                lines.append("")
            lines.append('echo "0" > "$REWARD"')
            lines.append('echo "Score: 0"')
        else:
            lines.append('echo "0" > "$REWARD"')
            lines.append('echo "Score: 0"')

    return "\n".join(lines) + "\n"


def _read_template(template_dir: Path, name: str) -> str:
    return (template_dir / name).read_text(encoding="utf-8")


def _render(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


# Tasks excluded from export — broken on our VM image.
# Each entry: task UUID → reason (what needs to change to unblock it).
_EXCLUDED_TASK_IDS: dict[str, str] = {
    # Keystroke pre_command: TCC accessibility permission not granted.
    # Fix: grant "System Events" accessibility in base image via TCC.db
    "1df5d9f4-5b69-5e98-a8ab-d415dae9ca83": "pre_command keystroke blocked by TCC",
    # Grader hangs: Contacts osascript hangs on fresh VM (never writes reward file).
    # Fix: investigate why osascript queries to Contacts hang — possibly needs first-launch
    "89f88c8c-2c64-91a0-444f-fa71315cd92c": "grader: Contacts osascript hangs",
    "b071a2dc-6b16-5e76-16a8-72803d557495": "grader: Contacts osascript hangs",
    "64824755-362d-64b6-fcac-e7c5348a09f0": "grader: Contacts osascript hangs",
    "9d73393d-14a3-1259-70fb-32aae63b3b80": "grader: Contacts osascript hangs",
    "a204124e-c06a-08b1-13fa-5e61f689180c": "grader: Contacts osascript hangs",
    # Grader hangs: Reminders osascript hangs on fresh VM (never writes reward file).
    # Fix: investigate why osascript queries to Reminders hang
    "e8bc0bd1-1618-ea00-345b-589b1fef3f99": "grader: Reminders osascript hangs",
    "b431b94b-193c-0478-1e26-709f7d3aa453": "grader: Reminders osascript hangs",
    "2f72dba5-c364-d21b-7040-684f4d6d40b7": "grader: Reminders osascript hangs",
    "48cf0af3-0612-dbcd-14da-d5202eed6ce9": "grader: Reminders osascript hangs",
    "4a89fe83-8a93-ecd4-7841-c0bb470571c2": "grader: Reminders osascript hangs",
    "e7491382-6e93-288d-9d67-24369c382e86": "grader: Reminders osascript hangs",
    # Grader hangs: Media tasks — QuickTime/Preview osascript queries timeout.
    # Fix: investigate why these specific osascript queries hang
    "1ea85063-355f-bfad-3de7-3ab038261d62": "grader: QuickTime osascript hangs",
    "e2a6a1ac-e352-6680-00ae-d191a817d143": "grader: media osascript hangs",
    "efd1a2f9-b0b5-f14b-1fac-70a7f8e8a9e7": "grader: media osascript hangs",
    # Grader hangs: multi_apps tasks — grader osascript doesn't write reward file.
    # Fix: investigate grader commands
    "48a16605-74e7-c9d8-01d7-3001bc0b0c1d": "grader: osascript hangs, no reward file",
    "f0147143-27e5-e942-cd8c-ab723dc6e490": "grader: osascript hangs, no reward file",
    # Non-deterministic failures (~5-10% flake rate across pass@50).
    # Fail on random Minis with RewardFileNotFoundError or RuntimeError.
    # Grader osascript sometimes finishes in time, sometimes doesn't.
    # productivity__6a40e62d: 45/50 ok — 3 RewardFileNotFound + 2 RuntimeError
    "6a40e62d-c990-7d55-81fb-f2ced53664b7": "flaky grader: 90% pass rate, osascript timeout",
    # safety__7f81662d: 47/50 ok — 3 RewardFileNotFound
    "7f81662d-5c57-9078-b0c4-e1af24d63304": "flaky grader: 94% pass rate, osascript timeout",
}


# Per-task pre_command fixes for our VM image.
# These run BEFORE the task's own pre_command to ensure the correct initial state.
_PRE_COMMAND_FIXES: dict[str, str] = {
    # Dark mode tasks: VM ships with dark mode, set light mode first
    "eb346395-b8fe-03bc-a6e5-a58719b1edce": "defaults delete -g AppleInterfaceStyle 2>/dev/null || true",
    "ce71ae98-6947-6c18-87ac-cdecb1750e5a": "defaults delete -g AppleInterfaceStyle 2>/dev/null || true",
}


# Keys that don't exist on fresh VMs — defaults delete fails with "Domain not found".
# The clean VM state is already the desired state for these.
_SKIP_DEFAULTS_DELETE = [
    "defaults delete -g KeyRepeat",
    "defaults delete -g InitialKeyRepeat",
    "defaults delete -g AppleAccentColor",
    "defaults delete com.apple.universalaccess hoverTextEnabled",
    "defaults delete com.apple.universalaccess hoverTypingEnabled",
    "defaults delete com.apple.universalaccess closeViewHotkeysEnabled",
]


def _skip_defaults_delete(cmd: str) -> bool:
    """Skip defaults delete for keys that don't exist on fresh VMs."""
    stripped = cmd.strip()
    return any(stripped.startswith(s) for s in _SKIP_DEFAULTS_DELETE)


def _split_chain(cmd: str) -> str:
    """Split && chains into separate lines so each runs independently."""
    return "\n".join(part.strip() for part in cmd.split(" && ") if part.strip())


def _embed_backup_files(cmd: str, files_dir: Path, dest_dir: Path) -> None:
    """Copy the Benchmark_Backup files this command needs into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in _BACKUP_ITEMS.items():
        if key not in cmd:
            continue
        src = files_dir / filename
        dst = dest_dir / filename
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


class MacOSWorldToHarbor:
    def __init__(
        self,
        macosworld_root: Path,
        harbor_tasks_root: Path,
        max_timeout_sec: float = 1800.0,
        template_dir: Optional[Path] = None,
        files_dir: Optional[Path] = None,
    ) -> None:
        self.loader = MacOSWorldLoader(macosworld_root)
        self.out_root = Path(harbor_tasks_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(template_dir or (Path(__file__).parent / "template"))
        self.max_timeout = float(max_timeout_sec)
        self.files_dir = Path(files_dir) if files_dir else None

    def get_all_ids(
        self, base_only: bool = False, ready_only: bool = False
    ) -> List[Tuple[str, str]]:
        return self.loader.all_task_ids(base_only=base_only, ready_only=ready_only)

    def generate_task(
        self, category: str, task_id: str, *, overwrite: bool = False
    ) -> Path:
        task = self.loader.load_task(category, task_id)
        local_name = f"{category}__{task_id}"
        task_dir = self.out_root / local_name

        if task_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Already exists: {task_dir}")
            shutil.rmtree(task_dir)

        task_dir.mkdir(parents=True)
        tests_dir = task_dir / "tests"
        tests_dir.mkdir()
        # Harbor requires environment/ dir to exist (even if empty for non-Docker envs)
        (task_dir / "environment").mkdir()

        # instruction.md
        instr_tpl = _read_template(self.template_dir, "instruction.md")
        instr = _render(
            instr_tpl,
            instruction=task.instruction,
            category=task.category,
            task_id=task.task_id,
        )
        (task_dir / "instruction.md").write_text(instr, encoding="utf-8")

        # task.toml
        cfg_tpl = _read_template(self.template_dir, "task.toml")
        cfg = _render(
            cfg_tpl, category=task.category, max_timeout=str(int(self.max_timeout))
        )
        (task_dir / "task.toml").write_text(cfg, encoding="utf-8")

        # tests/setup/ — pre-agent environment setup
        setup_dir = tests_dir / "setup"
        setup_dir.mkdir()

        pre_command = task.pre_command
        if isinstance(pre_command, dict):
            pre_command = pre_command.get("en", "")

        # Build pre_command.sh from task-specific command (path-rewritten for lume VMs)
        # Split && chains into separate lines so each command runs independently
        cmd_lines = []
        fix = _PRE_COMMAND_FIXES.get(task.task_id)
        if fix:
            cmd_lines.append(fix)
        if pre_command:
            # Strip stray backticks (bad source data in some tasks)
            rewritten = pre_command.lstrip("`")
            rewritten = rewritten.replace("/Users/ec2-user", "/Users/lume")
            rewritten = rewritten.replace("ec2-user", "lume")
            if self.files_dir and "Benchmark_Backup" in pre_command:
                _embed_backup_files(pre_command, self.files_dir, setup_dir / "files")
            split = _split_chain(rewritten)
            cmd_lines.append("\n".join(
                l for l in split.split("\n") if not _skip_defaults_delete(l)
            ))
        task_cmd = "\n".join(cmd_lines)

        pre_cmd_path = setup_dir / "pre_command.sh"
        pre_cmd_path.write_text(
            f"#!/bin/bash\n{task_cmd}\n" if task_cmd else "#!/bin/bash\n",
            encoding="utf-8",
        )
        pre_cmd_path.chmod(0o755)

        setup_config = {
            "task_id": task.task_id,
            "before_action_delay_seconds": task.before_action_delay_seconds,
            "before_grading_delay_seconds": task.before_grading_delay_seconds,
        }
        (setup_dir / "config.json").write_text(
            json.dumps(setup_config, indent=2) + "\n", encoding="utf-8"
        )

        # tests/test.sh — grading commands inlined
        grading_cmds = task.grading_command or []
        # Rewrite paths for lume VMs
        grading_cmds = [
            [cmd.replace("/Users/ec2-user", "/Users/lume").replace("ec2-user", "lume"), score]
            for cmd, score in grading_cmds
        ]
        test_sh = _build_test_sh(grading_cmds)
        test_path = tests_dir / "test.sh"
        test_path.write_text(test_sh, encoding="utf-8")
        test_path.chmod(0o755)

        return task_dir

    def generate_many(
        self,
        task_ids: Iterable[Tuple[str, str]],
        *,
        overwrite: bool = False,
    ) -> Tuple[List[Path], List[Tuple[str, str, str]]]:
        success: List[Path] = []
        failures: List[Tuple[str, str, str]] = []

        for idx, (category, task_id) in enumerate(task_ids, 1):
            try:
                out = self.generate_task(category, task_id, overwrite=overwrite)
                print(f"[{idx}] OK   {category}/{task_id} -> {out}")
                success.append(out)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"[{idx}] FAIL {category}/{task_id}: {msg}")
                failures.append((category, task_id, msg))

        return success, failures
