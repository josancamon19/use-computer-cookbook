"""
Convert macOSWorld benchmark tasks into Harbor task directories.

Reads task JSON files from the macOSWorld repo's tasks/ directory
and produces one Harbor task directory per task.

When a task references Benchmark_Backup files, the needed files are
copied into the task's tests/setup/files/ directory and the pre_command
is rewritten to use a VM-local staging path (/tmp/harbor/task_files/).
"""

from __future__ import annotations

import json
import re
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


def _build_test_sh(grading_cmds: list, before_grading_delay: int = 0) -> str:
    """Generate test.sh with grading commands inlined."""
    lines = [
        "#!/bin/bash",
        "# Auto-generated grading script",
        "set -e",
        "",
        '# Detect path prefix (macOS SIP blocks /tests, so mmini uses /tmp/harbor)',
        'PREFIX=""',
        '[ -d "/tmp/harbor/logs" ] && PREFIX="/tmp/harbor"',
        'REWARD="${PREFIX}/logs/verifier/reward.txt"',
        "",
    ]

    if before_grading_delay > 0:
        lines.append("# Wait for system to settle before grading")
        lines.append(f"sleep {before_grading_delay}")
        lines.append("")

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

        # Always create pre_command.sh with env init; append task-specific command if present
        env_init = (
            "# Dismiss crash dialogs, eject leftover drives\n"
            'find /Library/Logs/DiagnosticReports -type f -name "panic*.panic"'
            " -mmin -20 2>/dev/null | grep -q . && osascript -e "
            "'tell application \"System Events\" to click at {456,349}' && "
            "rm -rf /Library/Logs/DiagnosticReports/*.panic\n"
            "diskutil list | grep Creedence | awk '{print $NF}' | "
            "xargs -I {} diskutil eject {} 2>/dev/null\n"
            "true\n"
        )
        task_cmd = ""
        if pre_command:
            task_cmd = pre_command.replace("/Users/ec2-user", "/Users/lume")
            task_cmd = task_cmd.replace("ec2-user", "lume")
            if self.files_dir and "Benchmark_Backup" in pre_command:
                _embed_backup_files(pre_command, self.files_dir, setup_dir / "files")

        pre_cmd_path = setup_dir / "pre_command.sh"
        pre_cmd_path.write_text(
            f"#!/bin/bash\n{env_init}\n{task_cmd}\n" if task_cmd else f"#!/bin/bash\n{env_init}",
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
        test_sh = _build_test_sh(grading_cmds, task.before_grading_delay_seconds)
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
