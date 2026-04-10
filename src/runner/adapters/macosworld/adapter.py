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
- METAR API task: grading curls external API, empty response makes grep match anything.
    multi_apps/252f44fc
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
    "531b9a8b-5106-d64d-9e1e-126c2311729a": "needs Xcode (not in base image)",
    "9493a3da-c76d-4c96-787e-d636ff40563d": "needs Xcode (not in base image)",
    "9e9d2f46-8f9e-3ce7-7cba-b33b494f617b": "needs Xcode (not in base image)",
    "a0ef827d-8a52-57ae-3fd0-86aeec9ca7d7": "needs Xcode (not in base image)",
    "b2daa417-a51f-39f4-1218-4ea933e13a61": "needs Xcode (not in base image)",
    "b842ca44-34c0-a4ff-9af6-7e3f40630b95": "needs Xcode (not in base image)",
    "cc6a6287-b5f7-9e3d-a259-3b322b7dddbf": "needs Xcode (not in base image)",
    "d060948e-7731-b4dc-9286-a97bf350e81c": "needs Xcode (not in base image)",
    "d27cd859-5ba4-10e9-ab24-dee68981b2bd": "needs Xcode (not in base image)",
    "ff92d7fe-5eb5-fc7a-0c05-b3cf8bf80f64": "needs Xcode (not in base image)",
    "9974b2d0-1e74-36f2-a9f2-06329e40370c": "needs Xcode (not in base image)",
    "000b3117-0943-ec30-f8c7-7b978b80d6fd": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "249d882a-b365-cd2c-2606-a01d92114e3e": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "2b68aeb0-e88c-7df1-1ba1-57af2cbf65df": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "3ffb40f8-7e50-6e59-04dd-67d622636609": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "6277acb4-9fd2-d00d-7d79-a58ddae5cad3": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "7ba3bf86-beb3-1c4d-c30b-950855378b94": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "933c6a2f-2d33-b1d1-8336-e2e5bf539e7c": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "9e570e25-710a-1431-cb5e-c5539787b890": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "b25e4b35-03b8-3585-6cea-d2636a3e53b5": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "daef8548-e9e2-5883-c287-4c8ed11d6e83": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "ef40b520-3ae5-1af1-b698-127c3daf0b0e": "needs iMovie (LSMin 15.6 vs base 15.4)",
    "0d5e19c1-0c96-35ed-b70b-7c6a0f2bf5ad": "in_process: mid-task dialog injection unsupported",
    "125631b5-49cc-54af-19dc-83c6c8a96fb8": "in_process: mid-task dialog injection unsupported",
    "3ef3a054-c2cc-be98-dc08-a23fedbe68fe": "in_process: mid-task dialog injection unsupported",
    "4aace890-790b-d4b9-354d-808d1fb9cc4c": "in_process: mid-task dialog injection unsupported",
    "4b16b37c-2eee-cc0c-bc68-578f73213da2": "in_process: mid-task dialog injection unsupported",
    "513ec2f0-c6a7-9580-f549-631c52b244f5": "in_process: mid-task dialog injection unsupported",
    "61f8753f-7509-7df7-78a1-f9ae5b4ff3b9": "in_process: mid-task dialog injection unsupported",
    "72158f95-631f-8af0-20f3-ee0c5505f2d0": "in_process: mid-task dialog injection unsupported",
    "724f157a-e886-6536-ee1c-f84a70c3bf45": "in_process: mid-task dialog injection unsupported",
    "730c848f-c495-7737-4334-ea734615d6a6": "in_process: mid-task dialog injection unsupported",
    "7367605a-1dbf-347b-7d19-c7eacfd075a4": "in_process: mid-task dialog injection unsupported",
    "77fae9bf-061e-260c-becc-5aadd0805145": "in_process: mid-task dialog injection unsupported",
    "8c67c0bd-771a-70d3-15f6-7e1393af5e12": "in_process: mid-task dialog injection unsupported",
    "8f9f1a55-ff4b-bba0-975b-a6e336352fe7": "in_process: mid-task dialog injection unsupported",
    "b6a5e0fb-64b8-9b10-be0a-ed16f0ddaee5": "in_process: mid-task dialog injection unsupported",
    "ba242ccf-2fe1-eca8-8282-8adf67c7502c": "in_process: mid-task dialog injection unsupported",
    "c171c967-0ca8-3410-bffd-0ba6b106fe47": "in_process: mid-task dialog injection unsupported",
    "c3ab7a9b-c99b-c734-eeb4-c44169f650c6": "in_process: mid-task dialog injection unsupported",
    "c4964f3f-b32b-b3ba-670b-0aaef199cc57": "in_process: mid-task dialog injection unsupported",
    "cc652222-e5b1-0baa-f5af-df1070ba4693": "in_process: mid-task dialog injection unsupported",
    "cf20112d-60c2-c425-6fe1-59b6df9fa3b8": "in_process: mid-task dialog injection unsupported",
    "d89288fd-f56a-a127-5cc2-759ddbeae8de": "in_process: mid-task dialog injection unsupported",
    "de2ce6d7-b8dc-2507-5c7e-e35688568a92": "in_process: mid-task dialog injection unsupported",
    "e4123eb6-b64c-d772-38b3-e9b1692a1e27": "in_process: mid-task dialog injection unsupported",
    "e4b3fa45-01e7-4c8c-5c4d-f14f6a0ccfe1": "in_process: mid-task dialog injection unsupported",
    "eb346395-b8fe-03bc-a6e5-a58719b1edce": "in_process: mid-task dialog injection unsupported",
}


# Per-task pre_command fixes for our VM image.
# These run BEFORE the task's own pre_command to ensure the correct initial state.
_PRE_COMMAND_FIXES: dict[str, str] = {
    # Dark mode tasks: VM ships with dark mode, set light mode first
    "eb346395-b8fe-03bc-a6e5-a58719b1edce": "defaults delete -g AppleInterfaceStyle 2>/dev/null || true",
    "ce71ae98-6947-6c18-87ac-cdecb1750e5a": "defaults delete -g AppleInterfaceStyle 2>/dev/null || true",
    # TODO: these tasks auto-pass because our VM's Dock/settings don't match
    # macOSWorld's expected initial state. Need to figure out the right Dock layout.
    # "de2ce6d7-b8dc-2507-5c7e-c14f01c63a4c": "add Safari to Dock",
    # "1dac89be-62ba-1f2d-2a0f-3f1cd2587ef0": "add Safari to Dock",
    # "5a6b33d6-271f-3003-1f4f-e15dec4f8769": "add Freeform to Dock",
    # "17fed363-d017-b26c-22f2-ec1865306075": "set notification previews to always",
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
