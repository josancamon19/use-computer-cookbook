"""Convert macOSWorld benchmark tasks into Harbor task directories.

Reads task JSON from macOSWorld tasks/ and produces one Harbor task dir
per task. Benchmark_Backup files are embedded into the task dir and
pre_command is rewritten to a VM-local staging path.

Tasks that can't run on our VMs are listed in _EXCLUDED_TASK_IDS
(Xcode/iMovie not in base image, in_process mid-task dialogs unsupported).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from mmini.ax_transpile import patch_curl_timeouts, transpile

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
                # Capture output to a temp file instead of piping directly into
                # grep. When perl's alarm kills bash, any children bash spawned
                # (e.g. `defaults read | grep -q`) become orphaned and keep
                # bash's stdout (the pipe write-end) open — outer grep never
                # gets EOF → 28s alarm fires. With a temp file there is no pipe
                # to block on: the outer grep runs *after* perl returns.
                lines.append("_r=$(mktemp)")
                lines.append(f"perl -e 'alarm 5; exec @ARGV' -- bash -c '{escaped}' > \"$_r\" 2>/dev/null")
                lines.append("if grep -qi 'true' \"$_r\" 2>/dev/null; then")
                lines.append('  rm -f "$_r"')
                lines.append('  echo "1" > "$REWARD"')
                lines.append('  echo "Score: 1"')
                lines.append("  exit 0")
                lines.append("fi")
                lines.append('rm -f "$_r"')
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


# Tasks excluded from export. UUID → reason.
# Re-enable when the blocker is fixed (base image bump, adapter extension, etc.).
_EXCLUDED_TASK_IDS: dict[str, str] = {
    # --- Xcode: not in base image (LSMin 15.6, base is 15.4) ---
    "531b9a8b-5106-d64d-9e1e-126c2311729a": "xcode",
    "9493a3da-c76d-4c96-787e-d636ff40563d": "xcode",
    "9e9d2f46-8f9e-3ce7-7cba-b33b494f617b": "xcode",
    "a0ef827d-8a52-57ae-3fd0-86aeec9ca7d7": "xcode",
    "b2daa417-a51f-39f4-1218-4ea933e13a61": "xcode",
    "b842ca44-34c0-a4ff-9af6-7e3f40630b95": "xcode",
    "cc6a6287-b5f7-9e3d-a259-3b322b7dddbf": "xcode",
    "d060948e-7731-b4dc-9286-a97bf350e81c": "xcode",
    "d27cd859-5ba4-10e9-ab24-dee68981b2bd": "xcode",
    "ff92d7fe-5eb5-fc7a-0c05-b3cf8bf80f64": "xcode",
    "9974b2d0-1e74-36f2-a9f2-06329e40370c": "xcode",
    "cf20112d-60c2-c425-6fe1-59b6df9fa3b8": "xcode",
    "77fae9bf-061e-260c-becc-5aadd0805145": "xcode",
    # --- Pages templates: base image's Pages doesn't carry the template gallery ---
    "03dd4300-a9e4-8b44-1339-6333dd82066d": "pages-template",  # "Essay"
    "2db5297f-0f41-1a5e-7e5c-d109d2993b63": "pages-template",  # "Modern Letter"
    # --- iMovie: not in base image (LSMin 15.6, base is 15.4) ---
    "000b3117-0943-ec30-f8c7-7b978b80d6fd": "imovie",
    "249d882a-b365-cd2c-2606-a01d92114e3e": "imovie",
    "2b68aeb0-e88c-7df1-1ba1-57af2cbf65df": "imovie",
    "3ffb40f8-7e50-6e59-04dd-67d622636609": "imovie",
    "6277acb4-9fd2-d00d-7d79-a58ddae5cad3": "imovie",
    "7ba3bf86-beb3-1c4d-c30b-950855378b94": "imovie",
    "933c6a2f-2d33-b1d1-8336-e2e5bf539e7c": "imovie",
    "9e570e25-710a-1431-cb5e-c5539787b890": "imovie",
    "b25e4b35-03b8-3585-6cea-d2636a3e53b5": "imovie",
    "daef8548-e9e2-5883-c287-4c8ed11d6e83": "imovie",
    "ef40b520-3ae5-1af1-b698-127c3daf0b0e": "imovie",
}


# Per-task pre_command fixes applied BEFORE the task's own pre_command.
_PRE_COMMAND_FIXES: dict[str, str] = {
    # VM ships with dark mode; these tasks expect light mode as starting state
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
                line for line in split.split("\n") if not _skip_defaults_delete(line)
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
            "in_process": task.in_process,
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
        # Transpile osascript AX patterns and patch any existing base64-encoded
        # curl calls to include -m 5 timeout. Baking this into the dataset means
        # the verifier always runs the hardened script regardless of which
        # environment uploads it — no runtime patching needed.
        test_sh, _ = transpile(test_sh)
        test_sh, _ = patch_curl_timeouts(test_sh)
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
