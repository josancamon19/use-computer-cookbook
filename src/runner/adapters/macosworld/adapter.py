"""
Convert macOSWorld benchmark tasks into Harbor task directories.

Reads task JSON files from the macOSWorld repo's tasks/ directory
and produces one Harbor task directory per task.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class MacOSWorldTask:
    task_id: str
    category: str
    instruction: str
    snapshot: dict = field(default_factory=dict)
    pre_command: str | dict = ""
    grading_command: list = field(default_factory=list)
    before_action_delay_seconds: int = 10
    before_grading_delay_seconds: int = 30
    force_snapshot_recovery: bool = False
    in_process: list | None = None

    @classmethod
    def from_json(cls, path: Path, category: str) -> "MacOSWorldTask":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            task_id=data["id"],
            category=category,
            instruction=data.get("task", {}).get("en", ""),
            snapshot=data.get("snapshot", {}),
            pre_command=data.get("pre_command", ""),
            grading_command=data.get("grading_command", []),
            before_action_delay_seconds=data.get("before_action_delay_seconds", 10),
            before_grading_delay_seconds=data.get("before_grading_delay_seconds", 30),
            force_snapshot_recovery=data.get("force_snapshot_recovery", False),
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
                if ready_only and task.needs_benchmark_backup:
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


def _read_template(template_dir: Path, name: str) -> str:
    return (template_dir / name).read_text(encoding="utf-8")


def _render(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


class MacOSWorldToHarbor:
    def __init__(
        self,
        macosworld_root: Path,
        harbor_tasks_root: Path,
        max_timeout_sec: float = 1800.0,
        template_dir: Optional[Path] = None,
    ) -> None:
        self.loader = MacOSWorldLoader(macosworld_root)
        self.out_root = Path(harbor_tasks_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(template_dir or (Path(__file__).parent / "template"))
        self.max_timeout = float(max_timeout_sec)

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

        # tests/task_config.json — task JSON with paths rewritten for lume VMs
        src_json = self.loader.task_json_path(category, task_id)
        task_json = src_json.read_text(encoding="utf-8")
        task_json = task_json.replace("/Users/ec2-user", "/Users/lume")
        task_json = task_json.replace("ec2-user", "lume")
        (tests_dir / "task_config.json").write_text(task_json, encoding="utf-8")

        # tests/test.sh
        test_tpl = _read_template(self.template_dir, "test.sh")
        test = _render(test_tpl, task_id=task.task_id, category=task.category)
        test_path = tests_dir / "test.sh"
        test_path.write_text(test, encoding="utf-8")
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
