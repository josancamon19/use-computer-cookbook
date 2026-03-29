"""Gateway collected tasks → Harbor format."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from mmini import Mmini, TaskSummary
from mmini.tasks import task_to_harbor


class CollectedTasksAdapter:
    def __init__(
        self,
        gateway_url: str = "http://localhost:8080",
        api_key: str = "",
    ):
        self._client = Mmini(api_key=api_key or None, base_url=gateway_url)

    def list_tasks(self, limit: int = 50) -> List[TaskSummary]:
        return self._client.tasks.list(limit=limit)

    def list_runnable(self, limit: int = 50) -> List[TaskSummary]:
        return [t for t in self.list_tasks(limit=limit) if t.runnable]

    def list_not_runnable(self, limit: int = 50) -> List[TaskSummary]:
        return [t for t in self.list_tasks(limit=limit) if not t.runnable]

    def export_task(
        self, task_id: str, output_dir: str | Path, *, overwrite: bool = False
    ) -> Path:
        task = self._client.tasks.get(task_id)
        return task_to_harbor(task, Path(output_dir), overwrite=overwrite)

    def export_all(
        self,
        output_dir: str | Path,
        *,
        runnable_only: bool = False,
        platform: Optional[str] = None,
        overwrite: bool = False,
        limit: int = 200,
    ) -> tuple[list[Path], list[tuple[str, str]]]:
        """Returns (success_paths, [(task_id, error_msg), ...])."""
        tasks = self.list_tasks(limit=limit)

        if runnable_only:
            tasks = [t for t in tasks if t.runnable]
        if platform:
            tasks = [t for t in tasks if t.platform == platform]

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        successes: list[Path] = []
        failures: list[tuple[str, str]] = []

        for i, summary in enumerate(tasks, 1):
            try:
                full_task = self._client.tasks.get(summary.id)
                path = task_to_harbor(full_task, output, overwrite=overwrite)
                status = "runnable" if summary.runnable else "NO GRADER"
                print(f"[{i}/{len(tasks)}] {status:>10}  {summary.id}  →  {path.name}")
                successes.append(path)
            except Exception as e:
                print(f"[{i}/{len(tasks)}]      FAIL  {summary.id}: {e}")
                failures.append((summary.id, str(e)))

        print(f"\nExported {len(successes)}/{len(tasks)} tasks to {output}")
        if failures:
            print(f"  {len(failures)} failures")

        runnable_count = sum(1 for t in tasks if t.runnable)
        not_runnable = len(tasks) - runnable_count
        if not_runnable > 0:
            print(f"  {not_runnable} tasks skipped or missing graders (not runnable)")

        return successes, failures

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
