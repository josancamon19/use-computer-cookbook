"""
mmini environment — runs Harbor tasks on macOS VMs via the mmini gateway.

Usage in job config:
    environment:
      import_path: "runner.environments.mmini:MminiEnvironment"
      kwargs:
        gateway_url: "http://localhost:8080"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from pathlib import Path

import httpx
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from mmini.sandbox import AsyncSandbox as AsyncMacOSSandbox


class MminiEnvironment(BaseEnvironment):
    """A macOS VM environment backed by the mmini gateway."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        gateway_url: str = "http://localhost:8080",
        api_key: str = "",
        ssh_user: str = "lume",
        **kwargs,
    ):
        self._gateway_url = gateway_url
        self._ssh_user = ssh_user
        self._sandbox_id: str | None = None
        self._vm_ip: str | None = None
        self._task_dir: Path = environment_dir.parent
        headers = {}
        api_key = api_key or os.environ.get("MMINI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            base_url=gateway_url, headers=headers, timeout=300,
        )
        self._sandbox: AsyncMacOSSandbox | None = None

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER  # Placeholder — used via import_path

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    @property
    def sandbox(self) -> AsyncMacOSSandbox:
        if self._sandbox is None:
            raise RuntimeError("sandbox not available — call start() first")
        return self._sandbox

    @property
    def vm_ip(self) -> str | None:
        return self._vm_ip

    def _validate_definition(self) -> None:
        cfg = self.task_env_config

        if cfg.cpus > 1 and cfg.cpus != 4:
            self.logger.warning(
                f"task requests {cfg.cpus} CPUs — mmini VMs are fixed at 4 cores"
            )
        if cfg.memory_mb > 0 and cfg.memory_mb != 8192:
            self.logger.warning(
                f"task requests {cfg.memory_mb}MB memory — mmini VMs are fixed at 8GB"
            )
        if cfg.storage_mb and cfg.storage_mb > 80 * 1024:
            self.logger.warning(
                f"task requests {cfg.storage_mb}MB storage — mmini VMs have 80GB disks"
            )

        # TODO: implement skills_dir — copy skills directory to the VM
        if cfg.skills_dir:
            self.logger.warning(
                "task defines skills_dir — not yet supported in mmini"
            )

    async def start(self, force_build: bool = False) -> None:
        if self._sandbox_id is not None:
            return

        resp = await self._http.post("/v1/sandboxes?wait=true")
        resp.raise_for_status()
        data = resp.json()
        self._sandbox_id = data["sandbox_id"]
        self._vm_ip = data.get("vm_ip", "")
        self._sandbox = AsyncMacOSSandbox(
            sandbox_id=self._sandbox_id,
            http=self._http,
            vnc_url=data.get("vnc_url", ""),
            ssh_url=data.get("ssh_url", ""),
        )
        self.logger.info(
            f"mmini sandbox created: {self._sandbox_id} (vm_ip={self._vm_ip})"
        )

        # Create Harbor's expected directory structure
        # macOS SIP prevents writing to root dirs, so we use /tmp/harbor
        await self.sandbox.exec_ssh(
            "mkdir -p /tmp/harbor/logs/agent /tmp/harbor/logs/verifier "
            "/tmp/harbor/logs/artifacts /tmp/harbor/tests /tmp/harbor/solution "
            "/Users/lume/workspace && "
            "sudo mkdir -p /usr/local/bin && sudo chown lume /usr/local/bin"
        )

        # Run task setup (pre_command, delays) — runs before any agent
        await self._setup_task()

        # TODO: start MCP servers on the VM if configured
        # (upload scripts, start background processes, etc.)

    async def _setup_task(self) -> None:
        """Run pre-agent task setup from tests/setup/ if present."""
        setup_dir = self._task_dir / "tests" / "setup"

        # Upload test.sh for the verifier (Harbor doesn't mount task dirs in remote envs)
        test_script = self._task_dir / "tests" / "test.sh"
        if test_script.exists():
            await self.upload_file(test_script, "/tests/test.sh")

        # Upload task-specific files to VM if present
        files_dir = setup_dir / "files"
        if files_dir.is_dir() and any(files_dir.iterdir()):
            self.logger.info("Uploading task files to VM...")
            await self.upload_dir(files_dir, "/Users/lume/Benchmark_Backup")

        pre_cmd_path = setup_dir / "pre_command.sh"
        if pre_cmd_path.exists():
            cmd = pre_cmd_path.read_text().strip()

            lines = [line for line in cmd.split("\n") if not line.startswith("#!")]
            cmd = "\n".join(lines).strip()
            if cmd:
                self.logger.info("Running pre_command...")
                for attempt in range(3):
                    result = await self.exec(cmd, timeout_sec=60)
                    if result.return_code == 0:
                        break
                    self.logger.warning(
                        f"pre_command attempt {attempt + 1} failed"
                    )

        config_path = setup_dir / "config.json"
        delay = 10
        if config_path.exists():
            data = json.loads(config_path.read_text())
            delay = data.get("before_action_delay_seconds", 10)
        if delay:
            await asyncio.sleep(delay)

    async def stop(self, delete: bool = True) -> None:
        if self._sandbox_id is None:
            return
        if delete:
            try:
                resp = await self._http.delete(
                    f"/v1/sandboxes/{self._sandbox_id}"
                )
                resp.raise_for_status()
                self.logger.info(
                    f"mmini sandbox destroyed: {self._sandbox_id}"
                )
            except Exception as e:
                self.logger.error(f"failed to destroy sandbox: {e}")
        self._sandbox_id = None
        self._vm_ip = None
        self._sandbox = None

    # macOS SIP prevents writing to root dirs (/logs, /tests, etc.)
    # Remap Harbor's expected paths to /tmp/harbor/*
    _PATH_PREFIX = "/tmp/harbor"

    # /app and /workspace → home dir (for coding agents like claude-code)
    _WORKSPACE_DIR = "/Users/lume"

    def _remap_str(self, s: str) -> str:
        """Remap all known root paths in an arbitrary string."""
        return (
            s.replace("/logs/", f"{self._PATH_PREFIX}/logs/")
            .replace("/logs\"", f"{self._PATH_PREFIX}/logs\"")
            .replace("/logs'", f"{self._PATH_PREFIX}/logs'")
            .replace("/tests/", f"{self._PATH_PREFIX}/tests/")
            .replace("/solution/", f"{self._PATH_PREFIX}/solution/")
            .replace("/installed-agent", f"{self._PATH_PREFIX}/installed-agent")
            .replace("/app/", f"{self._WORKSPACE_DIR}/")
            .replace("/workspace/", f"{self._WORKSPACE_DIR}/")
        )

    def _remap(self, path: str) -> str:
        """Remap absolute paths for macOS."""
        if path.startswith(("/logs", "/tests", "/solution", "/installed-agent")):
            return self._PATH_PREFIX + path
        if path.startswith(("/app", "/workspace")):
            return self._WORKSPACE_DIR
        return path

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a command on the VM via the gateway's exec endpoint."""
        merged_env = self._merge_env(env)

        remapped_cmd = self._remap_str(command)

        parts = []
        if merged_env:
            remapped_env = {k: self._remap_str(v) for k, v in merged_env.items()}
            exports = " ".join(
                f"{k}={shlex.quote(v)}" for k, v in remapped_env.items()
            )
            parts.append(f"export {exports};")
        if cwd:
            parts.append(f"cd {shlex.quote(self._remap(cwd))} &&")
        parts.append(remapped_cmd)
        full_cmd = " ".join(parts)

        try:
            return_code, stdout = await self.sandbox.exec_ssh(
                full_cmd, timeout=timeout_sec or 120
            )
            return ExecResult(stdout=stdout, stderr=None, return_code=return_code)
        except Exception as e:
            return ExecResult(stdout=None, stderr=str(e), return_code=1)

    async def upload_file(
        self,
        source_path: Path | str,
        target_path: str,
    ) -> None:
        await self.sandbox.upload(str(source_path), self._remap(target_path))

    async def upload_dir(
        self,
        source_dir: Path | str,
        target_dir: str,
    ) -> None:
        await self.sandbox.upload_dir(
            str(source_dir), self._remap(target_dir)
        )

    async def download_file(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        await self.sandbox.download_file(
            self._remap(source_path), str(target_path)
        )

    async def download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
    ) -> None:
        await self.sandbox.download_dir(
            self._remap(source_dir), str(target_dir)
        )
