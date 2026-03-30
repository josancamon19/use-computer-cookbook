"""
mmini environment — runs Harbor tasks on macOS VMs via the mmini gateway.

Usage in job config:
    environment:
      import_path: "runner.environments.mmini:MminiEnvironment"
      kwargs:
        gateway_url: "http://localhost:8080"
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from mmini.client import AsyncMmini
from mmini.sandbox import AsyncMacOSSandbox


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
        api_key = api_key or os.environ.get("MMINI_API_KEY", "")
        self._client = AsyncMmini(api_key=api_key, base_url=gateway_url)
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

        self.logger.info("creating mmini sandbox...")
        self._sandbox = await self._client.create(type="macos", wait=True)
        self._sandbox_id = self._sandbox.sandbox_id
        self._vm_ip = self._sandbox.vm_ip
        self.logger.info(f"sandbox ready: {self._sandbox_id} vm={self._vm_ip} host={self._sandbox.host}")

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
            self.logger.info("uploading test.sh")
            await self.upload_file(test_script, "/tests/test.sh")

        pre_cmd_path = setup_dir / "pre_command.sh"
        if pre_cmd_path.exists():
            raw = pre_cmd_path.read_text().strip()

            # Upload Benchmark_Backup files referenced by pre_command
            await self._upload_benchmark_files(raw)

            lines = [
                line.strip()
                for line in raw.split("\n")
                if line.strip() and not line.startswith("#!") and not line.strip().startswith("#")
            ]
            for i, line in enumerate(lines):
                self.logger.info(f"pre_command [{i+1}/{len(lines)}]: {line[:80]}")
                result = await self.exec(line)
                if result.return_code != 0:
                    raise RuntimeError(
                        f"pre_command [{i+1}/{len(lines)}] failed "
                        f"(rc={result.return_code}):\n"
                        f"cmd: {line}\n"
                        f"stdout: {result.stdout or ''}\n"
                        f"stderr: {result.stderr or ''}"
                    )
                else:
                    self.logger.info(f"pre_command [{i+1}/{len(lines)}] ok")

        self.logger.info("task setup complete")

    # Shared benchmark assets used by macOSWorld tasks
    _BENCHMARK_ASSETS_DIR = Path(__file__).resolve().parents[3] / "macosworld" / "files"
    _BENCHMARK_VM_DIR = "/Users/lume/Benchmark_Backup"

    async def _upload_benchmark_files(self, pre_command_text: str) -> None:
        """Parse pre_command for Benchmark_Backup/ references and upload only needed files."""
        # Extract top-level asset names from Benchmark_Backup/ references
        # Handles quoted paths (spaces) and subpaths (BenchmarkApp/BenchmarkApp)
        refs = set()
        # Quoted: "Benchmark_Backup/iMovie Library.imovielibrary"
        for m in re.finditer(r'"[^"]*Benchmark_Backup/([^"]+)"', pre_command_text):
            refs.add(m.group(1).split("/")[0].rstrip())
        # Unquoted: Benchmark_Backup/benchmark_files or Benchmark_Backup/X/Y
        for m in re.finditer(r'(?<!")Benchmark_Backup/(\S+)', pre_command_text):
            refs.add(m.group(1).split("/")[0].rstrip(";"))
        if not refs:
            return

        # Ensure target dir exists
        await self.sandbox.exec_ssh(f"mkdir -p {self._BENCHMARK_VM_DIR}")

        for name in sorted(refs):
            local = self._BENCHMARK_ASSETS_DIR / name
            remote = f"{self._BENCHMARK_VM_DIR}/{name}"
            if not local.exists():
                self.logger.warning(f"benchmark asset not found: {local}")
                continue
            if local.is_dir():
                self.logger.info(f"uploading {name}/ -> {remote}")
                await self.upload_dir(local, remote)
            else:
                self.logger.info(f"uploading {name} -> {remote}")
                await self.upload_file(local, remote)

        # Verify uploads landed correctly
        result = await self.sandbox.exec_ssh(
            f"find {self._BENCHMARK_VM_DIR} -maxdepth 2 -type f | head -30 && echo '---' && du -sh {self._BENCHMARK_VM_DIR}/*"
        )
        self.logger.info(f"Benchmark_Backup contents:\n{result.stdout}")

    async def stop(self, delete: bool = True) -> None:
        if self._sandbox is None:
            return
        if delete:
            try:
                await self._sandbox.close()
                self.logger.info(f"mmini sandbox destroyed: {self._sandbox_id}")
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

        result = await self.sandbox.exec_ssh(full_cmd, timeout=30)
        self.logger.debug(f"exec rc={result.return_code} cmd={full_cmd}")
        return ExecResult(stdout=result.stdout, stderr=None, return_code=result.return_code)

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
