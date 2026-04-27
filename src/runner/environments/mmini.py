"""Harbor environment that drives mmini sandboxes (macOS VMs or iOS sims)."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from mmini.client import AsyncMmini
from mmini.sandbox import AsyncMacOSSandbox, AsyncSandbox

from runner.environments import ios_runtime, macos_runtime
from runner.environments import setup as setup_helpers


@contextmanager
def _timer():
    """Yield an object with `.elapsed` (seconds since context entered)."""
    class _T:
        def __init__(self): self.start = time.monotonic()
        @property
        def elapsed(self): return time.monotonic() - self.start
    yield _T()


class MminiEnvironment(BaseEnvironment):
    """Runs harbor tasks on a mmini sandbox: macOS VMs or iOS simulators."""

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
        host: str = "",
        platform: str = "macos",
        **kwargs,
    ):
        self._gateway_url = gateway_url
        self._ssh_user = ssh_user
        self._host = host or os.environ.get("MMINI_HOST", "")
        self._platform = platform
        self._sandbox_id: str | None = None
        self._vm_ip: str | None = None
        self._task_dir: Path = environment_dir.parent
        api_key = api_key or os.environ.get("MMINI_API_KEY", "")
        self._api_key = api_key
        self._client = AsyncMmini(api_key=api_key, base_url=gateway_url)
        self._sandbox: AsyncSandbox | None = None
        self._in_process_cmd: str | None = None
        self._in_process_step: int | None = None

        # Wire SDK logs (mmini.*) into harbor's trial logger.
        sdk_logger = logging.getLogger("mmini")
        if logger and not sdk_logger.handlers:
            sdk_logger.setLevel(logging.DEBUG)
            for handler in logger.handlers:
                sdk_logger.addHandler(handler)

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
        return EnvironmentType.DOCKER  # placeholder — used via import_path

    @property
    def is_mounted(self) -> bool:
        # iOS writes <trial>/verifier/* directly via exec_local; harbor's
        # download_dir round-trip is wasted, tell harbor to skip it.
        return self._platform == "ios"

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    @property
    def sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("sandbox not available — call start() first")
        return self._sandbox

    @property
    def macos_sandbox(self) -> AsyncMacOSSandbox:
        """Narrow `sandbox` to AsyncMacOSSandbox for macOS-only call sites."""
        sb = self.sandbox
        if not isinstance(sb, AsyncMacOSSandbox):
            raise RuntimeError(
                f"macos_sandbox accessed on non-macOS env (platform={self._platform})"
            )
        return sb

    @property
    def vm_ip(self) -> str | None:
        return self._vm_ip

    def _validate_definition(self) -> None:
        # iOS sandbox sizes are whatever lume-emu launches; cpus/memory_mb in
        # task.toml don't apply.
        if self._platform == "ios":
            return
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
        if cfg.skills_dir:
            self.logger.warning("task defines skills_dir — not yet supported in mmini")

    async def start(self, force_build: bool = False) -> None:
        if self._sandbox_id is not None:
            return

        self.logger.info(f"creating mmini sandbox (platform={self._platform})...")
        with _timer() as t:
            if self._platform == "ios":
                device_type, runtime = ios_runtime.read_ios_pin(self._task_dir)
                if device_type or runtime:
                    self.logger.info(
                        f"ios pin: device_type={device_type!r} runtime={runtime!r}"
                    )
                self._sandbox = await self._client.create(
                    type="ios", device_type=device_type, runtime=runtime,
                )
            else:
                self._sandbox = await self._client.create(type="macos", host=self._host)

        self._sandbox_id = self._sandbox.sandbox_id
        if isinstance(self._sandbox, AsyncMacOSSandbox):
            self._vm_ip = self._sandbox.vm_ip
            self.logger.info(
                f"macos sandbox ready in {t.elapsed:.1f}s: {self._sandbox_id} "
                f"vm={self._vm_ip} host={self._sandbox.host}"
            )
        else:
            self.logger.info(
                f"ios sandbox ready in {t.elapsed:.1f}s: {self._sandbox_id}"
            )

        await self._sandbox.start_keepalive(interval=30)

        if self._platform == "ios":
            return

        # macOS bootstrap: prep harbor's expected paths inside the VM, then
        # run the task's pre_command (if any).
        await self.macos_sandbox.exec_ssh(
            "mkdir -p /tmp/harbor/logs/agent /tmp/harbor/logs/verifier "
            "/tmp/harbor/logs/artifacts /tmp/harbor/tests /tmp/harbor/solution "
            "/Users/lume/workspace && "
            "sudo mkdir -p /usr/local/bin && sudo chown lume /usr/local/bin && "
            "echo 0 > /tmp/harbor/logs/verifier/reward.txt"
        )
        await setup_helpers.run_macos_setup(self)

    async def stop(self, delete: bool = True) -> None:
        if self._sandbox is None:
            return
        if delete:
            try:
                await self._sandbox.close()
                self.logger.info(f"mmini sandbox destroyed: {self._sandbox_id}")
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"failed to destroy sandbox: {e}")
        self._sandbox_id = None
        self._vm_ip = None
        self._sandbox = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        merged_env = self._merge_env(env) or {}
        if self._platform == "ios":
            return await ios_runtime.exec_local(self, command, merged_env, timeout_sec)
        return await macos_runtime.exec_on_vm(
            self, command, cwd, merged_env, timeout_sec, user
        )

    async def fire_in_process(self, step: int) -> None:
        if self._platform == "ios":
            return
        await macos_runtime.fire_in_process(self, step)

    # --- transfer helpers (no-ops on iOS) ---

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self._platform == "ios":
            return
        await macos_runtime.upload_file(self, source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if self._platform == "ios":
            return
        await macos_runtime.upload_dir(self, source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if self._platform == "ios":
            return
        await macos_runtime.download_file(self, source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if self._platform == "ios":
            return
        await macos_runtime.download_dir(self, source_dir, target_dir)
