"""mmini environment — runs Harbor tasks on macOS VMs via the mmini gateway."""

from __future__ import annotations

import logging
import os
import re
import shlex
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _timer():
    """Yields an object with .elapsed property (seconds since context entered)."""
    class _T:
        def __init__(self): self.start = time.monotonic()
        @property
        def elapsed(self): return time.monotonic() - self.start
    yield _T()

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
        host: str = "",
        **kwargs,
    ):
        self._gateway_url = gateway_url
        self._ssh_user = ssh_user
        self._host = host or os.environ.get("MMINI_HOST", "")
        self._sandbox_id: str | None = None
        self._vm_ip: str | None = None
        self._task_dir: Path = environment_dir.parent
        api_key = api_key or os.environ.get("MMINI_API_KEY", "")
        self._client = AsyncMmini(api_key=api_key, base_url=gateway_url)
        self._sandbox: AsyncMacOSSandbox | None = None

        # Propagate SDK logs (mmini.*) into harbor's trial logger
        # so retries and transient errors are visible in trial.log
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
        with _timer() as t:
            self._sandbox = await self._client.create(type="macos", host=self._host)
        self._sandbox_id = self._sandbox.sandbox_id
        self._vm_ip = self._sandbox.vm_ip
        self.logger.info(f"sandbox ready in {t.elapsed:.1f}s: {self._sandbox_id} vm={self._vm_ip} host={self._sandbox.host}")

        await self.sandbox.start_keepalive(interval=30)

        await self.sandbox.exec_ssh(
            "mkdir -p /tmp/harbor/logs/agent /tmp/harbor/logs/verifier "
            "/tmp/harbor/logs/artifacts /tmp/harbor/tests /tmp/harbor/solution "
            "/Users/lume/workspace && "
            "sudo mkdir -p /usr/local/bin && sudo chown lume /usr/local/bin && "
            "echo 0 > /tmp/harbor/logs/verifier/reward.txt"
        )
        await self._setup_task()

    async def _setup_task(self) -> None:
        """Run pre-agent task setup from tests/setup/ if present."""
        setup_dir = self._task_dir / "tests" / "setup"

        # Upload test.sh for the verifier (Harbor doesn't mount task dirs in
        # remote envs). Transpile osascript+System Events Accessibility
        # patterns into python3.12 invocations *before* upload — see
        # mmini.ax_transpile for the why and the supported patterns.
        test_script = self._task_dir / "tests" / "test.sh"
        if test_script.exists():
            from mmini.ax_transpile import transpile, needs_exec_ax, PRE_COMMAND_OSASCRIPT_TIMEOUT_S

            raw = test_script.read_text()
            rewritten, n = transpile(raw)
            if n > 0:
                # Stage the rewritten file in the trial dir so the upload path
                # has a real file to send. Trial dir is per-trial so this is safe.
                staged = self.trial_paths.trial_dir / ".test.transpiled.sh"
                staged.write_text(rewritten)
                self.logger.info(f"uploading test.sh (transpiled {n} AX call(s))")
                # Preview shows what actually shipped — debugging a hung trial
                # otherwise requires re-deriving transpiler output.
                self.logger.info(f"  transpiled preview: {rewritten[:300]!r}")
                await self.upload_file(staged, "/tests/test.sh")
            else:
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
                # Transpile osascript AX patterns (e.g. keystroke) and route
                # through exec_ax when needed — same as test.sh but per-line.
                transpiled_line, n = transpile(line, fallback_timeout_s=PRE_COMMAND_OSASCRIPT_TIMEOUT_S)
                if n > 0:
                    self.logger.info(f"pre_command [{i+1}/{len(lines)}] (transpiled {n}): {line[:80]}")
                    self.logger.info(f"  transpiled: {transpiled_line[:300]!r}")
                    # Only ax_helper emissions need exec_ax (TCC Accessibility via
                    # cua-server). Fallback osascript-with-timeout runs directly in
                    # bash and works via regular exec without the 30s hard limit.
                    if needs_exec_ax(transpiled_line):
                        result = await self._exec_ax(transpiled_line)
                    else:
                        result = await self.exec(transpiled_line)
                else:
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

        # Load in_process config (if any) for fire_in_process()
        self._in_process_cmd = None
        self._in_process_step = None
        config_path = setup_dir / "config.json"
        if config_path.exists():
            import json as _json
            cfg = _json.loads(config_path.read_text())
            ip = cfg.get("in_process")
            if ip and isinstance(ip, list) and len(ip) >= 2:
                self._in_process_cmd = ip[0]
                self._in_process_step = ip[1]

        self.logger.info("task setup complete")

    # Shared benchmark assets used by macOSWorld tasks
    _BENCHMARK_ASSETS_DIR = Path(__file__).resolve().parents[3] / "macosworld" / "files"
    _BENCHMARK_VM_DIR = "/Users/lume/Benchmark_Backup"

    async def _upload_benchmark_files(self, pre_command_text: str) -> None:
        """Upload Benchmark_Backup files referenced by pre_command."""
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

    @staticmethod
    def _wrap_with_timeout(cmd: str, timeout: int) -> str:
        """Wrap command with perl alarm to kill hung processes on the VM."""
        kill_after = max(timeout - 2, 5)
        escaped = cmd.replace("'", "'\\''")
        return f"perl -e 'alarm {kill_after}; exec @ARGV' -- bash -c '{escaped}'"

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a command on the VM via the gateway's exec endpoint."""
        merged_env = self._merge_env(env)

        remapped_cmd = self._remap_str(command)

        parts = []
        if user is not None:
            parts.append(f"sudo -u {shlex.quote(str(user))}")
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

        timeout = timeout_sec or 30

        # TODO: ugly hack — perl alarm wraps all exec to kill hung osascript.
        # The right fix is server-side timeouts in lume's exec handler.
        full_cmd = self._wrap_with_timeout(full_cmd, timeout)

        # Verifier scripts (test.sh) get transpiled at upload time to invoke
        # python3.12 with AX calls, but those calls only get TCC's
        # Accessibility grant when invoked under cua-server's launchd-domain
        # responsibility chain. The SSH-backed exec_ssh puts
        # sshd-keygen-wrapper in the chain (which has Accessibility=denied)
        # and TCC propagates the deny to children. Routing the verifier
        # exec via exec_ax (cua-server's run_command) gives us the right
        # chain. Detection: the command path contains "test.sh".
        with _timer() as t:
            if "test.sh" in full_cmd:
                result = await self.sandbox.exec_ax(full_cmd, timeout=timeout)
            else:
                result = await self.sandbox.exec_ssh(full_cmd, timeout=timeout)
        # rc=-14 is SIGALRM — our perl-alarm wrapper killed the inner command
        # before it finished. When this happens on a verifier exec, the trial's
        # reward.txt keeps the "0" written at setup, and the trial looks like
        # a plain task failure in result.json. Shout it loudly so post-hoc
        # log grep can separate "agent failed the task" from "verifier hung
        # and got killed".
        is_verifier = "test.sh" in full_cmd or "_ax_" in full_cmd
        alarm_killed = result.return_code == -14
        if is_verifier and alarm_killed:
            self.logger.warning(
                f"verifier ALARM-KILLED after {t.elapsed:.1f}s (timeout={timeout}s) — "
                f"reward=0.0 is a verifier hang, NOT a task failure. cmd={full_cmd[:120]}"
            )
            # Drop a marker in the trial's verifier dir so artifact collection
            # picks it up. Fire-and-forget via exec_ssh (a new 5s-wrapped exec),
            # not exec_ax — we don't want another hang inside a hang.
            try:
                await self.sandbox.exec_ssh(
                    "touch /tmp/harbor/logs/verifier/alarm-killed.marker",
                    timeout=5,
                )
            except Exception as exc:  # best-effort, don't fail the trial on it
                self.logger.debug(f"alarm-killed marker write failed: {exc}")
        elif is_verifier:
            self.logger.info(
                f"verifier exec rc={result.return_code} ({t.elapsed:.1f}s) cmd={full_cmd[:100]}"
            )
        else:
            self.logger.debug(
                f"exec rc={result.return_code} ({t.elapsed:.1f}s) cmd={full_cmd[:100]}"
            )
        return ExecResult(
            stdout=result.stdout, stderr=None, return_code=result.return_code
        )

    async def fire_in_process(self, step: int) -> None:
        """Fire the in_process dialog if the current step matches.

        Agents call this after each step. If the task has an in_process
        field and step matches, the dialog osascript is spawned in the
        background so it appears on screen for the agent's next screenshot.
        """
        if self._in_process_cmd and step == self._in_process_step:
            self.logger.info(f"in_process: firing dialog at step {step}")
            await self.sandbox.exec_ssh(
                f"nohup {self._in_process_cmd} > /dev/null 2>&1 &"
            )

    async def _exec_ax(self, command: str, timeout_sec: int = 30) -> ExecResult:
        """Run a command via exec_ax (cua-server chain) for AX-needing calls."""
        full_cmd = self._wrap_with_timeout(self._remap_str(command), timeout_sec)
        with _timer() as t:
            result = await self.sandbox.exec_ax(full_cmd, timeout=timeout_sec)
        self.logger.debug(f"exec_ax rc={result.return_code} ({t.elapsed:.1f}s) cmd={full_cmd[:100]}")
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
        await self.sandbox.upload_dir(str(source_dir), self._remap(target_dir))

    async def download_file(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        await self.sandbox.download_file(self._remap(source_path), str(target_path))

    async def download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
    ) -> None:
        await self.sandbox.download_dir(self._remap(source_dir), str(target_dir))
