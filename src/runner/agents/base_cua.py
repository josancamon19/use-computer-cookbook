"""Shared CUA logic — action execution, prompts, base class."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from mmini.sandbox import AsyncMacOSSandbox

# Anthropic CUA outputs key names that don't match computer-server's expected names.
_KEY_ALIASES = {
    "Escape": "esc",
    "escape": "esc",
    "Return": "enter",
    "return": "enter",
    "super": "cmd",
    "meta": "cmd",
    "Super_L": "cmd",
    "Meta_L": "cmd",
    "Control": "ctrl",
    "Shift": "shift",
    "Alt": "alt",
    "Backspace": "backspace",
    "Delete": "delete",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "Page_Up": "pageup",
    "Page_Down": "pagedown",
}


def _normalize_key(key: str) -> str:
    """Map agent key names to computer-server key names."""
    if "+" in key:
        return "+".join(_KEY_ALIASES.get(k, k) for k in key.split("+"))
    return _KEY_ALIASES.get(key, key)


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()


def build_system_prompt(
    template: str,
    width: int,
    height: int,
    password: str = "lume",
    instruction: str = "",
    step: int = 1,
    max_steps: int = 50,
) -> str:
    return (
        template.replace("{OS_TYPE}", "macOS")
        .replace("{TASK_INSTRUCTION}", instruction)
        .replace("{SCREENSHOT_WIDTH}", str(width))
        .replace("{SCREENSHOT_HEIGHT}", str(height))
        .replace("{SCREENSHOT_MAX_X}", str(width - 1))
        .replace("{SCREENSHOT_MAX_Y}", str(height - 1))
        .replace("{SCREENSHOT_CENTER_X}", str(width // 2))
        .replace("{SCREENSHOT_CENTER_Y}", str(height // 2))
        .replace("{STEP_NUMBER}", str(step))
        .replace("{MAX_STEPS}", str(max_steps))
        .replace("{CLIENT_PASSWORD}", password)
        .replace(
            "{CREDENTIALS_SECTION}",
            f"Username: lume\nPassword: {password}",
        )
    )


async def execute_action(
    sandbox: AsyncMacOSSandbox,
    action: dict[str, Any],
    images_dir: Path,
    step_idx: int,
) -> tuple[str, bytes | None, str | None]:
    """Execute a computer-use action.

    Returns (result_text, screenshot_bytes, screenshot_path).
    screenshot_path is relative to agent dir (e.g. "images/step_001.png").
    """
    import logging as _logging
    import time as _time
    _log = _logging.getLogger("mmini.agent")

    action_type = action.get("action", "")
    img_name = f"step_{step_idx:03d}.png"
    _t0 = _time.monotonic()

    if action_type == "screenshot":
        ss = await sandbox.screenshot.take_full_screen()
        (images_dir / img_name).write_bytes(ss)
        _log.info(f"action screenshot: {_time.monotonic()-_t0:.2f}s, {len(ss)} bytes")
        return "Screenshot taken", ss, f"images/{img_name}"

    if action_type in ("left_click", "click"):
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.click(coord[0], coord[1], button="left")
    elif action_type == "right_click":
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.click(coord[0], coord[1], button="right")
    elif action_type == "double_click":
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.click(coord[0], coord[1], button="left", double=True)
    elif action_type == "triple_click":
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.click(coord[0], coord[1], button="left", double=True)
        await asyncio.sleep(0.05)
        await sandbox.mouse.click(coord[0], coord[1], button="left")
    elif action_type == "middle_click":
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.click(coord[0], coord[1], button="middle")
    elif action_type == "type":
        await sandbox.keyboard.type(action.get("text", ""))
    elif action_type == "key":
        key = _normalize_key(action.get("key", "") or action.get("text", ""))
        if "+" in key:
            await sandbox.keyboard.hotkey(key)
        else:
            await sandbox.keyboard.press(key)
    elif action_type == "scroll":
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.scroll(
            coord[0],
            coord[1],
            action.get("direction", "down"),
            action.get("amount", 3),
        )
    elif action_type == "move":
        coord = action.get("coordinate", [0, 0])
        await sandbox.mouse.move(coord[0], coord[1])
    elif action_type == "drag":
        start = action.get("start_coordinate", [0, 0])
        end = action.get("coordinate", [0, 0])
        await sandbox.mouse.drag(start[0], start[1], end[0], end[1])
    elif action_type == "wait":
        await asyncio.sleep(action.get("duration", 1))
    else:
        return f"Unknown action: {action_type}", None, None

    _action_elapsed = _time.monotonic() - _t0
    await asyncio.sleep(2)
    ss = await sandbox.screenshot.take_full_screen()
    (images_dir / img_name).write_bytes(ss)
    _total_elapsed = _time.monotonic() - _t0
    _log.info(f"action {action_type}: {_action_elapsed:.2f}s + screenshot {_total_elapsed-_action_elapsed:.2f}s = {_total_elapsed:.2f}s total")
    return f"Action '{action_type}' executed", ss, f"images/{img_name}"


def _resolve_task_dir(logs_dir: Path) -> Path | None:
    """Discover task_dir from the trial's config.json."""
    trial_dir = logs_dir.parent
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
        task_path = data.get("task", {}).get("path")
        if task_path:
            p = Path(task_path)
            if p.exists():
                return p
    except Exception:
        pass
    return None


def write_trajectory(
    logs_dir: Path,
    steps: list[dict[str, Any]],
    total_input: int,
    total_output: int,
    model_name: str,
    agent_name: str = "cua",
) -> None:
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": "1.0.0",
            "model_name": model_name,
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_input,
            "total_completion_tokens": total_output,
            "total_steps": len(steps),
        },
    }
    (logs_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))


class BaseCUAAgent(BaseAgent):
    """Shared init/setup for all CUA agents."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = 50,
        screen_width: int = 1920,
        screen_height: int = 1080,
        task_dir: str | Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.task_dir = Path(task_dir) if task_dir else _resolve_task_dir(logs_dir)
        self._recording_id: str | None = None
        self.sandbox: AsyncMacOSSandbox | None = None
        self.images_dir: Path = (logs_dir / "images").resolve()
        self.steps: list[dict[str, Any]] = []
        self.total_in = 0
        self.total_out = 0

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def pre_run(self, environment: BaseEnvironment) -> AsyncMacOSSandbox:
        """Common setup: get sandbox, start recording, run task setup, create images dir.

        Returns the sandbox for convenience.
        """
        sandbox: AsyncMacOSSandbox | None = getattr(environment, "sandbox", None)
        if sandbox is None:
            raise RuntimeError("CUA agents require an environment with a .sandbox property")
        self.sandbox = sandbox
        await self.start_recording(sandbox)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.steps = [{"step_id": 1, "source": "user", "message": ""}]
        self.total_in = 0
        self.total_out = 0
        return sandbox

    async def _fire_in_process(self, environment: BaseEnvironment, step_idx: int) -> None:
        """Fire in_process dialog if the environment supports it and step matches."""
        fn = getattr(environment, "fire_in_process", None)
        if fn:
            await fn(step_idx)

    def checkpoint(self, context: AgentContext, model: str, agent_name: str) -> None:
        """Persist trajectory + update AgentContext fields. Safe to call per step.
        Ensures partial state survives harbor's asyncio cancellation at agent_timeout_sec —
        otherwise result.json ends up with empty agent_result (no tokens, no rollout)."""
        write_trajectory(
            self.logs_dir, self.steps, self.total_in, self.total_out, model, agent_name
        )
        context.n_input_tokens = self.total_in
        context.n_output_tokens = self.total_out
        context.n_cache_tokens = (
            getattr(self, "_total_cache_read", 0) + getattr(self, "_total_cache_write", 0)
        )

    async def post_run(self, context: AgentContext, model: str, agent_name: str) -> None:
        """Common teardown: stop recording, write trajectory, set token counts."""
        assert self.sandbox is not None
        await self.stop_recording(self.sandbox)
        self.checkpoint(context, model, agent_name)

    async def start_recording(self, sandbox: AsyncMacOSSandbox) -> None:
        try:
            result = await sandbox.recording.start(name="trial")
            self._recording_id = result.id
            self.logger.info(f"Recording started: {self._recording_id}")
        except Exception as e:
            self.logger.warning(f"Failed to start recording: {e}")

    async def stop_recording(self, sandbox: AsyncMacOSSandbox) -> None:
        if not self._recording_id:
            return
        try:
            import time as _time

            info = await sandbox.recording.stop(self._recording_id)
            self.logger.info(
                f"Recording stopped: id={self._recording_id} "
                f"name={info.name!r} filename={info.filename!r} "
                f"server_size={info.file_size}B "
                f"({info.file_size / (1024 * 1024):.2f} MB)"
            )

            await asyncio.sleep(2)

            rec_path = self.logs_dir / "recording.mp4"
            self.logger.info(f"Downloading recording → {rec_path}")
            t_dl = _time.monotonic()
            await sandbox.recording.download(
                self._recording_id, str(rec_path), timeout=180.0
            )
            dl_elapsed = _time.monotonic() - t_dl
            local_size = rec_path.stat().st_size
            self.logger.info(
                f"Recording saved: {rec_path} "
                f"local_size={local_size}B ({local_size / (1024 * 1024):.2f} MB) "
                f"download_elapsed={dl_elapsed:.1f}s"
            )
        except Exception as e:
            self.logger.warning(
                f"Recording save failed: {type(e).__name__}: {e}"
            )
