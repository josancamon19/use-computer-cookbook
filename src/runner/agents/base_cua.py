"""Shared CUA logic — action execution, prompts, base class."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from mmini.sandbox import AsyncMacOSSandbox, AsyncSandbox  # noqa: F401
from PIL import Image

_log = logging.getLogger("mmini.agent")


def vision_target_long_edge(model: str) -> int:
    """Long-edge cap for client-side resize before vision input — per Anthropic
    docs, model coords come back in resized-image space so we resize ourselves
    to a known target rather than letting the server downscale silently."""
    m = (model or "").lower()
    if "opus-4-7" in m or "opus-4-8" in m or "opus-5" in m:
        return 2576  # Anthropic Opus 4.7+ caps at 2576px
    if "claude" in m or "anthropic/" in m:
        return 1568  # Other Anthropic models cap at 1568px
    if "kimi" in m or "fireworks" in m:
        return 896  # Kimi: y-coord accuracy degrades at >1024 (probed on tall iOS shots)
    return 1280  # OpenAI/Gemini fallback


def resize_for_vision(
    image_bytes: bytes, model: str
) -> tuple[bytes, int, int, float, float]:
    """Aspect-preserving resize to model's vision cap.
    Returns (bytes, api_w, api_h, sx, sy) — multiply model coords by sx/sy at dispatch.
    Pass-through if image already fits within the cap."""
    img = Image.open(io.BytesIO(image_bytes))
    native_w, native_h = img.size
    target = vision_target_long_edge(model)
    scale = min(target / native_w, target / native_h, 1.0)
    api_w = int(native_w * scale)
    api_h = int(native_h * scale)
    sx = native_w / api_w
    sy = native_h / api_h
    if scale == 1.0:
        return image_bytes, api_w, api_h, sx, sy
    resized = img.resize((api_w, api_h), Image.LANCZOS)
    buf = io.BytesIO()
    fmt = (img.format or "PNG").upper()
    if fmt == "JPEG":
        resized.convert("RGB").save(buf, format="JPEG", quality=85)
    else:
        resized.save(buf, format="PNG")
    return buf.getvalue(), api_w, api_h, sx, sy

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
    """Execute a computer-use action; returns (result_text, ss_bytes, ss_path)."""
    action_type = action.get("action", "")
    img_name = f"step_{step_idx:03d}.png"
    t0 = time.monotonic()

    if action_type == "screenshot":
        ss = await sandbox.screenshot.take_full_screen()
        (images_dir / img_name).write_bytes(ss)
        _log.info(f"action screenshot: {time.monotonic()-t0:.2f}s, {len(ss)} bytes")
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

    action_elapsed = time.monotonic() - t0
    await asyncio.sleep(2)
    ss = await sandbox.screenshot.take_full_screen()
    (images_dir / img_name).write_bytes(ss)
    total_elapsed = time.monotonic() - t0
    _log.info(f"action {action_type}: {action_elapsed:.2f}s + screenshot {total_elapsed-action_elapsed:.2f}s = {total_elapsed:.2f}s total")
    return f"Action '{action_type}' executed", ss, f"images/{img_name}"


def _task_dir_from_env(environment: BaseEnvironment) -> Path:
    """Pull the task dir from the environment — always set at harbor init."""
    p = getattr(environment, "_task_dir", None) or getattr(environment, "task_dir", None)
    if p is None:
        raise RuntimeError(
            "environment exposes neither _task_dir nor task_dir; "
            "agent can't locate the task directory"
        )
    return Path(p).resolve()


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
        # task_dir is bound from the environment in pre_run(); see comment there.
        self.task_dir: Path | None = Path(task_dir) if task_dir else None
        self._recording_id: str | None = None
        # Sandbox is platform-agnostic at the base level. macOS subclasses cast
        # to AsyncMacOSSandbox; iOS subclasses cast to AsyncIOSSandbox.
        self.sandbox: AsyncSandbox | None = None
        self.images_dir: Path = (logs_dir / "images").resolve()
        self.steps: list[dict[str, Any]] = []
        self.total_in = 0
        self.total_out = 0

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def pre_run(self, environment: BaseEnvironment) -> AsyncSandbox:
        """Common setup: get sandbox, start recording, create images dir.

        Returns the sandbox for convenience. The concrete sandbox type
        (macOS vs iOS) depends on the environment — subclasses that need
        macOS-only methods (mouse/keyboard) should narrow the cast themselves.
        """
        sandbox: AsyncSandbox | None = getattr(environment, "sandbox", None)
        if sandbox is None:
            raise RuntimeError("CUA agents require an environment with a .sandbox property")
        self.sandbox = sandbox
        # Bind task_dir from the env (always set at harbor init); harbor doesn't
        # write the trial's config.json until after agent __init__, so reading
        # it from there is racy. The env's _task_dir is reliable.
        if self.task_dir is None:
            self.task_dir = _task_dir_from_env(environment)
            self.logger.info(f"task_dir={self.task_dir}")
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
        await self._save_final_ax_tree(self.sandbox)
        await self.stop_recording(self.sandbox)
        self.checkpoint(context, model, agent_name)

    async def _save_final_ax_tree(self, sandbox: AsyncSandbox) -> None:
        """Snapshot the sandbox's AX tree to <agent>/final_ax_tree.json. Useful
        for adhoc/no-verifier runs where the trajectory alone doesn't tell you
        what state the device ended in."""
        try:
            tree = await sandbox.display.get_windows()
            (self.logs_dir / "final_ax_tree.json").write_text(json.dumps(tree, indent=2))
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to capture final AX tree: {e}")

    async def start_recording(self, sandbox: AsyncSandbox) -> None:
        try:
            result = await sandbox.recording.start(name="trial")
            self._recording_id = result.id
            self.logger.info(f"Recording started: {self._recording_id}")
        except Exception as e:
            self.logger.warning(f"Failed to start recording: {e}")

    async def stop_recording(self, sandbox: AsyncSandbox) -> None:
        if not self._recording_id:
            return
        await asyncio.sleep(3.0)
        try:
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
            t_dl = time.monotonic()
            await sandbox.recording.download(
                self._recording_id, str(rec_path), timeout=180.0
            )
            dl_elapsed = time.monotonic() - t_dl
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
