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

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Dismiss crash dialogs, eject extra drives (from macOSWorld)
ENV_INIT_COMMAND = (
    'find /Library/Logs/DiagnosticReports -type f -name "panic*.panic"'
    " -mmin -20 2>/dev/null | grep -q . && osascript -e "
    "'tell application \"System Events\" to click at {456,349}' && "
    "rm -rf /Library/Logs/DiagnosticReports/*.panic; "
    "diskutil list | grep Creedence | awk '{print $NF}' | "
    "xargs -I {} diskutil eject {} 2>/dev/null; true"
)


def get_sandbox(environment: BaseEnvironment) -> Any:
    """Extract sandbox from environment. Works with any environment
    that exposes a .sandbox property (e.g. MminiEnvironment)."""
    sandbox = getattr(environment, "sandbox", None)
    if sandbox is None:
        raise RuntimeError("CUA agents require an environment with a .sandbox property")
    return sandbox


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
    sandbox: Any,
    action: dict[str, Any],
    images_dir: Path,
    step_idx: int,
) -> tuple[str, bytes | None, str | None]:
    """Execute a computer-use action.

    Returns (result_text, screenshot_bytes, screenshot_path).
    screenshot_path is relative to agent dir (e.g. "images/step_001.png").
    """
    action_type = action.get("action", "")
    img_name = f"step_{step_idx:03d}.png"

    if action_type == "screenshot":
        ss = await sandbox.screenshot.take_full_screen()
        (images_dir / img_name).write_bytes(ss)
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
        key = action.get("key", "")
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

    await asyncio.sleep(2)
    ss = await sandbox.screenshot.take_full_screen()
    (images_dir / img_name).write_bytes(ss)
    return f"Action '{action_type}' executed", ss, f"images/{img_name}"


def _resolve_task_dir(logs_dir: Path) -> Path | None:
    """Discover task_dir from the trial's config.json.

    Harbor writes config.json to the trial directory (parent of agent/).
    It contains task.path — the local task directory.
    """
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


async def run_task_setup(
    environment: BaseEnvironment,
    task_dir: Path | None,
    logger: Any,
) -> None:
    """Run env_init + pre_command from task_config.json."""
    # Always run env_init (dismiss crash dialogs, etc.)
    await environment.exec(ENV_INIT_COMMAND, timeout_sec=15)

    if not task_dir:
        return

    config_path = task_dir / "tests" / "task_config.json"
    if not config_path.exists():
        return

    data = json.loads(config_path.read_text())
    pre_command = data.get("pre_command", "")
    if isinstance(pre_command, dict):
        pre_command = pre_command.get("en", "")

    if pre_command:
        logger.info("Running pre_command...")
        for attempt in range(3):
            result = await environment.exec(pre_command, timeout_sec=60)
            if result.return_code == 0:
                break
            logger.warning(f"pre_command attempt {attempt + 1} failed")

    delay = data.get("before_action_delay_seconds", 10)
    if delay:
        await asyncio.sleep(delay)


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
        # task_dir passed by Harbor fork, or auto-resolved from trial config.json
        self.task_dir = Path(task_dir) if task_dir else _resolve_task_dir(logs_dir)
        self._recording_id: str | None = None
        self._sandbox: Any = None

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def start_recording(self, sandbox: Any) -> None:
        """Start screen recording. Call at the beginning of run()."""
        try:
            result = await sandbox.recording.start(name="trial")
            self._recording_id = result.get("id") or result.get("recording_id")
            self.logger.info(f"Recording started: {self._recording_id}")
        except Exception as e:
            self.logger.warning(f"Failed to start recording: {e}")

    async def stop_recording(self, sandbox: Any) -> None:
        """Stop recording and download. Call at the end of run()."""
        if not self._recording_id:
            return
        try:
            await sandbox.recording.stop(self._recording_id)
            self.logger.info(f"Recording stopped: {self._recording_id}")
            await asyncio.sleep(2)
            rec_path = str(self.logs_dir / "recording.mp4")
            await sandbox.recording.download(self._recording_id, rec_path)
            self.logger.info(f"Recording saved: {rec_path}")
        except Exception as e:
            self.logger.warning(f"Recording save failed: {e}")
