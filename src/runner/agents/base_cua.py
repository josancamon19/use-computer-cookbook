"""Shared CUA logic — action execution, prompts, base class."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
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


def _extract_coords(action: dict[str, Any]) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Pull (point, secondary_point) from any of the action shapes we use:
       macOS-style:  coordinate=[x,y], start_coordinate=[x,y]
       iOS-style:    x, y / from_x, from_y, to_x, to_y
    Returns (single_or_drag_end, drag_start)."""
    point: tuple[int, int] | None = None
    secondary: tuple[int, int] | None = None
    if "coordinate" in action and action["coordinate"]:
        c = action["coordinate"]
        point = (int(c[0]), int(c[1]))
    if "start_coordinate" in action and action["start_coordinate"]:
        s = action["start_coordinate"]
        secondary = (int(s[0]), int(s[1]))
    if point is None and "x" in action and "y" in action:
        try:
            point = (int(float(action["x"])), int(float(action["y"])))
        except (TypeError, ValueError):
            pass
    if "to_x" in action and "to_y" in action:
        try:
            point = (int(float(action["to_x"])), int(float(action["to_y"])))
        except (TypeError, ValueError):
            pass
    if "from_x" in action and "from_y" in action:
        try:
            secondary = (int(float(action["from_x"])), int(float(action["from_y"])))
        except (TypeError, ValueError):
            pass
    return point, secondary


def _annotate_click(png_bytes: bytes, action: dict[str, Any]) -> bytes:
    """Draw a red marker at the click/move/scroll/drag coordinate so the
    rendered screenshot shows where the agent intended to act. Pass-through
    when the action has no coordinate."""
    point, start = _extract_coords(action)
    if point is None and start is None:
        return png_bytes
    try:
        from PIL import ImageDraw

        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        red = (255, 30, 30, 230)
        if point and start:
            draw.line([start, point], fill=red, width=4)
            draw.ellipse([start[0]-12, start[1]-12, start[0]+12, start[1]+12], outline=red, width=3)
            draw.ellipse([point[0]-12, point[1]-12, point[0]+12, point[1]+12], outline=red, width=3)
        else:
            cx, cy = point or start  # type: ignore
            draw.ellipse([cx-18, cy-18, cx+18, cy+18], outline=red, width=4)
            draw.line([(cx-26, cy), (cx+26, cy)], fill=red, width=2)
            draw.line([(cx, cy-26), (cx, cy+26)], fill=red, width=2)
        out = Image.alpha_composite(img, overlay).convert("RGB")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        _log.warning(f"annotate_click failed: {e}")
        return png_bytes


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
    annotated = _annotate_click(ss, action)
    (images_dir / img_name).write_bytes(annotated)
    total_elapsed = time.monotonic() - t0
    _log.info(f"action {action_type}: {action_elapsed:.2f}s + screenshot {total_elapsed-action_elapsed:.2f}s = {total_elapsed:.2f}s total")
    # Return the unannotated bytes so vision-input downscaling sees a clean shot;
    # the annotation lives only in the persisted file for human review.
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
        await self._log_final_state_metrics(self.sandbox)
        await self._capture_artifacts(context)
        await self.stop_recording(self.sandbox)
        self.checkpoint(context, model, agent_name)

    async def _capture_artifacts(self, context: AgentContext) -> None:
        """For every file the runner uploaded at setup time, snapshot the
        original (already on disk) and pull the post-run state back from the VM
        to <trial>/artifacts/{uploaded,final}/<basename>. Writes a manifest with
        sizes + a 'changed' flag derived from sha256.

        Best-effort — skipped silently when there are no uploaded files (no
        manifest), and individual download failures don't kill the run."""
        if self.task_dir is None:
            return
        manifest_path = self.task_dir / "tests" / "setup" / "files" / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            entries = json.loads(manifest_path.read_text())
        except Exception as e:
            self.logger.warning(f"artifacts: manifest parse failed: {e}")
            return
        if not entries:
            return

        env = getattr(context, "environment", None)
        artifacts_dir = self.logs_dir.parent / "artifacts"
        uploaded_dir = artifacts_dir / "uploaded"
        final_dir = artifacts_dir / "final"
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        captured: list[dict[str, Any]] = []
        for entry in entries:
            remote = entry.get("remote_path")
            local_name = entry.get("local_name")
            if not remote or not local_name:
                continue
            src_local = self.task_dir / "tests" / "setup" / "files" / local_name
            basename = Path(remote).name
            uploaded_target = uploaded_dir / basename
            final_target = final_dir / basename
            uploaded_sha: str | None = None
            final_sha: str | None = None
            uploaded_size = 0
            final_size = 0

            try:
                if src_local.exists():
                    data = src_local.read_bytes()
                    uploaded_target.write_bytes(data)
                    uploaded_sha = hashlib.sha256(data).hexdigest()
                    uploaded_size = len(data)
            except Exception as e:
                self.logger.warning(f"artifacts: copy uploaded {basename} failed: {e}")

            if env is not None:
                try:
                    await env.download_file(remote, final_target)
                    fbytes = final_target.read_bytes()
                    final_sha = hashlib.sha256(fbytes).hexdigest()
                    final_size = len(fbytes)
                except Exception as e:
                    self.logger.warning(f"artifacts: download final {remote} failed: {e}")

            captured.append({
                "name": basename,
                "remote_path": remote,
                "uploaded_size": uploaded_size,
                "final_size": final_size,
                "changed": (uploaded_sha is not None and final_sha is not None and uploaded_sha != final_sha),
                "uploaded_sha256": uploaded_sha,
                "final_sha256": final_sha,
            })

        if captured:
            (artifacts_dir / "manifest.json").write_text(json.dumps(captured, indent=2))
            self.logger.info(f"artifacts: captured {len(captured)} file(s) under {artifacts_dir}")

    async def _save_final_ax_tree(self, sandbox: AsyncSandbox) -> None:
        """Snapshot the sandbox's AX tree to <agent>/final_ax_tree.json. Useful
        for adhoc/no-verifier runs where the trajectory alone doesn't tell you
        what state the device ended in."""
        try:
            tree = await sandbox.display.get_windows()
            (self.logs_dir / "final_ax_tree.json").write_text(json.dumps(tree, indent=2))
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Failed to capture final AX tree: {e}")

    async def _log_final_state_metrics(self, sandbox: AsyncSandbox) -> None:
        """Compute screenshot_similarity + a11y_match_ratio against the human's
        recorded final state. Only runs when expected_final.json (shipped by
        the gateway for collected tasks) is present in the task dir.

        Metrics land in:
          - logger.info (visible in runner stdout / harbor.log)
          - <logs_dir>/verifier-metrics.json (machine-readable)
        """
        if self.task_dir is None:
            return
        expected_path = self.task_dir / "expected_final.json"
        if not expected_path.exists():
            return
        try:
            expected = json.loads(expected_path.read_text())
        except Exception as e:
            self.logger.warning(f"expected_final.json parse failed: {e}")
            return

        metrics: dict[str, Any] = {}

        # Final screenshot vs expected
        try:
            actual_bytes = await sandbox.screenshot.take_full_screen()
            ss_path = self.logs_dir / "final_screenshot.png"
            ss_path.write_bytes(actual_bytes)
            ref_filename = expected.get("screenshot_filename")
            ref_task_id = expected.get("task_id")
            if ref_filename and ref_task_id:
                ref_bytes = await self._fetch_reference_screenshot(ref_task_id, ref_filename)
                if ref_bytes:
                    sim = _screenshot_similarity(actual_bytes, ref_bytes)
                    metrics["screenshot_similarity"] = sim
                    self.logger.info(f"final-state metrics: screenshot_similarity={sim:.3f}")
        except Exception as e:
            self.logger.warning(f"screenshot similarity failed: {e}")

        # Final a11y vs expected
        try:
            ref_a11y = expected.get("a11y_after")
            if ref_a11y is not None:
                actual_a11y = await sandbox.display.get_windows()
                matched, total = _a11y_match(actual_a11y, ref_a11y)
                metrics["a11y_match"] = f"{matched}/{total}"
                metrics["a11y_match_ratio"] = (matched / total) if total else 0.0
                self.logger.info(f"final-state metrics: a11y_match={matched}/{total}")
        except Exception as e:
            self.logger.warning(f"a11y match failed: {e}")

        if metrics:
            (self.logs_dir / "verifier-metrics.json").write_text(json.dumps(metrics, indent=2))

    async def _fetch_reference_screenshot(self, task_id: str, filename: str) -> bytes | None:
        """GET <gateway>/admin/tasks/{task_id}/images/{filename}. Returns None on
        any failure — metrics are best-effort."""
        gateway = os.environ.get("MMINI_GATEWAY_URL") or os.environ.get("GATEWAY_URL", "")
        api_key = os.environ.get("MMINI_API_KEY", "")
        if not gateway:
            return None
        url = f"{gateway.rstrip('/')}/admin/tasks/{task_id}/images/{filename}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code != 200:
                    self.logger.warning(f"reference screenshot fetch {resp.status_code}: {url}")
                    return None
                return resp.content
        except Exception as e:
            self.logger.warning(f"reference screenshot fetch failed: {e}")
            return None

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


def _screenshot_similarity(a: bytes, b: bytes) -> float:
    """Pixel-wise similarity in [0,1]. 1.0 = identical, 0.0 = max difference.
    Resizes b to a's dimensions if they differ. Uses mean absolute pixel
    difference normalized by 255."""
    try:
        ia = Image.open(io.BytesIO(a)).convert("RGB")
        ib = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return 0.0
    if ib.size != ia.size:
        ib = ib.resize(ia.size)
    pa = list(ia.getdata())
    pb = list(ib.getdata())
    if not pa:
        return 0.0
    diff_sum = 0
    for (r1, g1, b1), (r2, g2, b2) in zip(pa, pb):
        diff_sum += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
    max_diff = len(pa) * 255 * 3
    return 1.0 - (diff_sum / max_diff if max_diff else 0.0)


def _a11y_match(actual: Any, expected: Any) -> tuple[int, int]:
    """Count node signatures from `expected` that also appear in `actual`.
    Returns (matched, total). Signature = (role, label-or-name-or-title).
    Lossy on purpose — coordinates / IDs are too volatile for replay."""
    actual_sigs = _collect_a11y_signatures(actual)
    expected_sigs = _collect_a11y_signatures(expected)
    if not expected_sigs:
        return (0, 0)
    matched = sum(1 for sig in expected_sigs if sig in actual_sigs)
    return (matched, len(expected_sigs))


def _collect_a11y_signatures(node: Any, out: set | None = None) -> set:
    if out is None:
        out = set()
    if isinstance(node, dict):
        role = (
            node.get("role")
            or node.get("AXRole")
            or node.get("type")
            or ""
        )
        ident = (
            node.get("AXLabel")
            or node.get("AXTitle")
            or node.get("AXValue")
            or node.get("label")
            or node.get("name")
            or node.get("title")
            or node.get("text")
            or ""
        )
        if role or ident:
            out.add((str(role)[:40], str(ident)[:80]))
        for v in node.values():
            _collect_a11y_signatures(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_a11y_signatures(v, out)
    return out
