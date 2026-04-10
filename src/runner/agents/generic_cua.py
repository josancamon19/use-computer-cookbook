"""Generic CUA agent — any vision LLM via litellm or tinker, pyautogui-style output."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from pathlib import Path
from typing import Any

import litellm
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from PIL import Image

from runner.agents.base_cua import (
    BaseCUAAgent,
    build_system_prompt,
    execute_action,
    load_prompt,
)


class GenericCUAAgent(BaseCUAAgent):
    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        prompt_template: str = "pyautogui.txt",
        llm_backend: str = "litellm",
        max_tokens: int = 4096,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._prompt_template = load_prompt(prompt_template)
        self._llm_backend = llm_backend
        self._max_tokens = max_tokens
        # If api_base points at Fireworks, fall back to FIREWORKS_API_KEY env var.
        # litellm with openai/* prefix would otherwise look for OPENAI_API_KEY.
        if api_key is None and api_base and "fireworks" in api_base:
            api_key = os.environ.get("FIREWORKS_API_KEY")
        self._api_key = api_key
        self._api_base = api_base
        self._tinker_llm = None

    @staticmethod
    def name() -> str:
        return "generic-cua"

    async def _get_completion(self, model: str, messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
        """Call LLM and return (text, prompt_tokens, completion_tokens)."""
        if self._llm_backend == "tinker":
            return await self._tinker_completion(model, messages, max_tokens)
        else:
            return self._litellm_completion(model, messages, max_tokens)

    def _litellm_completion(self, model: str, messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, usage.prompt_tokens or 0, usage.completion_tokens or 0

    async def _tinker_completion(self, model: str, messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
        if self._tinker_llm is None:
            from harbor.llms.tinker import TinkerLLM
            self._tinker_llm = TinkerLLM(
                model_name=model,
                max_tokens=max_tokens,
            )

        # TinkerLLM.call() expects (prompt, message_history)
        # Split system + history from the final user message
        history = messages[:-1] if len(messages) > 1 else []
        prompt = ""
        last_msg = messages[-1] if messages else {"content": ""}
        if isinstance(last_msg.get("content"), str):
            prompt = last_msg["content"]
        elif isinstance(last_msg.get("content"), list):
            # Extract text parts from multimodal content
            prompt = " ".join(
                p.get("text", "") for p in last_msg["content"] if p.get("type") == "text"
            )

        response = await self._tinker_llm.call(prompt=prompt, message_history=history)
        content = response.content or ""
        text = content if isinstance(content, str) else "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        return text, prompt_tokens, completion_tokens

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        sandbox = await self.pre_run(environment)
        self.steps[0]["message"] = instruction

        model = self.model_name or "openai/gpt-4o"

        # The model sees a 1280x800 image (resized from VM's 1920x1080) so the
        # prompt MUST tell it the screen is 1280x800 — otherwise the model emits
        # coords for the wrong space. We then scale model coords back to VM
        # pixel space before clicking. (Mirrors anthropic_cua.)
        api_w, api_h = 1280, 800
        sx = self.screen_width / api_w
        sy = self.screen_height / api_h

        def _resize_for_api(png_bytes: bytes) -> bytes:
            """Resize and re-encode as JPEG q80. JPEG is ~17x smaller than PNG
            for screenshots and vision models read it identically. Fireworks
            rejected even single 1280x800 PNGs (~1.4MB b64) as 'Payload Too
            Large'; JPEG keeps the full sliding history under 1MB total."""
            img = Image.open(io.BytesIO(png_bytes))
            if img.size != (api_w, api_h):
                img = img.resize((api_w, api_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=80)
            return buf.getvalue()

        def _scale_action(a: dict[str, Any]) -> dict[str, Any]:
            """Scale a parsed action's coordinates from API space (1280x800) to VM space."""
            for field in ("coordinate", "start_coordinate"):
                if field in a and isinstance(a[field], list) and len(a[field]) == 2:
                    a[field] = [int(a[field][0] * sx), int(a[field][1] * sy)]
            return a

        ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / "step_000.png").write_bytes(ss)

        # Conversation history (sliding window)
        max_history = 6
        history: list[dict[str, Any]] = []

        for step_idx in range(self.max_steps):
            await self._fire_in_process(environment, step_idx)
            system = build_system_prompt(
                self._prompt_template,
                api_w,
                api_h,
                instruction=instruction,
                step=step_idx + 1,
                max_steps=self.max_steps,
            )
            system += f"\n\nYou are on step {step_idx + 1} of {self.max_steps}. Act efficiently."

            ss_b64 = base64.b64encode(_resize_for_api(ss)).decode()
            user_text = "Here is the current screenshot. Complete the task." if step_idx == 0 else "What's the next step?"

            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ss_b64}"}},
                        {"type": "text", "text": user_text},
                    ],
                }
            )

            if len(history) > max_history:
                history = history[-max_history:]

            messages = [{"role": "system", "content": system}, *history]

            text, in_tok, out_tok = await self._get_completion(model, messages, self._max_tokens)
            self.total_in += in_tok
            self.total_out += out_tok

            history.append({"role": "assistant", "content": text})

            code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
            code_match = bool(code_blocks)
            has_done = "DONE" in text or "```DONE```" in text
            has_fail = "FAIL" in text or "```FAIL```" in text
            has_wait = "WAIT" in text or "```WAIT```" in text

            if not code_match and has_done:
                self.steps.append({"step_id": len(self.steps) + 1, "source": "agent", "message": text[:500]})
                break
            if not code_match and has_fail:
                self.steps.append({"step_id": len(self.steps) + 1, "source": "agent", "message": text[:500]})
                break
            if not code_match and has_wait:
                await asyncio.sleep(2)
                ss = await sandbox.screenshot.take_full_screen()
                continue
            if not code_match:
                self.logger.warning(f"No code block: {text[:200]}")
                continue

            all_code = "\n".join(code_blocks)
            # Parse using API dims (1280x800) — the model thinks that's the screen.
            # Then scale each coordinate up to VM pixel space.
            actions = _parse_pyautogui(all_code, api_w, api_h)
            actions = [_scale_action(a) for a in actions]

            last_ss_path: str | None = None
            for action in actions:
                _, ss_bytes, ss_path = await execute_action(
                    sandbox, action, self.images_dir, step_idx + 1,
                )
                if ss_bytes:
                    ss = ss_bytes
                    last_ss_path = ss_path

            step_data: dict[str, Any] = {
                "step_id": len(self.steps) + 1,
                "source": "agent",
                "message": text[:500],
                "metrics": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                },
            }
            if last_ss_path:
                step_data["observation"] = {
                    "results": [
                        {
                            "content": [
                                {"type": "text", "text": "After execution"},
                                {"type": "image", "source": {"media_type": "image/png", "path": last_ss_path}},
                            ],
                        }
                    ],
                }
            self.steps.append(step_data)

            if not actions:
                ss = await sandbox.screenshot.take_full_screen()

        await self.post_run(context, model, "generic-cua")


# ── pyautogui parser ────────────────────────────────────────────────


def _parse_coord(s: str, dim: int) -> int:
    """Parse a coordinate from a model-emitted string.

    Tolerates floats (some models ignore "use integers" prompts) and
    normalized [0, 1] floats (scaled to `dim`). Falls back to 0 on garbage.
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return 0
    # Normalized coordinate (0..1) — scale to screen dim. Use < 1 (not <= 1)
    # so that "1" or "1.0" stays a literal pixel, not the right edge.
    if 0.0 <= f < 1.0:
        return int(f * dim)
    return int(f)


def _parse_int(s: str) -> int:
    """Parse an int field that should not be scaled (e.g. scroll clicks, drag delta)."""
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


_MODIFIER_KEYS = {"ctrl", "control", "alt", "option", "shift", "cmd", "command", "win", "super", "meta"}


def _parse_pyautogui(code: str, screen_w: int = 1920, screen_h: int = 1080) -> list[dict[str, Any]]:
    """Parse pyautogui code into action dicts."""
    actions: list[dict[str, Any]] = []
    held_mods: list[str] = []  # ordered list of held modifier keys for keyDown/keyUp pairs

    for line in code.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "pyautogui.tripleClick(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append(
                    {
                        "action": "triple_click",
                        "coordinate": [_parse_coord(args[0], screen_w), _parse_coord(args[1], screen_h)],
                    }
                )

        elif "pyautogui.doubleClick(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append(
                    {
                        "action": "double_click",
                        "coordinate": [_parse_coord(args[0], screen_w), _parse_coord(args[1], screen_h)],
                    }
                )

        elif "pyautogui.click(" in line:
            args = _args(line)
            if len(args) >= 2:
                a: dict[str, Any] = {
                    "action": "click",
                    "coordinate": [_parse_coord(args[0], screen_w), _parse_coord(args[1], screen_h)],
                }
                if "'right'" in line or '"right"' in line:
                    a["action"] = "right_click"
                actions.append(a)

        elif "pyautogui.moveTo(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append(
                    {
                        "action": "move",
                        "coordinate": [_parse_coord(args[0], screen_w), _parse_coord(args[1], screen_h)],
                    }
                )

        elif "pyautogui.scroll(" in line:
            args = _args(line)
            clicks = _parse_int(args[0]) if args else -3
            x = _parse_coord(args[1], screen_w) if len(args) > 1 else 0
            y = _parse_coord(args[2], screen_h) if len(args) > 2 else 0
            actions.append(
                {
                    "action": "scroll",
                    "coordinate": [x, y],
                    "direction": "up" if clicks > 0 else "down",
                    "amount": abs(clicks),
                }
            )

        elif "pyautogui.typewrite(" in line or "pyautogui.write(" in line:
            text = _str_arg(line)
            if text:
                actions.append({"action": "type", "text": text})

        elif "pyautogui.press(" in line:
            key = _str_arg(line)
            if key:
                actions.append({"action": "key", "key": key})

        elif "pyautogui.hotkey(" in line:
            keys = _str_args(line)
            if keys:
                actions.append({"action": "key", "key": "+".join(keys)})

        elif "pyautogui.keyDown(" in line:
            key = _str_arg(line)
            if key:
                if key.lower() in _MODIFIER_KEYS:
                    held_mods.append(key)
                else:
                    # non-modifier key — fire it now combined with any held mods
                    combo = "+".join(held_mods + [key])
                    actions.append({"action": "key", "key": combo})

        elif "pyautogui.keyUp(" in line:
            key = _str_arg(line)
            if key and key in held_mods:
                held_mods.remove(key)
            # otherwise: partner of an already-emitted keyDown, ignore

        elif "pyautogui.drag(" in line:
            args = _args(line)
            if len(args) >= 2:
                # drag uses relative deltas, not absolute coordinates — don't normalize
                actions.append(
                    {
                        "action": "drag",
                        "start_coordinate": [0, 0],
                        "coordinate": [_parse_int(args[0]), _parse_int(args[1])],
                    }
                )

        elif "time.sleep(" in line or "pyautogui.sleep(" in line:
            args = _args(line)
            if args:
                try:
                    duration = float(args[0])
                except ValueError:
                    duration = 1.0
                actions.append({"action": "wait", "duration": duration})

    return actions


def _args(line: str) -> list[str]:
    m = re.search(r"\(([^)]*)\)", line)
    if not m:
        return []
    parts = []
    for p in m.group(1).split(","):
        p = p.strip()
        if "=" in p:
            p = p.split("=", 1)[1].strip()
        p = p.strip("'\"")
        if p:
            parts.append(p)
    return parts


def _str_arg(line: str) -> str:
    """Extract first string literal. Handles ''' """ + '"""' + """, ''', ", '."""
    # triple-quoted forms first (longer match wins)
    m = re.search(r'"""(.*?)"""', line, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"'''(.*?)'''", line, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'"([^"]*)"', line)
    if m:
        return m.group(1)
    m = re.search(r"'([^']*)'", line)
    if m:
        return m.group(1)
    return ""


def _str_args(line: str) -> list[str]:
    """Extract all string literals. Handles triple-quoted and single-quoted forms."""
    out: list[str] = []
    # Strip triple-quoted matches first so they don't get double-counted
    remaining = line
    for pat in (r'"""(.*?)"""', r"'''(.*?)'''"):
        for m in re.finditer(pat, remaining, re.DOTALL):
            out.append(m.group(1))
        remaining = re.sub(pat, "", remaining, flags=re.DOTALL)
    for m in re.finditer(r'"([^"]*)"', remaining):
        out.append(m.group(1))
    for m in re.finditer(r"'([^']*)'", remaining):
        out.append(m.group(1))
    return out
