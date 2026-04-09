"""Generic CUA agent — any vision LLM via litellm or tinker, pyautogui-style output."""

from __future__ import annotations

import asyncio
import base64
import os
import re
from pathlib import Path
from typing import Any

import litellm
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

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

        ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / "step_000.png").write_bytes(ss)

        # Conversation history (sliding window)
        max_history = 6
        history: list[dict[str, Any]] = []

        for step_idx in range(self.max_steps):
            system = build_system_prompt(
                self._prompt_template,
                self.screen_width,
                self.screen_height,
                instruction=instruction,
                step=step_idx + 1,
                max_steps=self.max_steps,
            )
            system += f"\n\nYou are on step {step_idx + 1} of {self.max_steps}. Act efficiently."

            ss_b64 = base64.b64encode(ss).decode()
            user_text = "Here is the current screenshot. Complete the task." if step_idx == 0 else "What's the next step?"

            history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ss_b64}"}},
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
            actions = _parse_pyautogui(all_code)

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


def _parse_pyautogui(code: str) -> list[dict[str, Any]]:
    """Parse pyautogui code into action dicts."""
    actions: list[dict[str, Any]] = []

    for line in code.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "pyautogui.click(" in line:
            args = _args(line)
            if len(args) >= 2:
                a: dict[str, Any] = {"action": "click", "coordinate": [int(args[0]), int(args[1])]}
                if "'right'" in line or '"right"' in line:
                    a["action"] = "right_click"
                actions.append(a)

        elif "pyautogui.doubleClick(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append({"action": "double_click", "coordinate": [int(args[0]), int(args[1])]})

        elif "pyautogui.moveTo(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append({"action": "move", "coordinate": [int(args[0]), int(args[1])]})

        elif "pyautogui.scroll(" in line:
            args = _args(line)
            clicks = int(args[0]) if args else -3
            x = int(args[1]) if len(args) > 1 else 0
            y = int(args[2]) if len(args) > 2 else 0
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

        elif "pyautogui.drag(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append(
                    {"action": "drag", "start_coordinate": [0, 0], "coordinate": [int(args[0]), int(args[1])]}
                )

        elif "time.sleep(" in line:
            args = _args(line)
            if args:
                actions.append({"action": "wait", "duration": float(args[0])})

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
    m = re.search(r"""['"](.*?)['"]""", line)
    return m.group(1) if m else ""


def _str_args(line: str) -> list[str]:
    return re.findall(r"""['"]([^'"]+)['"]""", line)
