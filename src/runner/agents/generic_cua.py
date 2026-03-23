"""Generic CUA agent — any vision LLM via litellm, pyautogui-style output."""

from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path
from typing import Any

from runner.agents.base_cua import (
    BaseCUAAgent,
    build_system_prompt,
    execute_action,
    get_sandbox,
    load_prompt,
    run_task_setup,
    write_trajectory,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class GenericCUAAgent(BaseCUAAgent):
    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        prompt_template: str = "pyautogui.txt",
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._prompt_template = load_prompt(prompt_template)

    @staticmethod
    def name() -> str:
        return "generic-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        import litellm

        sandbox = get_sandbox(environment)
        await self.start_recording(sandbox)
        await run_task_setup(environment, self.task_dir, self.logger)

        images_dir = self.logs_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        model = self.model_name or "openai/gpt-4o"

        ss = await sandbox.screenshot.take_full_screen()
        (images_dir / "step_000.png").write_bytes(ss)

        steps: list[dict[str, Any]] = [
            {"step_id": 1, "source": "user", "message": instruction}
        ]
        total_in = 0
        total_out = 0

        # Conversation history (sliding window)
        max_history = 6  # keep last 3 turns (user+assistant pairs)
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
            system += (
                f"\n\nYou are on step {step_idx + 1} of "
                f"{self.max_steps}. Act efficiently."
            )

            ss_b64 = base64.b64encode(ss).decode()
            if step_idx == 0:
                user_text = "Here is the current screenshot. Complete the task."
            else:
                user_text = "What's the next step?"

            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{ss_b64}",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            )

            # Sliding window
            if len(history) > max_history:
                history = history[-max_history:]

            messages = [
                {"role": "system", "content": system},
                *history,
            ]

            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=1024,
            )

            history.append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                }
            )

            usage = response.usage
            total_in += usage.prompt_tokens or 0
            total_out += usage.completion_tokens or 0

            text = response.choices[0].message.content or ""

            # Check for terminal responses (anywhere in text)
            # Find ALL code blocks (model may output multiple actions)
            code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
            code_match = bool(code_blocks)
            has_done = "DONE" in text or "```DONE```" in text
            has_fail = "FAIL" in text or "```FAIL```" in text
            has_wait = "WAIT" in text or "```WAIT```" in text

            if not code_match and has_done:
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "agent",
                        "message": text[:500],
                    }
                )
                break
            if not code_match and has_fail:
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "agent",
                        "message": text[:500],
                    }
                )
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
                    sandbox,
                    action,
                    images_dir,
                    step_idx + 1,
                )
                if ss_bytes:
                    ss = ss_bytes
                    last_ss_path = ss_path

            step_data: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": text[:500],
                "metrics": {
                    "prompt_tokens": usage.prompt_tokens or 0,
                    "completion_tokens": usage.completion_tokens or 0,
                },
            }
            if last_ss_path:
                step_data["observation"] = {
                    "results": [
                        {
                            "content": [
                                {"type": "text", "text": "After execution"},
                                {
                                    "type": "image",
                                    "source": {
                                        "media_type": "image/png",
                                        "path": last_ss_path,
                                    },
                                },
                            ],
                        }
                    ],
                }
            steps.append(step_data)

            if not actions:
                ss = await sandbox.screenshot.take_full_screen()

        await self.stop_recording(sandbox)
        write_trajectory(
            self.logs_dir,
            steps,
            total_in,
            total_out,
            model,
            "generic-cua",
        )
        context.n_input_tokens = total_in
        context.n_output_tokens = total_out


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
                a: dict[str, Any] = {
                    "action": "click",
                    "coordinate": [int(args[0]), int(args[1])],
                }
                if "'right'" in line or '"right"' in line:
                    a["action"] = "right_click"
                actions.append(a)

        elif "pyautogui.doubleClick(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append(
                    {
                        "action": "double_click",
                        "coordinate": [int(args[0]), int(args[1])],
                    }
                )

        elif "pyautogui.moveTo(" in line:
            args = _args(line)
            if len(args) >= 2:
                actions.append(
                    {
                        "action": "move",
                        "coordinate": [int(args[0]), int(args[1])],
                    }
                )

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
                    {
                        "action": "drag",
                        "start_coordinate": [0, 0],
                        "coordinate": [int(args[0]), int(args[1])],
                    }
                )

        elif "time.sleep(" in line:
            args = _args(line)
            if args:
                actions.append(
                    {
                        "action": "wait",
                        "duration": float(args[0]),
                    }
                )

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
