"""OpenAI CUA agent — GPT computer-use via Responses API."""

from __future__ import annotations

import base64
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


class OpenAICUAAgent(BaseCUAAgent):
    @staticmethod
    def name() -> str:
        return "openai-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from openai import OpenAI

        sandbox = get_sandbox(environment)
        await self.start_recording(sandbox)
        await run_task_setup(environment, self.task_dir, self.logger)

        images_dir = self.logs_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        client = OpenAI()
        model = self._parsed_model_name or "computer-use-preview"

        system = build_system_prompt(
            load_prompt("openai.txt"),
            self.screen_width,
            self.screen_height,
        )

        # Initial screenshot
        ss = await sandbox.screenshot.take_full_screen()
        (images_dir / "step_000.png").write_bytes(ss)

        tools = [
            {
                "type": "computer_use_preview",
                "display_width": self.screen_width,
                "display_height": self.screen_height,
                "environment": "mac",
            }
        ]

        ss_b64 = base64.b64encode(ss).decode()
        input_items: list[Any] = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{ss_b64}",
                    }
                ],
            },
        ]

        steps: list[dict[str, Any]] = [
            {"step_id": 1, "source": "user", "message": instruction}
        ]
        total_in = 0
        total_out = 0

        for step_idx in range(self.max_steps):
            response = client.responses.create(
                model=model,
                tools=tools,  # type: ignore
                input=input_items,  # type: ignore
                truncation="auto",
            )

            total_in += response.usage.input_tokens if response.usage else 0
            total_out += response.usage.output_tokens if response.usage else 0

            computer_calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "computer_call"
            ]

            if not computer_calls:
                text_parts = [
                    getattr(item, "text", "")
                    for item in response.output
                    if getattr(item, "type", None) == "message"
                ]
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "agent",
                        "message": "\n".join(text_parts) or "Task complete.",
                    }
                )
                break

            for call in computer_calls:
                action = call.action
                mapped = _map_openai_action(action)

                result_text, ss_bytes, ss_path = await execute_action(
                    sandbox,
                    mapped,
                    images_dir,
                    step_idx + 1,
                )

                step_data: dict[str, Any] = {
                    "step_id": len(steps) + 1,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": call.call_id,
                            "function_name": action.type,
                            "arguments": mapped,
                        }
                    ],
                }
                if ss_path:
                    step_data["observation"] = {
                        "results": [
                            {
                                "content": [
                                    {"type": "text", "text": result_text},
                                    {
                                        "type": "image",
                                        "source": {
                                            "media_type": "image/png",
                                            "path": ss_path,
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                steps.append(step_data)

                if ss_bytes:
                    b64 = base64.b64encode(ss_bytes).decode()
                    input_items = response.output + [
                        {
                            "type": "computer_call_output",
                            "call_id": call.call_id,
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64}",
                            },
                        }
                    ]

        await self.stop_recording(sandbox)
        write_trajectory(
            self.logs_dir,
            steps,
            total_in,
            total_out,
            model,
            "openai-cua",
        )
        context.n_input_tokens = total_in
        context.n_output_tokens = total_out


def _map_openai_action(action: Any) -> dict[str, Any]:
    """Map OpenAI computer_use_preview action to our standard format."""
    action_type = action.type

    type_map = {
        "keypress": "key",
        "move": "move",
        "click": "click",
        "double_click": "double_click",
        "drag": "drag",
        "screenshot": "screenshot",
        "type": "type",
        "scroll": "scroll",
        "wait": "wait",
    }
    mapped: dict[str, Any] = {"action": type_map.get(action_type, action_type)}

    if hasattr(action, "x") and hasattr(action, "y"):
        x = getattr(action, "x", None)
        y = getattr(action, "y", None)
        if x is not None and y is not None:
            mapped["coordinate"] = [x, y]

    if hasattr(action, "keys") and action.keys:
        mapped["key"] = "+".join(action.keys)

    if hasattr(action, "text") and action.text:
        mapped["text"] = action.text

    if hasattr(action, "scroll_x"):
        sx = getattr(action, "scroll_x", 0) or 0
        sy = getattr(action, "scroll_y", 0) or 0
        if sy < 0:
            mapped["direction"] = "down"
            mapped["amount"] = abs(sy)
        elif sy > 0:
            mapped["direction"] = "up"
            mapped["amount"] = sy
        elif sx < 0:
            mapped["direction"] = "left"
            mapped["amount"] = abs(sx)
        else:
            mapped["direction"] = "right"
            mapped["amount"] = abs(sx) or 3

    if hasattr(action, "start_x") and action.start_x is not None:
        mapped["start_coordinate"] = [action.start_x, action.start_y]
        if hasattr(action, "x"):
            mapped["coordinate"] = [action.x, action.y]

    if hasattr(action, "button") and action.button == "right":
        mapped["action"] = "right_click"

    return mapped
