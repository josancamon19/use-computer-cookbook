"""Anthropic CUA agent — Claude computer-use with structured tool calls."""

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


class AnthropicCUAAgent(BaseCUAAgent):
    @staticmethod
    def name() -> str:
        return "anthropic-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from anthropic import Anthropic

        sandbox = get_sandbox(environment)
        await self.start_recording(sandbox)
        await run_task_setup(environment, self.task_dir, self.logger)

        images_dir = self.logs_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        client = Anthropic()
        model = self._parsed_model_name or "claude-sonnet-4-6"

        uses_new = any(t in model for t in ("opus-4-6", "sonnet-4-6"))
        tool_type = "computer_20251124" if uses_new else "computer_20250124"
        beta = "computer-use-2025-11-24" if uses_new else "computer-use-2025-01-24"

        computer_tool: dict[str, Any] = {
            "type": tool_type,
            "name": "computer",
            "display_width_px": self.screen_width,
            "display_height_px": self.screen_height,
            "display_number": 1,
        }

        system = build_system_prompt(
            load_prompt("anthropic.txt"),
            self.screen_width,
            self.screen_height,
        )

        # Initial screenshot
        ss = await sandbox.screenshot.take_full_screen()
        (images_dir / "step_000.png").write_bytes(ss)

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(ss).decode(),
                        },
                    },
                ],
            }
        ]

        steps: list[dict[str, Any]] = [
            {"step_id": 1, "source": "user", "message": instruction}
        ]
        total_in = 0
        total_out = 0

        for step_idx in range(self.max_steps):
            response = client.beta.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                tools=[computer_tool],  # type: ignore
                messages=messages,  # type: ignore
                betas=[beta],
            )

            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                if not any(b.type == "tool_use" for b in response.content):
                    steps.append(
                        {
                            "step_id": len(steps) + 1,
                            "source": "agent",
                            "message": _text(response.content),
                        }
                    )
                    break

            tool_results: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []
            obs_results: list[dict[str, Any]] = []

            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                action = block.input
                tool_calls.append(
                    {
                        "tool_call_id": block.id,
                        "function_name": action.get("action", ""),
                        "arguments": action,
                    }
                )

                result_text, ss_bytes, ss_path = await execute_action(
                    sandbox,
                    action,
                    images_dir,
                    step_idx + 1,
                )

                # Tool result for the LLM conversation
                content: list[dict[str, Any]] = [{"type": "text", "text": result_text}]
                if ss_bytes:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(ss_bytes).decode(),
                            },
                        }
                    )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                )

                # Observation for the trajectory (with image path)
                obs: dict[str, Any] = {
                    "source_call_id": block.id,
                    "content": result_text,
                }
                if ss_path:
                    obs["content"] = [
                        {"type": "text", "text": result_text},
                        {
                            "type": "image",
                            "source": {
                                "media_type": "image/png",
                                "path": ss_path,
                            },
                        },
                    ]
                obs_results.append(obs)

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            step_data: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": _text(response.content) or None,
                "tool_calls": tool_calls,
                "metrics": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                },
            }
            if obs_results:
                step_data["observation"] = {"results": obs_results}
            steps.append(step_data)

        await self.stop_recording(sandbox)
        write_trajectory(
            self.logs_dir,
            steps,
            total_in,
            total_out,
            model,
            "anthropic-cua",
        )
        context.n_input_tokens = total_in
        context.n_output_tokens = total_out


def _text(content: Any) -> str:
    parts = []
    if isinstance(content, list):
        for b in content:
            if hasattr(b, "text"):
                parts.append(b.text)
    return "\n".join(parts) or "Task complete."
