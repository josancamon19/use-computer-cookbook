"""Anthropic CUA agent — Claude computer-use with structured tool calls."""

from __future__ import annotations

import base64
from typing import Any

import anthropic
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from runner.agents.base_cua import (
    BaseCUAAgent,
    build_system_prompt,
    execute_action,
    load_prompt,
)


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
        sandbox = await self.pre_run(environment)
        return
        self.steps[0]["message"] = instruction

        client = anthropic.Anthropic()
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
        (self.images_dir / "step_000.png").write_bytes(ss)

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

        for step_idx in range(self.max_steps):
            _truncate_old_screenshots(messages, keep=5)

            response = client.beta.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                tools=[computer_tool],  # type: ignore
                messages=messages,  # type: ignore
                betas=[beta],
            )

            self.total_in += response.usage.input_tokens
            self.total_out += response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                if not any(b.type == "tool_use" for b in response.content):
                    self.steps.append(
                        {
                            "step_id": len(self.steps) + 1,
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
                    sandbox, action, self.images_dir, step_idx + 1,
                )

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
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

                obs: dict[str, Any] = {"source_call_id": block.id, "content": result_text}
                if ss_path:
                    obs["content"] = [
                        {"type": "text", "text": result_text},
                        {"type": "image", "source": {"media_type": "image/png", "path": ss_path}},
                    ]
                obs_results.append(obs)

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            step_data: dict[str, Any] = {
                "step_id": len(self.steps) + 1,
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
            self.steps.append(step_data)

        await self.post_run(context, model, "anthropic-cua")


def _truncate_old_screenshots(messages: list[dict[str, Any]], keep: int = 5) -> None:
    """Remove base64 images from older messages, keeping only the last `keep`."""
    image_indices: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "image":
                image_indices.append((i, j))
            elif isinstance(item, dict) and item.get("type") == "tool_result":
                for k, sub in enumerate(
                    item.get("content", []) if isinstance(item.get("content"), list) else []
                ):
                    if isinstance(sub, dict) and sub.get("type") == "image":
                        image_indices.append((i, (j, k)))

    to_remove = len(image_indices) - keep
    if to_remove <= 0:
        return

    for idx in image_indices[:to_remove]:
        if isinstance(idx[1], tuple):
            j, k = idx[1]
            content_list = messages[idx[0]]["content"][j].get("content", [])
            if isinstance(content_list, list) and k < len(content_list):
                content_list[k] = {"type": "text", "text": "[screenshot removed]"}
        else:
            messages[idx[0]]["content"][idx[1]] = {"type": "text", "text": "[screenshot removed]"}


def _text(content: Any) -> str:
    parts = []
    if isinstance(content, list):
        for b in content:
            if hasattr(b, "text"):
                parts.append(b.text)
    return "\n".join(parts) or "Task complete."
