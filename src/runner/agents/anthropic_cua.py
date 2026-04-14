"""Anthropic CUA agent — Claude computer-use with structured tool calls."""

from __future__ import annotations

import base64
import io
from typing import Any

import anthropic
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from PIL import Image

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
        self.steps[0]["message"] = instruction

        client = anthropic.Anthropic()
        model = self._parsed_model_name or "claude-sonnet-4-6"

        uses_new = any(t in model for t in ("opus-4-6", "sonnet-4-6"))
        tool_type = "computer_20251124" if uses_new else "computer_20250124"
        beta = "computer-use-2025-11-24" if uses_new else "computer-use-2025-01-24"

        # Anthropic's vision model internally resizes screenshots.
        # Declaring 1280x800 ensures coordinates match the model's internal space.
        # We scale coordinates to actual VM resolution (1920x1080) in execute_action.
        self._api_width = 1280
        self._api_height = 800
        computer_tool: dict[str, Any] = {
            "type": tool_type,
            "name": "computer",
            "display_width_px": self._api_width,
            "display_height_px": self._api_height,
            "display_number": 1,
        }

        # Scale factors: model coords → VM coords
        self._sx = self.screen_width / self._api_width
        self._sy = self.screen_height / self._api_height

        system = build_system_prompt(
            load_prompt("anthropic.txt"),
            self._api_width,
            self._api_height,
        )

        def resize_ss(png_bytes: bytes) -> bytes:
            """Resize screenshot to API dimensions."""
            img = Image.open(io.BytesIO(png_bytes))
            if img.size != (self._api_width, self._api_height):
                img = img.resize((self._api_width, self._api_height), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

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
                            "data": base64.b64encode(resize_ss(ss)).decode(),
                        },
                    },
                ],
            }
        ]

        for step_idx in range(self.max_steps):
            await self._fire_in_process(environment, step_idx)
            _truncate_old_screenshots(messages, keep=5)

            self.logger.info(f"step {step_idx+1}/{self.max_steps}: calling Anthropic API...")
            import time as _time
            _t0 = _time.monotonic()
            try:
                response = client.beta.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system,
                    tools=[computer_tool],  # type: ignore
                    messages=messages,  # type: ignore
                    betas=[beta],
                    # Top-level cache_control is the 4.6-era API for prompt caching
                    # (block-level cache_control on system/tools is silently ignored on 4.6).
                    # Caches system + tools + stable message prefix; _truncate_old_screenshots
                    # invalidates the message suffix but system+tools survive = ~60% input cost cut.
                    cache_control={"type": "ephemeral"},
                )
            except Exception as e:
                self.logger.error(f"step {step_idx+1}: Anthropic API error after {_time.monotonic()-_t0:.1f}s: {type(e).__name__}: {e}")
                raise
            u = response.usage
            cache_r = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_w = getattr(u, "cache_creation_input_tokens", 0) or 0
            self.logger.info(f"step {step_idx+1}: API responded in {_time.monotonic()-_t0:.1f}s, in={u.input_tokens} out={u.output_tokens} cache_r={cache_r} cache_w={cache_w}, stop={response.stop_reason}")

            self.total_in += u.input_tokens
            self.total_out += u.output_tokens
            self._total_cache_read = getattr(self, "_total_cache_read", 0) + cache_r
            self._total_cache_write = getattr(self, "_total_cache_write", 0) + cache_w
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

                # Scale coordinates from API space (1280x800) to VM space (1920x1080)
                scaled = dict(action)
                if "coordinate" in scaled:
                    x, y = scaled["coordinate"]
                    scaled["coordinate"] = [int(x * self._sx), int(y * self._sy)]
                if "start_coordinate" in scaled:
                    x, y = scaled["start_coordinate"]
                    scaled["start_coordinate"] = [int(x * self._sx), int(y * self._sy)]

                result_text, ss_bytes, ss_path = await execute_action(
                    sandbox, scaled, self.images_dir, step_idx + 1,
                )

                content: list[dict[str, Any]] = [{"type": "text", "text": result_text}]
                if ss_bytes:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(resize_ss(ss_bytes)).decode(),
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
            self.checkpoint(context, model, "anthropic-cua")

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
