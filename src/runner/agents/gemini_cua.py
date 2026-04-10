"""Gemini CUA agent — Google's computer-use tool via google-genai SDK.

Converts Gemini's normalized (0-999) coordinate actions into our standard
action format for execution via the SDK, following the same BaseCUAAgent
pattern as the Anthropic and OpenAI agents.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Content, FunctionResponse, FunctionResponseBlob, FunctionResponsePart, Part
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from runner.agents.base_cua import (
    BaseCUAAgent,
    execute_action,
    load_prompt,
)

logger = logging.getLogger(__name__)

API_RETRY_TIMES = 5
API_RETRY_INTERVAL = 5

# Browser-only actions that don't apply to desktop VMs
_EXCLUDED_ACTIONS = ["navigate", "go_back", "go_forward", "open_web_browser", "search"]


def _denormalize(coord: int, screen_dim: int) -> int:
    """Convert Gemini's 0-999 normalized coordinate to pixel coordinate."""
    return int(coord / 1000 * screen_dim)


def _map_gemini_action(name: str, args: dict, screen_w: int, screen_h: int) -> dict[str, Any] | None:
    """Convert a Gemini function call into our standard action dict."""
    x = _denormalize(args.get("x", 0), screen_w)
    y = _denormalize(args.get("y", 0), screen_h)

    if name == "click_at":
        return {"action": "click", "coordinate": [x, y]}

    if name == "double_click_at":
        return {"action": "double_click", "coordinate": [x, y]}

    if name == "right_click_at":
        return {"action": "right_click", "coordinate": [x, y]}

    if name == "type_text_at":
        text = args.get("text", "")
        # Click first to focus, then type
        actions_text = text
        return {"action": "type", "text": actions_text, "_click_first": [x, y],
                "_clear": args.get("clear_before_typing", False),
                "_enter": args.get("press_enter", False)}

    if name == "hover_at":
        return {"action": "move", "coordinate": [x, y]}

    if name in ("scroll_document", "scroll_at"):
        direction = args.get("direction", "down")
        amount = args.get("magnitude", 3)
        coord = [x, y] if name == "scroll_at" else [screen_w // 2, screen_h // 2]
        return {"action": "scroll", "coordinate": coord, "direction": direction, "amount": amount}

    if name == "key_combination":
        keys_str = args.get("keys", "")
        keys = [k.strip().lower() for k in keys_str.split("+")]
        key_map = {
            "control": "ctrl", "meta": "command", "command": "command",
            "arrowup": "up", "arrowdown": "down", "arrowleft": "left",
            "arrowright": "right", "escape": "esc",
        }
        mapped = [key_map.get(k, k) for k in keys]
        if len(mapped) == 1:
            return {"action": "key", "key": mapped[0]}
        return {"action": "key", "key": "+".join(mapped)}

    if name == "drag_and_drop":
        dest_x = _denormalize(args.get("destination_x", 0), screen_w)
        dest_y = _denormalize(args.get("destination_y", 0), screen_h)
        return {"action": "drag", "start_coordinate": [x, y], "coordinate": [dest_x, dest_y]}

    if name == "wait_5_seconds":
        return {"action": "wait", "duration": 5}

    if name in _EXCLUDED_ACTIONS:
        logger.warning("Gemini returned browser action '%s' — skipping", name)
        return None

    logger.warning("Unknown Gemini action: %s (args=%s)", name, args)
    return None


class GeminiCUAAgent(BaseCUAAgent):
    @staticmethod
    def name() -> str:
        return "gemini-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        sandbox = await self.pre_run(environment)
        self.steps[0]["message"] = instruction

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        model = self._parsed_model_name or "gemini-2.5-flash-preview-04-17"

        prompt_text = load_prompt("gemini.txt")
        system_instruction = (
            prompt_text
            .replace("{DATE}", time.strftime("%A, %B %d, %Y"))
            .replace("{CLIENT_PASSWORD}", "lume")
        )

        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_UNSPECIFIED,
                        excluded_predefined_functions=_EXCLUDED_ACTIONS,
                    )
                )
            ],
            system_instruction=system_instruction,
        )

        # Initial screenshot
        ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / "step_000.png").write_bytes(ss)

        contents: list = [
            Content(
                role="user",
                parts=[
                    Part(text=instruction),
                    Part.from_bytes(data=ss, mime_type="image/png"),
                ],
            )
        ]

        pending_calls: list = []

        for step_idx in range(self.max_steps):
            await self._fire_in_process(environment, step_idx)
            # Feed back screenshot for pending function calls
            if step_idx > 0 and pending_calls:
                parts = []
                for fc in pending_calls:
                    fr = FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"url": "about:blank"},
                        parts=[
                            FunctionResponsePart(
                                inline_data=FunctionResponseBlob(
                                    mime_type="image/png",
                                    data=ss,
                                )
                            )
                        ],
                    )
                    parts.append(Part(function_response=fr))
                contents.append(Content(role="user", parts=parts))

            # API call with retry
            response = None
            for attempt in range(API_RETRY_TIMES):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    if "400" in err_str or "INVALID_ARGUMENT" in err_str:
                        logger.error("Gemini client error (not retryable): %s", err_str[:500])
                        self.steps.append({
                            "step_id": len(self.steps) + 1,
                            "source": "agent",
                            "message": f"API error: {err_str[:300]}",
                        })
                        await self.post_run(context, model, "gemini-cua")
                        return
                    logger.warning("Gemini API error (attempt %d/%d): %s", attempt + 1, API_RETRY_TIMES, err_str[:300])
                    if attempt < API_RETRY_TIMES - 1:
                        time.sleep(API_RETRY_INTERVAL)

            if response is None:
                self.steps.append({
                    "step_id": len(self.steps) + 1,
                    "source": "agent",
                    "message": "All API attempts failed",
                })
                break

            # Track usage
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta:
                self.total_in += getattr(usage_meta, "prompt_token_count", 0) or 0
                self.total_out += getattr(usage_meta, "candidates_token_count", 0) or 0

            # Parse response
            candidate = response.candidates[0] if response.candidates else None
            content = candidate.content if candidate else None
            if content is None or content.parts is None:
                self.steps.append({
                    "step_id": len(self.steps) + 1,
                    "source": "agent",
                    "message": "No content from model",
                })
                # Take new screenshot and retry
                ss = await sandbox.screenshot.take_full_screen()
                pending_calls = []
                continue

            contents.append(content)
            pending_calls = []

            # Extract actions and text
            tool_calls: list[dict[str, Any]] = []
            obs_results: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for part in content.parts:
                if part.function_call:
                    fc = part.function_call
                    pending_calls.append(fc)
                    args = dict(fc.args) if fc.args else {}
                    mapped = _map_gemini_action(fc.name, args, self.screen_width, self.screen_height)

                    if mapped is None:
                        continue

                    tool_calls.append({
                        "tool_call_id": getattr(fc, "id", fc.name),
                        "function_name": fc.name,
                        "arguments": mapped,
                    })

                    # Handle type_text_at special: click first, optionally clear, then type
                    if mapped.get("_click_first"):
                        cx, cy = mapped["_click_first"]
                        await sandbox.mouse.click(cx, cy)
                        await asyncio.sleep(0.3)
                        if mapped.get("_clear"):
                            await sandbox.keyboard.hotkey("command+a")
                            await asyncio.sleep(0.1)
                            await sandbox.keyboard.press("Delete")
                            await asyncio.sleep(0.1)
                        # Now execute the type action
                        clean_action = {"action": "type", "text": mapped["text"]}
                        result_text, ss_bytes, ss_path = await execute_action(
                            sandbox, clean_action, self.images_dir, step_idx + 1,
                        )
                        if mapped.get("_enter"):
                            await sandbox.keyboard.press("Return")
                    else:
                        result_text, ss_bytes, ss_path = await execute_action(
                            sandbox, mapped, self.images_dir, step_idx + 1,
                        )

                    if ss_bytes:
                        ss = ss_bytes

                    obs: dict[str, Any] = {"content": result_text}
                    if ss_path:
                        obs["content"] = [
                            {"type": "text", "text": result_text},
                            {"type": "image", "source": {"media_type": "image/png", "path": ss_path}},
                        ]
                    obs_results.append(obs)

                elif part.text:
                    text_parts.append(part.text)
                    if "INFEASIBLE" in part.text:
                        self.steps.append({
                            "step_id": len(self.steps) + 1,
                            "source": "agent",
                            "message": part.text[:500],
                        })
                        await self.post_run(context, model, "gemini-cua")
                        return

            # No function calls = model is done
            if not tool_calls:
                self.steps.append({
                    "step_id": len(self.steps) + 1,
                    "source": "agent",
                    "message": "\n".join(text_parts) or "Task complete.",
                })
                break

            # Record step
            step_data: dict[str, Any] = {
                "step_id": len(self.steps) + 1,
                "source": "agent",
                "message": "\n".join(text_parts) or None,
                "tool_calls": tool_calls,
                "metrics": {
                    "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0 if usage_meta else 0,
                    "completion_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0 if usage_meta else 0,
                },
            }
            if obs_results:
                step_data["observation"] = {"results": obs_results}
            self.steps.append(step_data)

        await self.post_run(context, model, "gemini-cua")
