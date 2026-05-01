"""iOS CUA agent — provider-agnostic via litellm.

Uses a custom tool schema that mirrors the mmini iOS DSL surface (tap,
swipe, type_text, press_button, wait, done). Works against any
litellm-supported backend: anthropic, openai, gemini, fireworks, etc.
The model is whatever string `model_name` resolves to ("anthropic/...",
"openai/...", "gemini/...", etc.) — litellm dispatches.

Tool schema is OpenAI-shaped (`{type: function, function: {...}}`) — litellm
translates it to each provider's native format under the hood. Messages are
also OpenAI-shaped: screenshots come back as a fresh `role: user` observation
after each turn rather than embedded in tool_results, because tool-result-with-
image is Anthropic-only and doesn't survive the litellm normalization.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from typing import Any, cast

import litellm
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from mmini.sandbox import AsyncIOSSandbox
from PIL import Image

from runner.agents.base_cua import BaseCUAAgent, _annotate_click, load_prompt, resize_for_vision

litellm.drop_params = True  # Fireworks doesn't accept tool_choice etc; drop instead of erroring


def _build_ios_system_prompt(
    template: str, width: int, height: int, step: int = 1, max_steps: int = 50
) -> str:
    return (
        template.replace("{SCREENSHOT_WIDTH}", str(width))
        .replace("{SCREENSHOT_HEIGHT}", str(height))
        .replace("{SCREENSHOT_MAX_X}", str(width - 1))
        .replace("{SCREENSHOT_MAX_Y}", str(height - 1))
        .replace("{STEP_NUMBER}", str(step))
        .replace("{MAX_STEPS}", str(max_steps))
    )


IOS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tap",
            "description": "Single tap at the given (x, y) coordinate in screen pixels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "swipe",
            "description": (
                "Drag a finger from (from_x, from_y) to (to_x, to_y). The content "
                "under the finger moves WITH the finger. See <swipe> in the "
                "system prompt for direction conventions and back-gesture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_x": {"type": "number"},
                    "from_y": {"type": "number"},
                    "to_x": {"type": "number"},
                    "to_y": {"type": "number"},
                },
                "required": ["from_x", "from_y", "to_x", "to_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into the currently focused input field. Tap the field first.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_button",
            "description": "Press a hardware button.",
            "parameters": {
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["home", "lock", "siri", "side-button", "apple-pay"],
                    },
                },
                "required": ["button"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait N seconds (e.g. for an app launch or animation).",
            "parameters": {
                "type": "object",
                "properties": {"duration": {"type": "number"}},
                "required": ["duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the task is complete. Set success=false if you decided it cannot be completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                },
                "required": ["success"],
            },
        },
    },
]


async def _execute_ios_tool(
    sandbox: AsyncIOSSandbox,
    tool_name: str,
    tool_input: dict[str, Any],
) -> tuple[str, bool]:
    """Run a single iOS tool call. Returns (result_text, is_done).

    Result text never echoes coordinates — the model emits in api space, we
    dispatch in device space, and feeding either back caused confusion. The
    next observation screenshot is the real feedback signal.
    """
    if tool_name == "tap":
        await sandbox.input.tap(float(tool_input["x"]), float(tool_input["y"]))
        return "tap", False
    if tool_name == "swipe":
        await sandbox.input.swipe(
            float(tool_input["from_x"]),
            float(tool_input["from_y"]),
            float(tool_input["to_x"]),
            float(tool_input["to_y"]),
        )
        return "swipe", False
    if tool_name == "type_text":
        await sandbox.input.type_text(tool_input["text"])
        return f"type_text({tool_input['text']!r})", False
    if tool_name == "press_button":
        await sandbox.input.press_button(tool_input["button"])
        return f"press_button({tool_input['button']})", False
    if tool_name == "wait":
        await asyncio.sleep(float(tool_input["duration"]))
        return f"wait({tool_input['duration']}s)", False
    if tool_name == "done":
        msg = tool_input.get("message", "")
        return f"done(success={tool_input['success']}) {msg}", True
    return f"unknown tool: {tool_name}", False


_PAIR_KEYS = {"x": "y", "from_x": "from_y", "to_x": "to_y"}


def _coerce_coord_args(args: dict[str, Any]) -> dict[str, Any]:
    """Tolerate models that pack two coords into one field or emit junk.

    Sonnet occasionally emits {'x': '204, 907'}, {'x': '666, '} (trailing
    comma), or {'x': [204, 907]} despite the schema asking for separate
    numbers. Coerce to floats; only fill the y-pair when we actually got two
    valid numbers out.
    """
    for x_key, y_key in _PAIR_KEYS.items():
        v = args.get(x_key)
        if v is None or isinstance(v, (int, float)):
            continue
        nums: list[float] = []
        if isinstance(v, str):
            for p in v.split(","):
                p = p.strip()
                if not p:
                    continue
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
        elif isinstance(v, (list, tuple)):
            for p in v:
                try:
                    nums.append(float(p))
                except (TypeError, ValueError):
                    pass
        if not nums:
            continue
        args[x_key] = nums[0]
        if len(nums) >= 2 and y_key not in args:
            args[y_key] = nums[1]
    return args


def _data_url(image_bytes: bytes) -> str:
    """Detect the image format (PIL) and emit a matching data URL.

    The iOS gateway returns JPEGs by default; we used to hardcode image/png and
    Anthropic's strict validator would reject the request.
    """
    fmt = (Image.open(io.BytesIO(image_bytes)).format or "PNG").lower()
    return f"data:image/{fmt};base64," + base64.b64encode(image_bytes).decode()


class IOSAgent(BaseCUAAgent):
    """iOS CUA agent — provider-agnostic, uses the mmini iOS DSL via litellm."""

    @staticmethod
    def name() -> str:
        return "ios-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        sandbox = await self.pre_run(environment)
        if not isinstance(sandbox, AsyncIOSSandbox):
            raise RuntimeError(
                f"IOSAgent requires an iOS sandbox, got {type(sandbox).__name__}"
            )
        self.steps[0]["message"] = instruction

        model = self.model_name
        if not model:
            raise RuntimeError(
                "IOSAgent requires model_name (e.g. 'anthropic/claude-sonnet-4-6', "
                "'openai/gpt-4o', 'gemini/gemini-2.0-flash'); set it in job.yaml."
            )

        ss = await sandbox.screenshot.take_full_screen()
        screen_w, screen_h = Image.open(io.BytesIO(ss)).size
        self.screen_width = screen_w
        self.screen_height = screen_h
        (self.images_dir / "step_000.png").write_bytes(ss)

        ss_api, api_w, api_h, _, _ = resize_for_vision(ss, model)
        # axe tap/swipe interpret coords as logical POINTS, not pixels.
        # Override sx/sy to map api-space → points (e.g. iPhone 17 Pro: 402×874).
        info = await sandbox.display.get_info()
        point_w, point_h = info.width, info.height
        sx = point_w / api_w
        sy = point_h / api_h
        self.logger.info(
            f"iOS screen={screen_w}x{screen_h}px points={point_w}x{point_h} "
            f"api={api_w}x{api_h} sx={sx:.4f} sy={sy:.4f} model={model}"
        )

        prompt_template = load_prompt("ios.txt")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": _data_url(ss_api)}},
                ],
            },
        ]

        for step_idx in range(self.max_steps):
            await self._fire_in_process(environment, step_idx)
            _truncate_old_screenshots(messages, keep=5)

            messages[0] = {
                "role": "system",
                "content": _build_ios_system_prompt(
                    prompt_template,
                    api_w,
                    api_h,
                    step=step_idx + 1,
                    max_steps=self.max_steps,
                ),
            }

            self.logger.info(f"step {step_idx+1}/{self.max_steps}: calling {model}...")
            t0 = time.monotonic()
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=IOS_TOOLS,
                    max_tokens=4096,
                )
            except Exception as e:
                self.logger.error(
                    f"step {step_idx+1}: completion error after "
                    f"{time.monotonic()-t0:.1f}s: {type(e).__name__}: {e}"
                )
                raise

            choice = response.choices[0]
            assistant_msg = choice.message
            usage = getattr(response, "usage", None)
            in_t = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            out_t = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            self.total_in += in_t
            self.total_out += out_t
            self.logger.info(
                f"step {step_idx+1}: responded in {time.monotonic()-t0:.1f}s, "
                f"in={in_t} out={out_t} stop={choice.finish_reason}"
            )

            tool_calls = list(getattr(assistant_msg, "tool_calls", None) or [])

            assistant_text = getattr(assistant_msg, "content", None) or ""
            # Kimi (and other reasoning models) leave content empty and put
            # chain-of-thought in reasoning_content. Surface it so trajectories
            # are debuggable. Doesn't go back into the wire — model produces
            # it fresh each turn.
            reasoning = getattr(assistant_msg, "reasoning_content", None)
            if reasoning and not assistant_text:
                assistant_text = reasoning
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text,
            }
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_entry)

            if not tool_calls:
                self.steps.append(
                    {
                        "step_id": len(self.steps) + 1,
                        "source": "agent",
                        "message": assistant_text or "Task complete.",
                    }
                )
                break

            tool_calls_record: list[dict[str, Any]] = []
            obs_results: list[dict[str, Any]] = []
            done_flag = False

            for tc in tool_calls:
                try:
                    raw_args = json.loads(tc.function.arguments or "{}")
                except (ValueError, json.JSONDecodeError):
                    raw_args = {}

                scaled = _coerce_coord_args(dict(raw_args))
                coord_err: str | None = None
                for k in ("x", "y", "from_x", "from_y", "to_x", "to_y"):
                    if k in scaled:
                        factor = sx if k.endswith("x") else sy
                        try:
                            scaled[k] = float(scaled[k]) * factor
                        except (TypeError, ValueError):
                            coord_err = f"non-numeric {k}={scaled[k]!r}"
                            break

                tool_calls_record.append(
                    {
                        "tool_call_id": tc.id,
                        "function_name": tc.function.name,
                        "arguments": scaled,
                    }
                )

                if coord_err:
                    self.logger.warning(f"skipping {tc.function.name}: {coord_err}")
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id,
                         "content": f"skipped (bad coords: {coord_err})"}
                    )
                    obs_results.append(
                        {"source_call_id": tc.id, "content": f"skipped: {coord_err}"}
                    )
                    continue

                try:
                    result_text, is_done = await _execute_ios_tool(
                        sandbox, tc.function.name, scaled
                    )
                except Exception as e:
                    result_text = f"tool error: {type(e).__name__}: {e}"
                    is_done = False
                    self.logger.warning(result_text)

                if is_done:
                    done_flag = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )
                obs_results.append({"source_call_id": tc.id, "content": result_text})

            if not done_flag:
                await asyncio.sleep(1.0)
                ss_bytes = await sandbox.screenshot.take_full_screen()
                img_name = f"step_{step_idx + 1:03d}.png"
                # Annotate with the most recent click/swipe coords for human review.
                annotated = ss_bytes
                for tcr in reversed(tool_calls_record):
                    annotated = _annotate_click(ss_bytes, tcr.get("arguments", {}))
                    if annotated is not ss_bytes:
                        break
                (self.images_dir / img_name).write_bytes(annotated)
                ss_bytes_api, _, _, _, _ = resize_for_vision(ss_bytes, model)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Observation:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url(ss_bytes_api)},
                            },
                        ],
                    }
                )
                if obs_results:
                    last = obs_results[-1]
                    last["content"] = [
                        {"type": "text", "text": last["content"]},
                        {
                            "type": "image",
                            "source": {
                                "media_type": "image/png",
                                "path": f"images/{img_name}",
                            },
                        },
                    ]

            step_data: dict[str, Any] = {
                "step_id": len(self.steps) + 1,
                "source": "agent",
                "message": assistant_text or None,
                "tool_calls": tool_calls_record,
                "metrics": {"prompt_tokens": in_t, "completion_tokens": out_t},
            }
            if obs_results:
                step_data["observation"] = {"results": obs_results}
            self.steps.append(step_data)
            self.checkpoint(context, model, "ios-cua")

            if done_flag:
                break

        await self.post_run(context, model, "ios-cua")


def _truncate_old_screenshots(messages: list[dict[str, Any]], keep: int = 5) -> None:
    """Keep only the last `keep` image_url items in user-content arrays.

    Older screenshots are replaced with a small text placeholder so the
    conversation history stays linear but doesn't blow the token budget.
    """
    image_locations: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, item in enumerate(content):
            if isinstance(item, dict) and cast(dict[str, Any], item).get("type") == "image_url":
                image_locations.append((i, j))
    to_remove = len(image_locations) - keep
    if to_remove <= 0:
        return
    for i, j in image_locations[:to_remove]:
        messages[i]["content"][j] = {"type": "text", "text": "[screenshot removed]"}
