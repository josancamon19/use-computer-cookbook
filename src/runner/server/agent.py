"""agent_spec_for: pick the right harbor agent for a given (model, platform)."""
from __future__ import annotations


def agent_spec_for(model: str, max_steps: int = 30, platform: str = "macos") -> dict:
    # "replay" — deterministic playback of recorded actions. Used by the
    # gateway's task-detail Replay button. Reads actions.json next to
    # task.toml and walks each step via the SDK; no LLM call.
    if model == "replay":
        return {
            "import_path": "runner.agents.debug_cua:DebugCUAAgent",
            "model_name": "debug",
            "kwargs": {"max_steps": max_steps, "replay": True},
        }
    if platform == "ios":
        # iOS uses a custom tool schema (tap/swipe/button/...) so generic and
        # macOS-shaped agents don't apply. Only Anthropic for now.
        return {
            "import_path": "runner.agents.ios:IOSAgent",
            "model_name": model if "/" in model else f"anthropic/{model}",
            "kwargs": {"max_steps": max_steps},
        }
    if "kimi" in model or "moonshot" in model or model.startswith("openai/"):
        return {
            "import_path": "runner.agents.generic_cua:GenericCUAAgent",
            "model_name": model if "/" in model else f"openai/{model}",
            "kwargs": {
                "max_steps": max_steps,
                "max_tokens": 4096,
                "api_base": "https://api.fireworks.ai/inference/v1",
            },
        }
    return {
        "import_path": "runner.agents.anthropic_cua:AnthropicCUAAgent",
        "model_name": model if "/" in model else f"anthropic/{model}",
        "kwargs": {"max_steps": max_steps},
    }


def strip_ios_prefix(s: str, kind: str) -> str:
    """Drop the boilerplate `com.apple.CoreSimulator.<kind>.` prefix so the
    pin in task.toml stays readable. `kind` is "SimDeviceType" or "SimRuntime"."""
    prefix = f"com.apple.CoreSimulator.{kind}."
    return s[len(prefix):] if s.startswith(prefix) else s
