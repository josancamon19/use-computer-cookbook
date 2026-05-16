"""agent_spec_for: pick the right harbor agent for a given (model, platform)."""
from __future__ import annotations


def agent_spec_for(model: str, max_steps: int = 30, platform: str = "macos") -> dict:
    # "replay" — deterministic playback of recorded actions. Used by the
    # gateway's task-detail Replay button. Reads actions.json next to
    # task.toml and walks each step via the SDK; no LLM call.
    if model == "replay":
        return {
            "import_path": "runner.agents.debug:DebugCUAAgent",
            "model_name": "debug",
            "kwargs": {"max_steps": max_steps, "replay": True},
        }
    if platform == "ios":
        # iOS uses a custom tool schema (tap/swipe/button/...). IOSAgent
        # dispatches via litellm so the model prefix selects the provider:
        # anthropic/* → Claude, openai/* → GPT, gemini/* → Gemini.
        return {
            "import_path": "runner.agents.ios.agent:IOSAgent",
            "model_name": model if "/" in model else f"anthropic/{model}",
            "kwargs": {"max_steps": max_steps},
        }
    # macOS: dispatch to a provider-native CUA agent based on the model prefix.
    if model.startswith("gemini/") or "gemini" in model:
        return {
            "import_path": "runner.agents.macos.gemini:GeminiCUAAgent",
            "model_name": model if "/" in model else f"gemini/{model}",
            "kwargs": {"max_steps": max_steps},
        }
    if model.startswith("openai/") or "gpt" in model:
        return {
            "import_path": "runner.agents.macos.openai:OpenAICUAAgent",
            "model_name": model if "/" in model else f"openai/{model}",
            "kwargs": {"max_steps": max_steps},
        }
    if "kimi" in model or "moonshot" in model:
        # Fireworks-hosted open models — keep the generic path.
        return {
            "import_path": "runner.agents.macos.generic:GenericCUAAgent",
            "model_name": model if "/" in model else f"openai/{model}",
            "kwargs": {
                "max_steps": max_steps,
                "max_tokens": 4096,
                "api_base": "https://api.fireworks.ai/inference/v1",
            },
        }
    return {
        "import_path": "runner.agents.macos.anthropic:AnthropicCUAAgent",
        "model_name": model if "/" in model else f"anthropic/{model}",
        "kwargs": {"max_steps": max_steps},
    }


def strip_ios_prefix(s: str, kind: str) -> str:
    """Drop the boilerplate `com.apple.CoreSimulator.<kind>.` prefix so the
    pin in task.toml stays readable. `kind` is "SimDeviceType" or "SimRuntime"."""
    prefix = f"com.apple.CoreSimulator.{kind}."
    return s[len(prefix):] if s.startswith(prefix) else s
