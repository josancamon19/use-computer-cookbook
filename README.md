# use-computer-cookbook

Reference recipes for running computer-use agents on [use.computer](https://use.computer): macOS / iOS sandboxes, Anthropic + OpenAI + Gemini agents, and adapters that turn external task sources into the [Harbor](https://harborframework.com) task format.

For one-purpose SDK snippets (screenshot, recording, file transfer, keepalive), see [`use-computer-python/examples/`](https://github.com/josancamon19/use-computer-python/tree/main/examples). For walkthroughs of each recipe, see the [examples section](https://docs.use.computer/docs/examples) of the docs.

## Quickstart

```bash
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY + MMINI_API_KEY

uv run harbor run -c src/runner/configs/job-macosworld.yaml --env-file .env
```

| Config                 | Dataset                                               | Default agent             |
| ---------------------- | ----------------------------------------------------- | ------------------------- |
| `job-macosworld.yaml`  | [macOSWorld](https://github.com/microsoft/MacOSWorld) | Claude Sonnet 4.6 (macOS) |
| `job-adhoc-macos.yaml` | hand-written macOS prompts                            | Claude Sonnet 4.6 (macOS) |
| `job-adhoc-ios.yaml`   | hand-written iOS prompts                              | Claude Sonnet 4.6 (iOS)   |

## Layout

```
src/runner/
├── adapters/      external task sources → Harbor task dirs
├── agents/        macos/, ios/, shared base + debug
├── configs/       Harbor job YAMLs (above)
├── environments/  UseComputerEnvironment (Harbor ↔ gateway bridge)
└── server/        internal HTTP sidecar; ignore unless deploying it
```

## Links

- SDK: <https://github.com/josancamon19/use-computer-python>
- Docs: <https://docs.use.computer>
- API reference (Swagger): <https://api.use.computer/docs>
