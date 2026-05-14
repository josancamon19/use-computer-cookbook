# use-computer-cookbook

Reference recipes for running computer-use agents on [use.computer](https://use.computer): macOS / iOS sandboxes, Anthropic + OpenAI + Gemini agents, and adapters that turn external task sources into the [Harbor](https://harborframework.com) task format.

For one-purpose SDK snippets (screenshot, recording, file transfer, keepalive), see [`use-computer-python/examples/`](https://github.com/josancamon19/use-computer-python/tree/main/examples). For walkthroughs of each recipe, see the [examples section](https://docs.use.computer/docs/examples) of the docs.

## Quickstart

```bash
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY + USE_COMPUTER_API_KEY (mk_live_…)

uv run harbor run -c src/runner/configs/job-macosworld.yaml --env-file .env
```

| Config                | Dataset                                     | Default agent                                                                  |
| --------------------- | ------------------------------------------- | ------------------------------------------------------------------------------ |
| `job-macosworld.yaml` | [macOSWorld](https://macos-world.github.io) | Claude Sonnet 4.6 + Kimi K2.5 + Qwen 3.6                                       |
| `job-adhoc.yaml`      | hand-written prompts (macOS or iOS)         | Claude Sonnet 4.6 — flip `platform:` and the agent `import_path` for iOS       |

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
