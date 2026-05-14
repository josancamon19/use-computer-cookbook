# use-computer-cookbook

Reference recipes for running computer-use agents on [use.computer](https://use.computer): macOS and iOS sandboxes, Anthropic / OpenAI / Gemini agents, and three adapters that turn external task sources into the [Harbor](https://harborframework.com) task format.

> Looking for one-purpose SDK snippets (screenshot, recording, file transfer, keepalive)? Those live in [use-computer-python/examples/](https://github.com/josancamon19/use-computer-python/tree/main/examples). This repo is for full agent loops and evals.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY and MMINI_API_KEY
```

- `ANTHROPIC_API_KEY` — for the macOS / iOS Anthropic CUA agents.
- `MMINI_API_KEY` — either a customer `mk_live_…` (for `https://api.use.computer`) or the gateway admin key (for the internal `10.10.10.2:808x` configs).

## Run a job

```bash
uv run harbor run -c src/runner/configs/job-macosworld.yaml --env-file .env
```

Each config wires together a dataset, an environment (the use.computer gateway), and exactly one agent. Pick one:

| Config | Dataset | Default agent |
| --- | --- | --- |
| `job-macosworld.yaml` | curated [macOSWorld](https://github.com/microsoft/MacOSWorld) tasks | Claude Sonnet 4.6 (macOS) |
| `job-adhoc-macos.yaml` | hand-written macOS prompts (`datasets/adhoc/macos`) | Claude Sonnet 4.6 (macOS) |
| `job-adhoc-ios.yaml` | hand-written iOS prompts (`datasets/adhoc/ios`) | Claude Sonnet 4.6 (iOS) |
| `job-collected.yaml` | macOS tasks recorded via the `/collect` UI | Debug replay |
| `job-collected-ios.yaml` | iOS tasks recorded via the `/collect-ios` UI | Debug replay |

Full Harbor CLI docs: <https://harborframework.com/docs/run-jobs/run-evals>

## Adapters — making datasets

All three live under [`src/runner/adapters/`](src/runner/adapters/) and write Harbor task dirs to `datasets/<source>/<platform>/`.

- **`adhoc/`** — hand-written prompt lists in `tasks/{ios,macos}.json`.
  ```bash
  uv run python -m runner.adapters.adhoc.export src/runner/adapters/adhoc/tasks/macos.json
  ```
- **`collected/`** — tasks recorded with the use.computer `/collect` / `/collect-ios` UI on the gateway. Pulls from `/admin/tasks`, bundles each task's recorded actions into `actions.json` for replay.
  ```bash
  uv run python -m runner.adapters.collected.adapter --all --platform macos
  ```
- **`macosworld/`** — the [macOSWorld](https://github.com/microsoft/MacOSWorld) benchmark. Needs a local clone of the upstream repo passed in via `--macosworld-root`.
  ```bash
  uv run python -m runner.adapters.macosworld.run_adapter \
    --macosworld-root <path-to-macosworld-clone> \
    --task-dir datasets/macosworld_ready --ready-only
  ```

## Agents

In [`src/runner/agents/`](src/runner/agents/), split by target platform plus a couple of shared helpers.

**macOS** ([`agents/macos/`](src/runner/agents/macos/)):

- `anthropic.py` → `AnthropicCUAAgent` — Claude's computer-use tool with mouse + keyboard + screenshot.
- `openai.py` → `OpenAICUAAgent` — OpenAI computer-use preview.
- `gemini.py` → `GeminiCUAAgent` — Gemini's computer-use tool.
- `generic.py` → `GenericCUAAgent` — any OpenAI-compatible chat/completions endpoint (e.g. Fireworks).

**iOS** ([`agents/ios/`](src/runner/agents/ios/)):

- `agent.py` → `IOSAgent` — Claude-driven iOS simulator agent using tap, swipe, app launch, environment, and screenshot tools.

**Shared** ([`agents/`](src/runner/agents/)):

- `base.py` → `BaseCUAAgent` — the tool-loop scaffolding every CUA agent inherits.
- `debug.py` → `DebugCUAAgent` — replays a task's recorded `actions.json` with no LLM. Use this against the `collected/` datasets.
- `prompts/` — system prompt templates loaded by `base.load_prompt(...)`.

## Task shape

Every adapter writes Harbor's standard layout:

```text
<task-name>/
├── instruction.md
├── task.toml
├── tests/
│   ├── test.sh
│   └── setup/
│       ├── pre_command.sh
│       └── files/                # optional, populated by collected/
├── actions.json                  # optional — debug-agent replay
└── expected_final.json           # optional — reference state
```

Harbor docs: <https://harborframework.com/docs/tasks>

## Layout

```
runner/
├── src/runner/
│   ├── adapters/      ← see adapters/README.md
│   ├── agents/        ← macos/, ios/, shared base + debug
│   ├── configs/       ← Harbor job YAMLs
│   ├── environments/  ← UseComputerEnvironment (the Harbor↔gateway bridge)
│   └── server/        ← internal HTTP sidecar; ignore unless deploying it
├── datasets/          ← generated Harbor task dirs (gitignored)
└── jobs/              ← generated run outputs (gitignored)
```

## Links

- API docs (Swagger): <https://api.use.computer/docs>
- Python SDK: <https://github.com/josancamon19/use-computer-python>
- Docs site: <https://docs.use.computer>
