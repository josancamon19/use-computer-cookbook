# use-computer examples

Runnable examples for use.computer: macOS tasks, iOS simulator tasks, replay/debug agents, and Harbor job configs.

## Setup

Install deps with `uv sync`.

Environment variables:

- `ANTHROPIC_API_KEY`: required for Anthropic agents.
- `MMINI_API_KEY`: use `mk_live_...` for `https://api.use.computer`, or the master key for admin/dev gateway configs.

Point customer configs at `https://api.use.computer`. Point internal/dev configs at the relevant admin/dev gateway.

## Run

```bash
uv run harbor run -c src/runner/configs/job.yaml --env-file .env
```

Use a different config for iOS or collected tasks:

```bash
uv run harbor run -c src/runner/configs/job-collected-ios.yaml --env-file .env
```

For more Harbor CLI options, see the [Harbor docs](https://harborframework.com/docs/run-jobs/run-evals).

## Configs

- `src/runner/configs/job.yaml`: curated macOSWorld set.
- `src/runner/configs/job-adhoc-macos.yaml`: free-form macOS prompts from `datasets/adhoc/macos`.
- `src/runner/configs/job-collected.yaml`: collected macOS tasks, defaulting to replay/debug.
- `src/runner/configs/job-collected-ios.yaml`: collected iOS simulator tasks.

Each config sets the task dataset, gateway URL, platform, concurrency, cleanup behavior, and active agent. See the [Harbor run docs](https://harborframework.com/docs/run-jobs/run-evals) for the full config format.

## Agents

- `AnthropicCUAAgent`: Claude computer-use agent for macOS screenshots, mouse, keyboard, and shell-backed tasks.
- `IOSAgent`: Claude-driven iOS simulator agent using tap, swipe, app, appearance, and screenshot tools.
- `DebugCUAAgent` with `replay: true`: replays collected `actions.json` without an LLM.
- `DebugCUAAgent` with `realistic: true`: deterministic endpoint smoke test for infra.
- `GenericCUAAgent`: OpenAI-compatible chat/completions agent for Fireworks or other OpenAI-style model endpoints.

## Task Shape

Each task directory is normal [Harbor task format](https://harborframework.com/docs/tasks):

```text
instruction.md
task.toml
tests/test.sh
tests/setup/pre_command.sh
actions.json          # optional replay steps
expected_final.json   # optional visual/reference state
```

Docs: https://api.use.computer/docs

Python SDK: https://github.com/josancamon19/use-computer-python
