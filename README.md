# use-computer examples

Runnable examples for use.computer: macOS tasks, iOS simulator tasks, replay/debug agents, and Harbor job configs.

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=...
export MMINI_API_KEY=...      # Harbor configs against admin/dev gateway
export GATEWAY_API_KEY=...    # run.py against admin/dev gateway
```

For reservation API keys, point configs or SDK calls at `https://api.use.computer` and use the `mk_live_...` key from the reservation dashboard.

## Run One macOS Task

```bash
uv run python run.py --task "productivity__*" --agent anthropic --model claude-sonnet-4-6 --max-steps 30
```

`run.py` creates a sandbox, runs the task setup, drives the agent, grades when a verifier exists, and writes outputs under `results/`.

## Run Harbor Jobs

```bash
# macOSWorld curated set
uv run harbor run -c src/runner/configs/job.yaml -p datasets/macosworld_ready

# collected iOS task
uv run harbor run -c src/runner/configs/job-collected-ios.yaml -p datasets/collected/ios/<task-name>

# ad hoc macOS prompts
uv run python adhoc/export.py adhoc/tasks/macos.json --clean
uv run harbor run -c src/runner/configs/job-adhoc-macos.yaml -p datasets/adhoc/macos
```

Useful knobs live in the YAML files: `gateway_url`, `platform`, `n_concurrent_trials`, model, and `max_steps`.

## Task Shape

Each task directory is normal Harbor format:

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
