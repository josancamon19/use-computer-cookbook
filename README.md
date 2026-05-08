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

## job.yaml

A job config tells Harbor what to run:

```yaml
jobs_dir: jobs
n_attempts: 1
orchestrator:
    type: local
    n_concurrent_trials: 16
environment:
    import_path: runner.environments.mmini:MminiEnvironment
    kwargs:
        gateway_url: https://api.use.computer
        platform: macos
    delete: true
agents:
    - import_path: runner.agents.anthropic_cua:AnthropicCUAAgent
      model_name: anthropic/claude-sonnet-4-6
      kwargs:
          max_steps: 50
datasets:
    - path: datasets/macosworld_ready
```

Useful fields: `gateway_url`, `platform`, `n_concurrent_trials`, `model_name`, `max_steps`, and `datasets`.

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
