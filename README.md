# mmini-runner

Harbor-based runner for evaluating CUA (Computer Use Agent) models on macOS VMs via the [mmini](https://use.computer) gateway. Ships with the [macOSWorld](https://github.com/anthropics/macosworld) benchmark dataset and adapters for Anthropic, OpenAI, and generic (LiteLLM/Tinker) agents.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- The `mmini-sandbox` SDK checked out alongside this repo (see `pyproject.toml` for the expected path)

## Setup

```bash
# Install dependencies
uv sync
```

### Environment variables

Copy the template below into a `.env` file in the project root (it's already in `.gitignore`):

```bash
# .env

# Required — mmini gateway key (get one at https://use.computer)
MMINI_API_KEY=sk-...

# Required for Anthropic CUA agent
ANTHROPIC_API_KEY=sk-ant-...

# Required for OpenAI CUA agent
OPENAI_API_KEY=sk-...

# Required for Gemini CUA agent
GEMINI_API_KEY=...

# Optional — override the mmini host
# MMINI_HOST=
```

Load them before running:

```bash
source .env
# or
export $(cat .env | xargs)
```

## Usage

Run a benchmark evaluation with a config file:

```bash
uv run harbor run -c src/runner/configs/debug.yaml
```

### Config files

Configs live in `src/runner/configs/`. Each one specifies the environment, agent(s), and dataset tasks to run:


| Config                | Agent                  | Description                        |
| --------------------- | ---------------------- | ---------------------------------- |
| `debug.yaml`          | Anthropic Claude       | Single task, good for testing      |
| `macosworld-cua.yaml` | Anthropic Claude       | Full macOSWorld benchmark suite    |
| `kimi-cua.yaml`       | Kimi K2.5 (via Tinker) | Generic agent with LiteLLM backend |


### Running a specific task

Edit the `task_names` list in your config file, or pass tasks on the command line:

```bash
uv run harbor run \
  -c src/runner/configs/macosworld-cua.yaml \
  --path datasets/macosworld_ready \
  -t "sys_apps__07709761*"
```

### Available agents


| Agent         | Import path                                     | Model example                 |
| ------------- | ----------------------------------------------- | ----------------------------- |
| Anthropic CUA | `runner.agents.anthropic_cua:AnthropicCUAAgent` | `anthropic/claude-sonnet-4-6` |
| OpenAI CUA    | `runner.agents.openai_cua:OpenAICUAAgent`       | `openai/computer-use-preview` |
| Generic CUA   | `runner.agents.generic_cua:GenericCUAAgent`     | Any vision model via LiteLLM  |
| Gemini CUA    | `runner.agents.gemini_cua:GeminiCUAAgent`       | `gemini-2.5-pro`              |


## Output

Results are written to the `jobs/` directory, including:

- Agent step logs (actions, tool calls, token usage)
- Screenshots at each step
- Verifier test results

