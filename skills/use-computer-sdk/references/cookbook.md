# Cookbook integration (Harbor jobs)

When the user is editing this repo (`use-computer-cookbook`), they're usually running an agent against a _dataset_ of tasks via [Harbor](https://harborframework.com). The bridge is `runner.environments.use_computer:UseComputerEnvironment` — Harbor calls `start()` / `stop()`, the environment creates a sandbox via this SDK and hands it to the agent's `step()`.

## Job YAML shape

```yaml
jobs_dir: ./jobs
n_attempts: 1
orchestrator:
    type: local
    n_concurrent_trials: 2 # ≤ macOS slot count of the reservation

environment:
    import_path: runner.environments.use_computer:UseComputerEnvironment
    kwargs:
        gateway_url: https://api.use.computer
        platform: macos # or "ios"
    delete: true # destroy sandbox on trial finish

agents:
    - import_path: runner.agents.macos.anthropic:AnthropicCUAAgent
      model_name: anthropic/claude-sonnet-4-6
      kwargs:
          max_steps: 50
```

Run with:

```bash
uv run harbor run -c src/runner/configs/job-adhoc.yaml -p datasets/adhoc/macos --env-file .env
```

`n_concurrent_trials` is the hard concurrency cap; keep it ≤ the reservation's slot count (Macs × 2 macOS + Macs × 2 iOS) or trials will block waiting for a free sandbox.

## Available agents

| `import_path`                                     | What it is                                             |
| ------------------------------------------------- | ------------------------------------------------------ |
| `runner.agents.macos.anthropic:AnthropicCUAAgent` | Claude computer-use (macOS)                            |
| `runner.agents.macos.openai:OpenAICUAAgent`       | OpenAI computer-use preview                            |
| `runner.agents.macos.gemini:GeminiCUAAgent`       | Gemini computer-use                                    |
| `runner.agents.macos.generic:GenericCUAAgent`     | Any OpenAI-compatible endpoint (Fireworks, vLLM, etc.) |
| `runner.agents.ios.agent:IOSAgent`                | Claude-driven iOS simulator agent                      |

All five inherit from `runner.agents.base` (shared screenshot prep, coord scaling, action dispatch). Switching providers is a 1-line YAML change.

## Coordinate scaling — `runner.agents.base`

`src/runner/agents/base.py` exposes two helpers every cookbook agent uses before sending a screenshot to a vision model. They live in the cookbook, not the SDK, because the per-model cap table is a registry that grows as new families ship — adding a model shouldn't require an SDK release.

- `screenshot_cap_for_model(model)` — looks up the per-family long-edge cap (Claude Opus 4.7+ → 2576 px, other Claude → 1568, Kimi/Fireworks → 896, OpenAI/Gemini fallback → 1280).
- `scale_screenshot_for_model(image_bytes, model) → (bytes, api_w, api_h, sx, sy)` — aspect-preserving resize to that cap, returning the scale factors that map _model coords → native display points_.

Why we do this client-side:

1. **Vision-model coords come back in resized-image space.** Per Anthropic's CUA docs, if you send a 3200×2000 screenshot and the model resizes it to 1568×980 internally, click coordinates come back relative to 1568×980 — not your 3200×2000. The SDK's `mouse.click()` takes native display points, so you have to multiply back.
2. **Resizing on the server is silent.** Different model families resize to different targets, and the resize is only loosely documented. Doing the resize ourselves to a known target makes click coords reproducible and lets us pick the long-edge that we've actually probed for accuracy.
3. **Model-specific tuning matters.** Kimi/Fireworks at 1024+ px showed y-coord drift on tall iOS shots; capping at 896 fixed it. Opus 4.7+ supports a larger 2576 cap and benefits from the extra detail. The cap lives in `screenshot_cap_for_model()`, change in one place.

Pattern used in every agent:

```python
from runner.agents.base import scale_screenshot_for_model

png = mac.screenshot.take_full_screen()
api_bytes, api_w, api_h, sx, sy = scale_screenshot_for_model(png, model_name)
# send api_bytes to the model (api_w × api_h)
# when the model returns coordinate=[x, y]:
mac.mouse.click(int(x * sx), int(y * sy))
```

If you're writing a custom agent (in this cookbook or elsewhere), **use these helpers instead of doing your own resize**. Tweaks to per-model caps belong in `screenshot_cap_for_model()`, not scattered across agent files. If you're consuming the SDK outside this cookbook, the implementation is ~40 lines of Pillow — copy it over (and the per-model registry) into your project.

## What `UseComputerEnvironment` does

`runner/environments/use_computer.py` is a thin Harbor adapter:

- On `start()`: instantiates `Computer(api_key=…, base_url=gateway_url)`, calls `cc.create(type=platform, ...)`, exposes the sandbox to the agent via `self.sandbox`.
- On `stop()`: calls `sandbox.destroy()` (skipped if `delete: false`). The reservation's idle reaper handles anything missed.
- It also reads optional iOS `device_type` and `runtime` from the task JSON when `platform: ios`, so per-task iOS configs work without YAML changes.

For local dev (no real reservation), point `gateway_url` at `http://localhost:<port>` and the same code works against a local gateway.

## macOSWorld: AppleScript verifier transpiling

macOSWorld tasks ship `pre_command` setup scripts and `grading_command` lists that lean heavily on `osascript -e 'tell application "System Events" ...'`. Run over `exec_ssh` those silently fail because TCC denies Accessibility to anything in the `sshd-keygen-wrapper` responsibility chain (return code `-25211`).

The macosworld adapter (`src/runner/adapters/macosworld/adapter.py`) doesn't ship the raw scripts — it pipes both `pre_command` and the `tests/test.sh` it generates through the SDK's transpiler before writing them into the task dir:

```python
from use_computer.ax_transpile import patch_curl_timeouts, transpile

rewritten, _ = transpile(raw_pre_command)
rewritten, _ = patch_curl_timeouts(rewritten)
# write rewritten into <task-dir>/setup.sh
```

`transpile()` rewrites recognized AppleScript shapes (attribute reads, `keystroke`, dock item lists, etc.) into calls against `/usr/local/bin/ax_helper.py` invoked through the CUA server's `/cmd run_command` endpoint, where the TCC grant on `python3.12` actually applies. `patch_curl_timeouts()` wraps any `curl` in the grader with `alarm()` so a wedged endpoint can't hang a verifier past Harbor's per-trial timeout.

If you're writing a new adapter that runs macOS-side AppleScript setup or graders, apply the same two passes — it's the difference between "graders work" and "60% of trials silently 0".

## Datasets and adapters

Tasks live in `datasets/<benchmark>/` as Harbor task directories (one folder per task with `task.json`, optional `setup.sh`, optional `tests/test.sh` for graders). Adapters in `src/runner/adapters/` turn external task sources into that layout:

```
src/runner/adapters/
├── adhoc/          ad-hoc prompt lists (datasets/adhoc/macos, datasets/adhoc/ios)
├── macosworld/     macOSWorld benchmark task export
└── collected/      tasks recorded interactively via the gateway's collector
```

Each adapter has its own `export` entry point — e.g.:

```bash
uv run python -m runner.adapters.adhoc.export src/runner/adapters/adhoc/tasks/macos.json --clean
```

## Running side-by-side comparisons

Each YAML agent runs independently. To benchmark two models on the same trajectories, add both under `agents:` and Harbor will fan out:

```yaml
agents:
    - import_path: runner.agents.macos.anthropic:AnthropicCUAAgent
      model_name: anthropic/claude-sonnet-4-6
    - import_path: runner.agents.macos.generic:GenericCUAAgent
      model_name: accounts/fireworks/models/kimi-k2-instruct
      kwargs:
          base_url: https://api.fireworks.ai/inference/v1
```

Both run the same task dirs and Harbor's summary reports rewards per agent.

## Troubleshooting Harbor runs

- **All trials fail with `ConnectError`**: the machine running Harbor can't reach `gateway_url`. Test with `curl https://api.use.computer/` first.
- **All trials fail with `unauthorized`**: wrong env's key. `mk_live_*` is environment-scoped — a key minted on prod won't work against `api.dev.use.computer` and vice versa.
- **Trials stall waiting for sandboxes**: `n_concurrent_trials` exceeds the reservation's slot count, or earlier trials' sandboxes didn't get cleaned up. Reduce concurrency or kill stragglers via the dashboard's per-row Reset button.
- **iOS trials with reward 0.0**: graders are typically stub `Score: 0`-printers for `adhoc/` tasks (no real grader exists). Real macOSWorld tasks ship real graders; check `tests/test.sh` in the task dir.
