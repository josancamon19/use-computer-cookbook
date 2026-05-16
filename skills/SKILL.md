---
name: use-computer-sdk
description: Write Python that drives macOS or iOS sandboxes through the use-computer SDK (use.computer). Use this skill whenever the user is building, scripting, or debugging anything that touches `use_computer.Computer()`, `mac.exec_ssh`, `mac.mouse`, `mac.keyboard`, `mac.screenshot`, `ios.input.tap`/`long_press`/`swipe`/`type_text`, `ios.apps.open_url`, screen recording, file upload/download, or sandbox keepalive — including computer-use agents, evals, demos, and the use-computer-cookbook's Harbor jobs. Also trigger any time the user wants to use a **macOS sandbox or iOS emulator sandbox** for any reason — debugging an iOS app, testing UI on real macOS, exercising a flow on a fresh simulator, exploring an app interactively — they need this SDK to get one. Trigger even when the user just says "rent a Mac mini", "open Safari in a sandbox", "tap the screen on iOS", or "drive a simulator from Python" — they almost certainly want this SDK.
---

# use-computer SDK

The use-computer Python SDK rents an ephemeral macOS or iOS sandbox from a real Mac mini fleet at [use.computer](https://use.computer) and gives you a tight DSL to drive it. Every call hits `https://api.use.computer/v1/...` with a `mk_live_*` bearer token. The cookbook (this repo) wraps that with a Harbor environment + agents for benchmark-style runs.

This SKILL.md is the entry point. Read the section that matches what the user is doing, then load the relevant `references/` file for the full surface; don't dump everything into your reply.

## Mental model

```
Computer(api_key=…) ──create()──▶ MacOSSandbox    .mouse / .keyboard / .screenshot / .recording / .exec_ssh / .upload / .download_file
                                  IOSSandbox      .input  / .apps     / .screenshot / .recording / .upload / .download_file
                                  (both share .display, keepalive, file transfer, destroy)
```

- `Computer()` is a synchronous client. `AsyncComputer()` is the asyncio variant — same surface, every method becomes `await`-able. Pick one and don't mix.
- `create()` blocks until the sandbox is allocated (typical ~1 s warm-pool for macOS, up to ~45 s if an iOS sim has to boot). Type is `"macos"` by default; pass `type="ios"` for an iOS simulator.
- Use the sandbox as a context manager — `with Computer().create() as mac:` — so it destroys on exit. The sandbox-side server is reaped after ~2 min of no API touches even without an explicit destroy.
- Errors surface as `UseComputerError` subclasses (`use_computer.errors`). HTTP layer auto-retries idempotent calls, so a `ConnectError` reaching your code usually means the gateway is genuinely unreachable.

## Setup

```bash
pip install use-computer
export USE_COMPUTER_API_KEY=mk_live_…       # from use.computer/r/<reservation-id>
```

```python
from use_computer import Computer

with Computer().create() as mac:        # defaults: type="macos"
    mac.exec_ssh("open -a TextEdit")
    mac.keyboard.type("hello")
    png = mac.screenshot.take_full_screen()
```

`Computer()` reads `USE_COMPUTER_API_KEY` from the env. Pass `api_key=` or `base_url=` explicitly when scripting against dev (`https://api.dev.use.computer`) or a local tunnel.

## Vision-model coordinate caps (read this before clicking)

Every CUA-style model returns click coordinates **in resized-image space**, not the original screenshot's pixel space. If you feed a 3200×2000 retina screenshot straight to a model that resizes to 1568 px long-edge internally, the model returns `[x, y]` relative to 1568×980 — and `mac.mouse.click(x, y)` takes native display points, so your clicks land at half the intended position. The model's resize is silent and only loosely documented, which is why we cap on the client.

Known per-family long-edge caps:

| Model                                    | Cap (px) | Notes                                                                 |
| ---------------------------------------- | -------- | --------------------------------------------------------------------- |
| Claude Opus 4.7+, 4.8+, 5                | 2576     | Anthropic's larger cap on newer Opus models                           |
| Other Claude (Sonnet, Haiku, older Opus) | 1568     | Standard Anthropic CUA cap                                            |
| Kimi K2.5 / Fireworks                    | 896      | y-coord accuracy degrades >1024 px on tall iOS shots (probed)         |
| OpenAI computer-use-preview, Gemini      | 1280     | Fallback for everyone else                                            |

**If you're building agent code**, use the helper instead of duplicating the registry:

```python
from runner.agents.base import scale_screenshot_for_model

png = mac.screenshot.take_full_screen()
api_bytes, api_w, api_h, sx, sy = scale_screenshot_for_model(png, model_name)
# send api_bytes to the model (api_w × api_h)
# when the model returns coordinate=[x, y]:
mac.mouse.click(int(x * sx), int(y * sy))   # or ios.input.tap / long_press(int(x * sx), int(y * sy))
```

The helper does an aspect-preserving Pillow resize and returns `sx`, `sy` to multiply model coords back into native space. See `references/cookbook.md` for the rationale and how to extend it for a new model.

**If you're the agent reading this skill to drive a sandbox directly** (not writing agent code, just acting on a screenshot you've been handed): the same rule applies to you. Before reasoning about pixel positions in a screenshot, check whether the image you received was already resized to the cap above. If it wasn't, you're looking at a higher-resolution image than the coordinates you produce will refer to — divide by the scale or ask the harness for a pre-scaled version. When in doubt, use the helper above and trust the `sx`, `sy` it returns.

## When to read which reference

| Situation                                                               | Read                       |
| ----------------------------------------------------------------------- | -------------------------- |
| Driving a macOS sandbox (click, type, exec, upload, screenshot)         | `references/macos.md`      |
| Driving an iOS / iPad / Watch / TV / Vision simulator                   | `references/ios.md`        |
| Long-running session, slow agent think time, errors, retries, recording | `references/lifecycle.md`  |
| Running an agent over a dataset (Harbor jobs, this cookbook's recipes)  | `references/cookbook.md`   |

If the user's question spans multiple sections (e.g. "build an iOS agent that runs in Harbor"), load both — they're short.

## Async variant

The async API mirrors the sync API exactly. Use it when the caller is already an asyncio loop (an MCP server, an aiohttp service, etc.):

```python
import asyncio
from use_computer import AsyncComputer

async def main():
    async with AsyncComputer() as cc:
        async with await cc.create() as mac:
            await mac.keyboard.type("hello")
            png = await mac.screenshot.take_full_screen()

asyncio.run(main())
```

`start_keepalive` is sync-only by design — the heartbeat thread doesn't need an event loop.

## Where to look in source

- SDK source: `sdk/use_computer/` (sync + async DSL). Start with `sandbox.py` and `client.py`.
- One-shot examples: `sdk/examples/_1_hello_macos.py` … `_5_keepalive.py`.
- Cookbook recipes: `src/runner/agents/`, `src/runner/environments/use_computer.py`, `src/runner/configs/job-*.yaml`.
- API reference: <https://docs.use.computer/docs/sdk> and Swagger at <https://api.use.computer/docs>.
