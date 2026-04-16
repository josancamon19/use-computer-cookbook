"""Debug CUA agent — smoke-tests infra without calling any LLM.

Two modes:
  - default (realistic=False): 1 screenshot + 1 click + 1 exec per step.
    Minimum viable traffic, useful for quick end-to-end smoke tests.
  - realistic=True: mimics real CUA agent per-step traffic so concurrency
    stress reflects production load. Deterministic — no RNG, no per-trial
    variance. Runs a fixed canonical action sequence rotating across steps.
"""

from __future__ import annotations

import asyncio

import httpx
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from runner.agents.base_cua import BaseCUAAgent

# Realistic-mode tunables. Fixed values — deterministic runs so two back-to-back
# trials of the same task emit the identical HTTP sequence, and concurrency
# breakages can be reproduced from a single failure.
REAL_ACTIONS_PER_STEP = 2           # mouse/keyboard actions per step
REAL_TYPE_LEN = 100                 # chars per type call
REAL_THINK_DELAY_SEC = 10           # between-step "thinking" — INTEGER (httpbin needs int)
REAL_EXEC_SLEEP_SEC = 2.0           # seconds the remote exec_ssh command sleeps
# httpbin.org/delay/<N> sleeps server-side for N seconds and returns. Used in
# place of asyncio.sleep so each step's "thinking" actually exercises an
# outbound HTTPS round-trip — same pattern as real CUA agents calling Anthropic
# / Fireworks per step. Falls back to local sleep on network errors.
REAL_THINK_URL = "https://httpbin.org/delay"

# Canonical action pool. Each step consumes REAL_ACTIONS_PER_STEP consecutive
# slots starting at (step-1) * REAL_ACTIONS_PER_STEP (mod len), so the full
# set is exercised across a trial.
ACTION_POOL = [
    "click", "type", "scroll", "hotkey", "move", "press", "drag", "rclick",
]


class DebugCUAAgent(BaseCUAAgent):
    def __init__(self, *args, realistic: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.realistic = realistic

    @staticmethod
    def name() -> str:
        return "debug-cua"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        sandbox = await self.pre_run(environment)

        for step in range(1, self.max_steps + 1):
            if self.realistic:
                await self._realistic_step(sandbox, step)
            else:
                await self._minimal_step(sandbox, step)

            await self._fire_in_process(environment, step)
            if step < self.max_steps:
                # if self.realistic:
                #     await self._think(REAL_THINK_DELAY_SEC)
                # else:
                await asyncio.sleep(1.0)

        await self.post_run(context, "debug", "debug-cua")

    async def _think(self, delay_sec: int) -> None:
        """Outbound HTTPS round-trip that takes ~delay_sec to simulate the
        LLM API call real CUA agents make between actions. Falls back to a
        local sleep on network errors so transient blips don't fail trials."""
        try:
            async with httpx.AsyncClient(timeout=delay_sec + 10) as http:
                await http.get(f"{REAL_THINK_URL}/{delay_sec}")
        except Exception as e:
            # log + sleep the remainder so timing stays consistent
            print(f"[debug-realistic] think network err: {type(e).__name__}: {e}")
            await asyncio.sleep(delay_sec)

    async def _minimal_step(self, sandbox, step: int) -> None:
        ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / f"step_{step:03d}.png").write_bytes(ss)
        await sandbox.mouse.click(960, 540)
        result = await sandbox.exec_ssh("echo hello world")
        print(f"[debug] step={step}/{self.max_steps} screenshot={len(ss)} bytes, exec={result}")

    async def _realistic_step(self, sandbox, step: int) -> None:
        """One step: screenshot + REAL_ACTIONS_PER_STEP canonical actions + exec.
        Fully deterministic — same step number always runs the same actions."""

        # Screenshot (always fetched — exercises the bandwidth-dominant call)
        # but only persisted every 10 steps so long trials don't blow disk.
        # Filmstrip in the viewer still shows the trajectory shape.
        ss = await sandbox.screenshot.take_full_screen()
        if step == 1 or step % 10 == 0 or step == self.max_steps:
            (self.images_dir / f"step_{step:03d}.png").write_bytes(ss)

        # Pick actions for this step by rotating through ACTION_POOL
        start = ((step - 1) * REAL_ACTIONS_PER_STEP) % len(ACTION_POOL)
        names = [
            ACTION_POOL[(start + i) % len(ACTION_POOL)]
            for i in range(REAL_ACTIONS_PER_STEP)
        ]

        executed = []
        for name in names:
            try:
                await self._run_action(sandbox, name)
                executed.append(name)
            except Exception as e:
                executed.append(f"{name}!err={type(e).__name__}")

        # One exec with a fixed sleep — stresses the long-lived gateway exec path.
        try:
            await sandbox.exec_ssh(
                f"printf 'step_{step}'; sleep {REAL_EXEC_SLEEP_SEC:.1f}"
            )
        except Exception as e:
            print(f"[debug-realistic] exec err: {type(e).__name__}: {e}")

        print(
            f"[debug-realistic] step={step}/{self.max_steps} "
            f"ss={len(ss)}B actions={','.join(executed)} "
            f"exec_sleep={REAL_EXEC_SLEEP_SEC}s"
        )

    async def _run_action(self, sandbox, name: str) -> None:
        """Canonical action implementations — fixed coordinates and payloads,
        no randomness. The specific values are arbitrary but stable so every
        run emits byte-identical HTTP request bodies (makes wire-level diffs
        across runs meaningful)."""
        if name == "click":
            await sandbox.mouse.click(960, 540)
        elif name == "rclick":
            await sandbox.mouse.click(960, 540, button="right")
        elif name == "scroll":
            await sandbox.mouse.scroll(960, 540, direction="down", amount=3)
        elif name == "move":
            await sandbox.mouse.move(500, 500)
        elif name == "drag":
            await sandbox.mouse.drag(200, 200, 800, 800)
        elif name == "type":
            await sandbox.keyboard.type("x" * REAL_TYPE_LEN)
        elif name == "press":
            await sandbox.keyboard.press("return")
        elif name == "hotkey":
            await sandbox.keyboard.hotkey("cmd+a")
        else:
            raise ValueError(f"unknown debug action: {name}")
