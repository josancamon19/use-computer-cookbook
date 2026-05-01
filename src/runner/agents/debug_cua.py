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
import json
from pathlib import Path

import httpx
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from mmini.sandbox import AsyncIOSSandbox

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

# iOS DSL has a smaller surface — no mouse, no keyboard hotkeys, no exec.
ACTION_POOL_IOS = ["tap", "swipe", "type", "button", "tap", "swipe"]


class DebugCUAAgent(BaseCUAAgent):
    """No-LLM agent for infra smoke tests.

    Three modes (mutually exclusive — replay wins, then realistic, then minimal):
      - replay=True (iOS only today): walks `<task_dir>/actions.json` and replays
        every recorded tool call against the live sandbox, with the same 2s
        spacing as the gateway's replay path. Used to validate the export
        pipeline + grader end-to-end without burning model tokens.
      - realistic=True: deterministic canonical action stream, exercises real
        per-step traffic patterns to repro concurrency bugs.
      - default: 1 screenshot + 1 click/tap + 1 exec per step.
    """

    def __init__(
        self,
        *args,
        realistic: bool = False,
        replay: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.realistic = realistic
        self.replay = replay

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
        is_ios = isinstance(sandbox, AsyncIOSSandbox)

        if self.replay:
            if is_ios:
                await self._replay_ios(sandbox)
            else:
                await self._replay_macos(sandbox)
            await self.post_run(context, "debug", "debug-cua-replay")
            return

        for step in range(1, self.max_steps + 1):
            if self.realistic:
                if is_ios:
                    await self._realistic_step_ios(sandbox, step)
                else:
                    await self._realistic_step(sandbox, step)
            else:
                if is_ios:
                    await self._minimal_step_ios(sandbox, step)
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

    # ---------------- iOS replay (no LLM) ----------------

    async def _replay_ios(self, sandbox) -> None:
        """Walk `<task_dir>/actions.json` and dispatch each recorded tool call.

        Mirrors the gateway-side runReplay flow: 2s between actions, type
        coalescing (legacy collections recorded one step per character; iOS
        Maps and friends don't render single-char inputs cleanly when paced
        like that, so we concatenate consecutive `type` runs into one call).
        """
        actions_path = self.task_dir / "actions.json" if self.task_dir else None
        if actions_path is None or not actions_path.exists():
            raise RuntimeError(
                f"replay: actions.json missing at {actions_path} "
                "— re-export the task with collected/export.py"
            )
        data = json.loads(actions_path.read_text())
        actions = self._coalesce_type_runs(data.get("steps") or [])
        total = len(actions)
        self.logger.info(f"[replay-ios] replaying {total} action(s)")

        for i, action in enumerate(actions, start=1):
            if i > 1:
                await asyncio.sleep(2.0)
            ss = await sandbox.screenshot.take_full_screen()
            ss_path = self.images_dir / f"step_{i:03d}.png"
            ss_path.write_bytes(ss)
            try:
                await self._dispatch_ios_action(sandbox, action)
                self.logger.info(
                    f"[replay-ios] {i}/{total} {action['function']}({action.get('args')})"
                )
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"[replay-ios] {i}/{total} {action['function']} ERR: "
                    f"{type(e).__name__}: {e}"
                )
            self._append_replay_step(action, ss_path)

        # Capture the post-final-action state.
        await asyncio.sleep(2.0)
        final_ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / f"step_{total + 1:03d}.png").write_bytes(final_ss)

    async def _dispatch_ios_action(self, sandbox, action: dict) -> None:
        """Translate one collected action into an SDK call. Maps the gateway's
        recorded vocabulary (tap/swipe/type/button/key/open_url) onto the
        AsyncIOSSandbox surface."""
        fn = action["function"]
        args = action.get("args") or {}
        if fn == "tap":
            await sandbox.input.tap(float(args["x"]), float(args["y"]))
        elif fn == "swipe":
            # Gateway records camelCase; SDK takes snake_case.
            from_x = float(args.get("fromX") or args.get("from_x", 0))
            from_y = float(args.get("fromY") or args.get("from_y", 0))
            to_x = float(args.get("toX") or args.get("to_x", 0))
            to_y = float(args.get("toY") or args.get("to_y", 0))
            await sandbox.input.swipe(from_x, from_y, to_x, to_y)
        elif fn in ("type", "type_text"):
            await sandbox.input.type_text(args.get("text", ""))
        elif fn in ("button", "press_button"):
            btn = (args.get("button") or "").lower().replace("_", "-")
            await sandbox.input.press_button(btn)
        elif fn in ("key", "press_key"):
            await sandbox.input.press_key(int(args["keycode"]))
        elif fn == "open_url":
            await sandbox.apps.open_url(args.get("url", ""))
        else:
            raise ValueError(f"unsupported replay action: {fn!r}")

    @staticmethod
    def _coalesce_type_runs(actions: list[dict]) -> list[dict]:
        """Merge consecutive type/type_text actions into a single type_text call.
        Old collections recorded one step per character; sending nine separate
        /type calls 2s apart breaks rendering on iOS Maps and friends."""
        out: list[dict] = []
        i = 0
        while i < len(actions):
            a = actions[i]
            if a.get("function") in ("type", "type_text"):
                buf = (a.get("args") or {}).get("text", "")
                j = i + 1
                while j < len(actions) and actions[j].get("function") in (
                    "type", "type_text"
                ):
                    buf += (actions[j].get("args") or {}).get("text", "")
                    j += 1
                out.append({"function": "type_text", "args": {"text": buf}})
                i = j
            else:
                out.append(a)
                i += 1
        return out

    # ---------------- macOS replay (no LLM) ----------------

    async def _replay_macos(self, sandbox) -> None:
        """Walk <task>/actions.json and dispatch each recorded action via the macOS SDK."""
        actions_path = self.task_dir / "actions.json" if self.task_dir else None
        if actions_path is None or not actions_path.exists():
            raise RuntimeError(
                f"replay: actions.json missing at {actions_path} "
                "— re-export the task with collected/export.py"
            )
        data = json.loads(actions_path.read_text())
        actions = data.get("steps") or []
        total = len(actions)
        self.logger.info(f"[replay-macos] replaying {total} action(s)")

        for i, action in enumerate(actions, start=1):
            if i > 1:
                await asyncio.sleep(2.0)
            ss = await sandbox.screenshot.take_full_screen()
            ss_path = self.images_dir / f"step_{i:03d}.png"
            ss_path.write_bytes(ss)
            try:
                await self._dispatch_macos_action(sandbox, action)
                self.logger.info(
                    f"[replay-macos] {i}/{total} {action['function']}({action.get('args')})"
                )
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"[replay-macos] {i}/{total} {action['function']} ERR: "
                    f"{type(e).__name__}: {e}"
                )
            self._append_replay_step(action, ss_path)

        # Capture the post-final-action state so we can see what the last
        # click actually did (verifier needs this signal too).
        await asyncio.sleep(2.0)
        final_ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / f"step_{total + 1:03d}.png").write_bytes(final_ss)

    def _append_replay_step(self, action: dict, ss_path: Path) -> None:
        """ATIF step entry per replayed action so trajectory.json reflects the run.

        Stores the screenshot path RELATIVE to the trial dir (e.g.
        agent/images/step_001.png). Absolute paths break after the runner
        sidecar flattens <work_dir>/jobs/<ts>/<trial>/ → <work_dir>/<trial>/.

        Shape mirrors anthropic_cua (message field, text+image content) so
        harbor-viewer's code-block component doesn't crash on .split() of
        undefined fields.
        """
        try:
            rel = ss_path.relative_to(self.logs_dir.parent)
        except ValueError:
            rel = Path("agent/images") / ss_path.name
        fn = action.get("function") or "?"
        args = action.get("args") or {}
        self.steps.append({
            "step_id": len(self.steps) + 1,
            "source": "agent",
            "message": f"[replay] {fn}({args})",
            "tool_calls": [{"function": fn, "args": args}],
            "observation": {"results": [{
                "content": [
                    {"type": "text", "text": f"step screenshot: {rel}"},
                    {"type": "image", "source": {"media_type": "image/png", "path": str(rel)}},
                ]
            }]},
        })

    async def _dispatch_macos_action(self, sandbox, action: dict) -> None:
        """Translate a recorded action into a macOS SDK call."""
        fn = action["function"]
        args = action.get("args") or {}
        if fn in ("left_click", "click"):
            await sandbox.mouse.click(
                float(args["x"]), float(args["y"]),
                button=args.get("button", "left"),
                double=bool(args.get("double", False)),
            )
        elif fn == "double_click":
            await sandbox.mouse.click(
                float(args["x"]), float(args["y"]), button="left", double=True
            )
        elif fn == "right_click":
            await sandbox.mouse.click(
                float(args["x"]), float(args["y"]), button="right"
            )
        elif fn in ("type_text", "type"):
            await sandbox.keyboard.type(args.get("text", ""))
        elif fn in ("press_key", "key", "press"):
            await sandbox.keyboard.press(args.get("key") or args.get("text", ""))
        elif fn == "hotkey":
            keys = args.get("keys") or args.get("key", "")
            await sandbox.keyboard.hotkey(keys)
        elif fn == "scroll":
            await sandbox.mouse.scroll(
                float(args.get("x", 0)), float(args.get("y", 0)),
                args.get("direction", "down"),
                int(args.get("amount", 3)),
            )
        elif fn == "move":
            await sandbox.mouse.move(float(args["x"]), float(args["y"]))
        elif fn == "drag":
            await sandbox.mouse.drag(
                float(args.get("startX") or args.get("from_x", 0)),
                float(args.get("startY") or args.get("from_y", 0)),
                float(args.get("endX") or args.get("to_x", 0)),
                float(args.get("endY") or args.get("to_y", 0)),
            )
        else:
            raise ValueError(f"unsupported macos replay action: {fn!r}")

    # ---------------- iOS variants ----------------

    async def _minimal_step_ios(self, sandbox, step: int) -> None:
        ss = await sandbox.screenshot.take_full_screen()
        (self.images_dir / f"step_{step:03d}.png").write_bytes(ss)
        await sandbox.input.tap(500, 1000)
        print(f"[debug-ios] step={step}/{self.max_steps} screenshot={len(ss)} bytes")

    async def _realistic_step_ios(self, sandbox, step: int) -> None:
        """iOS analog of _realistic_step: screenshot + REAL_ACTIONS_PER_STEP
        canonical iOS DSL actions. No exec — iOS sims have no shell."""
        ss = await sandbox.screenshot.take_full_screen()
        if step == 1 or step % 10 == 0 or step == self.max_steps:
            (self.images_dir / f"step_{step:03d}.png").write_bytes(ss)

        start = ((step - 1) * REAL_ACTIONS_PER_STEP) % len(ACTION_POOL_IOS)
        names = [
            ACTION_POOL_IOS[(start + i) % len(ACTION_POOL_IOS)]
            for i in range(REAL_ACTIONS_PER_STEP)
        ]

        executed = []
        for name in names:
            try:
                await self._run_action_ios(sandbox, name)
                executed.append(name)
            except Exception as e:
                executed.append(f"{name}!err={type(e).__name__}")

        print(
            f"[debug-realistic-ios] step={step}/{self.max_steps} "
            f"ss={len(ss)}B actions={','.join(executed)}"
        )

    async def _run_action_ios(self, sandbox, name: str) -> None:
        """iOS DSL canonical actions — fixed coordinates, deterministic."""
        if name == "tap":
            await sandbox.input.tap(500, 1000)
        elif name == "swipe":
            # swipe up = drag content upward (reveal what's below)
            await sandbox.input.swipe(500, 1500, 500, 500)
        elif name == "type":
            await sandbox.input.type_text("x" * REAL_TYPE_LEN)
        elif name == "button":
            await sandbox.input.press_button("home")
        else:
            raise ValueError(f"unknown debug ios action: {name}")

    # ---------------- macOS variants ----------------

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
