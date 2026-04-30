"""
Standalone runner — run a single macOSWorld task without Harbor.

Usage:
    uv run python run.py                                    # fzf pick task + default agent
    uv run python run.py --task "productivity__0a1b981d*"   # glob match
    uv run python run.py --agent openai --model gpt-5.4     # pick agent
    uv run python run.py --agent anthropic --model claude-sonnet-4-6 --max-steps 30
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import httpx
import litellm
from PIL import Image

DATASETS_DIR = Path(__file__).parent / "datasets" / "macosworld_ready"
RESULTS_DIR = Path(__file__).parent / "results"
PROMPTS_DIR = Path(__file__).parent / "src" / "runner" / "agents" / "prompts"

GATEWAY_URL = "http://localhost:8080"
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# When true, key lifecycle moments are also emitted as NDJSON on stdout so a
# wrapping process (e.g. the runner HTTP sidecar / gateway) can track live
# state. The human-readable prints are left in place for CLI users.
EMIT_EVENTS = False


def _emit(event: str, **fields) -> None:
    if EMIT_EVENTS:
        print(json.dumps({"event": event, **fields}), flush=True)


def pick_task(pattern: str | None) -> Path:
    tasks = sorted(DATASETS_DIR.iterdir()) if DATASETS_DIR.exists() else []
    if not tasks:
        print(f"No tasks found in {DATASETS_DIR}")
        sys.exit(1)

    if pattern:
        matched = [t for t in tasks if glob(str(t), recursive=False) or pattern in t.name]
        if not matched:
            matched = [t for t in tasks if pattern.rstrip("*") in t.name]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            tasks = matched

    # Try fzf
    try:
        names = "\n".join(t.name for t in tasks)
        result = subprocess.run(
            ["fzf", "--prompt=task> "],
            input=names, capture_output=True, text=True,
        )
        if result.returncode == 0:
            choice = result.stdout.strip()
            return DATASETS_DIR / choice
    except FileNotFoundError:
        pass

    # Fallback: numbered list
    for i, t in enumerate(tasks[:20]):
        print(f"  {i:3d}  {t.name}")
    if len(tasks) > 20:
        print(f"  ... ({len(tasks)} total)")
    idx = int(input("\nPick task number: "))
    return tasks[idx]


def load_task(task_dir: Path) -> dict:
    instruction = (task_dir / "instruction.md").read_text().strip()
    config_path = task_dir / "tests" / "task_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {"instruction": instruction, "config": config, "dir": task_dir}


async def run_pre_command(http: httpx.AsyncClient, sandbox_id: str, config: dict):
    pre_command = config.get("pre_command", "")
    if isinstance(pre_command, dict):
        pre_command = pre_command.get("en", "")
    if not pre_command:
        return

    print(f"  Running pre_command...")
    resp = await http.post(
        f"/v1/sandboxes/{sandbox_id}/exec",
        json={"command": pre_command}, timeout=60,
    )
    if resp.status_code == 200:
        data = resp.json()
        if data.get("return_code", 0) != 0:
            print(f"  pre_command warning: exit {data['return_code']}")
    else:
        print(f"  pre_command failed: {resp.status_code}")

    print("  Waiting 10s for setup...")
    await asyncio.sleep(10)


async def run_grading(http: httpx.AsyncClient, sandbox_id: str, config: dict, task_dir: Path | None = None) -> float:
    await asyncio.sleep(5)

    grading = config.get("grading_command", [])
    if grading:
        total_score = 0.0
        total_weight = 0.0
        for entry in grading:
            cmd, weight = entry[0], entry[1] if len(entry) > 1 else 100
            resp = await http.post(
                f"/v1/sandboxes/{sandbox_id}/exec",
                json={"command": cmd}, timeout=30,
            )
            if resp.status_code == 200:
                stdout = resp.json().get("stdout", "").strip()
                passed = stdout.lower() == "true"
                total_score += weight if passed else 0
                total_weight += weight
                print(f"  Grade [{'PASS' if passed else 'FAIL'}] (weight={weight})")
            else:
                total_weight += weight
                print(f"  Grade [ERROR] {resp.status_code}")
        return total_score / total_weight if total_weight > 0 else 0

    # Fall back to test.sh
    test_sh = (task_dir / "tests" / "test.sh") if task_dir else None
    if test_sh and test_sh.exists():
        content = test_sh.read_bytes()
        await http.put(
            f"/v1/sandboxes/{sandbox_id}/files?path=/tmp/grade_test.sh",
            content=content, headers={"Content-Type": "application/octet-stream"},
        )
        resp = await http.post(
            f"/v1/sandboxes/{sandbox_id}/exec",
            json={"command": "bash /tmp/grade_test.sh"}, timeout=60,
        )
        if resp.status_code == 200:
            stdout = resp.json().get("stdout", "")
            print(f"  Grade stdout: {stdout.strip()[:100]}")
            if "Score: 1" in stdout or "1" == stdout.strip():
                return 1.0
            return 0.0
        print(f"  Grade [ERROR] {resp.status_code}")
        return 0.0

    print("  No grading command — skipping verification")
    return -1


async def run_anthropic(
    http: httpx.AsyncClient,
    sandbox_id: str,
    instruction: str,
    model: str,
    max_steps: int,
    output_dir: Path,
    cache: bool = False,
):
    from anthropic import Anthropic

    client = Anthropic()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    uses_new = any(t in model for t in ("opus-4-6", "sonnet-4-6"))
    tool_type = "computer_20251124" if uses_new else "computer_20250124"
    beta = "computer-use-2025-11-24" if uses_new else "computer-use-2025-01-24"

    computer_tool = {
        "type": tool_type,
        "name": "computer",
        "display_width_px": SCREEN_WIDTH,
        "display_height_px": SCREEN_HEIGHT,
        "display_number": 1,
    }

    prompt_template = (PROMPTS_DIR / "anthropic.txt").read_text()
    system = (
        prompt_template
        .replace("{SCREENSHOT_WIDTH}", str(SCREEN_WIDTH))
        .replace("{SCREENSHOT_HEIGHT}", str(SCREEN_HEIGHT))
        .replace("{CLIENT_PASSWORD}", "lume")
    )

    # Initial screenshot
    resp = await http.get(f"/v1/sandboxes/{sandbox_id}/screenshot")
    ss = resp.content
    (images_dir / "step_000.png").write_bytes(ss)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(ss).decode()}},
        ],
    }]

    total_in, total_out, total_cache_read, total_cache_write = 0, 0, 0, 0
    extra: dict = {"cache_control": {"type": "ephemeral"}} if cache else {}

    for step in range(max_steps):
        print(f"  Step {step + 1}/{max_steps}...", end="", flush=True)
        _emit("agent_step", idx=step + 1)

        response = client.beta.messages.create(
            model=model, max_tokens=4096, system=system,
            tools=[computer_tool], messages=messages, betas=[beta],
            **extra,
        )

        u = response.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        total_in += u.input_tokens
        total_out += u.output_tokens
        total_cache_read += cr
        total_cache_write += cw
        print(f" [in={u.input_tokens} out={u.output_tokens} cache_r={cr} cache_w={cw}]", end="", flush=True)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" and not any(b.type == "tool_use" for b in response.content):
            text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
            print(f" done — {text[:80]}")
            break

        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue

            action = block.input
            action_type = action.get("action", "")
            print(f" {action_type}", end="", flush=True)

            ss_bytes = await execute_action_direct(http, sandbox_id, action)
            img_name = f"step_{step + 1:03d}.png"
            if ss_bytes:
                (images_dir / img_name).write_bytes(ss_bytes)

            content = [{"type": "text", "text": f"Action '{action_type}' executed"}]
            if ss_bytes:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(ss_bytes).decode()}})

            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        print()

    print(f"  Tokens: in={total_in} out={total_out} cache_read={total_cache_read} cache_write={total_cache_write}")


async def run_generic(
    http: httpx.AsyncClient,
    sandbox_id: str,
    instruction: str,
    model: str,
    max_steps: int,
    output_dir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = (PROMPTS_DIR / "pyautogui.txt").read_text()

    api_w, api_h = 1280, 800
    sx = SCREEN_WIDTH / api_w
    sy = SCREEN_HEIGHT / api_h

    def _resize(png_bytes: bytes) -> bytes | None:
        try:
            img = Image.open(io.BytesIO(png_bytes))
        except Exception:
            return None
        if img.size != (api_w, api_h):
            img = img.resize((api_w, api_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def _scale(a: dict) -> dict:
        for field in ("coordinate", "start_coordinate"):
            if field in a and isinstance(a[field], list) and len(a[field]) == 2:
                a[field] = [int(a[field][0] * sx), int(a[field][1] * sy)]
        return a

    resp = await http.get(f"/v1/sandboxes/{sandbox_id}/screenshot")
    ss = resp.content
    if _resize(ss) is None:
        print("  [screenshot failed on init — aborting]")
        return
    (images_dir / "step_000.png").write_bytes(ss)

    history: list[dict] = []
    total_in = total_out = 0

    for step in range(max_steps):
        system = (
            prompt_template
            .replace("{OS_TYPE}", "macOS")
            .replace("{TASK_INSTRUCTION}", instruction)
            .replace("{SCREENSHOT_WIDTH}", str(api_w))
            .replace("{SCREENSHOT_HEIGHT}", str(api_h))
            .replace("{SCREENSHOT_MAX_X}", str(api_w - 1))
            .replace("{SCREENSHOT_MAX_Y}", str(api_h - 1))
            .replace("{SCREENSHOT_CENTER_X}", str(api_w // 2))
            .replace("{SCREENSHOT_CENTER_Y}", str(api_h // 2))
            .replace("{STEP_NUMBER}", str(step + 1))
            .replace("{MAX_STEPS}", str(max_steps))
            .replace("{CLIENT_PASSWORD}", "lume")
            .replace("{CREDENTIALS_SECTION}", "Username: lume\nPassword: lume")
        )
        system += f"\n\nYou are on step {step + 1} of {max_steps}. Act efficiently."

        resized = _resize(ss)
        if resized is None:
            print(f" [bad screenshot at step {step+1} — stopping]")
            break
        ss_b64 = base64.b64encode(resized).decode()
        user_text = "Here is the current screenshot. Complete the task." if step == 0 else "What's the next step?"
        history.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ss_b64}"}},
                {"type": "text", "text": user_text},
            ],
        })
        if len(history) > 6:
            history = history[-6:]

        messages = [{"role": "system", "content": system}, *history]

        print(f"  Step {step + 1}/{max_steps}...", end="", flush=True)
        kwargs: dict = {"model": model, "messages": messages, "max_tokens": 4096}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        for attempt in range(5):
            try:
                response = litellm.completion(**kwargs)
                break
            except litellm.RateLimitError:
                wait = 60 * (attempt + 1)
                print(f" [rate_limit, retry in {wait}s]", end="", flush=True)
                await asyncio.sleep(wait)
        else:
            print(" [rate_limit_fatal]")
            break
        text = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = usage.prompt_tokens or 0
        out_tok = usage.completion_tokens or 0
        total_in += in_tok
        total_out += out_tok
        print(f" [in={in_tok} out={out_tok}]", end="", flush=True)
        history.append({"role": "assistant", "content": text})

        code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not code_blocks:
            if "DONE" in text or "FAIL" in text:
                print(f" terminal")
                break
            print(f" no-code")
            continue

        actions = [_scale(a) for a in _parse_pyautogui("\n".join(code_blocks), api_w, api_h)]
        img_name = f"step_{step + 1:03d}.png"
        for action in actions:
            print(f" {action.get('action', '?')}", end="", flush=True)
            ss_bytes = await execute_action_direct(http, sandbox_id, action)
            if ss_bytes:
                ss = ss_bytes
                (images_dir / img_name).write_bytes(ss_bytes)
        if not actions:
            r = await http.get(f"/v1/sandboxes/{sandbox_id}/screenshot")
            ss = r.content
        print()

    print(f"  Tokens: in={total_in} out={total_out}")


# ── pyautogui parser (ported from runner.agents.generic_cua) ─────────────────

_MODIFIER_KEYS = {"ctrl", "control", "alt", "option", "shift", "cmd", "command", "win", "super", "meta"}


def _parse_pyautogui(code: str, screen_w: int = 1920, screen_h: int = 1080) -> list[dict]:
    actions: list[dict] = []
    held_mods: list[str] = []
    for line in code.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "pyautogui.tripleClick(" in line:
            args = _pg_args(line)
            if len(args) >= 2:
                actions.append({"action": "triple_click", "coordinate": [_pc(args[0], screen_w), _pc(args[1], screen_h)]})
        elif "pyautogui.doubleClick(" in line:
            args = _pg_args(line)
            if len(args) >= 2:
                actions.append({"action": "double_click", "coordinate": [_pc(args[0], screen_w), _pc(args[1], screen_h)]})
        elif "pyautogui.click(" in line:
            args = _pg_args(line)
            if len(args) >= 2:
                a: dict = {"action": "click", "coordinate": [_pc(args[0], screen_w), _pc(args[1], screen_h)]}
                if "'right'" in line or '"right"' in line:
                    a["action"] = "right_click"
                actions.append(a)
        elif "pyautogui.moveTo(" in line:
            args = _pg_args(line)
            if len(args) >= 2:
                actions.append({"action": "move", "coordinate": [_pc(args[0], screen_w), _pc(args[1], screen_h)]})
        elif "pyautogui.scroll(" in line:
            args = _pg_args(line)
            clicks = _pi(args[0]) if args else -3
            x = _pc(args[1], screen_w) if len(args) > 1 else 0
            y = _pc(args[2], screen_h) if len(args) > 2 else 0
            actions.append({"action": "scroll", "coordinate": [x, y], "direction": "up" if clicks > 0 else "down", "amount": abs(clicks)})
        elif "pyautogui.typewrite(" in line or "pyautogui.write(" in line:
            t = _pg_str(line)
            if t:
                actions.append({"action": "type", "text": t})
        elif "pyautogui.press(" in line:
            k = _pg_str(line)
            if k:
                actions.append({"action": "key", "key": k})
        elif "pyautogui.hotkey(" in line:
            keys = _pg_strs(line)
            if keys:
                actions.append({"action": "key", "key": "+".join(keys)})
        elif "pyautogui.keyDown(" in line:
            k = _pg_str(line)
            if k:
                if k.lower() in _MODIFIER_KEYS:
                    held_mods.append(k)
                else:
                    actions.append({"action": "key", "key": "+".join(held_mods + [k])})
        elif "pyautogui.keyUp(" in line:
            k = _pg_str(line)
            if k and k in held_mods:
                held_mods.remove(k)
        elif "pyautogui.drag(" in line:
            args = _pg_args(line)
            if len(args) >= 2:
                actions.append({"action": "drag", "start_coordinate": [0, 0], "coordinate": [_pi(args[0]), _pi(args[1])]})
        elif "time.sleep(" in line or "pyautogui.sleep(" in line:
            args = _pg_args(line)
            if args:
                try:
                    dur = float(args[0])
                except ValueError:
                    dur = 1.0
                actions.append({"action": "wait", "duration": dur})
    return actions


def _pc(s: str, dim: int) -> int:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        return 0
    return int(f * dim) if 0.0 <= f < 1.0 else int(f)


def _pi(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


def _pg_args(line: str) -> list[str]:
    m = re.search(r"\(([^)]*)\)", line)
    if not m:
        return []
    parts = []
    for p in m.group(1).split(","):
        p = p.strip()
        if "=" in p:
            p = p.split("=", 1)[1].strip()
        p = p.strip("'\"")
        if p:
            parts.append(p)
    return parts


def _pg_str(line: str) -> str:
    for pat in (r'"""(.*?)"""', r"'''(.*?)'''", r'"([^"]*)"', r"'([^']*)'"):
        m = re.search(pat, line, re.DOTALL)
        if m:
            return m.group(1)
    return ""


def _pg_strs(line: str) -> list[str]:
    out: list[str] = []
    remaining = line
    for pat in (r'"""(.*?)"""', r"'''(.*?)'''"):
        for m in re.finditer(pat, remaining, re.DOTALL):
            out.append(m.group(1))
        remaining = re.sub(pat, "", remaining, flags=re.DOTALL)
    for m in re.finditer(r'"([^"]*)"', remaining):
        out.append(m.group(1))
    for m in re.finditer(r"'([^']*)'", remaining):
        out.append(m.group(1))
    return out


async def execute_action_direct(
    http: httpx.AsyncClient, sandbox_id: str, action: dict,
) -> bytes | None:
    prefix = f"/v1/sandboxes/{sandbox_id}"
    action_type = action.get("action", "")

    if action_type == "screenshot":
        resp = await http.get(f"{prefix}/screenshot")
        return resp.content

    if action_type in ("left_click", "click"):
        coord = action.get("coordinate", [0, 0])
        await http.post(f"{prefix}/mouse/click", json={"x": coord[0], "y": coord[1], "button": "left"})
    elif action_type == "right_click":
        coord = action.get("coordinate", [0, 0])
        await http.post(f"{prefix}/mouse/click", json={"x": coord[0], "y": coord[1], "button": "right"})
    elif action_type == "double_click":
        coord = action.get("coordinate", [0, 0])
        await http.post(f"{prefix}/mouse/click", json={"x": coord[0], "y": coord[1], "button": "left", "double": True})
    elif action_type == "type":
        await http.post(f"{prefix}/keyboard/type", json={"text": action.get("text", "")})
    elif action_type == "key":
        key = action.get("key", "")
        if "+" in key:
            await http.post(f"{prefix}/keyboard/hotkey", json={"keys": key})
        else:
            await http.post(f"{prefix}/keyboard/press", json={"key": key})
    elif action_type == "scroll":
        coord = action.get("coordinate", [0, 0])
        await http.post(f"{prefix}/mouse/scroll", json={"x": coord[0], "y": coord[1], "direction": action.get("direction", "down"), "amount": action.get("amount", 3)})
    elif action_type == "move":
        coord = action.get("coordinate", [0, 0])
        await http.post(f"{prefix}/mouse/move", json={"x": coord[0], "y": coord[1]})
    elif action_type == "drag":
        start = action.get("start_coordinate", [0, 0])
        end = action.get("coordinate", [0, 0])
        await http.post(f"{prefix}/mouse/drag", json={"startX": start[0], "startY": start[1], "endX": end[0], "endY": end[1]})
    elif action_type == "wait":
        await asyncio.sleep(action.get("duration", 1))
        return None

    await asyncio.sleep(2)
    resp = await http.get(f"{prefix}/screenshot")
    return resp.content


async def main():
    parser = argparse.ArgumentParser(description="Run a single macOSWorld task")
    parser.add_argument("--task", "-t", help="Task name or glob pattern")
    parser.add_argument("--task-dir", help="Absolute path to a task dir (bypasses dataset lookup)")
    parser.add_argument("--agent", "-a", default="anthropic", choices=["anthropic", "generic"], help="Agent type")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6", help="Model name")
    parser.add_argument("--api-base", default=None, help="LLM API base URL (generic agent)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps")
    parser.add_argument("--gateway", default=GATEWAY_URL, help="Gateway URL")
    parser.add_argument("--no-grade", action="store_true", help="Skip grading")
    parser.add_argument("--events", action="store_true", help="Emit NDJSON events on stdout")
    parser.add_argument("--cache", action="store_true", help="Enable Anthropic prompt caching (system + tools)")
    parser.add_argument("--gateway-api-key", default=None, help="Bearer token for gateway API (defaults to $GATEWAY_API_KEY)")
    args = parser.parse_args()

    global EMIT_EVENTS
    EMIT_EVENTS = args.events

    import os as _os
    api_key = args.gateway_api_key or _os.environ.get("GATEWAY_API_KEY", "")
    auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    task_dir: Path = Path(args.task_dir) if args.task_dir else pick_task(args.task)
    task = load_task(task_dir)
    print(f"\n{'='*60}")
    print(f"Task:  {task_dir.name}")
    print(f"Agent: {args.agent} ({args.model})")
    print(f"{'='*60}")
    print(f"\n{task['instruction'][:200]}...")

    output_dir = RESULTS_DIR / f"{task_dir.name}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(base_url=args.gateway, timeout=300, headers=auth_headers) as http:
            # Create sandbox
            print("\n[1] Creating sandbox...")
            resp = await http.post("/v1/sandboxes?wait=true")
            resp.raise_for_status()
            sandbox_id = resp.json()["sandbox_id"]
            print(f"  sandbox: {sandbox_id}")
            _emit("sandbox_created", sandbox_id=sandbox_id)

            try:
                # Run pre_command
                print("\n[2] Setting up task environment...")
                _emit("pre_command_start")
                await run_pre_command(http, sandbox_id, task["config"])
                _emit("pre_command_done")

                # Run agent
                print(f"\n[3] Running {args.agent} agent...")
                _emit("agent_start", model=args.model, max_steps=args.max_steps)
                if args.agent == "anthropic":
                    await run_anthropic(http, sandbox_id, task["instruction"], args.model, args.max_steps, output_dir, cache=args.cache)
                elif args.agent == "generic":
                    import os as _os2
                    api_base = args.api_base or "https://api.fireworks.ai/inference/v1"
                    fw_key = _os2.environ.get("FIREWORKS_API_KEY") if "fireworks" in api_base else None
                    model = args.model if "/" in args.model else f"openai/accounts/fireworks/models/{args.model}"
                    await run_generic(http, sandbox_id, task["instruction"], model, args.max_steps, output_dir, api_base=api_base, api_key=fw_key)
                _emit("agent_done")

                # Grade
                score = None
                if not args.no_grade:
                    print("\n[4] Grading...")
                    _emit("grading_start")
                    score = await run_grading(http, sandbox_id, task["config"], task_dir)
                    print(f"\n  Score: {score:.0%}")
                    (output_dir / "score.txt").write_text(f"{score}\n")
                    _emit("grading_done", reward=score)

                _emit("done", reward=score if score is not None else -1)

            finally:
                print("\n[5] Destroying sandbox...")
                await http.delete(f"/v1/sandboxes/{sandbox_id}")

        print(f"\nResults saved to {output_dir}/")
    except Exception as e:
        _emit("error", message=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
