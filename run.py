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
import json
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import httpx

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

    delay = config.get("before_action_delay_seconds", 10)
    if delay:
        print(f"  Waiting {delay}s for setup...")
        await asyncio.sleep(delay)


async def run_grading(http: httpx.AsyncClient, sandbox_id: str, config: dict) -> float:
    grading = config.get("grading_command", [])
    if not grading:
        print("  No grading command — skipping verification")
        return -1

    delay = config.get("before_grading_delay_seconds", 5)
    if delay:
        await asyncio.sleep(delay)

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
            status = "PASS" if passed else "FAIL"
            print(f"  Grade [{status}] (weight={weight})")
        else:
            total_weight += weight
            print(f"  Grade [ERROR] {resp.status_code}")

    return total_score / total_weight if total_weight > 0 else 0


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
    parser.add_argument("--agent", "-a", default="anthropic", choices=["anthropic"], help="Agent type")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6", help="Model name")
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

    task_dir = Path(args.task_dir) if args.task_dir else pick_task(args.task)
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
                _emit("agent_done")

                # Grade
                score = None
                if not args.no_grade:
                    print("\n[4] Grading...")
                    _emit("grading_start")
                    score = await run_grading(http, sandbox_id, task["config"])
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
