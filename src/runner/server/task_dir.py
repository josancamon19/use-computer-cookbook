"""Build the on-disk harbor task dir from a /run payload."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import yaml

from runner.server.agent import strip_ios_prefix
from runner.server.grader import build_test_sh


def write_task_dir(task_dir: Path, task: dict, platform: str = "macos") -> None:
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    # harbor's TaskPaths.is_valid() requires an environment/ subdir to mark
    # this as a runnable task — empty is fine, it's just a marker.
    (task_dir / "environment").mkdir(exist_ok=True)

    (task_dir / "instruction.md").write_text(task.get("instruction", ""))
    # No [environment] block — use.computer sandbox sizes (cpus/memory) are fixed by
    # the warm-pool config on the gateway side; setting them here is ignored
    # at best and conflicts with UseComputerEnvironment._validate_definition.
    toml_lines = [
        "[verifier]\ntimeout_sec = 90\n",
        "[agent]\ntimeout_sec = 900\n",
    ]
    # Pin the simulator the task was collected with so replay coords match.
    # UseComputerEnvironment.start() reads this back, re-expands the prefix, and
    # passes the full id to client.create(type=ios). We store the short form
    # ("iPhone-17-Pro" / "iOS-26-4") so the toml is readable; the long
    # "com.apple.CoreSimulator.*" prefix is reconstructed at runtime.
    if platform == "ios":
        device_type = strip_ios_prefix((task.get("device_type") or "").strip(), "SimDeviceType")
        runtime = strip_ios_prefix((task.get("runtime") or "").strip(), "SimRuntime")
        if device_type or runtime:
            ios_block = ["[ios]\n"]
            if device_type:
                ios_block.append(f'device_type = "{device_type}"\n')
            if runtime:
                ios_block.append(f'runtime = "{runtime}"\n')
            toml_lines.append("".join(ios_block))
    (task_dir / "task.toml").write_text("".join(toml_lines))

    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text(build_test_sh(task.get("grading_command") or [], platform))
    test_sh.chmod(0o755)

    # Pre-command is bash-via-SSH — only meaningful on macOS. Skip for iOS.
    if platform != "ios":
        pre = (task.get("pre_command") or "").strip()
        if pre:
            setup_dir = task_dir / "tests" / "setup"
            setup_dir.mkdir(parents=True, exist_ok=True)
            pc = setup_dir / "pre_command.sh"
            pc.write_text("#!/bin/bash\n" + pre + "\n")
            pc.chmod(0o755)

    # actions.json — used by the replay agent (model="replay"). Walks each
    # step via the SDK with no LLM call.
    actions = task.get("actions")
    if actions:
        (task_dir / "actions.json").write_text(json.dumps({"steps": actions}, indent=2))

    # expected_final.json — base_cua reads this in post_run to compute
    # screenshot_similarity + a11y_match_ratio against the human's final
    # state. Set only for collected-task runs; adhoc has no reference.
    expected = task.get("expected_final")
    if expected:
        (task_dir / "expected_final.json").write_text(json.dumps(expected, indent=2))

    _write_files(task_dir, task.get("files") or [], platform)


def _write_files(task_dir: Path, files: list, platform: str) -> None:
    """Files captured at collect time. Persisted under setup/files/<sha>.<ext>
    plus a manifest mapping back to the remote_path. The runner's setup.py
    uploads each before pre_command runs."""
    if not files or platform == "ios":
        return
    files_dir = task_dir / "tests" / "setup" / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in files:
        remote = f.get("remote_path") or ""
        content_b64 = f.get("content_b64") or ""
        if not remote or not content_b64:
            continue
        ext = Path(remote).suffix
        local_name = hashlib.sha256(remote.encode()).hexdigest()[:16] + ext
        (files_dir / local_name).write_bytes(base64.b64decode(content_b64))
        manifest.append({"remote_path": remote, "local_name": local_name})
    if manifest:
        (files_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_job_yaml(
    job_yaml: Path,
    jobs_dir: Path,
    agent_spec: dict,
    gateway_url: str,
    platform: str = "macos",
) -> None:
    job_yaml.write_text(yaml.safe_dump({
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "orchestrator": {"type": "local", "n_concurrent_trials": 1},
        "environment": {
            "import_path": "runner.environments.use_computer:UseComputerEnvironment",
            "kwargs": {"gateway_url": gateway_url, "platform": platform},
            "delete": True,
        },
        "agents": [agent_spec],
    }))
