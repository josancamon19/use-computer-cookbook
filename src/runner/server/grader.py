"""Build test.sh content from a task's grader spec.

Two grader shapes are supported:

  - JSON checker spec (iOS DSL). POSTs to /v1/sandboxes/<id>/grade and
    captures the structured response into verifier/grader_checks.json.
  - Raw bash. Runs the command and greps stdout for "true".
"""
from __future__ import annotations

import json


def try_parse_spec(text: str) -> list[dict] | None:
    """Return the parsed check spec if `text` is a JSON list of {kind: ...}
    objects. Otherwise None — the grader is treated as raw bash."""
    s = text.strip()
    if not s.startswith("["):
        return None
    try:
        spec = json.loads(s)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(spec, list) or not all(
        isinstance(x, dict) and "kind" in x for x in spec
    ):
        return None
    return spec


def build_test_sh(graders: list, platform: str) -> str:
    """Compose the verifier script. graders is a list of (cmd, weight) pairs."""
    header = ["#!/bin/bash"]
    if platform == "ios":
        header += [
            "# iOS task — pre-commands are not supported (no shell access on",
            "# the simulator). Grading runs on the host: POST the check spec to",
            "# /v1/sandboxes/<id>/grade, the gateway evaluates against the live",
            "# AX tree (same evaluator the replay path uses) and returns",
            "# {\"passed\": bool, \"results\": [...]}.",
        ]
    grader_sh = header + [
        'PREFIX=""',
        '[ -d "/tmp/harbor/logs" ] && PREFIX="/tmp/harbor"',
        'REWARD="${PREFIX}/logs/verifier/reward.txt"',
    ]
    if not graders:
        # No grader yet — fail by default so the missing verifier surfaces
        # instead of silently passing every run.
        grader_sh += [
            'echo "no grader for this task yet — defaulting reward to 0" >&2',
            'echo "0" > "$REWARD"',
            'echo "Score: 0"',
            'exit 0',
        ]
    else:
        # JSON-spec graders (iOS, the new shape) and bash graders (macOS or
        # legacy iOS) coexist — sniff each entry and emit the matching wrapper.
        for cmd, _weight in graders:
            spec = try_parse_spec(cmd)
            if spec is not None:
                payload = json.dumps({"specs": spec})
                grader_sh += [
                    f"PAYLOAD={json.dumps(payload)}",
                    # Capture full response so per-checker results survive
                    # for the dashboard to display.
                    'RESP=$(curl -sf -H "Authorization: Bearer $USE_COMPUTER_API_KEY" '
                    '-H "Content-Type: application/json" '
                    '-X POST "$GATEWAY_URL/v1/sandboxes/$SANDBOX_ID/grade" '
                    '-d "$PAYLOAD" 2>/dev/null)',
                    'mkdir -p "${PREFIX}/logs/verifier"',
                    'echo "$RESP" > "${PREFIX}/logs/verifier/grader_checks.json"',
                    'if ! echo "$RESP" | grep -qE \'"passed"[[:space:]]*:[[:space:]]*true\'; then',
                    '  echo "0" > "$REWARD"; echo "Score: 0"; exit 0',
                    "fi",
                ]
                continue
            # Raw-bash grader (macOS osascript, or older iOS curl-style).
            # Older iOS curl graders may lack the bearer token — auto-inject so
            # legacy tasks keep grading. Mirrors gateway runIOSGrader().
            if platform == "ios" and "curl " in cmd and "Authorization" not in cmd:
                cmd = cmd.replace(
                    'curl -s "$GATEWAY_URL',
                    'curl -s -H "Authorization: Bearer $USE_COMPUTER_API_KEY" "$GATEWAY_URL',
                )
            grader_sh += [
                f"if ! bash -c {json.dumps(cmd)} 2>/dev/null | grep -qi 'true'; then",
                '  echo "0" > "$REWARD"; echo "Score: 0"; exit 0',
                "fi",
            ]
        grader_sh += ['echo "1" > "$REWARD"', 'echo "Score: 1"', 'exit 0']
    return "\n".join(grader_sh) + "\n"
