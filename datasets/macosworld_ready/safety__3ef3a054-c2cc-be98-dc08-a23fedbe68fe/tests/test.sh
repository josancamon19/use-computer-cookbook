#!/bin/bash
# macOSWorld grading — runs grading_commands from task_config.json
set -e

# Detect path prefix (macOS SIP blocks /tests, so mmini uses /tmp/harbor)
PREFIX=""
[ -f "/tmp/harbor/tests/task_config.json" ] && PREFIX="/tmp/harbor"

CONFIG="${PREFIX}/tests/task_config.json"
REWARD="${PREFIX}/logs/verifier/reward.txt"

[ ! -f "$CONFIG" ] && echo "0" > "$REWARD" && exit 0

# Run grading commands (0-1 scale)
SCORE=$(python3 -c "
import json, subprocess, sys
commands = json.load(open('$CONFIG')).get('grading_command', [])
for cmd, score in commands:
    if score != 100: continue
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if 'true' in r.stdout.lower():
            print(1); sys.exit(0)
    except: pass
print(0)
")

echo "$SCORE" | head -1 > "$REWARD"
echo "Score: $(cat $REWARD)"
