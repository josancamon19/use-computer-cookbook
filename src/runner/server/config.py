"""Process-wide constants for the runner sidecar."""
from __future__ import annotations

import os
import re
from pathlib import Path

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/data/jobs"))
REPO_ROOT = Path("/repo")
RUNNER_DIR = REPO_ROOT / "runner"

# Matches "sb-<32 hex>" sandbox IDs in trial logs (used to surface the
# active sandbox to the dashboard before harbor finishes writing result.json).
SANDBOX_RE = re.compile(r"(sb-[0-9a-f]{16,})")
