"""HTTP sidecar for "Run with model" / Adhoc / Replay.

  POST /run          start a harbor run in the background; returns {job_id}
  GET  /jobs/{id}    current status pulled from disk; terminal state has reward
  GET  /health

Jobs land under JOBS_DIR (shared with harbor-viewer-debug) so once a run
completes the dashboard can deep-link straight to the trial for inspection.
"""
from runner.server.handlers import make_app

__all__ = ["make_app"]
