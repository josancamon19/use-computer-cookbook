"""Entrypoint: `python -m runner.server`."""
from __future__ import annotations

import os

from aiohttp import web

from runner.server.config import JOBS_DIR
from runner.server.handlers import make_app


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "8090"))
    web.run_app(make_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
