"""Entrypoint: `python -m runner.server`."""
from __future__ import annotations

import os

from aiohttp import web

from runner.server.config import JOBS_DIR
from runner.server.handlers import make_app


def main() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "8090"))
    host = os.environ.get("BIND_HOST", "127.0.0.1")
    web.run_app(make_app(), host=host, port=port)


if __name__ == "__main__":
    main()
