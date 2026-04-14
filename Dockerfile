FROM python:3.12-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Build context is the deploy dir (see docker-compose.yml). Copy runner + sdk
# so pyproject's editable path dep (`mmini = { path = "../sdk" }`) resolves.
COPY runner /repo/runner
COPY sdk /repo/sdk

WORKDIR /repo/runner
RUN uv sync --frozen
# aiohttp / pyyaml for the thin HTTP shim; everything else comes via uv sync
RUN uv pip install aiohttp==3.10.10 pyyaml

EXPOSE 8090
CMD ["uv", "run", "python", "server.py"]
