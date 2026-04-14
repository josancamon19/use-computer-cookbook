FROM python:3.12-slim
WORKDIR /app

# Slim deps — only what run.py / server.py actually need for the
# sonnet/anthropic path. No uv, no harbor, no mmini, no litellm.
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
       aiohttp==3.10.10 \
       anthropic>=0.86.0 \
       httpx>=0.27.0

COPY run.py server.py /app/

EXPOSE 8090
CMD ["python", "server.py"]
