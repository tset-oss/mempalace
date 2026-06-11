# Dockerfile — MemPalace central HTTP MCP server (FastMCP over streamable-http).
#
# Built from the repository root (compose sets `context: ..`), so the whole
# mempalace package is available to pip. Installs the `postgres` + `serve`
# extras: the server talks to the central PostgreSQL `db` service and exposes
# the MCP endpoint at /mcp behind a shared bearer token. Runs alongside `db` in
# docker-compose.yml.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY . /app

# psycopg[binary], chromadb, tokenizers, onnxruntime and the mcp SDK all publish
# manylinux wheels, so no compiler toolchain is needed on slim.
#
# Upgrade pip first: a months-old cached python:3.12-slim ships a pip whose
# resolver gives up (ResolutionImpossible) on dependency graphs that current
# pip solves — exactly what broke this build on 2026-06-11. The constraints
# file bounds the volatile httpx/httpcore stack so one bad transitive release
# cannot make the floating-resolution build unsolvable.
RUN pip install --upgrade pip && \
    pip install -c deploy/server/constraints.txt ".[postgres,serve]"

# Pre-bake the embeddinggemma-300m ONNX model (~300 MB) into the image so the
# server never downloads it from HuggingFace at request time (no runtime egress
# to HF, no cold-start latency on the first add_drawer/search). The download is
# cached under the default HF cache (/root/.cache/huggingface), reused at run
# time since the container runs as root. This matches MEMPALACE_EMBEDDING_MODEL
# in docker-compose.yml; if you switch the compose model, the new model
# lazy-downloads on first use instead.
RUN python -c "from mempalace.embedding import get_embedding_function as g; g(model='embeddinggemma')(['warm-up'])"

EXPOSE 8080

# Bind all interfaces *inside the container*; the published port + the shared
# bearer token (MEMPALACE_AUTH_TOKEN, read by `serve` when --auth-token is
# omitted) are the security boundary. Team routing is per-session via the
# X-Mempalace-Team header each client sends — the server default stays "default".
CMD ["mempalace", "serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
