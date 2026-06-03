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
RUN pip install ".[postgres,serve]"

EXPOSE 8080

# Bind all interfaces *inside the container*; the published port + the shared
# bearer token (MEMPALACE_AUTH_TOKEN, read by `serve` when --auth-token is
# omitted) are the security boundary. Team routing is per-session via the
# X-Mempalace-Team header each client sends — the server default stays "default".
CMD ["mempalace", "serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
