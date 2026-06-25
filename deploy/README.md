# Central MemPalace on PostgreSQL

This directory hosts the **central, team-vaulted** deployment of MemPalace: a
single PostgreSQL 18 instance with `pgvector` (vectors), `pg_trgm` (the trigram
GIN that backs keyword recall), and Apache AGE (graph), plus an **HTTP MCP
server** (FastMCP, streamable-http) that every engineer's Claude Code connects
to. Every engineering team gets its own **vault** (an isolated Postgres schema,
e.g. `team_frontend`). ParadeDB `pg_search` is also provisioned (see the
Dockerfile) but the query path does not use it for retrieval — keyword
candidates are scoreless rows from the trigram GIN, ranked by the shared Python
Okapi-BM25 reranker.

It is built for an internal engineering network. Postgres itself has **no auth
beyond its static credentials** — never expose port 5432 publicly. The MCP
server (port 8080) is gated by a **shared static bearer token**
(`MEMPALACE_AUTH_TOKEN`); keep it behind the org network / a TLS-terminating
reverse proxy before exposing it.

The client wiring engineers run (the `claude mcp add` recipes, team routing) is
in section 2 below; a reference consumer-side integration (recall/save hooks and
the in-session usage skill) is vendored under
[`../integrations/agent-commons/`](../integrations/agent-commons/).

## Layout

| File | Purpose |
|------|---------|
| `postgres/Dockerfile` | PG18 + pgvector 0.8.2 + pg_search 0.22.5 + AGE 1.7.0-rc0 |
| `postgres/postgresql.conf` | Tuning; `shared_preload_libraries='pg_search,age'` |
| `postgres/initdb/10-extensions.sql` | First-boot `CREATE EXTENSION` (idempotent) |
| `server/Dockerfile` | MCP server image; `pip install .[postgres,serve]` + pre-baked embeddinggemma ONNX model |
| `docker-compose.yml` | The `db` + `mcp` services + volume |
| `.env.example` | Server token + per-machine client env |
| `../.dockerignore` | Trims the MCP server build context |

## 1. Start the stack (database + MCP server)

```bash
cd deploy
cp .env.example .env                     # then set MEMPALACE_AUTH_TOKEN
#   openssl rand -hex 32   →  a good shared bearer token
docker compose up -d --build             # first db build compiles pgvector + AGE (~5-10 min)
docker compose ps                        # wait for db healthy + mcp up
```

`docker compose up` fails fast if `MEMPALACE_AUTH_TOKEN` is unset, so the server
is never started unauthenticated by accident.

Verify the database extensions:

```bash
docker compose exec db psql -U mempalace -d mempalace -c \
  "CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_search;
   CREATE EXTENSION IF NOT EXISTS age;
   LOAD 'age';
   SELECT extname, extversion FROM pg_extension ORDER BY extname;"
```

You should see `age`, `pg_search`, and `vector` listed.

Verify the MCP server is up and enforcing auth:

```bash
# Unauthenticated → 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8080/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

# Authenticated initialize → 200 (a real MCP handshake)
curl -s -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer $MEMPALACE_AUTH_TOKEN" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}'
```

## 2. Connect Claude Code (per engineer)

The central server is consumed as a **thin HTTP MCP client** — engineers do not
install the fork. The MCP config carries the bearer token and the team vault:
`user` scope = the developer's machine-wide default, `project` scope (committed
`.mcp.json`) = a repo's default. Full recipes and team-routing details are in
the in-session skill under
[`../integrations/agent-commons/skills/mempalace/SKILL.md`](../integrations/agent-commons/skills/mempalace/SKILL.md).
The short version:

```bash
# Machine-wide default vault for this developer (user scope)
claude mcp add --transport http --scope user mempalace https://mempalace.internal/mcp \
  --header "Authorization: Bearer <shared-token>" \
  --header "X-Mempalace-Team: <your-team>"
```

`X-Mempalace-Team` seeds the session's default vault; `switch_team` overrides it
at runtime, and a `vault=` argument overrides a single call. `vault: "all"` on
`mempalace_search` reads across every team vault. Team vaults are created lazily
— the first write for `frontend` provisions the `team_frontend` schema, tables,
and indexes automatically.

### Legacy: run the fork locally instead of the HTTP server

The pre-server path (each machine installs `mempalace[postgres]` and talks to
Postgres directly) still works:

```bash
pip install "mempalace[postgres]"
mempalace team set frontend --database-url postgresql://mempalace:mempalace@DB_HOST:5432/mempalace
```

or via env (`MEMPALACE_BACKEND=postgres`, `MEMPALACE_TEAM=frontend`,
`MEMPALACE_DATABASE_URL=...`). Prefer the central HTTP server for distribution —
it keeps the fork on one host.

## Operator runbook

After the stack is up, follow **[`RUNBOOK.md`](RUNBOOK.md)** for the six
post-deploy steps: connection wiring + token rotation, WAL audit-sink
configuration, entity-index backfill, embedding-model choices, entity-registry
seeding, and the `topic-coverage` health check.

## Scaling notes

- Bump `shared_buffers` / `effective_cache_size` in `postgresql.conf` for a real
  server (25% / 60% of RAM is a good starting point).
- Keep `MEMPALACE_PG_POOL_MAX` small per client (≈5) so concurrent engineers do
  not exhaust `max_connections` (100). The `mcp` server holds one pool for all
  the sessions it multiplexes.
