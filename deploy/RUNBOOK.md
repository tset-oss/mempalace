# MemPalace Central Server — Operator Runbook

This checklist covers what you need to do **after** the stack is up (see
`README.md` for `docker compose up -d --build`, extension verification, and the
auth smoke-test). Work through sections 1–6 in order for a first deploy; use
individual sections as reference for later maintenance.

---

## 1. Connection and auth wiring

### What to set on the client side

Agents connect via two values: the server URL and the shared bearer token.

**The committed plugin entry** (`agent-commons/.claude-plugin/plugin.json`,
key `mcpServers.mempalace.url`) currently holds the placeholder
`https://mempalace.internal/mcp`. Replace it with the actual endpoint once the
server is reachable:

```json
"mcpServers": {
  "mempalace": {
    "type": "http",
    "url": "https://your-real-host/mcp",
    "headers": {
      "Authorization": "Bearer <shared-token>"
    }
  }
}
```

Per-repo overrides go in a committed `.mcp.json` at the repo root (same
structure; add `"X-Mempalace-Team": "<team>"` to default to that vault).
Per-engineer machine-wide defaults go in user scope:

```bash
claude mcp add --transport http --scope user mempalace https://your-real-host/mcp \
  --header "Authorization: Bearer <shared-token>" \
  --header "X-Mempalace-Team: <your-team>"
```

### Token rotation

The token in `plugin.json` and the server's `MEMPALACE_AUTH_TOKEN` are a
**matched pair**. The token is an internal/VPN guard, not real auth — it is
committed intentionally so every engineer connects without per-person secret
handling.

Rotate both together:

1. Generate a new token: `openssl rand -hex 32`
2. Set `MEMPALACE_AUTH_TOKEN=<new-token>` in `deploy/.env` and restart the
   `mcp` service: `docker compose up -d mcp`
3. Update the `Authorization: Bearer` value in `plugin.json` (and any
   per-repo `.mcp.json` files) and push the change.

Engineers pick up the new token when they pull `agent-commons`.

Confirm auth is working:

```bash
# Unauthenticated → 401 (expected)
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://your-real-host/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

# Authenticated → 200
curl -s -X POST https://your-real-host/mcp \
  -H "Authorization: Bearer $MEMPALACE_AUTH_TOKEN" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}'
```

---

## 2. WAL audit sink

The write-ahead audit log records every write operation. Two sinks are
supported: `jsonl` (a host-global file) and `postgres` (a team-tagged table
in the central database).

The `mcp` service in `docker-compose.yml` sets:

```
MEMPALACE_WAL_SINK=postgres
```

This is already the right value for a central deploy. Do **not** change it to
`jsonl` — the jsonl path is a host-global file shared by all teams and loses
the team-scoping that makes the audit useful.

If you run the server outside Docker, export the variable before starting:

```bash
export MEMPALACE_WAL_SINK=postgres
mempalace serve --transport streamable-http
```

`config.py` accepts `postgres` or `jsonl`; any other value falls back to the
backend-aware default (`postgres` when `MEMPALACE_BACKEND=postgres`).

---

## 3. Entity-index backfill

The entity index (the table backing `mempalace_entities` and entity-scoped
search) is built **incrementally as memories are filed**. New writes index
automatically.

**First deploy with an empty database:** no backfill is needed. The index
starts empty and populates as agents write memories.

**After upgrading a server that already holds memories** (you added the entity
index to an existing vault), run the backfill once per vault:

```bash
# Backfill the entity index for one team vault
mempalace reindex-entities --vault frontend

# Or backfill every vault at once
mempalace reindex-entities --all-vaults
```

The command requires `MEMPALACE_BACKEND=postgres` (it exits with an error on
the local chroma backend). It is idempotent — safe to re-run.

---

## 4. Embeddings

The `mcp` service starts with:

```
MEMPALACE_EMBEDDING_MODEL=embeddinggemma
```

`embeddinggemma` is the multilingual model (100+ languages, 384-dim vectors),
pre-baked into the server image so no model is downloaded at request time.

The alternative value is `minilm` (English-only, ChromaDB's
all-MiniLM-L6-v2). The docker-compose file ships `embeddinggemma` as the
default for the central deploy; do not change it unless you have a specific
reason to use English-only embeddings.

**Important:** set `MEMPALACE_EMBEDDING_MODEL` **before the first ingest**.
If you change the model after memories have been filed, the existing vectors
live in a different embedding space and semantic search degrades. There is no
in-place re-embed: the drawers must be re-embedded from their source text under
the new model, which in practice means rebuilding the vault's contents. (Note:
`mempalace repair` is a local-chroma tool that rebuilds a palace's vector index
from its already-stored embeddings — it does not re-run the model on text, so it
does **not** fix a model change.) Treat the embedding model as fixed at deploy
time.

---

## 5. Entity registry seeding (FU1)

The per-team entity registry stores known people, projects, and aliases so the
server can resolve ambiguous names (e.g. "Dana" → Dana Okonkwo, senior
engineer). It is distinct from the entity **index** (section 3): the index is
built from what agents write; the registry is the disambiguation table agents
read.

**The server never seeds this automatically.** The registry for a new team
vault starts empty. It is populated by agents calling `mempalace_entity_seed`.

### Expected behaviour on a fresh vault

- `mempalace_disambiguate` returns an empty result until at least one
  `mempalace_entity_seed` call has been made for the team.
- `mempalace_entities` lists whatever has been indexed from filed memories (the
  incremental index from section 3); this is separate from the disambiguation
  registry.

### How seeding works

Agents call `mempalace_entity_seed` with the team's known people, projects, and
aliases. The tool is **read-merge-write**: it reads the existing registry,
merges in the new entries, and writes the result back, so existing data is
never clobbered. Re-seeding is safe and idempotent.

### Concurrency caveat

If two agents call `mempalace_entity_seed` for the **same team** at the same
moment, the result is last-writer-wins: one write may overwrite the other's
merge before it completes. In practice this is rare — seeding is a one-time or
infrequent operation — and because `mempalace_entity_seed` is idempotent, the
next seed call will restore any dropped entries.

---

## 6. Health check — topic coverage

Topic labels power cross-wing tunnels: when two wings share confirmed topic
labels, the server creates a tunnel between them so agents can navigate between
related projects without an explicit search.

The `mempalace topic-coverage` command is the operator's way to confirm agents
are filing topic labels, and that tunnels can therefore form:

```bash
# Check one vault
mempalace topic-coverage --vault frontend

# Check every vault
mempalace topic-coverage --all-vaults
```

Output:

```
  frontend: 142 wing_topics rows, 8 distinct topics, 11 wings with topics
```

- **0 rows** means no topic labels have been filed yet (agents have not used
  topic-aware tools, or the vault is new).
- **Growing rows over time** means agents are actively filing labels and tunnels
  will form between overlapping wings.

Requires `MEMPALACE_BACKEND=postgres` (exits with an informational message on
the local chroma backend).

---

## Quick-reference: env vars

| Variable | Required | Value in docker-compose | Notes |
|---|---|---|---|
| `MEMPALACE_AUTH_TOKEN` | Yes | set in `deploy/.env` | Shared bearer token; stack fails to start if unset |
| `MEMPALACE_BACKEND` | Yes (server) | `postgres` | Selects the central backend |
| `MEMPALACE_WAL_SINK` | Recommended | `postgres` | Audit log sink; use `postgres` on the central server |
| `MEMPALACE_EMBEDDING_MODEL` | Recommended | `embeddinggemma` | Set before first ingest; changing later needs re-embedding from source (no in-place path) — treat as fixed at deploy |
| `MEMPALACE_DATABASE_URL` | Yes (server) | set in compose | Postgres DSN; or use `MEMPALACE_PG_*` vars |
| `MEMPALACE_TEAM` | Per-machine | not set on server | Client-side default vault |

## Quick-reference: CLI commands

| Command | What it does | When to run |
|---|---|---|
| `mempalace serve --transport streamable-http` | Start the HTTP MCP server | On deploy (docker-compose handles this) |
| `mempalace reindex-entities --all-vaults` | Backfill entity index from existing drawers | Once after upgrading an existing server; skip on first deploy |
| `mempalace reindex-entities --vault <name>` | Backfill one vault | After adding a new vault to a running server with existing data |
| `mempalace topic-coverage --all-vaults` | Report topic-label coverage per vault | Periodic health check |
| `mempalace topic-coverage --vault <name>` | Report coverage for one vault | Per-team health check |
| `mempalace team list` | List vaults on the central server | Confirm vaults exist after first writes |
