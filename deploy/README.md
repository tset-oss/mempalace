# Central MemPalace on PostgreSQL

This directory hosts the **central, team-vaulted** deployment of MemPalace: a
single PostgreSQL 18 instance with `pgvector` (vectors), ParadeDB `pg_search`
(BM25), and Apache AGE (graph). Every engineering team gets its own **vault**
(an isolated Postgres schema, e.g. `team_frontend`). It is built for an internal
engineering network — **there is no authentication layer** beyond Postgres
itself, so do not expose it to the public internet.

## Layout

| File | Purpose |
|------|---------|
| `postgres/Dockerfile` | PG18 + pgvector 0.8.2 + pg_search 0.22.5 + AGE 1.7.0-rc0 |
| `postgres/postgresql.conf` | Tuning; `shared_preload_libraries='pg_search,age'` |
| `postgres/initdb/10-extensions.sql` | First-boot `CREATE EXTENSION` (idempotent) |
| `docker-compose.yml` | The `db` service + volume |
| `.env.example` | Client env: backend, team, connection |

## 1. Start the central database

```bash
cd deploy
docker compose up -d --build      # first build compiles pgvector + AGE (~5-10 min)
docker compose ps                 # wait for healthy
```

Verify all three extensions are available:

```bash
docker compose exec db psql -U mempalace -d mempalace -c \
  "CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_search;
   CREATE EXTENSION IF NOT EXISTS age;
   LOAD 'age';
   SELECT extname, extversion FROM pg_extension ORDER BY extname;"
```

You should see `age`, `pg_search`, and `vector` listed.

## 2. Point a machine at it (per engineer)

Install the Postgres extra:

```bash
pip install "mempalace[postgres]"
```

Then choose **one** of two equivalent ways to configure the backend + primary team:

**a) The `team` command (writes `~/.mempalace/config.json`):**

```bash
mempalace team set frontend --database-url postgresql://mempalace:mempalace@DB_HOST:5432/mempalace
mempalace team            # show current backend / team / db
mempalace team list       # list team vaults on the server
```

This is equivalent to copying `mempalace-config.example.json` to
`~/.mempalace/config.json` and editing the `team` / `database_url` fields.

**b) Environment variables (e.g. in the MCP launcher):**

```bash
export MEMPALACE_BACKEND=postgres
export MEMPALACE_TEAM=frontend       # this machine's primary vault
export MEMPALACE_DATABASE_URL=postgresql://mempalace:mempalace@DB_HOST:5432/mempalace
```

The team you set is the vault MemPalace reads and writes by default, so Claude
Code automatically files and recalls into your team's vault. Team vaults are
created lazily — the first write for `frontend` provisions the `team_frontend`
schema, its drawer/KG tables, and indexes automatically.

To read or write another team's vault for a single call, the MCP tools take a
`vault` parameter (`mempalace_search`, `mempalace_add_drawer`), and
`mempalace_list_vaults` discovers what exists. `vault: "all"` searches every team.

## Scaling notes

- Bump `shared_buffers` / `effective_cache_size` in `postgresql.conf` for a real
  server (25% / 60% of RAM is a good starting point).
- Keep `MEMPALACE_PG_POOL_MAX` small per client (≈5) so concurrent engineers do
  not exhaust `max_connections` (100).
