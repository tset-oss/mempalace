# CLI Commands

All commands accept `--palace <path>` to override the default palace location.

## `mempalace init`

Scan a project directory for people, projects, and rooms, and set up the palace.

```bash
mempalace init <dir>                 # <dir> is required
mempalace init <dir> --yes           # non-interactive mode
mempalace init ~/projects/myapp      # example
mempalace init .                     # initialize from the current directory
```

| Option  | Description                                                                  |
|---------|------------------------------------------------------------------------------|
| `<dir>` | **Required.** Project directory to scan. Pass `.` for the current directory. |
| `--yes` | Auto-accept all detected entities                                            |

What it does:

1. Scans `<dir>` for people and projects in file content
2. Detects rooms from `<dir>`'s folder structure
3. Saves detected entities to `<dir>/entities.json`
4. Ensures the global `~/.mempalace/` config directory exists

Running `mempalace init` with no argument will exit with
`error: the following arguments are required: dir`.

## `mempalace mine`

Mine files into the palace.

```bash
mempalace mine <dir>
mempalace mine <dir> --mode convos
mempalace mine <dir> --mode convos --extract general
mempalace mine <dir> --wing myapp
```

| Option | Default | Description |
|--------|---------|-------------|
| `<dir>` | — | Directory to mine |
| `--mode` | `projects` | `projects` for code/docs, `convos` for chat exports |
| `--wing` | directory name | Wing name override |
| `--agent` | `mempalace` | Agent name tag |
| `--limit` | `0` (all) | Max files to process |
| `--dry-run` | — | Preview without filing |
| `--extract` | `exchange` | `exchange` or `general` (for convos mode) |
| `--no-gitignore` | — | Don't respect .gitignore |
| `--include-ignored` | — | Always scan these paths even if ignored |

## `mempalace search`

Find anything by semantic search.

```bash
mempalace search "query"
mempalace search "query" --wing myapp
mempalace search "query" --wing myapp --room auth
mempalace search "query" --results 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `"query"` | — | What to search for |
| `--wing` | all | Filter by wing |
| `--room` | all | Filter by room |
| `--results` | `5` | Number of results |

## `mempalace split`

Split concatenated transcript mega-files into per-session files.

```bash
mempalace split <dir>
mempalace split <dir> --dry-run
mempalace split <dir> --min-sessions 3
mempalace split <dir> --output-dir ~/split-output/
```

| Option | Default | Description |
|--------|---------|-------------|
| `<dir>` | — | Directory with transcript files |
| `--output-dir` | same dir | Write split files here |
| `--dry-run` | — | Preview without writing |
| `--min-sessions` | `2` | Only split files with N+ sessions |

## `mempalace wake-up`

Show L0 + L1 wake-up context (~600–900 tokens).

```bash
mempalace wake-up
mempalace wake-up --wing driftwood
```

| Option | Description |
|--------|-------------|
| `--wing` | Project-specific wake-up |

## `mempalace compress`

Compress drawers using AAAK Dialect.

```bash
mempalace compress --wing myapp
mempalace compress --wing myapp --dry-run
mempalace compress --config entities.json
```

| Option | Description |
|--------|-------------|
| `--wing` | Wing to compress (default: all) |
| `--dry-run` | Preview without storing |
| `--config` | Entity config JSON file |

## `mempalace status`

Show what's been filed — drawer count, wing/room breakdown.

```bash
mempalace status
```

## `mempalace repair`

Rebuild palace vector index from stored data. Fixes segfaults after database corruption.

```bash
mempalace repair
```

Creates a backup at `<palace_path>.backup` before rebuilding.

## `mempalace mcp`

Helper command that outputs setup syntax (like `claude mcp add...`) to connect MemPalace to your AI client, automatically handling paths.

```bash
mempalace mcp
mempalace mcp --palace ~/.custom-palace
```

## `mempalace serve`

Run the MCP server. Defaults to **stdio** (a local install); `--transport streamable-http` runs the **central HTTP server** at `/mcp` that engineers' Claude Code connects to.

```bash
mempalace serve                                   # stdio (local)
mempalace serve --transport streamable-http \     # central server
  --host 0.0.0.0 --port 8080 \
  --default-team default --auth-token "$MEMPALACE_AUTH_TOKEN"
```

| Option | Default | Description |
|--------|---------|-------------|
| `--transport` | `stdio` | `stdio` (local IDE) or `streamable-http` (central HTTP server) |
| `--host` | `127.0.0.1` | Bind address for streamable-http; use `0.0.0.0` to expose |
| `--port` | `8080` | Port for streamable-http |
| `--default-team` | `default` | Process-global default vault (sets `MEMPALACE_TEAM`); the bottom of the routing chain below the `X-Mempalace-Team` header |
| `--auth-token` | _(none)_ | Shared static bearer token (or `MEMPALACE_AUTH_TOKEN`); when set, HTTP requests must send `Authorization: Bearer <token>` |

Team routing is per session: a client's `X-Mempalace-Team` header seeds its default vault, `mempalace_switch_team` overrides it at runtime, and a tool's `vault` parameter overrides one call. See [deploy/README.md](https://github.com/MemPalace/mempalace/blob/main/deploy/README.md) for the full central deployment.

## `mempalace team`

Show or set this machine's primary team vault for central (postgres) mode.

```bash
mempalace team                       # show backend / primary team / db
mempalace team set frontend \        # point this machine at a team vault
  --database-url postgresql://mempalace:mempalace@DB_HOST:5432/mempalace
mempalace team list                  # list team vaults on the central server
```

`set` writes `~/.mempalace/config.json` (`backend=postgres`, `team`, optional `database_url`). Team names are lowercase `[a-z0-9_]`. This is the per-machine default the local fork uses; the central HTTP server instead takes the team from each client's `X-Mempalace-Team` header.

## `mempalace reindex-entities`

Backfill the per-vault entity index from drawers already filed (central postgres mode only). New writes index entities incrementally; this one-time operator command populates the index for drawers that predate the feature. Idempotent — safe to re-run.

```bash
mempalace reindex-entities                  # this machine's configured team
mempalace reindex-entities --vault backend  # a specific team vault
mempalace reindex-entities --all-vaults     # every vault on the server
```

Run it once per vault after upgrading a server that already holds memories, e.g. inside the deploy container: `docker compose exec mcp mempalace reindex-entities --all-vaults`. The index then powers `mempalace_entities` and the `entity=` filter on `mempalace_search` (see [MCP tools](mcp-tools.md)).

## `mempalace hook`

Run hook logic for Claude Code / Codex integration.

```bash
mempalace hook run --hook stop --harness claude-code
mempalace hook run --hook precompact --harness claude-code
mempalace hook run --hook session-start --harness codex
```

| Option | Values | Description |
|--------|--------|-------------|
| `--hook` | `session-start`, `stop`, `precompact` | Hook name |
| `--harness` | `claude-code`, `codex` | Harness type |

## `mempalace instructions`

Output skill instructions to stdout.

```bash
mempalace instructions init
mempalace instructions search
mempalace instructions mine
mempalace instructions help
mempalace instructions status
```
