# MCP Tools Reference

Detailed parameter schemas for the 32 core MCP tools. The central HTTP server
(`mempalace serve`) additionally exposes `mempalace_switch_team` for per-session
team-vault routing (documented below).

## Palace — Read Tools

### `mempalace_status`

Palace overview: total drawers, wing and room counts, AAAK spec, and memory protocol.

**Parameters:** None

**Returns:** `{ total_drawers, wings, rooms, protocol, aaak_dialect }`

---

### `mempalace_list_wings`

List all wings with drawer counts.

**Parameters:** None

**Returns:** `{ wings: { "wing_name": count } }`

---

### `mempalace_list_rooms`

List rooms within a wing (or all rooms if no wing given).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | No | Wing to list rooms for |

**Returns:** `{ wing, rooms: { "room_name": count } }`

---

### `mempalace_get_taxonomy`

Full wing → room → drawer count tree.

**Parameters:** None

**Returns:** `{ taxonomy: { "wing": { "room": count } } }`

---

### `mempalace_search`

Semantic search. Returns verbatim drawer content with similarity scores.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | **Yes** | What to search for |
| `limit` | integer | No | Max results (default: 5) |
| `wing` | string | No | Filter by wing |
| `room` | string | No | Filter by room |
| `vault` | string | No | Team vault to search (central/postgres only): omit for your primary, `<team>` for one team, `all` to sweep every vault. Ignored on local installs. |
| `entity` | string | No | Scope recall to drawers mentioning this entity before ranking (central/postgres only; not supported with `vault: "all"`). See `mempalace_entities`. |

**Returns:** `{ query, filters, results: [{ text, wing, room, source_file, similarity }] }`

---

### `mempalace_check_duplicate`

Check if content already exists in the palace before filing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | **Yes** | Content to check |
| `threshold` | number | No | Similarity threshold 0–1 (default: 0.85–0.87) |

**Returns:** `{ is_duplicate, matches: [{ id, wing, room, similarity, content }] }`

---

### `mempalace_get_aaak_spec`

Returns the AAAK dialect specification.

**Parameters:** None

**Returns:** `{ aaak_spec: "..." }`

---

## Palace — Write Tools

### `mempalace_add_drawer`

File verbatim content into the palace. Identical content (same deterministic drawer ID) is silently skipped. For similarity-based duplicate detection before filing, use `mempalace_check_duplicate`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | **Yes** | Wing (project name) |
| `room` | string | **Yes** | Room (aspect: backend, decisions, etc.) |
| `content` | string | **Yes** | Verbatim content to store |
| `source_file` | string | No | Where this came from |
| `added_by` | string | No | Who is filing (default: "mcp") |

**Returns:** `{ success, drawer_id, wing, room }`

---

### `mempalace_delete_drawer`

Delete a drawer by ID. Irreversible.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `drawer_id` | string | **Yes** | ID of the drawer to delete |

**Returns:** `{ success, drawer_id }`

---

### `mempalace_sync`

Prune drawers whose source files are gitignored, deleted, or moved. Returns a dry-run report by default; pass `apply=true` to commit deletions.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_dir` | string | No | Project root to scope the sync (auto-detected from drawer metadata if omitted) |
| `wing` | string | No | Limit to one wing |
| `apply` | boolean | No | Actually delete drawers; default is dry-run preview |

**Returns:** `{ scanned, kept, gitignored, missing, no_source, out_of_scope, removed_drawers, removed_closets, dry_run, by_source }`

---

### `mempalace_get_drawer`

Fetch a single drawer by ID — returns full content and metadata.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `drawer_id` | string | **Yes** | ID of the drawer to fetch |

**Returns:** `{ drawer_id, content, wing, room, metadata }` where `metadata.source_file`, when present, is the basename only — the absolute path written by the miners is reduced before the dict is returned to MCP clients.

---

### `mempalace_list_drawers`

List drawers with pagination. Optional wing/room filter. Returns IDs, wings, rooms, and content previews.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | No | Filter by wing |
| `room` | string | No | Filter by room |
| `limit` | integer | No | Max results per page (default 20, max 100) |
| `offset` | integer | No | Offset for pagination (default 0) |

**Returns:** `{ drawers: [...], total, limit, offset }`

---

### `mempalace_update_drawer`

Update an existing drawer's content and/or metadata (wing, room). Fetches the existing drawer first; returns an error if not found.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `drawer_id` | string | **Yes** | ID of the drawer to update |
| `content` | string | No | New content (omit to keep existing) |
| `wing` | string | No | New wing (omit to keep existing) |
| `room` | string | No | New room (omit to keep existing) |

**Returns:** `{ success, drawer_id, updated_fields }`

---

## Knowledge Graph Tools

### `mempalace_kg_query`

Query entity relationships with time filtering.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity` | string | **Yes** | Entity to query (e.g. "Max", "MyProject") |
| `as_of` | string | No | Date filter — only facts valid at this date (YYYY-MM-DD) |
| `direction` | string | No | `outgoing`, `incoming`, or `both` (default: `both`) |

**Returns:** `{ entity, as_of, facts: [{ direction, subject, predicate, object, valid_from, valid_to, current }], count }`

---

### `mempalace_kg_neighbors`

Walk the knowledge graph multiple hops out from an entity to surface indirect connections a single-hop query misses.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity` | string | **Yes** | Entity to start the walk from (e.g. "Max", "MyProject") |
| `depth` | integer | No | How many hops to follow out (default: 2, clamped to 1–4) |
| `direction` | string | No | `outgoing`, `incoming`, or `both` (default: `outgoing`) |
| `as_of` | string | No | Date filter — only facts valid at this point in time, applied to every hop |
| `target` | string | No | Keep only paths that reach this entity |
| `predicates` | array | No | Restrict every hop to these relationship types |

**Returns:** `{ entity, depth, direction, as_of, neighbors: [{ hop, direction, subject, predicate, object, valid_from, valid_to }], count, truncated }`

Postgres backend only; the local SQLite knowledge graph returns a structured "unsupported on local backend" result.

---

### `mempalace_kg_add`

Add a fact to the knowledge graph.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | **Yes** | The entity doing/being something |
| `predicate` | string | **Yes** | Relationship type (e.g. "loves", "works_on") |
| `object` | string | **Yes** | The entity being connected to |
| `valid_from` | string | No | When this became true (YYYY-MM-DD) |
| `source_closet` | string | No | Closet ID where this fact appears |

**Returns:** `{ success, triple_id, fact }`

---

### `mempalace_kg_invalidate`

Mark a fact as no longer true.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | **Yes** | Entity |
| `predicate` | string | **Yes** | Relationship |
| `object` | string | **Yes** | Connected entity |
| `ended` | string | No | When it stopped being true (default: today) |

**Returns:** `{ success, fact, ended }`

---

### `mempalace_kg_timeline`

Chronological timeline of facts.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity` | string | No | Entity to get timeline for (omit for full timeline) |

**Returns:** `{ entity, timeline: [{ subject, predicate, object, valid_from, valid_to, current }], count }`

---

### `mempalace_kg_stats`

Knowledge graph overview.

**Parameters:** None

**Returns:** `{ entities, triples, current_facts, expired_facts, relationship_types }`

---

## Navigation Tools

### `mempalace_traverse`

Walk the palace graph from a room. Find connected ideas across wings.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_room` | string | **Yes** | Room to start from |
| `max_hops` | integer | No | How many connections to follow (default: 2) |

**Returns:** `[{ room, wings, halls, count, hop, connected_via }]`

---

### `mempalace_find_tunnels`

Find rooms that bridge two wings.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing_a` | string | No | First wing |
| `wing_b` | string | No | Second wing |

**Returns:** `[{ room, wings, halls, count, recent }]`

---

### `mempalace_graph_stats`

Palace graph overview: nodes, tunnels, edges, connectivity.

**Parameters:** None

**Returns:** `{ total_rooms, tunnel_rooms, total_edges, rooms_per_wing, top_tunnels }`

---

### `mempalace_create_tunnel`

Create a cross-wing tunnel linking two palace locations. Use when content in one project relates to another — e.g., an API design in `project_api` connects to a database schema in `project_database`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_wing` | string | **Yes** | Wing of the source |
| `source_room` | string | **Yes** | Room in the source wing |
| `target_wing` | string | **Yes** | Wing of the target |
| `target_room` | string | **Yes** | Room in the target wing |
| `label` | string | No | Description of the connection |
| `source_drawer_id` | string | No | Specific source drawer ID |
| `target_drawer_id` | string | No | Specific target drawer ID |

**Returns:** `{ success, tunnel_id, source, target }`

---

### `mempalace_list_tunnels`

List all explicit cross-wing tunnels. Optionally filter by wing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | No | Filter tunnels by wing (source or target) |

**Returns:** `{ tunnels: [...], count }`

---

### `mempalace_delete_tunnel`

Delete an explicit tunnel by its ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tunnel_id` | string | **Yes** | Tunnel ID to delete |

**Returns:** `{ success, tunnel_id }`

---

### `mempalace_follow_tunnels`

Follow tunnels from a room to see what it connects to in other wings. Returns connected rooms with drawer previews.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | **Yes** | Wing to start from |
| `room` | string | **Yes** | Room to follow tunnels from |

**Returns:** `[{ wing, room, label, previews }]`

---

## Agent Diary Tools

### `mempalace_diary_write`

Write to your personal agent diary.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_name` | string | **Yes** | Your name — each agent gets its own wing |
| `entry` | string | **Yes** | Diary entry (in AAAK format recommended) |
| `topic` | string | No | Topic tag (default: "general") |

**Returns:** `{ success, entry_id, agent, topic, timestamp }`

---

### `mempalace_diary_read`

Read recent diary entries.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_name` | string | **Yes** | Your name |
| `last_n` | integer | No | Number of recent entries (default: 10) |

**Returns:** `{ agent, entries: [{ date, timestamp, topic, content }], total, showing }`

---

## System Tools

### `mempalace_hook_settings`

Get or set auto-save hook behaviour. `silent_save=true` saves directly without MCP-level clutter; `silent_save=false` uses the legacy blocking path. `desktop_toast=true` surfaces a desktop notification when a save completes. Call with no arguments to view the current settings.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `silent_save` | boolean | No | `true` = silent direct save, `false` = blocking MCP calls |
| `desktop_toast` | boolean | No | `true` = show desktop toast via `notify-send` |

**Returns:** `{ silent_save, desktop_toast }`

---

### `mempalace_memories_filed_away`

Check whether a recent palace checkpoint was saved. Returns message count and timestamp of the last save.

**Parameters:** None

**Returns:** `{ filed, message_count, timestamp }`

---

### `mempalace_reconnect`

Force a reconnect to the palace database. Use this after external scripts or CLI commands modified the palace directly, which can leave the in-memory HNSW index stale.

**Parameters:** None

**Returns:** `{ success, message, drawers, vector_disabled[, vector_disabled_reason] }` (on no-palace: `{ success: false, message, drawers, vector_disabled }`; on exception: `{ success: false, error }`)

---

### `mempalace_list_vaults`

List the team vaults available on a centrally-hosted MemPalace (the `postgres` backend) and report which one is this machine's primary vault (from local config / `MEMPALACE_TEAM`). On a local (`chroma`) install there is a single implicit `local` vault. Call once per session before routing memories to a specific team.

Team-vault routing is also exposed through a `vault` parameter on `mempalace_search` and `mempalace_add_drawer`:

- omit `vault` → this machine's primary team vault (the default)
- `vault: "<team>"` → a specific team's vault (e.g. `frontend`, `backend`)
- `vault: "all"` (search only) → search every team vault, returned per-vault

**Parameters:** None

**Returns:** `{ backend, mode, primary, vaults[, hint] }`

On the central HTTP server the reported `primary` is the session's active vault
(the `X-Mempalace-Team` header, or whatever `mempalace_switch_team` last set),
not just local config.

---

### `mempalace_entities`

*Central HTTP server only (`postgres` backend).* Navigate the team vault's
entity index — the inverted index of which entities (people, projects,
services) appear in which drawers, maintained incrementally as drawers are
filed and seeded by `mempalace_kg_add`.

- with `entity` → the drawers that mention it (ids + wing/room) and a count.
  Pair with `mempalace_search(entity=...)` to retrieve the verbatim content.
- without `entity` → the vault's most-mentioned entities, optionally scoped to a
  `wing`. `min_count` (default 2) filters one-off extraction noise.

Use it to answer "what/who do we know about X" and to find the exact entity name
to scope a search by. On a local (`chroma`) install the entity index does not
exist (use `mempalace mine` + search there); the tool reports `available: false`.

**Parameters:**

- `entity` (string, optional) — entity to look up. Omit to list top entities.
- `wing` (string, optional) — scope the listing to one wing.
- `min_count` (integer, optional, default 2) — listing only: minimum drawers an
  entity must appear in. Pass 1 to see everything.
- `limit` (integer, optional, default 50) — max entities in the listing.
- `vault` (string, optional) — team vault to inspect (central deployments).

**Returns:** with `entity` → `{ backend, vault, entity, drawer_count, drawers[], hint }`;
without → `{ backend, vault, wing, entities[], count }`. On `chroma` →
`{ available: false, backend, reason }`.

The `entity` parameter on `mempalace_search` uses this same index to scope a
semantic query to one entity's drawers — verbatim recall narrowed to "what we
know about X".

---

### `mempalace_team_fact_add`

*Central HTTP server only (`postgres` backend).* Record a critical fact every
agent on this team should know — the small set of must-know facts the whole team
shares, e.g. "the prod DB is read-replica only" or "release freeze until Q3".
This is for team-wide facts, not per-conversation memories (file those with
`mempalace_add_drawer`).

The fact is stored in this team's vault (`team_<slug>.critical_facts`) and is
invisible to other teams — a fact added in one team is never visible to another.
This is DISTINCT from the personal local identity in `~/.mempalace/identity.txt`
(the L0 "who am I" layer): that file is host-local and per-developer and is never
vaulted; team critical-facts are the shared, central team layer.

Conservative defaults: a fact is capped at 2000 characters and a team holds at
most 200 facts; over either cap the call returns a structured error.

**Parameters:**

- `fact` (string, required) — the critical fact to share (a short must-know line,
  max 2000 chars).
- `created_by` (string, optional) — author/agent label recorded with the fact.

**Returns:** `{ backend, vault, added: { id, fact, created_at, created_by } }`.
On `chroma` → `{ available: false, backend, reason }`. On the central server a
call with no resolvable team raises a structured error (it never falls back to a
default vault).

---

### `mempalace_team_facts`

*Central HTTP server only (`postgres` backend).* List this team's critical facts
— the must-know facts every agent on the team shares, recorded via
`mempalace_team_fact_add`. Returned oldest first and isolated from other teams.

On a local (`chroma`) install there is no team concept, so the tool reports
`available: false`; the personal identity lives in `~/.mempalace/identity.txt`
instead.

**Parameters:** None

**Returns:** `{ backend, vault, facts[], count }` where each fact is
`{ id, fact, created_at, created_by }`. On `chroma` →
`{ available: false, backend, reason }`. On the central server a call with no
resolvable team raises a structured error (it never falls back to a default
vault).

---

### `mempalace_switch_team`

*Central HTTP server only (`mempalace serve`).* Set the active team vault for
the current session — all subsequent reads and writes route to it until you
switch again. Mirrors the primary shown by `mempalace_list_vaults`. Call with no
team (or `"default"`) to reset to the configured default (the `X-Mempalace-Team`
request header / the server default).

The session's default vault is normally seeded by the `X-Mempalace-Team` header
set in the MCP client config (user scope = a machine's default, project scope =
a repo's default); `switch_team` is the runtime override. To read across every
vault use `vault: "all"` on `mempalace_search`; to target a single call use that
call's `vault` parameter.

**Parameters:**

- `team` (string, optional) — the team vault to activate, e.g. `frontend`.
  Lowercase `[a-z0-9_]`, 1–40 chars, no leading/trailing underscore. Omitted or
  `"default"`/`"primary"` resets to the configured default.

**Returns:** `{ ok, active_team[, note] }` (on a rejected name: `{ ok: false, error }`)
