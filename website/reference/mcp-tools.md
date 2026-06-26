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

Advisory near-duplicate pre-check before filing — a best-effort similarity query, **not** a guard. Do not treat `is_duplicate: false` as proof that filing is safe; the authoritative protection against exact re-files is `mempalace_add_drawer`'s content-hash idempotency (identical wing/room/content returns `reason: "already_exists"` and creates no second row).

On the central (`postgres`) backend, a team vault that has had no writes yet returns `{ is_duplicate: false, empty_vault: true, vault, reason }` — there is simply nothing to compare against — rather than a "no palace" error.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | **Yes** | Content to check |
| `threshold` | number | No | Similarity threshold 0–1 (default: 0.85–0.87) |

**Returns:** `{ is_duplicate, matches: [{ id, wing, room, similarity, content }] }`. Situational keys may also appear: `empty_vault` / `vault` / `reason` (empty central vault) or `vector_disabled` (no usable vector index). Always read `is_duplicate` and `matches`; treat the rest as optional.

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
| `topics` | array of string | No | Topic labels for this drawer's wing (e.g. `["Angular", "OpenAPI"]`). Wings sharing a label are auto-linked by a cross-wing topic tunnel (central/postgres only). On the central/postgres backend a topic also makes this drawer findable via `mempalace_search(entity=...)` / `mempalace_entities(entity=...)` — even when the label never appears in the content — as a recall-only tag (it does not affect entity hallways/tunnels). Ignored on local installs. |

**Returns:** `{ success, drawer_id, wing, room, chunks }`. Content larger than the chunk size is split into physical `{drawer_id}_chunk_NNNNNN` rows; in that case the response also carries `chunk_ids` and the returned `drawer_id` is the LOGICAL group handle (pass it straight to `mempalace_get_drawer`, which reassembles the whole memory).

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

Fetch a single drawer by ID — returns full content and metadata. Accepts either a physical drawer/chunk ID **or** the LOGICAL group handle of a chunked drawer (the `drawer_id` that `mempalace_add_drawer` returns for oversized content): when no row exists under the ID, the chunks carrying that `parent_drawer_id` are reassembled, in order, into the whole verbatim memory. Reassembly is byte-exact and refuses (with an error, never silently) if a chunk is missing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `drawer_id` | string | **Yes** | Physical drawer/chunk ID, or the logical handle of a chunked drawer |

**Returns:** `{ drawer_id, content, wing, room, metadata }`. When the ID resolved via chunk reassembly, the response also carries `chunks` and `chunk_ids`. `metadata.source_file`, when present, is the basename only — the absolute path written by the miners is reduced before the dict is returned to MCP clients.

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

### `mempalace_mine`

Mine a directory into the palace — the MCP equivalent of `mempalace mine`. Runs synchronously and returns the miner's summary as `output`. The palace write lock is automatic; a concurrent mine returns a structured already-running error. Orphan cleanup is separate — use `mempalace_sync`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | **Yes** | Directory to mine |
| `mode` | string | No | Ingest mode: `projects` (code/docs, default), `convos` (chat transcripts), `extract` (office documents — PDF/DOCX/RTF, requires the `mempalace[extract]` extra) |
| `wing` | string | No | Target wing (default: source directory name) |
| `agent` | string | No | Recorded on every drawer (default: `mempalace`) |
| `limit` | integer | No | Max files to process (`0` = all). Default: `0` |
| `dry_run` | boolean | No | Report what would be filed without writing. Default: `false` |
| `extract` | string | No | Convos extraction strategy: `exchange` (default) or `general`. Ignored by other modes |

**Returns:** `{ success, mode, dry_run, output }`. A very large summary is tail-truncated to ~4000 chars with `output_truncated: true`. On failure: `{ success: false, error, error_class }`.

---

### `mempalace_delete_by_source`

Bulk-delete every drawer mined from one `source_file` (exact match). Use to clean up benchmark/test data accidentally mined into a user wing. Returns a dry-run match count and sample by default; pass `dry_run=false` to commit. Irreversible.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_file` | string | **Yes** | Exact `source_file` metadata value to remove (e.g. the full path that was mined) |
| `dry_run` | boolean | No | Preview the match count without deleting; default `true`. Pass `false` to actually delete |

**Returns (dry-run):** `{ success, dry_run: true, source_file, match_count, closet_match_count, sample, hint }`. **Returns (applied):** `{ success, dry_run: false, source_file, deleted, closets_deleted }`.

---

### `mempalace_checkpoint`

Save a whole session in one call: semantic-dedups each item, files the non-duplicates as drawers, then writes one diary entry. Use this instead of many separate `check_duplicate` / `add_drawer` / `diary_write` calls — it renders as a single tool-call card in the host UI.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `items` | array of object | **Yes** | Verbatim items to file. Each is `{ wing, room, content }` — content is the exact words, never summarized |
| `diary` | object | No | Optional diary entry written after filing: `{ agent_name, entry, topic?, wing? }`. `entry` is AAAK-format |
| `dedup_threshold` | number | No | Similarity threshold 0–1 for the per-item dedup check (default: `0.9`) |

**Returns:** `{ added: [...], duplicates: [...], errors: [...], diary? }`.

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

### `mempalace_list_hallways`

List within-wing hallway records (entity-to-entity co-occurrence links built at mine time). Optionally filter by wing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wing` | string | No | Filter hallways by wing |

**Returns:** `[{ hallway_id, wing, ... }]` — the matching hallway records.

---

### `mempalace_delete_hallway`

Delete a hallway record by its ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hallway_id` | string | **Yes** | Hallway ID to delete |

**Returns:** `{ deleted }` — `true` when a record was removed, `false` when the ID was not found.

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

### `mempalace_disambiguate`

*Central HTTP server only (`postgres` backend).* Resolve a surface form against this team's entity name-resolution registry — the lane that answers "is 'Max' the person Maxwell?" Resolution consults only the seeded registry of known people, projects, and aliases. It is local and offline: it never performs a network or Wikipedia lookup. Distinct from the knowledge graph (facts and relationships) and from team critical-facts (must-know lines).

Returns `found: false` (not an error) for an unseeded name. Returns an `ambiguous` flag when the name is marked as ambiguous in the registry. On a local (`chroma`) install there is no team vault, so the tool reports `available: false`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | **Yes** | The name or word to resolve to a canonical person or project |
| `context` | string | No | Surrounding sentence used to disambiguate a name that is also a common word |

**Returns (found):** `{ backend, vault, found: true, ambiguous, name, type, ... }` where `name` is the canonical form and `type` is `"person"` or `"project"`.

**Returns (not found):** `{ backend, vault, found: false, name, type: "unknown", needs_disambiguation: false }`

**Returns (local install):** `{ available: false, backend: "chroma", reason }`

---

### `mempalace_entity_seed`

*Central HTTP server only (`postgres` backend).* Populate this team's entity name-resolution registry with known people, projects, and aliases — the registry that `mempalace_disambiguate` reads. Writes name-resolution data **only**: it does not write knowledge-graph triples (`mempalace_kg_add`) or team critical-facts (`mempalace_team_fact_add`).

Semantics are **read-merge-write (additive, non-clobbering)**: existing projects, per-person contexts, and aliases are preserved and new data is unioned in, so re-seeding is safe and idempotent — it never drops prior entries. RAISES with no resolvable team (a team-less write must not leak into a shared vault). On a local (`chroma`) install the tool reports `available: false`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | Registry mode label (default `"personal"`). Set only on first seed; ignored on subsequent re-seeds of an existing registry |
| `people` | array of object | No | People to register: list of `{ name, relationship, context, aliases }` dicts. The optional per-person `aliases` is a list of alias strings for that person, e.g. `{ "name": "Markus Burger", "aliases": ["MB"] }` — each is registered exactly as if it appeared in the top-level `aliases` map pointing at that person |
| `projects` | array of string | No | Project names to register. A `{ "name": "..." }` dict is also accepted and coerced to its name |
| `aliases` | object | No | Alias map in the single fixed direction `{ alias: canonical }` — the key is the alias, the value is the canonical name it resolves to, e.g. `{ "MB": "Markus Burger" }` means `MB` → `Markus Burger`. There is no `{ canonical: [aliases] }` form |

Every alias (key, value, or embedded per-person entry) and every project entry must be a **non-empty string**; a malformed shape RAISES rather than being silently dropped.

**Returns:** `{ backend, vault, people, projects }` where `people` and `projects` are the post-merge counts.

**Returns (local install):** `{ available: false, backend: "chroma", reason }`

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
