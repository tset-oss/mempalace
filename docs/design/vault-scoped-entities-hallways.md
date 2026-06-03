# Design: vault-scoped, Postgres-backed, MCP-consumable entity index

Status: accepted (ultragoal G002, architect + critic gated). Scope: the central
Postgres backend only; the ChromaDB local backend keeps its current
miner-driven behaviour unchanged.

## What this is (and what it deliberately is not)

The one genuinely novel, on-mission capability the central server lacks is
**entity-scoped recall of verbatim content**: "what do we know about Dana / the
billing migration?" answered as the exact stored drawers, scoped to an entity.
That is the product's reason to exist (verbatim recall) and is the one thing
`kg_query` does *not* do (it returns relationships, not the stored text).

So this design ships a **per-vault entity index** that powers an **`entity=`
filter on `mempalace_search`** plus a noise-gated **`mempalace_entities`**
navigator, wired to a committed caller (the skill).

**Hallways are explicitly cut.** Within-wing regex co-occurrence "hallways"
would be a second, noisier entity-relationship system competing with
`kg_add`/`kg_query` (agent-curated, temporally-versioned triples) and the
already-existing `entity_tunnels_for_wing` primitive — and nothing would call
them. Cutting them also removes the chunk-id over-counting problem. The
`entity_occurrences` index keeps co-occurrence *derivable* later if a real need
appears; we just do not build/expose a hallway layer now.

## Problem (why the index does not exist today)

- Entity tagging (`miner._extract_entities_for_metadata`) sources its known set
  from `~/.mempalace/known_entities.json` — the *server's* home on a central
  deploy, so the registry branch is dead and extraction degrades to noisy
  capitalized-word frequency.
- `tool_add_drawer` / `tool_diary_write` stamp no entities, and no live MCP read
  path consumes `entities` (search filters wing/room only; tunnels key on
  room/wing). The data is inert over MCP.

## Data model (per `team_<slug>` schema)

One table, created by a **bespoke `PostgresEntityIndex` class** modelled on
`PostgresKnowledgeGraph` / `_ensure_wal_table` — NOT via `_ensure_collection`
(which hardcodes an `embedding vector(N)` column + HNSW index that an entity
table must not have):

```sql
CREATE TABLE team_<slug>.entity_occurrences (
  entity     text NOT NULL,
  drawer_id  text NOT NULL,   -- PHYSICAL row id (chunk id on the chunked path)
  wing       text,
  room       text,
  PRIMARY KEY (entity, drawer_id)
);
CREATE INDEX entity_occurrences_entity_idx ON team_<slug>.entity_occurrences (entity);
CREATE INDEX entity_occurrences_wing_idx   ON team_<slug>.entity_occurrences (wing);
```

`PostgresEntityIndex(backend, team)` computes `self._schema = team_schema(team)`,
shares `backend._conn()` / `_qi`, has its own `_ensured` flag, and exposes
`ensure()`, `add(drawer_id, wing, room, entities)`, `delete_by_drawer(ids)`,
`drawers_for_entity(entity)`, `top_entities(wing=None, min_count=N)`,
`known_entities()`. It is cached per-vault in `mcp_server` like `_kg_by_path`
and invalidated on `reconnect` the same way.

### Keying on the physical chunk id (the chunked-drawer fix)

`tool_add_drawer` stores chunked content under `f"{drawer_id}_chunk_{i:06d}"`
rows (no row exists under the logical id). The index keys on the **physical id
actually written** (the chunk id), so:

- delete/update can remove rows by the same physical ids the drawer write used,
- the `entity=` search intersect is a direct `id = ANY(:ids)` against the
  drawers table (no logical→physical expansion needed),
- and because we are NOT building hallways, the "same pair counted per chunk"
  over-count is a non-issue.

Entities are extracted once over the full `content` (pre-chunk) and the same
entity set is written for every chunk row of that drawer.

## Extraction on the server (per-vault, kg-seeded)

Make `miner._extract_entities_for_metadata(content, known=None)` accept an
injectable known set (default keeps the file-backed set — chroma unchanged; the
regex/COCA/stoplist tiers are untouched, pure, no circular-import risk:
`mcp_server -> miner` is a forward dep). The Postgres write path passes the
per-vault known set:

```sql
SELECT DISTINCT entity FROM team_<slug>.entity_occurrences
UNION
SELECT DISTINCT name FROM team_<slug>.kg_entities   -- kg_add is relational, NOT AGE
```

(`kg_entities.name` is the original-case display name; the extractor matches
case-insensitively.) Cached in-process per vault with a short TTL, invalidated
on write. Initially empty → extraction is the existing frequency+stoplist tier;
as `kg_add` triples and drawers accumulate, matching improves and stays isolated
per vault.

## Write-path integration (best-effort, never lose a drawer)

In `tool_add_drawer` (reuse the already-resolved `add_team`), after the drawer
upsert succeeds, extract entities and call `index.add(...)` inside a
`try/except` that logs and continues on failure. An entity-index error MUST NOT
fail the verbatim drawer write (verbatim-always). Same wiring in
`tool_diary_write`. `tool_delete_drawer` → `index.delete_by_drawer(ids)` for the
ids it deletes (using the same resolved team). `tool_update_drawer` on a content
change → delete + re-extract + re-insert. There is no shared transaction across
the collection write and the index (each pooled op autocommits); best-effort is
the honest, principle-aligned posture, matching the existing best-effort BM25 /
AGE provisioning.

## MCP surface

- **`mempalace_search` gains an optional `entity=` filter** — the load-bearing
  consumer. Resolve `entity -> drawer ids` via `index.drawers_for_entity()`,
  then restrict the search to those ids. Requires a small `restrict_ids` param
  threaded into `search_memories` and an `id = ANY(%s)` clause in
  `PostgresCollection.query` (it has id-restriction only in get/delete today).
- **`mempalace_entities`** navigator — `entity=` returns the verbatim drawers
  mentioning it (ids + wing/room); no entity returns the vault's top entities by
  occurrence count. **Noise-gated** by a minimum occurrence count (`min_count`,
  default 2) so one-off extraction noise stays out of the overview. (A further
  kg-seeded-intersection refinement is possible but deferred until the min-count
  gate proves insufficient.)

### Committed caller (so the data is not inert again)

In the same change, update the `/tset:mempalace` skill: *when the user names a
specific person, project, or service, set `entity=<name>` on `mempalace_search`
to scope recall.* This is the named caller whose absence sank the prior
proposal; without it we do not ship the navigator.

## Back-compat + rollout

- All new machinery gated to the Postgres backend (`_config.backend !=
  "chroma"`), mirroring `tool_status` / `tool_reconnect`. On chroma the tools
  fall back to existing `hallways.py` / `metadata.entities` reads (single-vault,
  fine).
- Tables created lazily on first write per vault (`_ensured` guard). The central
  server has no data yet, so there is nothing to migrate today.
- **Backfill** (G005): existing drawers filed before this feature have no entity
  rows (append-only history is never rewritten), so on an established vault the
  index would stay empty. A one-time, server-side per-vault backfill (iterate
  drawers, extract, populate `entity_occurrences`) closes that — distinct from
  an agent-invoked reindex; runs in-process, not as a daemon.

## Story breakdown (revised after the gate)

- **G003**: `PostgresEntityIndex` + injectable per-vault extraction + best-effort
  incremental maintenance on add/delete/update (physical-id keyed). DB-less
  unit tests for extraction injection + `importorskip`-guarded integration
  tests for the index.
- **G004**: `entity=` filter on `mempalace_search` (`restrict_ids` plumbing) +
  noise-gated `mempalace_entities` navigator + the `/tset:mempalace` skill
  caller. Tests.
- **G005** (repurposed from "hallways"): one-time per-vault backfill of
  `entity_occurrences` from existing drawers + the noise-gating thresholds.
  Tests.

## Cut / deferred

- `mempalace_hallways` and within-wing co-occurrence edges — redundant with
  `kg_query` + `entity_tunnels_for_wing`; co-occurrence remains derivable from
  `entity_occurrences` if a concrete need ever appears.
- L7 hallway "dynamics" fields (strength/stability/access_count) — local-only,
  no MCP consumer.
- A denormalised `entity_registry(entity PK, count)` — the `DISTINCT`/`UNION`
  seed is cheap + TTL-cached at memory scale; revisit only if it becomes hot.
