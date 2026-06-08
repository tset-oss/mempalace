"""Per-vault entity index (central, team-vaulted Postgres).

A lightweight inverted index mapping ``entity -> drawer`` inside a team's vault
schema (``team_<team>.entity_occurrences``). It is the server-side replacement
for the bulk miner's ``entities`` metadata + the single-user, non-vault-scoped
``hallways.json`` (see ``docs/design/vault-scoped-entities-hallways.md``).

It exists to power **entity-scoped recall of verbatim content** over MCP — the
one thing ``kg_query`` does not do (it returns relationships, not the stored
text). It is maintained best-effort and incrementally on the MCP write path
(``tool_add_drawer`` / ``tool_delete_drawer`` / ``tool_update_drawer``), keyed on
the **physical** row id actually written (the chunk id on the chunked path) so
delete/update stay consistent and an ``entity=`` search is a direct id filter.

This deliberately stays on a plain SQL table — no embeddings, no AGE — and
follows the bespoke-DDL-guard pattern of :class:`PostgresKnowledgeGraph` and
``PostgresBackend._ensure_wal_table`` rather than the embedding-shaped
``_ensure_collection``.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable, Optional

# Reuse identifier quoting + schema naming from the storage backend.
from .backends.postgres import _qi, team_schema

# The per-vault known-entity set seeds extraction. It is queried on the write
# path, so cache it briefly in-process and invalidate on any write rather than
# re-scanning DISTINCT on every add.
_KNOWN_TTL_SECONDS = 30.0


def _parent_match_params(parent_id: str) -> tuple[str, str]:
    """Return ``(exact_id, chunk_like_pattern)`` matching a drawer's bare id PLUS
    its ``{id}_chunk_NNNNNN`` rows, for ``WHERE drawer_id = %s OR drawer_id LIKE %s``.

    Shared by ``delete_by_parent`` (purge) and ``topic_labels_for_parent`` (capture)
    so the two ALWAYS target the identical row set — if the LIKE-escaping ever drifts
    between them, topic-recall rows would be captured-but-not-purged or vice versa.
    LIKE wildcards in the id are escaped so a literal ``%``/``_`` matches literally.
    """
    escaped = parent_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return parent_id, f"{escaped}\\_chunk\\_%"


class PostgresEntityIndex:
    """Entity -> drawer occurrence index stored in a team vault's schema."""

    def __init__(self, backend, team: Optional[str] = None):
        self._backend = backend
        self._team = team
        self._schema = team_schema(team)
        self._lock = threading.Lock()
        self._ensured = False
        self._known_cache: Optional[frozenset] = None
        self._known_cache_at = 0.0

    # -- schema -----------------------------------------------------------
    def _table(self) -> str:
        return f"{_qi(self._schema)}.{_qi('entity_occurrences')}"

    def _ensure(self) -> None:
        if self._ensured:
            return
        with self._lock:
            if self._ensured:
                return
            with self._backend._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_qi(self._schema)}")
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._table()} ("
                        "  entity text NOT NULL,"
                        "  drawer_id text NOT NULL,"
                        "  wing text,"
                        "  room text,"
                        "  is_topic boolean NOT NULL DEFAULT false,"
                        "  PRIMARY KEY (entity, drawer_id)"
                        ")"
                    )
                    # Migrate vaults whose entity_occurrences predates is_topic.
                    # Forward-only (no down-migration); a topic-only row can only
                    # be written by add(is_topic=True), which runs this _ensure
                    # first, so a column-less table provably has zero topic rows.
                    cur.execute(
                        f"ALTER TABLE {self._table()} "
                        "ADD COLUMN IF NOT EXISTS is_topic boolean NOT NULL DEFAULT false"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('entity_occurrences_entity')} "
                        f"ON {self._table()} (entity)"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('entity_occurrences_wing')} "
                        f"ON {self._table()} (wing)"
                    )
            self._ensured = True

    def _invalidate_known(self) -> None:
        self._known_cache = None

    # -- writes -----------------------------------------------------------
    def add(
        self,
        drawer_ids: Iterable[str],
        entities: Iterable[str],
        wing: Optional[str],
        room: Optional[str],
        is_topic: bool = False,
    ) -> int:
        """Insert (entity, drawer_id, wing, room) rows. Idempotent per pair.

        The same entity set is written for every physical drawer id (each chunk
        of a chunked drawer), so an ``entity=`` lookup hits whatever physical row
        the search returns. Returns the number of (entity, drawer) pairs offered
        (pre-dedup).

        ``is_topic`` marks RECALL-ONLY rows from caller-supplied topic labels:
        they are findable via ``drawers_for_entity`` but excluded from the
        co-occurrence derive (see ``link_store_postgres._co_occurrence_for_wing``)
        so a topic label never manufactures spurious entity hallways/tunnels.
        The conflict clause makes ``is_topic`` a MONOTONE meet (logical AND of
        every offer for a pair): any organically-extracted offer (``is_topic=
        False``) permanently demotes the row so it (re)joins co-occurrence,
        regardless of write order. A pure topic label stays ``is_topic=True``.
        """
        ids = [d for d in drawer_ids if d]
        ents = [e for e in entities if e]
        if not ids or not ents:
            return 0
        rows = [(e, d, wing, room, is_topic) for d in ids for e in ents]
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self._table()} (entity, drawer_id, wing, room, is_topic) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (entity, drawer_id) DO UPDATE "
                    "SET is_topic = entity_occurrences.is_topic AND EXCLUDED.is_topic",
                    rows,
                )
        self._invalidate_known()
        return len(rows)

    def delete_by_drawer(self, drawer_ids: Iterable[str]) -> None:
        """Remove all entity rows for the given physical drawer ids."""
        ids = [d for d in drawer_ids if d]
        if not ids:
            return
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table()} WHERE drawer_id = ANY(%s)", (ids,))
        self._invalidate_known()

    def delete_by_parent(self, parent_id: str) -> None:
        """Remove the bare ``parent_id`` row AND every ``{parent_id}_chunk_*`` row.

        Re-indexing a drawer on update must clear ALL its prior physical rows,
        not just the bare id: a drawer originally filed as multiple chunks keyed
        its entity rows on ``{parent_id}_chunk_NNNNNN``. Deleting only the bare id
        would orphan those chunk rows, and a re-chunk that shrinks the chunk count
        would leave stale high-index rows behind. Matching the parent id plus the
        chunk-id prefix clears them all regardless of the prior chunk count.
        """
        if not parent_id:
            return
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table()} WHERE drawer_id = %s OR drawer_id LIKE %s",
                    _parent_match_params(parent_id),
                )
        self._invalidate_known()

    # -- reads ------------------------------------------------------------
    def known_entities(self) -> frozenset:
        """Per-vault known-entity set: accumulated occurrences UNION kg_add names.

        Seeds the extractor so tagging is not pure regex guessing and improves
        within a vault over time. Briefly TTL-cached; invalidated on any write.
        kg seeding reads the relational ``kg_entities`` table (kg_add is plain
        SQL, not AGE) and is skipped when that table does not exist yet.
        """
        now = time.monotonic()
        if self._known_cache is not None and (now - self._known_cache_at) < _KNOWN_TTL_SECONDS:
            return self._known_cache
        self._ensure()
        names: set[str] = set()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT DISTINCT entity FROM {self._table()}")
                names.update(r[0] for r in cur.fetchall() if r[0])
                # Seed from this vault's kg_add entities when the KG table exists.
                cur.execute("SELECT to_regclass(%s)", (f"{self._schema}.kg_entities",))
                if cur.fetchone()[0] is not None:
                    cur.execute(
                        f"SELECT DISTINCT name FROM {_qi(self._schema)}.{_qi('kg_entities')}"
                    )
                    names.update(r[0] for r in cur.fetchall() if r[0])
        self._known_cache = frozenset(names)
        self._known_cache_at = now
        return self._known_cache

    def drawers_for_entity(self, entity: str) -> list[dict]:
        """Physical drawer ids (+ wing/room) that mention ``entity``.

        Case-insensitive match so a query for ``Dana`` finds rows stamped
        ``dana`` / ``DANA`` (extraction preserves the matched casing).
        """
        if not entity:
            return []
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT drawer_id, wing, room FROM {self._table()} "
                    "WHERE lower(entity) = lower(%s)",
                    (entity,),
                )
                return [{"drawer_id": r[0], "wing": r[1], "room": r[2]} for r in cur.fetchall()]

    def topic_labels_for_parent(self, parent_id: str) -> list[str]:
        """Distinct caller-topic labels (``is_topic=true`` rows) for a drawer and
        its chunks (bare ``parent_id`` plus ``{parent_id}_chunk_*``).

        ``update_drawer`` uses this to PRESERVE topic-based recall across a
        re-index: an edit re-derives extracted entities from the new content but
        cannot recover the caller's topic labels (they live only here and in
        ``wing_topics``), so they would otherwise be dropped by the delete+re-index.
        """
        if not parent_id:
            return []
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT entity FROM {self._table()} "
                    "WHERE is_topic = true AND (drawer_id = %s OR drawer_id LIKE %s)",
                    _parent_match_params(parent_id),
                )
                return [r[0] for r in cur.fetchall() if r[0]]

    def top_entities(
        self, wing: Optional[str] = None, min_count: int = 1, limit: int = 100
    ) -> list[dict]:
        """Most-mentioned entities in the vault (optionally scoped to a wing).

        Excludes recall-only topic-label rows (``is_topic=true``): a caller topic
        is stamped on every chunk of a drawer, so counting it here would let a
        single tagged drawer dominate the "most-mentioned" overview by its chunk
        count. Topic labels remain findable via ``drawers_for_entity`` /
        ``entity=``; this surface reflects organically-extracted entities. The
        ``_ensure`` above guarantees the column exists (this index owns it).
        """
        self._ensure()
        clauses = ["is_topic = false"]
        params: list = []
        if wing:
            clauses.append("wing = %s")
            params.append(wing)
        where = " WHERE " + " AND ".join(clauses)
        params.extend([max(1, int(min_count)), max(1, int(limit))])
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT entity, count(DISTINCT drawer_id) AS n FROM {self._table()}"
                    f"{where} GROUP BY entity HAVING count(DISTINCT drawer_id) >= %s "
                    "ORDER BY n DESC, entity ASC LIMIT %s",
                    params,
                )
                return [{"entity": r[0], "count": r[1]} for r in cur.fetchall()]

    # -- backfill ---------------------------------------------------------
    def backfill(self, drawers_col, extract, batch_size: int = 2000) -> dict:
        """One-time populate the index from an existing drawers collection.

        Drawers filed before this index existed (append-only history is never
        rewritten) carry no entity rows, so on an established vault the index
        stays empty until a backfill runs. This iterates the drawers collection
        in pages and indexes each physical row by the entities in its text.

        ``extract(content) -> list[str]`` is injected (so this module needs no
        miner dependency); the caller wraps the regex extractor with a known set
        captured once. Idempotent via ``add``'s ON CONFLICT, so re-running is
        safe. Keyed on the physical drawer/chunk id actually stored, matching
        the write path. Returns ``{"drawers": n, "entity_rows": m}``.
        """
        self._ensure()
        seen = 0
        rows = 0
        offset = 0
        try:
            total = drawers_col.count()
        except Exception:
            total = 0
        while offset < total:
            batch = drawers_col.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas"]
            )
            ids = list(getattr(batch, "ids", None) or [])
            if not ids:
                break
            docs = list(getattr(batch, "documents", None) or [])
            metas = list(getattr(batch, "metadatas", None) or [])
            for i, did in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                meta = meta if isinstance(meta, dict) else {}
                if meta.get("is_sentinel"):
                    continue
                content = docs[i] if i < len(docs) else ""
                seen += 1
                if not content:
                    continue
                ents = extract(content)
                if ents:
                    rows += self.add([did], ents, meta.get("wing"), meta.get("room"))
            offset += len(ids)
        return {"drawers": seen, "entity_rows": rows}
