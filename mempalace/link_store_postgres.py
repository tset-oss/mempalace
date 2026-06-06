"""Per-team Postgres explicit-tunnel store (central, team-vaulted).

The team-scoped counterpart to :class:`mempalace.link_store.JsonLinkStore`. It
owns the agent-authored explicit tunnels inside a team's vault schema
(``team_<slug>.tunnels``) instead of the host-global ``tunnels.json`` that the
chroma path uses. On the central multi-team deployment that single host-global
file is a cross-tenant leak (one file, one host, many teams); a per-team table
makes isolation structural — a store for team A physically cannot read or write
team B's rows because it queries a different schema.

This implements the same :class:`mempalace.link_store.LinkStore` contract as the
JSON store so it is a drop-in: the tunnel ids are the SAME symmetric
``palace_graph._canonical_tunnel_id`` (so a tunnel created on chroma and the
same tunnel created on Postgres share an id), the returned dicts carry the same
keys (``id`` / ``source`` / ``target`` / ``label`` / ``kind`` / ``created_at`` /
``updated_at`` + the four dynamics fields), and re-creating a tunnel preserves
its accumulated dynamics. On Postgres the four dynamics live as DISCRETE
COLUMNS; a re-create is an ``INSERT ... ON CONFLICT (id) DO UPDATE`` that touches
only ``label`` and ``updated_at`` and leaves the dynamics columns untouched —
the column-level equivalent of the JSON store's ``merge_dynamics`` (which copies
the four fields forward). A brand-new row gets the
``initialize_dynamics_fields`` defaults at insert time.

It mirrors :class:`mempalace.entity_index_postgres.PostgresEntityIndex`: the
schema comes from ``team_schema(team)``, identifiers are quoted with ``_qi``, the
DDL runs once behind ``self._ensured`` + a lock, and connections come from
``PostgresBackend._conn()``.

Team is mandatory here, for reads AND writes. Unlike the host-local chroma JSON
store (single-vault, no team), every operation on this store is scoped to one
team's schema — you cannot list, follow, or create tunnels without naming the
team whose vault to touch. The resolver in :mod:`mempalace.link_store` calls
``require_write_team`` before constructing this store so a missing team fails
loud rather than silently routing into a default vault.

No backfill: ``_ensure`` CREATES the table empty and copies NOTHING from the
host-global ``tunnels.json``. There is no deployment yet, so there is nothing to
migrate; the host-global file commingles every team's tunnels, so copying it
into one team's schema would contaminate that team with other teams' links.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from .backends.postgres import _qi, team_schema
from .config import normalize_wing_name
from .dynamics import initialize_dynamics_fields, merge_dynamics
from .link_store import LinkStore
from .palace_graph import (
    _canonical_tunnel_id,
    _require_name,
    entity_tunnels_for_wing,
    topic_tunnels_for_wing,
)

logger = logging.getLogger("mempalace.link_store_postgres")

# Pinned bound on the entity_occurrences rows scanned for ONE wing's derived
# rebuild. This is the entity-occurrence scan substrate the co-occurrence derive
# counts over — a row PER (entity, chunk) pair, so it is denser than the closet
# group scan (REBUILD_FETCH_CAP=500) or the closet reconcile scan
# (RECONCILE_SCAN_CAP=5000). We DELIBERATELY do not inherit either closet cap:
# those bound closet GROUPS, not a wing-wide co-occurrence scan. A wing with a
# few thousand drawers, each tagged with several entities per chunk, easily
# exceeds 5000 (entity, chunk) rows, so a too-small cap would silently drop real
# co-occurrence. 50_000 rows keeps the in-memory pair count bounded (a few MB)
# while comfortably covering a large single wing.
#
# Truncate-not-paginate: when a wing exceeds this cap the derive uses only the
# first WING_SCAN_CAP rows (ORDER BY drawer_id so the truncation is deterministic
# across rebuilds) and LOGS a truncation warning. We truncate rather than
# paginate because the derive is a NON-GATING analytic over a single wing (the
# verbatim drawers and the entity index are the recall floor); an unbounded
# multi-page scan on a pathological wing would be a worse failure mode than a
# bounded, observable approximation.
WING_SCAN_CAP = 50_000


def _row_to_tunnel(row: tuple) -> dict:
    """Map a ``tunnels`` row to the dict shape JsonLinkStore.create_tunnel returns.

    Column order matches ``_SELECT_COLUMNS`` below. Drawer ids are folded onto
    the ``source`` / ``target`` sub-dicts only when present, exactly as the JSON
    store does (it sets ``source["drawer_id"]`` only when a drawer id was given),
    so the two stores' return shapes are identical.
    """
    (
        id_,
        source_wing,
        source_room,
        target_wing,
        target_room,
        source_drawer_id,
        target_drawer_id,
        label,
        kind,
        created_at,
        updated_at,
        strength,
        stability,
        last_activated,
        access_count,
    ) = row

    source: dict = {"wing": source_wing, "room": source_room}
    if source_drawer_id:
        source["drawer_id"] = source_drawer_id
    target: dict = {"wing": target_wing, "room": target_room}
    if target_drawer_id:
        target["drawer_id"] = target_drawer_id

    # Normalize timestamptz columns to ISO strings so the return shape matches
    # JsonLinkStore byte-for-byte. JsonLinkStore uses datetime.isoformat()
    # (palace_graph.py:626/642, dynamics.py:91/100); psycopg returns raw
    # datetime objects from timestamptz columns. Leaving them as datetime would
    # break json.dumps (no default=str in mcp_server) and produce a space-
    # separated str() format instead of the 'T'-separated .isoformat() form.
    def _iso(dt) -> str:
        return dt.isoformat() if hasattr(dt, "isoformat") else dt

    tunnel: dict = {
        "id": id_,
        "source": source,
        "target": target,
        "label": label or "",
        "kind": kind,
        "created_at": _iso(created_at),
    }
    if updated_at is not None:
        tunnel["updated_at"] = _iso(updated_at)
    tunnel["strength"] = strength
    tunnel["stability"] = stability
    tunnel["last_activated"] = _iso(last_activated)
    tunnel["access_count"] = access_count
    return tunnel


# Columns selected (and the order ``_row_to_tunnel`` unpacks) on every read.
_SELECT_COLUMNS = (
    "id, source_wing, source_room, target_wing, target_room, "
    "source_drawer_id, target_drawer_id, label, kind, created_at, updated_at, "
    "strength, stability, last_activated, access_count"
)


def _hallway_id(wing: str, entity_a: str, entity_b: str) -> str:
    """Symmetric derived-hallway id — identical scheme to ``hallways._hallway_id``.

    Sorting the pair before hashing makes ``(Aya, Lumi)`` and ``(Lumi, Aya)``
    one record, so an idempotent rebuild upserts the same row instead of
    creating a parallel one. Kept byte-identical to the chroma hallway id so the
    two backends' hallway ids match for the same wing + pair.
    """
    import hashlib

    a, b = sorted([entity_a, entity_b])
    key = f"{wing}::{a}::{b}".encode("utf-8")
    suffix = hashlib.sha256(key).hexdigest()[:8]
    return f"hallway_{wing}_{a}_{b}_{suffix}"


def _row_to_hallway(row: tuple) -> dict:
    """Map a ``hallways`` row to the dict shape the JSON hallway store returns."""
    (
        id_,
        wing,
        entity_a,
        entity_b,
        co_occurrence_count,
        rooms,
        label,
        created_at,
        updated_at,
        strength,
        stability,
        last_activated,
        access_count,
    ) = row

    def _iso(dt) -> str:
        return dt.isoformat() if hasattr(dt, "isoformat") else dt

    record: dict = {
        "id": id_,
        "wing": wing,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "co_occurrence_count": co_occurrence_count,
        "rooms": list(rooms or []),
        "label": label or "",
        "created_at": _iso(created_at),
    }
    if updated_at is not None:
        record["updated_at"] = _iso(updated_at)
    record["strength"] = strength
    record["stability"] = stability
    record["last_activated"] = _iso(last_activated)
    record["access_count"] = access_count
    return record


class PostgresLinkStore(LinkStore):
    """Explicit-tunnel store backed by a team vault's ``tunnels`` table."""

    def __init__(self, backend, team: str):
        self._backend = backend
        self._team = team
        self._schema = team_schema(team)
        self._lock = threading.Lock()
        self._ensured = False

    # -- schema -----------------------------------------------------------
    def _table(self) -> str:
        return f"{_qi(self._schema)}.{_qi('tunnels')}"

    def _hallways_table(self) -> str:
        return f"{_qi(self._schema)}.{_qi('hallways')}"

    def _wing_topics_table(self) -> str:
        return f"{_qi(self._schema)}.{_qi('wing_topics')}"

    def _ensure(self) -> None:
        """Create schema + ``tunnels`` table + wing-endpoint indexes once.

        Guarded behind ``self._ensured`` + a lock (the PostgresEntityIndex
        pattern). NO-BACKFILL: this CREATES the table EMPTY and copies nothing
        from the host-global ``tunnels.json`` — there is no deployment yet, so
        there is nothing to migrate, and the host-global file commingles every
        team's tunnels (copying it would contaminate this one team).
        """
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
                        "  id text PRIMARY KEY,"
                        "  source_wing text NOT NULL,"
                        "  source_room text NOT NULL,"
                        "  target_wing text NOT NULL,"
                        "  target_room text NOT NULL,"
                        "  source_drawer_id text,"
                        "  target_drawer_id text,"
                        "  label text NOT NULL DEFAULT '',"
                        "  kind text NOT NULL DEFAULT 'explicit',"
                        "  created_at timestamptz NOT NULL,"
                        "  updated_at timestamptz,"
                        "  strength double precision NOT NULL,"
                        "  stability double precision NOT NULL,"
                        "  last_activated timestamptz NOT NULL,"
                        "  access_count integer NOT NULL"
                        ")"
                    )
                    # Wing-endpoint indexes back list_tunnels(wing=...) and
                    # follow_tunnels (both filter on either endpoint's wing).
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('tunnels_source_wing')} "
                        f"ON {self._table()} (source_wing)"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('tunnels_target_wing')} "
                        f"ON {self._table()} (target_wing)"
                    )
                    # The derived within-wing entity hallways table. A hallway
                    # is the co-occurrence fact "entity_a and entity_b travel
                    # together inside this wing"; the cross-wing entity tunnels
                    # (kind='entity', in the tunnels table above) are derived
                    # FROM these rows. The four dynamics fields live as discrete
                    # columns (mirroring the tunnels table) so an incremental
                    # rebuild can preserve accumulated weights via ON CONFLICT.
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._hallways_table()} ("
                        "  id text PRIMARY KEY,"
                        "  wing text NOT NULL,"
                        "  entity_a text NOT NULL,"
                        "  entity_b text NOT NULL,"
                        "  co_occurrence_count integer NOT NULL,"
                        "  rooms text[] NOT NULL DEFAULT '{}',"
                        "  label text NOT NULL DEFAULT '',"
                        "  created_at timestamptz NOT NULL,"
                        "  updated_at timestamptz,"
                        "  strength double precision NOT NULL,"
                        "  stability double precision NOT NULL,"
                        "  last_activated timestamptz NOT NULL,"
                        "  access_count integer NOT NULL"
                        ")"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('hallways_wing')} "
                        f"ON {self._hallways_table()} (wing)"
                    )
                    # Per-wing TOPIC labels — the substrate for topic tunnels.
                    # Agents/the miner supply the labels (no LLM at tunnel time);
                    # topic-tunnel MATCHING is pure case-insensitive string
                    # overlap of these per-wing label sets (the unchanged
                    # palace_graph.compute_topic_tunnels). On chroma the same
                    # labels live in the host-global
                    # known_entities.json["topics_by_wing"]; on the central
                    # multi-team deployment that one host file is a cross-tenant
                    # leak, so each team's labels live in its own schema instead.
                    # PK (topic, wing) makes re-adding a label idempotent.
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._wing_topics_table()} ("
                        "  topic text NOT NULL,"
                        "  wing text NOT NULL,"
                        "  PRIMARY KEY (topic, wing)"
                        ")"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('wing_topics_wing')} "
                        f"ON {self._wing_topics_table()} (wing)"
                    )
            self._ensured = True

    # -- writes -----------------------------------------------------------
    def create_tunnel(
        self,
        source_wing: str,
        source_room: str,
        target_wing: str,
        target_room: str,
        label: str = "",
        source_drawer_id: str = None,
        target_drawer_id: str = None,
        kind: str = "explicit",
    ) -> dict:
        """Create (or re-create-and-update) a symmetric explicit tunnel.

        Computes the same symmetric ``_canonical_tunnel_id`` the JSON store uses
        (sorting the two endpoints before hashing) so ``create_tunnel(A, B)`` and
        ``create_tunnel(B, A)`` dedup to one row, and a tunnel created on chroma
        and on Postgres share an id.

        A brand-new row is inserted with ``initialize_dynamics_fields`` defaults
        for the four dynamics columns. A re-create with the same id runs
        ``ON CONFLICT (id) DO UPDATE`` that sets ONLY ``label`` and
        ``updated_at`` from the excluded row — the dynamics columns are left
        UNTOUCHED, so accumulated strength / stability / last_activated /
        access_count are preserved by simply not being overwritten. This is the
        column-level equivalent of the JSON store's ``merge_dynamics``.
        """
        source_wing = _require_name(source_wing, "source_wing")
        source_room = _require_name(source_room, "source_room")
        target_wing = _require_name(target_wing, "target_wing")
        target_room = _require_name(target_room, "target_room")

        tunnel_id = _canonical_tunnel_id(source_wing, source_room, target_wing, target_room)

        now = datetime.now(timezone.utc)
        # Build the brand-new-row dynamics defaults from one place. The dict
        # carries created_at so initialize_dynamics_fields seeds last_activated
        # from creation time (matching the JSON store's brand-new path).
        defaults: dict = {"created_at": now.isoformat()}
        initialize_dynamics_fields(defaults, now=now)

        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._table()} ("
                    "  id, source_wing, source_room, target_wing, target_room,"
                    "  source_drawer_id, target_drawer_id, label, kind,"
                    "  created_at, updated_at,"
                    "  strength, stability, last_activated, access_count"
                    ") VALUES ("
                    "  %(id)s, %(source_wing)s, %(source_room)s, %(target_wing)s, %(target_room)s,"
                    "  %(source_drawer_id)s, %(target_drawer_id)s, %(label)s, %(kind)s,"
                    "  %(created_at)s, NULL,"
                    "  %(strength)s, %(stability)s, %(last_activated)s, %(access_count)s"
                    ") ON CONFLICT (id) DO UPDATE SET"
                    "  label = EXCLUDED.label,"
                    "  updated_at = %(updated_at)s "
                    f"RETURNING {_SELECT_COLUMNS}",
                    {
                        "id": tunnel_id,
                        "source_wing": source_wing,
                        "source_room": source_room,
                        "target_wing": target_wing,
                        "target_room": target_room,
                        "source_drawer_id": source_drawer_id or None,
                        "target_drawer_id": target_drawer_id or None,
                        "label": label,
                        "kind": kind,
                        "created_at": now,
                        # updated_at is only consumed on the DO UPDATE branch;
                        # a brand-new INSERT writes NULL (literal above).
                        "updated_at": now,
                        "strength": defaults["strength"],
                        "stability": defaults["stability"],
                        "last_activated": defaults["last_activated"],
                        "access_count": defaults["access_count"],
                    },
                )
                row = cur.fetchone()
        return _row_to_tunnel(row)

    def delete_tunnel(self, tunnel_id: str) -> dict:
        """Delete a tunnel by ID. Returns ``{"deleted": <tunnel_id>}``."""
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table()} WHERE id = %s", (tunnel_id,))
        return {"deleted": tunnel_id}

    # -- reads ------------------------------------------------------------
    def list_tunnels(self, wing: str = None) -> list[dict]:
        """List explicit tunnels, optionally filtered by *wing*.

        Matches *wing* against either endpoint (tunnels are symmetric). Reads are
        team-scoped: this only ever sees the calling team's schema.
        """
        self._ensure()
        clauses = ""
        params: tuple = ()
        if wing:
            clauses = " WHERE source_wing = %s OR target_wing = %s"
            params = (wing, wing)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM {self._table()}{clauses} ORDER BY created_at",
                    params,
                )
                return [_row_to_tunnel(r) for r in cur.fetchall()]

    def follow_tunnels(self, wing: str, room: str, col=None, config=None) -> list[dict]:
        """Follow explicit tunnels from a room — returns connection dicts.

        Same return shape as ``JsonLinkStore.follow_tunnels``: each connection is
        ``{direction, connected_wing, connected_room, label, drawer_id,
        tunnel_id}`` (+ ``drawer_preview`` when *col* is supplied and the
        connected endpoint names a drawer id). A row matching on its source
        endpoint is ``outgoing`` (connected endpoint = target); a row matching on
        its target endpoint is ``incoming`` (connected endpoint = source).
        """
        self._ensure()
        connections: list[dict] = []
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                # Outgoing: this room is the source endpoint.
                cur.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM {self._table()} "
                    "WHERE source_wing = %s AND source_room = %s",
                    (wing, room),
                )
                for r in cur.fetchall():
                    t = _row_to_tunnel(r)
                    connections.append(
                        {
                            "direction": "outgoing",
                            "connected_wing": t["target"]["wing"],
                            "connected_room": t["target"]["room"],
                            "label": t["label"],
                            "drawer_id": t["target"].get("drawer_id"),
                            "tunnel_id": t["id"],
                        }
                    )
                # Incoming: this room is the target endpoint.
                cur.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM {self._table()} "
                    "WHERE target_wing = %s AND target_room = %s",
                    (wing, room),
                )
                for r in cur.fetchall():
                    t = _row_to_tunnel(r)
                    connections.append(
                        {
                            "direction": "incoming",
                            "connected_wing": t["source"]["wing"],
                            "connected_room": t["source"]["room"],
                            "label": t["label"],
                            "drawer_id": t["source"].get("drawer_id"),
                            "tunnel_id": t["id"],
                        }
                    )

        # Hydrate a drawer preview for connected drawers when a collection is in
        # hand — same best-effort shape as the JSON store.
        if col and connections:
            drawer_ids = [c["drawer_id"] for c in connections if c.get("drawer_id")]
            if drawer_ids:
                try:
                    results = col.get(ids=drawer_ids, include=["documents", "metadatas"])
                    drawer_map = dict(zip(results["ids"], results["documents"]))
                    for c in connections:
                        did = c.get("drawer_id")
                        if did and did in drawer_map:
                            c["drawer_preview"] = drawer_map[did][:300]
                except Exception:
                    pass

        return connections

    # -- derived hallways + entity tunnels --------------------------------
    def _co_occurrence_for_wing(self, wing: str) -> tuple[dict, dict, bool]:
        """Count entity-pair co-occurrence PER PHYSICAL CHUNK for one wing.

        Reads this team's ``entity_occurrences`` rows for ``wing`` (each row is
        one ``(entity, drawer_id)`` pair — and ``drawer_id`` is the PHYSICAL
        chunk id the per-chunk tagging from the write path stamped). Grouping by
        ``drawer_id`` reconstructs each chunk's own entity set, so two entities
        co-occur once per chunk they SHARE — no over-count, no whole-drawer
        inflation, and no ``COUNT(DISTINCT logical_key)`` column needed (the
        per-chunk tagging IS the substrate).

        Returns ``(pair_counts, pair_rooms, truncated)`` where:
          * ``pair_counts``: ``{(entity_a, entity_b): count}`` with the pair
            sorted (symmetric key — matches ``hallways._hallway_id``);
          * ``pair_rooms``: ``{(entity_a, entity_b): set(rooms)}``;
          * ``truncated``: ``True`` when the wing exceeded ``WING_SCAN_CAP``
            rows and the scan was clipped (an observability signal).

        The scan is bounded by ``WING_SCAN_CAP`` rows, ordered by ``drawer_id``
        so a clipped scan keeps whole chunks together and is deterministic
        across rebuilds.
        """
        from collections import defaultdict
        from itertools import combinations

        # Fetch one extra row past the cap to detect truncation without a second
        # COUNT query: if the cap+1th row exists, the wing exceeded the cap.
        rows: list[tuple] = []
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                # The entity_occurrences table is owned by PostgresEntityIndex and
                # only exists once that vault has indexed an entity. A vault that
                # never tagged an entity has no co-occurrence substrate, so derive
                # nothing rather than raise UndefinedTable.
                cur.execute("SELECT to_regclass(%s)", (f"{self._schema}.entity_occurrences",))
                if cur.fetchone()[0] is None:
                    return {}, {}, False
                cur.execute(
                    f"SELECT drawer_id, entity, room FROM {self._table_entities()} "
                    "WHERE wing = %s ORDER BY drawer_id LIMIT %s",
                    (wing, WING_SCAN_CAP + 1),
                )
                rows = cur.fetchall()

        truncated = len(rows) > WING_SCAN_CAP
        if truncated:
            rows = rows[:WING_SCAN_CAP]
            logger.warning(
                "derived-link rebuild truncated team=%s wing=%s cap=%s "
                "(co-occurrence derived from the first %s entity rows only)",
                self._team,
                wing,
                WING_SCAN_CAP,
                WING_SCAN_CAP,
            )

        # Group the entity rows back into per-chunk entity sets.
        chunk_entities: dict[str, set] = defaultdict(set)
        chunk_room: dict[str, Optional[str]] = {}
        for drawer_id, entity, room in rows:
            if not drawer_id or not entity:
                continue
            chunk_entities[drawer_id].add(entity)
            if room and isinstance(room, str) and room.strip():
                chunk_room.setdefault(drawer_id, room)

        pair_counts: dict[tuple, int] = defaultdict(int)
        pair_rooms: dict[tuple, set] = defaultdict(set)
        for drawer_id, ents in chunk_entities.items():
            if len(ents) < 2:
                continue
            room = chunk_room.get(drawer_id)
            for a, b in combinations(sorted(ents), 2):
                if a == b:
                    continue
                key = (a, b)
                pair_counts[key] += 1
                if room:
                    pair_rooms[key].add(room)

        return pair_counts, pair_rooms, truncated

    def _table_entities(self) -> str:
        return f"{_qi(self._schema)}.{_qi('entity_occurrences')}"

    def compute_hallways_for_wing(self, wing: str, col=None, min_count: int = 2) -> list[dict]:
        """Derive + persist this wing's entity hallways from entity_occurrences.

        Server-side counterpart to :func:`hallways.compute_hallways_for_wing`:
        instead of scanning a chroma collection's ``entities`` metadata, it
        counts co-occurrence over this team's vaulted ``entity_occurrences``
        rows (per physical chunk). It PURGES this wing's prior derived hallway
        rows and re-upserts the recomputed set, PRESERVING accumulated dynamics
        on records that survive the recompute (via the dynamics columns the
        ON CONFLICT leaves untouched). Records for other wings are untouched.

        ``col`` is accepted for signature parity with the JSON store but unused
        (the substrate is the entity index, not a collection). Returns this
        wing's hallway dicts in the same shape the JSON store returns.
        """
        if not isinstance(wing, str) or not wing.strip():
            return []
        min_count = max(1, int(min_count))
        self._ensure()

        pair_counts, pair_rooms, _truncated = self._co_occurrence_for_wing(wing)

        # Load the wing's CURRENT hallway dynamics BEFORE the purge, keyed by the
        # symmetric (entity_a, entity_b) pair, so accumulated strength/stability/
        # last_activated/access_count survive a recompute. Without this the purge
        # would wipe the living-connection weights every rebuild. This mirrors the
        # JSON store's pre-recompute existing-dynamics lookup + merge_dynamics.
        existing_dynamics: dict = {}
        for prior in self.list_hallways(wing):
            key = tuple(sorted([prior.get("entity_a"), prior.get("entity_b")]))
            existing_dynamics[key] = {
                k: prior[k]
                for k in ("strength", "stability", "last_activated", "access_count")
                if k in prior
            }

        now = datetime.now(timezone.utc)
        records: list[dict] = []
        for key in sorted(pair_counts.keys()):
            count = pair_counts[key]
            if count < min_count:
                continue
            entity_a, entity_b = key
            rooms = sorted(pair_rooms.get(key, set()))
            room_summary = ", ".join(rooms[:3]) if rooms else "(no room tags)"
            if len(rooms) > 3:
                room_summary += f", +{len(rooms) - 3} more"
            record: dict = {
                "id": _hallway_id(wing, entity_a, entity_b),
                "wing": wing,
                "entity_a": entity_a,
                "entity_b": entity_b,
                "co_occurrence_count": count,
                "rooms": rooms,
                "label": (
                    f"{entity_a} ↔ {entity_b} (co-occur in {count} drawers across "
                    f"{len(rooms) or 'no'} room{'s' if len(rooms) != 1 else ''}: {room_summary})"
                ),
                "created_at": now.isoformat(),
            }
            # Carry forward preserved dynamics (then backfill missing fields).
            # merge_dynamics is the single source of truth shared with the JSON
            # hallway recompute + the tunnel re-create path.
            merge_dynamics(record, existing_dynamics.get(key, {}), now=now)
            records.append(record)

        self._purge_wing_hallways(wing)
        self._upsert_hallways(wing, records, now)
        return records

    def _purge_wing_hallways(self, wing: str) -> None:
        """Delete this wing's prior derived hallway rows (other wings kept)."""
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._hallways_table()} WHERE wing = %s", (wing,))

    def _upsert_hallways(self, wing: str, records: list[dict], now: datetime) -> None:
        """Insert the recomputed hallway rows, carrying preserved dynamics.

        Each record already has its four dynamics fields set by
        ``merge_dynamics`` in :meth:`compute_hallways_for_wing` (preserved from
        the pre-purge row, or seeded fresh for a brand-new pair). The wing's rows
        were just purged so each is a fresh INSERT; the ``ON CONFLICT`` is a
        belt-and-braces guard for a concurrent rebuild re-creating the same id —
        it refreshes the recomputed count/rooms/label but, crucially, does NOT
        touch the dynamics columns, so accumulated weights are never clobbered.
        """
        if not records:
            return
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    cur.execute(
                        f"INSERT INTO {self._hallways_table()} ("
                        "  id, wing, entity_a, entity_b, co_occurrence_count, rooms,"
                        "  label, created_at, updated_at,"
                        "  strength, stability, last_activated, access_count"
                        ") VALUES ("
                        "  %(id)s, %(wing)s, %(entity_a)s, %(entity_b)s,"
                        "  %(co_occurrence_count)s, %(rooms)s, %(label)s, %(created_at)s, NULL,"
                        "  %(strength)s, %(stability)s, %(last_activated)s, %(access_count)s"
                        ") ON CONFLICT (id) DO UPDATE SET"
                        "  co_occurrence_count = EXCLUDED.co_occurrence_count,"
                        "  rooms = EXCLUDED.rooms,"
                        "  label = EXCLUDED.label,"
                        "  updated_at = %(now)s",
                        {
                            "id": rec["id"],
                            "wing": rec["wing"],
                            "entity_a": rec["entity_a"],
                            "entity_b": rec["entity_b"],
                            "co_occurrence_count": rec["co_occurrence_count"],
                            "rooms": list(rec["rooms"]),
                            "label": rec["label"],
                            "created_at": now,
                            "now": now,
                            "strength": rec["strength"],
                            "stability": rec["stability"],
                            "last_activated": rec["last_activated"],
                            "access_count": rec["access_count"],
                        },
                    )

    def list_hallways(self, wing: Optional[str] = None) -> list[dict]:
        """List this team's derived hallway rows, optionally filtered by wing."""
        self._ensure()
        clauses = ""
        params: tuple = ()
        if wing:
            clauses = " WHERE wing = %s"
            params = (wing,)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, wing, entity_a, entity_b, co_occurrence_count, rooms, label, "
                    "created_at, updated_at, strength, stability, last_activated, access_count "
                    f"FROM {self._hallways_table()}{clauses} ORDER BY entity_a, entity_b",
                    params,
                )
                return [_row_to_hallway(r) for r in cur.fetchall()]

    # -- per-wing topic labels (the topic-tunnel substrate) ---------------
    def add_topics(self, wing: str, topics) -> int:
        """Record one or more TOPIC labels for *wing* (idempotent upsert).

        Each ``(topic, wing)`` pair is inserted ``ON CONFLICT DO NOTHING``, so
        re-supplying a label an agent (or the miner) already filed is harmless.
        Blank/whitespace topics and a blank wing are skipped. Topics are stored
        verbatim (first-observed casing); the case-insensitive overlap match
        happens later in the unchanged ``compute_topic_tunnels``. Returns the
        number of pairs offered (pre-dedup).
        """
        if not isinstance(wing, str) or not wing.strip():
            return 0
        wing = wing.strip()
        labels = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
        if not labels:
            return 0
        self._ensure()
        rows = [(t, wing) for t in labels]
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self._wing_topics_table()} (topic, wing) "
                    "VALUES (%s, %s) ON CONFLICT (topic, wing) DO NOTHING",
                    rows,
                )
        return len(rows)

    def topics_by_wing(self) -> dict:
        """Return this team's ``{wing: [topic, ...]}`` map (verbatim casing).

        The exact shape ``palace_graph.compute_topic_tunnels`` consumes. Reads
        are team-scoped — this only ever sees the calling team's schema, so
        team A's labels can never link team B's wings.
        """
        self._ensure()
        out: dict = {}
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT wing, topic FROM {self._wing_topics_table()} ORDER BY wing, topic"
                )
                for wing, topic in cur.fetchall():
                    out.setdefault(wing, []).append(topic)
        return out

    # -- the incremental rebuild entrypoint -------------------------------
    def rebuild_derived_links_for_wing(self, wing: str, min_count: int = 2) -> dict:
        """Recompute one wing's DERIVED link layer from entity_occurrences.

        This is the per-wing recompute the debounced worker and the miner hook
        both call. In one pass it:

          1. recomputes + persists this wing's hallways
             (:meth:`compute_hallways_for_wing`, which purges the wing's prior
             hallway rows first);
          2. PURGES this wing's prior DERIVED entity tunnels — rows with
             ``kind='entity'`` where ``wing`` is one endpoint — and ONLY those.
             Explicit (``kind='explicit'``) and topic (``kind='topic'``) tunnels
             are NEVER touched: a derived rebuild must not delete user-authored
             links;
          3. re-derives the cross-wing entity tunnels by feeding ALL of this
             team's hallway rows (this wing's freshly recomputed set + every
             other wing's persisted rows) into the SHARED
             :func:`palace_graph.entity_tunnels_for_wing`, with its tunnel-write
             callback routed to THIS store's ``create_tunnel(kind='entity')`` so
             the construction is shared, not re-implemented, and lands in this
             team's ``tunnels`` table;
          4. rebuilds this wing's TOPIC tunnels the SAME way — PURGES this
             wing's prior ``kind='topic'`` tunnels (and ONLY those), reads this
             team's ``wing_topics`` into the ``{wing: [topic]}`` shape, and
             re-derives via the UNCHANGED
             :func:`palace_graph.topic_tunnels_for_wing` (pure case-insensitive
             string overlap of the per-wing labels — no LLM at tunnel time),
             with the tunnel-write callback routed to THIS store's
             ``create_tunnel(kind='topic')``. The labels are agent/miner-supplied
             (``tool_add_drawer``'s ``topics`` / ``tool_diary_write``'s ``topic``
             / the miner), so topic-tunnel parity with chroma holds without an
             offline LLM.

        Explicit (``kind='explicit'``) tunnels are NEVER touched by any of the
        purges — they are user-authored. Entity and topic tunnels are purged
        INDEPENDENTLY by their own ``kind``, so the three kinds coexist and a
        rebuild of one never deletes the others.

        The recompute is NOT atomic: the hallway purge+upsert, the entity- and
        topic-tunnel purges, and the per-tunnel re-creates each run on their own
        connection, so a crash mid-rebuild can leave this wing's derived layer
        half-rebuilt. That is acceptable here — the derived layer is non-gating
        (the verbatim drawers and the entity index are the recall floor) and the
        rebuild is a pure purge+recompute over the current ``entity_occurrences``
        / ``wing_topics`` state, so it is history-independent and the next
        rebuild for this wing fully heals it.

        Returns ``{"hallways": n, "entity_tunnels": m, "topic_tunnels": k}``.
        """
        if not isinstance(wing, str) or not wing.strip():
            return {"hallways": 0, "entity_tunnels": 0, "topic_tunnels": 0}
        self._ensure()

        hallways = self.compute_hallways_for_wing(wing, min_count=min_count)

        # Purge ONLY this wing's derived entity tunnels (kind='entity' with this
        # wing as an endpoint). Explicit + topic tunnels are user/agent data and
        # are left untouched.
        self._purge_wing_entity_tunnels(wing)

        # Re-derive cross-wing entity tunnels from the full hallway set so the
        # other endpoints of a cross-wing pair are visible. Route the tunnel
        # write through THIS store so the records land in the team vault.
        all_hallways = self.list_hallways()

        def _create_entity_tunnel(
            source_wing, source_room, target_wing, target_room, label="", kind="entity"
        ):
            return self.create_tunnel(
                source_wing,
                source_room,
                target_wing,
                target_room,
                label=label,
                kind=kind,
            )

        created = entity_tunnels_for_wing(
            wing, all_hallways, create_tunnel_fn=_create_entity_tunnel
        )

        topic_created = self._rebuild_topic_tunnels_for_wing(wing)
        return {
            "hallways": len(hallways),
            "entity_tunnels": len(created),
            "topic_tunnels": len(topic_created),
        }

    def _rebuild_topic_tunnels_for_wing(self, wing: str) -> list:
        """Purge + re-derive this wing's TOPIC tunnels from ``wing_topics``.

        Purges ONLY this wing's ``kind='topic'`` tunnels (never explicit or
        entity tunnels), then re-derives via the UNCHANGED
        :func:`palace_graph.topic_tunnels_for_wing` — pure case-insensitive
        string overlap of the per-wing labels — feeding the whole team's
        ``wing_topics`` map so the OTHER endpoint of a shared-topic pair is
        visible, and routing the tunnel write to THIS store's
        ``create_tunnel(kind='topic')`` so records land in the team vault.
        Returns the topic tunnels created/refreshed for this wing.
        """
        self._purge_wing_topic_tunnels(wing)

        topics_map = self.topics_by_wing()
        if not topics_map:
            return []

        def _create_topic_tunnel(
            source_wing, source_room, target_wing, target_room, label="", kind="topic"
        ):
            return self.create_tunnel(
                source_wing,
                source_room,
                target_wing,
                target_room,
                label=label,
                kind=kind,
            )

        return topic_tunnels_for_wing(wing, topics_map, create_tunnel_fn=_create_topic_tunnel)

    def _purge_wing_topic_tunnels(self, wing: str) -> None:
        """Delete this wing's TOPIC tunnels, never explicit/entity ones.

        Matches ``kind='topic'`` AND (``source_wing`` OR ``target_wing`` equals
        this wing). ``topic_tunnels_for_wing`` canonicalizes the wing slug it
        stamps on the endpoints (``normalize_wing_name``), so the purge matches
        BOTH the raw wing and its normalized form to clear the prior rows
        regardless of which form the enqueueing write used.
        """
        norm = normalize_wing_name(wing.strip()) if wing and wing.strip() else wing
        endpoints = {wing, norm}
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table()} "
                    "WHERE kind = 'topic' "
                    "AND (source_wing = ANY(%s) OR target_wing = ANY(%s))",
                    (list(endpoints), list(endpoints)),
                )

    def _purge_wing_entity_tunnels(self, wing: str) -> None:
        """Delete this wing's derived entity tunnels, never explicit/topic ones.

        Matches ``kind='entity'`` AND (``source_wing`` OR ``target_wing`` equals
        this wing). The ``kind`` predicate is what protects user-authored explicit
        tunnels (and agent/miner topic tunnels) from a derived rebuild — only the
        server-derived entity links are purged. ``entity_tunnels_for_wing`` stamps
        a normalized endpoint slug, so — like the topic purge — match BOTH the raw
        wing and its normalized form to clear prior rows regardless of which form
        the enqueueing write used (otherwise a slug-form mismatch could leave a
        stale derived entity tunnel behind a rebuild).
        """
        norm = normalize_wing_name(wing.strip()) if wing and wing.strip() else wing
        endpoints = {wing, norm}
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table()} "
                    "WHERE kind = 'entity' "
                    "AND (source_wing = ANY(%s) OR target_wing = ANY(%s))",
                    (list(endpoints), list(endpoints)),
                )
