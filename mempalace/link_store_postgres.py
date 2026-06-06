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

import threading
from datetime import datetime, timezone
from typing import Optional

from .backends.postgres import _qi, team_schema
from .dynamics import initialize_dynamics_fields
from .link_store import LinkStore
from .palace_graph import _canonical_tunnel_id, _require_name


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

    # -- derived hallways (a later story) ---------------------------------
    def compute_hallways_for_wing(self, wing: str, col=None, min_count: int = 2) -> list[dict]:
        """Not yet implemented on Postgres — the derived link layer lands later.

        Derived within-wing entity hallways are computed server-side from the
        vaulted entity occurrences in a later story. No production consumer
        routes hallway reads through this seam yet, so a clear placeholder is
        safe and deliberate (it is not a silent no-op).
        """
        raise NotImplementedError("derived hallways land in a later story")

    def list_hallways(self, wing: Optional[str] = None) -> list[dict]:
        """Not yet implemented on Postgres — the derived link layer lands later.

        See :meth:`compute_hallways_for_wing`.
        """
        raise NotImplementedError("derived hallways land in a later story")
