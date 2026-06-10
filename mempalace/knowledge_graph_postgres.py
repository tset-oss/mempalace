"""PostgreSQL-backed temporal knowledge graph (central, team-vaulted).

A faithful port of :class:`mempalace.knowledge_graph.KnowledgeGraph` (SQLite)
onto Postgres tables living in the team's vault schema
(``team_<team>.kg_entities`` / ``team_<team>.kg_triples``). The public method
surface and return shapes are identical, so it is a drop-in for the SQLite KG.

Temporal validity (``valid_from`` / ``valid_to``) is preserved exactly, including
the date-only normalization (a date-only ``valid_from`` compares as the start of
the day, a date-only ``valid_to`` as the end of the day) so legacy date-only
facts and canonical UTC datetimes interoperate.

This stays deliberately on plain SQL tables rather than Apache AGE: the temporal
semantics map cleanly to relational predicates, AGE has no native temporal
syntax, and the graph image already provisions AGE for traversal so the KG can
move to Cypher later behind this same interface without touching callers.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime
from typing import Optional

from .config import sanitize_iso_temporal
from .knowledge_graph import _temporal_end_key, _temporal_start_key

# Reuse identifier quoting + schema naming from the storage backend.
from .backends import PalaceNotFoundError
from .backends.postgres import _qi, team_schema


def _pg_temporal_start_expr(column: str) -> str:
    return (
        f"CASE WHEN length({column}) = 10 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"THEN {column} || 'T00:00:00Z' ELSE {column} END"
    )


def _pg_temporal_end_expr(column: str) -> str:
    return (
        f"CASE WHEN length({column}) = 10 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"THEN {column} || 'T23:59:59Z' ELSE {column} END"
    )


def _temporal_filter_sql(as_of: str) -> tuple[str, list]:
    """SQL + params for an as-of filter (psycopg %s placeholders)."""
    as_of_key = _temporal_start_key(as_of)
    vf = _pg_temporal_start_expr("t.valid_from")
    vt = _pg_temporal_end_expr("t.valid_to")
    return (
        f" AND (t.valid_from IS NULL OR {vf} <= %s) AND (t.valid_to IS NULL OR {vt} >= %s)",
        [as_of_key, as_of_key],
    )


def _temporal_filter_sql_named(column_prefix: str, key: str = "as_of") -> str:
    """As-of filter SQL referencing a NAMED parameter ``%(key)s``.

    Unlike :func:`_temporal_filter_sql` (which emits two positional ``%s``
    placeholders), this returns SQL that references the as-of value through a
    single named placeholder. The recursive multi-hop read references the
    same as-of value at both the anchor term and the recursive term, so a named
    placeholder lets one bound value satisfy every reference without the caller
    threading the parameter list in a fragile position-dependent order. The
    edge-validity test is applied once per hop, so a path whose edges have
    disjoint validity windows is traversable only when every edge is itself
    valid at the as-of instant.
    """
    vf = _pg_temporal_start_expr(f"{column_prefix}.valid_from")
    vt = _pg_temporal_end_expr(f"{column_prefix}.valid_to")
    return (
        f" AND ({column_prefix}.valid_from IS NULL OR {vf} <= %({key})s) "
        f"AND ({column_prefix}.valid_to IS NULL OR {vt} >= %({key})s)"
    )


class PostgresKnowledgeGraph:
    """Temporal KG stored in a team vault's Postgres schema."""

    def __init__(self, backend, team: Optional[str] = None):
        self._backend = backend
        self._team = team
        self._schema = team_schema(team)
        self._lock = threading.Lock()
        self._ensured = False

    # -- schema -----------------------------------------------------------
    def _entities(self) -> str:
        return f"{_qi(self._schema)}.{_qi('kg_entities')}"

    def _triples(self) -> str:
        return f"{_qi(self._schema)}.{_qi('kg_triples')}"

    def _ensure(self, *, create: bool = True) -> None:
        """Ensure the team vault's KG tables exist, gated per-call on ``create``.

        Mirrors ``PostgresBackend._ensure_collection``'s discipline: ``create=
        False`` on a never-materialized schema RAISES ``PalaceNotFoundError`` and
        never runs ``CREATE SCHEMA`` — reads must degrade to the central empty
        shape, not silently provision a tenant vault.

        ``create`` is a PER-CALL flag and is NEVER stored on the instance. This
        handle is cached per-team in ``mcp_server._kg_by_path`` and shared across
        read AND write requests, so a stored mode would alias: a read that arrived
        first could pin ``create=False`` and break the writer (PM#8). ``_ensured``
        may fast-path ONLY the materialized case — once the schema is known to
        exist, both modes are satisfied. While ``_ensured`` is False, a ``create=
        False`` call probes existence on EVERY call and raises if absent.
        """
        if self._ensured:
            # Schema is known materialized; both create=True and create=False are
            # satisfied. Do NOT branch on ``create`` here — see the docstring's
            # cache-aliasing note.
            return
        with self._lock:
            if self._ensured:
                return
            if not create:
                with self._backend._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                            (self._schema,),
                        )
                        schema_exists = cur.fetchone() is not None
                if not schema_exists:
                    # Never create, never set _ensured: an unmaterialized schema
                    # under create=False must raise on EVERY call.
                    raise PalaceNotFoundError(f"team vault {self._schema!r} does not exist")
                # Schema present -> its tables were created by whoever materialized
                # it (the only writer of this schema is this class's create=True
                # branch, which always creates the tables atomically below). Mark
                # ensured and return without issuing any DDL.
                self._ensured = True
                return
            # Backstop (PM#7): never CREATE SCHEMA off a None-derived/empty team.
            # team_schema(None) -> "team_default"; an empty slug would yield a bare
            # "team_" which must never be materialized as a vault.
            slug = self._schema[len("team_") :] if self._schema.startswith("team_") else ""
            if not slug:
                raise ValueError("no team resolved for KG schema materialization")
            with self._backend._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_qi(self._schema)}")
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._entities()} ("
                        "  id text PRIMARY KEY,"
                        "  name text NOT NULL,"
                        "  type text DEFAULT 'unknown',"
                        "  properties jsonb DEFAULT '{}'::jsonb,"
                        "  created_at timestamptz DEFAULT now()"
                        ")"
                    )
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._triples()} ("
                        "  id text PRIMARY KEY,"
                        "  subject text NOT NULL,"
                        "  predicate text NOT NULL,"
                        "  object text NOT NULL,"
                        "  valid_from text,"
                        "  valid_to text,"
                        "  confidence real DEFAULT 1.0,"
                        "  source_closet text,"
                        "  source_file text,"
                        "  source_drawer_id text,"
                        "  adapter_name text,"
                        "  extracted_at timestamptz DEFAULT now()"
                        ")"
                    )
                    for col in ("subject", "object", "predicate"):
                        cur.execute(
                            f"CREATE INDEX IF NOT EXISTS {_qi('kg_triples_' + col)} "
                            f"ON {self._triples()} ({col})"
                        )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi('kg_triples_valid')} "
                        f"ON {self._triples()} (valid_from, valid_to)"
                    )
            self._ensured = True

    def _entity_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    def close(self) -> None:
        # Connections belong to the shared backend pool; nothing to close here.
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- writes -----------------------------------------------------------
    def add_entity(self, name: str, entity_type: str = "unknown", properties: dict = None):
        self._ensure()
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._entities()} (id, name, type, properties) "
                    "VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, "
                    "type=EXCLUDED.type, properties=EXCLUDED.properties",
                    (eid, name, entity_type, props),
                )
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        valid_to: str = None,
        confidence: float = 1.0,
        source_closet: str = None,
        source_file: str = None,
        source_drawer_id: str = None,
        adapter_name: str = None,
        *,
        create: bool = True,
    ):
        self._ensure(create=create)
        valid_from = sanitize_iso_temporal(valid_from, "valid_from")
        valid_to = sanitize_iso_temporal(valid_to, "valid_to")
        if (
            valid_from is not None
            and valid_to is not None
            and _temporal_end_key(valid_to) < _temporal_start_key(valid_from)
        ):
            raise ValueError(
                f"valid_to={valid_to!r} is before valid_from={valid_from!r}; "
                "an inverted interval would be invisible to every KG query"
            )

        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._entities()} (id, name) VALUES (%s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (sub_id, subject),
                )
                cur.execute(
                    f"INSERT INTO {self._entities()} (id, name) VALUES (%s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (obj_id, obj),
                )
                cur.execute(
                    f"SELECT id FROM {self._triples()} "
                    "WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
                    (sub_id, pred, obj_id),
                )
                existing = cur.fetchone()
                if existing:
                    return existing[0]
                triple_id = (
                    f"t_{sub_id}_{pred}_{obj_id}_"
                    f"{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
                )
                cur.execute(
                    f"INSERT INTO {self._triples()} ("
                    "  id, subject, predicate, object, valid_from, valid_to,"
                    "  confidence, source_closet, source_file, source_drawer_id, adapter_name"
                    ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        triple_id,
                        sub_id,
                        pred,
                        obj_id,
                        valid_from,
                        valid_to,
                        confidence,
                        source_closet,
                        source_file,
                        source_drawer_id,
                        adapter_name,
                    ),
                )
                return triple_id

    def invalidate(
        self, subject: str, predicate: str, obj: str, ended: str = None, *, create: bool = True
    ):
        self._ensure(create=create)
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        ended = sanitize_iso_temporal(ended or date.today().isoformat(), "ended")

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, valid_from FROM {self._triples()} "
                    "WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
                    (sub_id, pred, obj_id),
                )
                for row in cur.fetchall():
                    valid_from = row[1]
                    if valid_from is not None and _temporal_end_key(ended) < _temporal_start_key(
                        valid_from
                    ):
                        raise ValueError(
                            f"valid_to={ended!r} is before valid_from={valid_from!r}; "
                            "an inverted interval would be invisible to every KG query"
                        )
                cur.execute(
                    f"UPDATE {self._triples()} SET valid_to=%s "
                    "WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
                    (ended, sub_id, pred, obj_id),
                )

    # -- queries ----------------------------------------------------------
    def query_entity(
        self, name: str, as_of: str = None, direction: str = "outgoing", *, create: bool = True
    ):
        self._ensure(create=create)
        as_of = sanitize_iso_temporal(as_of, "as_of")
        eid = self._entity_id(name)
        results = []
        temporal_sql, temporal_params = ("", [])
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(as_of)

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                if direction in ("outgoing", "both"):
                    cur.execute(
                        f"SELECT t.predicate, t.valid_from, t.valid_to, t.confidence, "
                        f"t.source_closet, e.name AS obj_name "
                        f"FROM {self._triples()} t JOIN {self._entities()} e ON t.object = e.id "
                        f"WHERE t.subject = %s" + temporal_sql,
                        [eid] + temporal_params,
                    )
                    for r in cur.fetchall():
                        results.append(
                            {
                                "direction": "outgoing",
                                "subject": name,
                                "predicate": r[0],
                                "object": r[5],
                                "valid_from": r[1],
                                "valid_to": r[2],
                                "confidence": r[3],
                                "source_closet": r[4],
                                "current": r[2] is None,
                            }
                        )
                if direction in ("incoming", "both"):
                    cur.execute(
                        f"SELECT t.predicate, t.valid_from, t.valid_to, t.confidence, "
                        f"t.source_closet, e.name AS sub_name "
                        f"FROM {self._triples()} t JOIN {self._entities()} e ON t.subject = e.id "
                        f"WHERE t.object = %s" + temporal_sql,
                        [eid] + temporal_params,
                    )
                    for r in cur.fetchall():
                        results.append(
                            {
                                "direction": "incoming",
                                "subject": r[5],
                                "predicate": r[0],
                                "object": name,
                                "valid_from": r[1],
                                "valid_to": r[2],
                                "confidence": r[3],
                                "source_closet": r[4],
                                "current": r[2] is None,
                            }
                        )
        return results

    def neighbors(
        self,
        name: str,
        depth: int = 2,
        direction: str = "outgoing",
        as_of: str = None,
        target: str = None,
        predicates: Optional[list] = None,
        limit: int = 500,
        expand_cap: int = 50,
        *,
        create: bool = True,
    ):
        """Bounded multi-hop neighborhood walk over the temporal triples.

        Starting at ``name``, walk up to ``depth`` hops along edges in the
        requested ``direction`` ("outgoing", "incoming", or "both"), returning
        one row per reached edge with its hop number, the edge endpoints as
        NAMES (joined from ``kg_entities``), and the triple-id path taken to
        reach it.

        Bounding (both bounds are required — a final row LIMIT alone does NOT
        constrain the recursive frontier):
        * ``depth`` is clamped to at most 4 hops.
        * ``expand_cap`` caps how many edges are followed out of each reached
          node, bounding the per-level frontier so a dense hub cannot blow up
          the working set.
        * ``limit`` caps the total rows returned.
        The returned dict carries ``truncated=True`` when EITHER cap clips the
        result (a node had more outgoing edges than ``expand_cap``, or the total
        row count reached ``limit``).

        Temporal semantics are point-in-time PER EDGE: when ``as_of`` is given,
        every hop's edge must independently satisfy the as-of validity test, so
        a path whose edges have disjoint validity windows is traversable only
        when each edge is itself valid at the single ``as_of`` instant.

        ``target`` restricts the result to paths that reach the named entity.
        ``predicates`` restricts every hop to the given predicate names.

        This is a read-only walk; it issues no DDL and never mutates the
        relational source of truth.
        """
        self._ensure(create=create)
        as_of = sanitize_iso_temporal(as_of, "as_of")
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(
                f"direction={direction!r} must be one of 'outgoing', 'incoming', or 'both'"
            )
        depth = max(1, min(int(depth), 4))
        expand_cap = max(1, int(expand_cap))
        limit = max(1, int(limit))

        start_id = self._entity_id(name)
        params: dict = {
            "start_id": start_id,
            "max_depth": depth,
            # Fetch one extra edge per node so a node whose real out-degree
            # exceeds ``expand_cap`` is detected (the surplus row is dropped
            # from the traversal but flips the truncation flag).
            "expand_probe": expand_cap + 1,
            "row_limit": limit,
        }
        if as_of:
            params["as_of"] = _temporal_start_key(as_of)
        temporal_sql = _temporal_filter_sql_named("e") if as_of else ""

        # Predicate typed-filter (named list param), applied to EVERY hop.
        pred_sql = ""
        if predicates:
            params["predicates"] = [p.lower().replace(" ", "_") for p in predicates]
            pred_sql = " AND e.predicate = ANY(%(predicates)s)"

        target_sql = ""
        if target:
            params["target_id"] = self._entity_id(target)
            target_sql = " WHERE w.endpoint = %(target_id)s"

        triples = self._triples()

        # One edge-expansion fragment, parameterized by the frontier-id and the
        # visited-path expressions so the anchor (start entity, empty path) and
        # the recursive term (frontier node, accumulated path) share identical
        # as-of, predicate, cycle-guard, and per-node-cap logic. ``next_id`` is
        # the entity an edge leads to in the walk direction: outgoing follows
        # subject -> object, incoming follows object -> subject. "both" is the
        # UNION ALL of the two; the cycle guard (the triple-id ``path``) is
        # shared, so an A->B->A cycle is rejected regardless of which side each
        # hop traverses. ``LIMIT %(expand_probe)s`` (= expand_cap + 1) bounds
        # the per-level frontier while still revealing a node that overflowed.
        def _expansion(frontier_expr: str, path_expr: str) -> str:
            def _leg(next_col: str, match_col: str, edge_dir: str) -> str:
                return (
                    "SELECT e.id AS triple_id, "
                    f"e.{next_col} AS next_id, e.predicate, "
                    "e.valid_from, e.valid_to, e.confidence, e.source_closet, "
                    "e.subject AS edge_subject, e.object AS edge_object, "
                    f"'{edge_dir}' AS edge_direction "
                    f"FROM {triples} e "
                    f"WHERE e.{match_col} = {frontier_expr} "
                    f"AND NOT (e.id = ANY({path_expr}))"
                    f"{temporal_sql}{pred_sql}"
                )

            if direction == "outgoing":
                body = _leg("object", "subject", "outgoing")
            elif direction == "incoming":
                body = _leg("subject", "object", "incoming")
            else:  # both
                body = (
                    _leg("object", "subject", "outgoing")
                    + " UNION ALL "
                    + _leg("subject", "object", "incoming")
                )
            return (
                "SELECT * FROM ( " + body + " ) step ORDER BY step.triple_id LIMIT %(expand_probe)s"
            )

        anchor = _expansion("%(start_id)s", "ARRAY[]::text[]")
        recursive = _expansion("w.endpoint", "w.path")

        # The walk carries the visited triple-id ``path`` (cycle guard) and the
        # ``hop`` counter (depth clamp). The outer query joins ``kg_entities``
        # twice to surface both edge endpoints as NAMES (mirroring
        # ``query_entity``'s edge-name join). ``LIMIT row_limit + 1`` lets the
        # caller see whether the total-row cap clipped the result.
        sql = (
            "WITH RECURSIVE walk AS ( "
            "  SELECT s.triple_id, s.next_id AS endpoint, s.predicate, "
            "    s.valid_from, s.valid_to, s.confidence, s.source_closet, "
            "    s.edge_subject, s.edge_object, s.edge_direction, "
            "    1 AS hop, ARRAY[s.triple_id] AS path "
            "  FROM ( " + anchor + " ) s "
            "  UNION ALL "
            "  SELECT n.triple_id, n.next_id AS endpoint, n.predicate, "
            "    n.valid_from, n.valid_to, n.confidence, n.source_closet, "
            "    n.edge_subject, n.edge_object, n.edge_direction, "
            "    w.hop + 1 AS hop, w.path || n.triple_id AS path "
            "  FROM walk w "
            "  CROSS JOIN LATERAL ( " + recursive + " ) n "
            "  WHERE w.hop < %(max_depth)s "
            ") "
            "SELECT w.triple_id, w.hop, w.predicate, w.valid_from, w.valid_to, "
            "  w.confidence, w.source_closet, w.edge_direction, w.path, "
            "  subj.name AS subject_name, obj.name AS object_name "
            "FROM walk w "
            f"JOIN {self._entities()} subj ON w.edge_subject = subj.id "
            f"JOIN {self._entities()} obj ON w.edge_object = obj.id "
            + target_sql
            + " ORDER BY w.hop, w.triple_id "
            "LIMIT %(row_limit)s + 1"
        )

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                fetched = cur.fetchall()
                # A per-node out-degree above ``expand_cap`` clips the frontier:
                # detected by any single frontier node yielding the probe count
                # (expand_cap + 1) of edges within the depth bound.
                cur.execute(
                    "WITH RECURSIVE walk AS ( "
                    "  SELECT s.triple_id, s.next_id AS endpoint, "
                    "    1 AS hop, ARRAY[s.triple_id] AS path, "
                    "    (SELECT count(*) FROM ( " + anchor + " ) ac) AS deg "
                    "  FROM ( " + anchor + " ) s "
                    "  UNION ALL "
                    "  SELECT n.triple_id, n.next_id AS endpoint, "
                    "    w.hop + 1 AS hop, w.path || n.triple_id AS path, "
                    "    (SELECT count(*) FROM ( " + recursive + " ) rc) AS deg "
                    "  FROM walk w "
                    "  CROSS JOIN LATERAL ( " + recursive + " ) n "
                    "  WHERE w.hop < %(max_depth)s "
                    ") "
                    "SELECT bool_or(deg >= %(expand_probe)s) FROM walk",
                    params,
                )
                clip_row = cur.fetchone()
                expand_clipped = bool(clip_row and clip_row[0])

        truncated = len(fetched) > limit or expand_clipped
        rows = []
        for r in fetched[:limit]:
            rows.append(
                {
                    "hop": r[1],
                    "direction": r[7],
                    "subject": r[9],
                    "predicate": r[2],
                    "object": r[10],
                    "valid_from": r[3],
                    "valid_to": r[4],
                    "confidence": r[5],
                    "source_closet": r[6],
                    "current": r[4] is None,
                    "path": list(r[8]),
                }
            )

        return {
            "neighbors": rows,
            "truncated": truncated,
            "depth": depth,
        }

    def query_relationship(self, predicate: str, as_of: str = None):
        self._ensure()
        as_of = sanitize_iso_temporal(as_of, "as_of")
        pred = predicate.lower().replace(" ", "_")
        sql = (
            f"SELECT t.valid_from, t.valid_to, s.name AS sub_name, o.name AS obj_name "
            f"FROM {self._triples()} t "
            f"JOIN {self._entities()} s ON t.subject = s.id "
            f"JOIN {self._entities()} o ON t.object = o.id "
            f"WHERE t.predicate = %s"
        )
        params = [pred]
        if as_of:
            tsql, tparams = _temporal_filter_sql(as_of)
            sql += tsql
            params.extend(tparams)
        results = []
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for r in cur.fetchall():
                    results.append(
                        {
                            "subject": r[2],
                            "predicate": pred,
                            "object": r[3],
                            "valid_from": r[0],
                            "valid_to": r[1],
                            "current": r[1] is None,
                        }
                    )
        return results

    def timeline(self, entity_name: str = None, *, create: bool = True):
        self._ensure(create=create)
        base = (
            f"SELECT t.predicate, t.valid_from, t.valid_to, s.name AS sub_name, o.name AS obj_name "
            f"FROM {self._triples()} t "
            f"JOIN {self._entities()} s ON t.subject = s.id "
            f"JOIN {self._entities()} o ON t.object = o.id "
        )
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                if entity_name:
                    eid = self._entity_id(entity_name)
                    cur.execute(
                        base + "WHERE (t.subject = %s OR t.object = %s) "
                        "ORDER BY t.valid_from ASC NULLS LAST LIMIT 100",
                        (eid, eid),
                    )
                else:
                    cur.execute(base + "ORDER BY t.valid_from ASC NULLS LAST LIMIT 100")
                rows = cur.fetchall()
        return [
            {
                "subject": r[3],
                "predicate": r[0],
                "object": r[4],
                "valid_from": r[1],
                "valid_to": r[2],
                "current": r[2] is None,
            }
            for r in rows
        ]

    def stats(self, *, create: bool = True):
        self._ensure(create=create)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._entities()}")
                entities = int(cur.fetchone()[0])
                cur.execute(f"SELECT COUNT(*) FROM {self._triples()}")
                triples = int(cur.fetchone()[0])
                cur.execute(f"SELECT COUNT(*) FROM {self._triples()} WHERE valid_to IS NULL")
                current = int(cur.fetchone()[0])
                cur.execute(f"SELECT DISTINCT predicate FROM {self._triples()} ORDER BY predicate")
                predicates = [r[0] for r in cur.fetchall()]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": triples - current,
            "relationship_types": predicates,
        }

    # Note: the SQLite KG also defines a ``seed_from_entity_facts`` helper, but
    # nothing in the package calls it, so it is intentionally not re-ported here
    # to avoid carrying duplicated dead code. Re-add it behind this interface if
    # a seeding path is introduced.
