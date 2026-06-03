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
        f" AND (t.valid_from IS NULL OR {vf} <= %s) "
        f"AND (t.valid_to IS NULL OR {vt} >= %s)",
        [as_of_key, as_of_key],
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
    ):
        self._ensure()
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
                        triple_id, sub_id, pred, obj_id, valid_from, valid_to,
                        confidence, source_closet, source_file, source_drawer_id, adapter_name,
                    ),
                )
                return triple_id

    def invalidate(self, subject: str, predicate: str, obj: str, ended: str = None):
        self._ensure()
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
    def query_entity(self, name: str, as_of: str = None, direction: str = "outgoing"):
        self._ensure()
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

    def timeline(self, entity_name: str = None):
        self._ensure()
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

    def stats(self):
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._entities()}")
                entities = int(cur.fetchone()[0])
                cur.execute(f"SELECT COUNT(*) FROM {self._triples()}")
                triples = int(cur.fetchone()[0])
                cur.execute(f"SELECT COUNT(*) FROM {self._triples()} WHERE valid_to IS NULL")
                current = int(cur.fetchone()[0])
                cur.execute(
                    f"SELECT DISTINCT predicate FROM {self._triples()} ORDER BY predicate"
                )
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
