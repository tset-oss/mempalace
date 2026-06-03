"""Conformance tests for the Postgres-backed temporal knowledge graph (G004).

Mirrors the behavioural contract of the SQLite KnowledgeGraph: temporal
add/query/invalidate, directionality, relationship + timeline queries, stats,
dedup, inverted-interval rejection, and per-team isolation. Skipped when no
Postgres is reachable.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.knowledge_graph_postgres import PostgresKnowledgeGraph  # noqa: E402


def _dsn():
    return (
        os.environ.get("MEMPALACE_TEST_PG_URL")
        or os.environ.get("MEMPALACE_DATABASE_URL")
        or "postgresql://mempalace:mempalace@localhost:5432/mempalace"
    )


def _reachable():
    try:
        with psycopg.connect(_dsn(), connect_timeout=3) as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="no reachable Postgres")


@pytest.fixture(scope="module")
def backend():
    be = PostgresBackend(dsn=_dsn())
    yield be
    be.close()


@pytest.fixture()
def kg(backend):
    team = "t" + uuid.uuid4().hex[:10]
    graph = PostgresKnowledgeGraph(backend, team=team)
    yield graph
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
        conn.commit()


def test_add_and_query_entity(kg):
    kg.add_triple("Max", "child_of", "Alice", valid_from="2015-04-01")
    kg.add_triple("Max", "loves", "chess", valid_from="2025-10-01")
    out = kg.query_entity("Max")  # outgoing
    preds = {(r["predicate"], r["object"]) for r in out}
    assert ("child_of", "Alice") in preds
    assert ("loves", "chess") in preds
    assert all(r["current"] for r in out)


def test_query_direction_incoming_and_both(kg):
    kg.add_triple("Max", "child_of", "Alice")
    incoming = kg.query_entity("Alice", direction="incoming")
    assert any(r["subject"] == "Max" and r["predicate"] == "child_of" for r in incoming)
    both = kg.query_entity("Alice", direction="both")
    assert len(both) >= 1


def test_temporal_as_of(kg):
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01", valid_to="2025-06-01")
    kg.add_triple("Max", "does", "chess", valid_from="2025-07-01")
    # In March 2025 only swimming was valid.
    as_mar = {r["object"] for r in kg.query_entity("Max", as_of="2025-03-15")}
    assert "swimming" in as_mar
    assert "chess" not in as_mar
    # In August 2025 only chess.
    as_aug = {r["object"] for r in kg.query_entity("Max", as_of="2025-08-15")}
    assert "chess" in as_aug
    assert "swimming" not in as_aug


def test_invalidate(kg):
    kg.add_triple("Max", "has_issue", "injury", valid_from="2026-01-01")
    kg.invalidate("Max", "has_issue", "injury", ended="2026-02-15")
    # No longer current as of March.
    current = {r["object"] for r in kg.query_entity("Max", as_of="2026-03-01")}
    assert "injury" not in current
    # But still visible in January.
    jan = {r["object"] for r in kg.query_entity("Max", as_of="2026-01-15")}
    assert "injury" in jan


def test_add_triple_dedup(kg):
    a = kg.add_triple("Max", "loves", "chess")
    b = kg.add_triple("Max", "loves", "chess")
    assert a == b  # identical still-valid triple returns the same id


def test_inverted_interval_rejected(kg):
    with pytest.raises(ValueError):
        kg.add_triple("X", "y", "Z", valid_from="2026-05-01", valid_to="2026-01-01")


def test_query_relationship(kg):
    kg.add_triple("Max", "child_of", "Alice")
    kg.add_triple("Sam", "child_of", "Alice")
    rows = kg.query_relationship("child_of")
    subs = {r["subject"] for r in rows}
    assert {"Max", "Sam"} <= subs


def test_timeline(kg):
    kg.add_triple("Max", "born", "Hospital", valid_from="2015-04-01")
    kg.add_triple("Max", "started", "School", valid_from="2021-09-01")
    tl = kg.timeline("Max")
    froms = [r["valid_from"] for r in tl if r["valid_from"]]
    assert froms == sorted(froms)  # chronological


def test_stats(kg):
    kg.add_triple("Max", "loves", "chess")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01", valid_to="2025-06-01")
    s = kg.stats()
    assert s["triples"] == 2
    assert s["current_facts"] == 1
    assert s["expired_facts"] == 1
    assert "loves" in s["relationship_types"]


def test_mcp_get_kg_routes_to_postgres(monkeypatch):
    """The MCP server's _get_kg returns a PostgresKnowledgeGraph for the
    configured team when the postgres backend is selected, and round-trips."""
    import mempalace.mcp_server as mcp
    from mempalace.config import MempalaceConfig

    team = "t" + uuid.uuid4().hex[:10]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", team)
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setattr(mcp, "_config", MempalaceConfig())
    mcp._kg_by_path.clear()
    try:
        kg = mcp._get_kg()
        assert isinstance(kg, PostgresKnowledgeGraph)
        kg.add_triple("Max", "loves", "chess")
        res = kg.query_entity("Max")
        assert any(r["object"] == "chess" for r in res)
    finally:
        mcp._kg_by_path.clear()
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
            conn.commit()


def test_team_isolation(backend):
    a = "t" + uuid.uuid4().hex[:10]
    b = "t" + uuid.uuid4().hex[:10]
    kga = PostgresKnowledgeGraph(backend, team=a)
    kgb = PostgresKnowledgeGraph(backend, team=b)
    try:
        kga.add_triple("Max", "loves", "chess")
        assert kgb.stats()["triples"] == 0
        assert kga.stats()["triples"] == 1
    finally:
        for t in (a, b):
            with psycopg.connect(_dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
                conn.commit()
