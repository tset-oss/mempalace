"""Slice 4 — read-path non-materialization tests (T-RD).

Pins the invariant that tool_entities, tool_kg_query, tool_kg_neighbors,
tool_kg_timeline, and tool_kg_stats NEVER create a vault schema as a side
effect of a read on the postgres backend.

For each tool:
  - Call it with _active_team_var set to a fresh uuid team that has never
    been written → assert the empty SUCCESS shape (no exception, no "error"
    key, empty_vault=True) AND NEG-DB: schema absent from
    information_schema.schemata afterward.
  - One positive control (kg_query): materialize a team via tool_kg_add, then
    kg_query returns the real fact — proving create=False reads work on a
    materialized vault.

Every NEG-DB case evicts the in-process caches before probing so a stale
handle cannot mask the ABSENT assertion.

Skipped when psycopg is absent or the DB is unreachable (same as the other
postgres integration tests).
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

import mempalace.mcp_server as m  # noqa: E402
from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.config import MempalaceConfig  # noqa: E402


def _dsn() -> str:
    return (
        os.environ.get("MEMPALACE_TEST_PG_URL")
        or os.environ.get("MEMPALACE_DATABASE_URL")
        or "postgresql://mempalace:mempalace@localhost:5432/mempalace"
    )


def _reachable(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(_dsn()),
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)


@pytest.fixture
def backend():
    be = PostgresBackend(dsn=_dsn())
    yield be
    be.close()


def _fresh_team() -> str:
    return "rd_" + uuid.uuid4().hex[:12]


def _pop_caches(team: str) -> None:
    """Evict all per-team cached handles from the in-process caches."""
    m._kg_by_path.pop(f"pgkg::{team}", None)
    m._entity_index_by_team.pop(f"pgentidx::{team}", None)
    m._link_store_by_team.pop(f"pglink::{team}", None)
    m._team_facts_by_team.pop(f"pgfacts::{team}", None)


def _schema_absent(team: str) -> bool:
    """Return True iff the postgres vault schema for *team* does NOT exist."""
    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = %s",
                (schema,),
            )
            return cur.fetchone()[0] == 0


def _drop(team: str) -> None:
    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.commit()


@pytest.fixture
def server_pg(monkeypatch, backend):
    """Put the server in postgres mode with a deterministic embedder."""
    import hashlib

    def _fake_embed(texts):
        out = []
        for t in texts:
            v = [0.0] * 384
            for i, b in enumerate(hashlib.sha256((t or "").encode()).digest()):
                v[i] = b / 255.0
            out.append(v)
        return out

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setattr(m, "_config", MempalaceConfig())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    prev_embedder = getattr(backend, "_embedder", None)
    backend._embedder = _fake_embed
    token = m._active_team_var.set(None)
    yield backend
    m._active_team_var.reset(token)
    backend._embedder = prev_embedder


# ─────────────────────────────────────────────────────────────────────────────
# T-RD-1: tool_kg_query — empty vault returns empty success shape, no schema
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_query_empty_vault_no_materialize(server_pg, monkeypatch):
    """tool_kg_query on a never-written vault returns empty shape; schema absent."""
    team = _fresh_team()
    _pop_caches(team)
    token = m._active_team_var.set(team)
    try:
        result = m.tool_kg_query("SomeEntity")
    finally:
        m._active_team_var.reset(token)

    # Empty success shape — no error key, no exception.
    assert "error" not in result, f"Unexpected error: {result}"
    assert result.get("facts") == [], f"Expected empty facts list: {result}"
    assert result.get("count") == 0, f"Expected count=0: {result}"
    assert result.get("empty_vault") is True, f"Expected empty_vault=True: {result}"

    # NEG-DB: schema must not have been created.
    _pop_caches(team)
    assert _schema_absent(team), f"Schema for {team!r} was created by a read — leak!"


# ─────────────────────────────────────────────────────────────────────────────
# T-RD-2: tool_kg_neighbors — empty vault returns empty success shape, no schema
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_neighbors_empty_vault_no_materialize(server_pg, monkeypatch):
    """tool_kg_neighbors on a never-written vault returns empty shape; schema absent."""
    team = _fresh_team()
    _pop_caches(team)
    token = m._active_team_var.set(team)
    try:
        result = m.tool_kg_neighbors("SomeEntity", depth=2)
    finally:
        m._active_team_var.reset(token)

    assert "error" not in result, f"Unexpected error: {result}"
    assert result.get("neighbors") == [], f"Expected empty neighbors: {result}"
    assert result.get("count") == 0, f"Expected count=0: {result}"
    assert result.get("truncated") is False, f"Expected truncated=False: {result}"
    assert result.get("empty_vault") is True, f"Expected empty_vault=True: {result}"

    _pop_caches(team)
    assert _schema_absent(team), f"Schema for {team!r} was created by a read — leak!"


# ─────────────────────────────────────────────────────────────────────────────
# T-RD-3: tool_kg_timeline — empty vault returns empty success shape, no schema
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_timeline_empty_vault_no_materialize(server_pg, monkeypatch):
    """tool_kg_timeline on a never-written vault returns empty shape; schema absent."""
    team = _fresh_team()
    _pop_caches(team)
    token = m._active_team_var.set(team)
    try:
        result = m.tool_kg_timeline()
    finally:
        m._active_team_var.reset(token)

    assert "error" not in result, f"Unexpected error: {result}"
    assert result.get("timeline") == [], f"Expected empty timeline: {result}"
    assert result.get("count") == 0, f"Expected count=0: {result}"
    assert result.get("empty_vault") is True, f"Expected empty_vault=True: {result}"

    _pop_caches(team)
    assert _schema_absent(team), f"Schema for {team!r} was created by a read — leak!"


# ─────────────────────────────────────────────────────────────────────────────
# T-RD-4: tool_kg_stats — empty vault returns empty success shape, no schema
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_stats_empty_vault_no_materialize(server_pg, monkeypatch):
    """tool_kg_stats on a never-written vault returns empty shape; schema absent."""
    team = _fresh_team()
    _pop_caches(team)
    token = m._active_team_var.set(team)
    try:
        result = m.tool_kg_stats()
    finally:
        m._active_team_var.reset(token)

    assert "error" not in result, f"Unexpected error: {result}"
    assert result.get("entities") == 0, f"Expected entities=0: {result}"
    assert result.get("triples") == 0, f"Expected triples=0: {result}"
    assert result.get("current_facts") == 0, f"Expected current_facts=0: {result}"
    assert result.get("expired_facts") == 0, f"Expected expired_facts=0: {result}"
    assert result.get("relationship_types") == [], f"Expected empty rel types: {result}"
    assert result.get("empty_vault") is True, f"Expected empty_vault=True: {result}"

    _pop_caches(team)
    assert _schema_absent(team), f"Schema for {team!r} was created by a read — leak!"


# ─────────────────────────────────────────────────────────────────────────────
# T-RD-5: tool_entities — empty vault returns empty success shape, no schema
# ─────────────────────────────────────────────────────────────────────────────


def test_entities_empty_vault_no_materialize(server_pg, monkeypatch):
    """tool_entities on a never-written vault returns empty shape; schema absent.

    Covers both the overview form (no entity arg) and the entity-lookup form.
    """
    team = _fresh_team()
    _pop_caches(team)
    token = m._active_team_var.set(team)
    try:
        overview = m.tool_entities()
        lookup = m.tool_entities(entity="SomeEntity")
    finally:
        m._active_team_var.reset(token)

    # Overview form
    assert "error" not in overview, f"Unexpected error in overview: {overview}"
    assert overview.get("entities") == [], f"Expected empty entities: {overview}"
    assert overview.get("empty_vault") is True, f"Expected empty_vault=True: {overview}"
    assert overview.get("vault") == team, f"Expected vault={team!r}: {overview}"

    # Entity-lookup form
    assert "error" not in lookup, f"Unexpected error in lookup: {lookup}"
    assert lookup.get("entities") == [], f"Expected empty entities in lookup: {lookup}"
    assert lookup.get("empty_vault") is True, f"Expected empty_vault=True in lookup: {lookup}"

    _pop_caches(team)
    assert _schema_absent(team), f"Schema for {team!r} was created by a read — leak!"


# ─────────────────────────────────────────────────────────────────────────────
# Positive control: kg_query on a MATERIALIZED vault returns real data
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_query_works_after_materialization(server_pg, monkeypatch):
    """After a write materializes the vault, kg_query returns real facts.

    Confirms that create=False reads work correctly on a vault that exists —
    no regression from the empty-vault path change.
    """
    team = _fresh_team()
    _pop_caches(team)
    try:
        # Write to materialize the vault.
        token = m._active_team_var.set(team)
        try:
            write_result = m.tool_kg_add(subject="Alice", predicate="knows", object="Bob")
        finally:
            m._active_team_var.reset(token)

        assert write_result.get("success") is True, f"Write failed: {write_result}"

        # Read on the same team now returns real data (vault materialized).
        _pop_caches(team)
        token = m._active_team_var.set(team)
        try:
            query_result = m.tool_kg_query("Alice")
        finally:
            m._active_team_var.reset(token)

        assert "error" not in query_result, f"Query error: {query_result}"
        assert query_result.get("count", 0) >= 1, f"Expected >=1 fact: {query_result}"
        assert query_result.get("empty_vault") is not True, (
            f"Materialized vault must not return empty_vault=True: {query_result}"
        )
    finally:
        _pop_caches(team)
        _drop(team)
