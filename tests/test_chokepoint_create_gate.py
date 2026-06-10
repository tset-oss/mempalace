"""Slice 0 — the CREATE SCHEMA chokepoint: per-call ``create`` discipline.

The two unconditional ``CREATE SCHEMA`` seams (``PostgresKnowledgeGraph._ensure``
and ``PostgresEntityIndex._ensure``) are the only places a read could silently
materialize a tenant vault — the leak class the rest of the suite never caught.
This file pins the seam invariant directly, mirroring the blessed
``backends/postgres.py`` ``_ensure_collection`` pattern:

* ``create=False`` on a never-materialized schema RAISES ``PalaceNotFoundError``
  and the schema stays ABSENT from ``information_schema.schemata`` afterward
  (asserted by a DIRECT psycopg probe, not a re-call) — the only shape that
  catches a raise-AFTER-materialize regression.
* ``create=True`` materializes; a fresh handle with ``create=False`` then
  succeeds (schema exists).
* The PER-CALL flag is never stored on the cached handle: on ONE instance,
  ``create=False`` raises, then ``create=True`` materializes, then
  ``create=False`` fast-paths — and the inverse holds (PM#8 cache-aliasing).
* The non-None-team backstop fires before any DDL (PM#7).

Every NEG-DB case uses a FRESH uuid team slug AND evicts the in-process caches
(``_kg_by_path`` / ``_entity_index_by_team``) before probing the DB, so a stale
handle can never mask the ABSENT assertion (PM#8). Schemas created by the
materializing legs are dropped on teardown.

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

import mempalace.mcp_server as m  # noqa: E402
from mempalace.backends import PalaceNotFoundError  # noqa: E402
from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.entity_index_postgres import PostgresEntityIndex  # noqa: E402
from mempalace.knowledge_graph_postgres import PostgresKnowledgeGraph  # noqa: E402


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


def _schema_present(schema: str) -> bool:
    """Direct DB probe (NOT a re-call into the gate) for the schema's existence.

    A re-call could itself materialize the schema; this opens an independent
    connection and reads ``information_schema.schemata`` so the ABSENT assertion
    is about the real DB state, immune to any seam-side caching.
    """
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema,),
            )
            return cur.fetchone() is not None


def _drop(*teams) -> None:
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            for t in teams:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
        conn.commit()


def _pop_caches(*teams) -> None:
    """Evict the per-team cached KG / entity-index handles (mirrors
    ``test_two_team_isolation._pop_caches``).

    A cached handle from a prior call could have ``_ensured=True`` set, which
    would let ``_ensure`` short-circuit and mask the ABSENT-schema assertion. A
    fresh uuid slug already avoids collisions, but popping is belt-and-braces.
    """
    for t in teams:
        m._kg_by_path.pop(f"pgkg::{t}", None)
        m._entity_index_by_team.pop(f"pgentidx::{t}", None)


def _fresh_team(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:12]


# ── T0a: KG create=False on a never-materialized team raises + stays ABSENT ──
def test_t0a_kg_create_false_unmaterialized_raises_and_absent(backend):
    team = _fresh_team("ckg_")
    _pop_caches(team)
    try:
        kg = PostgresKnowledgeGraph(backend, team=team)
        with pytest.raises(PalaceNotFoundError):
            kg.stats(create=False)
        # The raise must NOT have created the schema (raise-after-materialize).
        assert _schema_present(team_schema(team)) is False
    finally:
        _drop(team)


# ── T0b: entity-index create=False on a never-materialized team, same shape ──
def test_t0b_entity_index_create_false_unmaterialized_raises_and_absent(backend):
    team = _fresh_team("cei_")
    _pop_caches(team)
    try:
        idx = PostgresEntityIndex(backend, team=team)
        with pytest.raises(PalaceNotFoundError):
            idx.drawers_for_entity("Dana", create=False)
        assert _schema_present(team_schema(team)) is False
    finally:
        _drop(team)


# ── T0c: collection-seam regression pin — backend get_collection(create=False) ─
def test_t0c_collection_seam_create_false_missing_raises_and_absent(backend):
    """The blessed seam this slice mirrors must keep its discipline: a missing
    schema under create=False raises PalaceNotFoundError and does not create."""
    team = _fresh_team("ccol_")
    schema = team_schema(team)
    try:
        with pytest.raises(PalaceNotFoundError):
            backend.get_collection("/unused/palace/path", "memories", False, namespace=team)
        assert _schema_present(schema) is False
    finally:
        _drop(team)


# ── T0d: create=True materializes; a NEW handle with create=False then works ──
def test_t0d_kg_create_true_then_fresh_handle_create_false_succeeds(backend):
    team = _fresh_team("ckg_")
    _pop_caches(team)
    try:
        kg = PostgresKnowledgeGraph(backend, team=team)
        kg.add_triple("Max", "loves", "chess", create=True)
        assert _schema_present(team_schema(team)) is True

        # A brand-new handle (cold _ensured) with create=False must succeed now
        # that the schema exists — it probes, finds it, and serves the read.
        fresh = PostgresKnowledgeGraph(backend, team=team)
        s = fresh.stats(create=False)
        assert s["triples"] == 1
    finally:
        _drop(team)


# ── T0e: cache-aliasing guard (PM#8) — per-call flag, both directions ────────
def test_t0e_kg_per_call_flag_not_stored_on_handle(backend):
    """The create mode must be PER-CALL, never stored on the cached handle.

    Forward: one instance — create=False raises (unmaterialized), then create=
    True materializes, then create=False returns fine.
    Inverse: after create=True sets _ensured, a subsequent create=False
    fast-paths (the materialized case is the ONLY thing _ensured may short).
    """
    team = _fresh_team("ckg_")
    _pop_caches(team)
    try:
        kg = PostgresKnowledgeGraph(backend, team=team)

        # Forward: same instance, create=False first must raise (not aliased).
        with pytest.raises(PalaceNotFoundError):
            kg.stats(create=False)
        assert _schema_present(team_schema(team)) is False

        # Same instance materializes via create=True.
        kg.add_triple("Max", "loves", "chess", create=True)
        assert _schema_present(team_schema(team)) is True

        # Same instance, create=False now returns fine (schema exists).
        assert kg.stats(create=False)["triples"] == 1

        # Inverse: _ensured is now True, so create=False fast-paths the
        # materialized case without re-probing or raising.
        assert kg.query_entity("Max", create=False)
    finally:
        _drop(team)


def test_t0e_entity_index_per_call_flag_not_stored_on_handle(backend):
    """Entity-index mirror of T0e: per-call flag, both directions."""
    team = _fresh_team("cei_")
    _pop_caches(team)
    try:
        idx = PostgresEntityIndex(backend, team=team)

        with pytest.raises(PalaceNotFoundError):
            idx.drawers_for_entity("Dana", create=False)
        assert _schema_present(team_schema(team)) is False

        idx.add(["d1"], ["Dana"], "people", "r", create=True)
        assert _schema_present(team_schema(team)) is True

        assert {r["drawer_id"] for r in idx.drawers_for_entity("Dana", create=False)} == {"d1"}
        # _ensured fast-path: create=False after materialization just works.
        assert idx.top_entities(create=False)
    finally:
        _drop(team)


# ── T0f: backstop — an empty/None-derived team slug must never CREATE SCHEMA ──
def test_t0f_backstop_empty_slug_blocks_create(backend):
    """A KG/entity-index whose schema has an empty team slug must raise BEFORE
    any CREATE SCHEMA (PM#7), even on the create=True path.

    ``team_schema(None)`` resolves to ``team_default`` (a valid slug), so the
    backstop cannot be reached through the constructor. We construct the handle
    normally, then force the degenerate bare ``team_`` schema onto it to exercise
    the seam's own assertion — the class must not trust its schema name blindly.
    """
    kg = PostgresKnowledgeGraph(backend, team="placeholder")
    kg._schema = "team_"  # bare prefix, empty slug — must never materialize
    with pytest.raises(ValueError, match="no team resolved"):
        kg._ensure(create=True)
    assert _schema_present("team_") is False

    idx = PostgresEntityIndex(backend, team="placeholder")
    idx._schema = "team_"
    with pytest.raises(ValueError, match="no team resolved"):
        idx._ensure(create=True)
    assert _schema_present("team_") is False
