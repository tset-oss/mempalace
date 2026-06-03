"""Integration tests for the per-vault PostgresEntityIndex (G003).

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.entity_index_postgres import PostgresEntityIndex  # noqa: E402


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
    b = PostgresBackend(dsn=_dsn())
    yield b
    b.close()


@pytest.fixture
def team(backend):
    name = "eidx_" + uuid.uuid4().hex[:12]
    yield name
    # Tear down the whole vault schema (entity_occurrences + any kg tables).
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')


def test_add_query_top_and_delete(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1", "d2"], ["Dana", "Ingest"], "people", "2026-06-03")
    idx.add(["d3"], ["Dana"], "people", "2026-06-03")

    # Case-insensitive lookup returns every physical drawer mentioning the entity.
    rows = idx.drawers_for_entity("dana")
    assert {r["drawer_id"] for r in rows} == {"d1", "d2", "d3"}

    top = {t["entity"]: t["count"] for t in idx.top_entities()}
    assert top["Dana"] == 3  # d1, d2, d3 (count of DISTINCT drawer_id)
    assert top["Ingest"] == 2  # d1, d2

    # Deleting drawers removes their rows; Dana now only in d3.
    idx.delete_by_drawer(["d1", "d2"])
    rows = idx.drawers_for_entity("Dana")
    assert {r["drawer_id"] for r in rows} == {"d3"}


def test_add_is_idempotent_per_pair(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1"], ["Dana"], "w", "r")
    idx.add(["d1"], ["Dana"], "w", "r")  # ON CONFLICT DO NOTHING
    assert len(idx.drawers_for_entity("Dana")) == 1


def test_top_entities_wing_scope_and_min_count(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1"], ["Alpha", "Beta"], "wing_a", "r")
    idx.add(["d2"], ["Alpha"], "wing_a", "r")
    idx.add(["d3"], ["Gamma"], "wing_b", "r")

    wing_a = {t["entity"] for t in idx.top_entities(wing="wing_a")}
    assert wing_a == {"Alpha", "Beta"}  # Gamma is in wing_b
    frequent = {t["entity"] for t in idx.top_entities(min_count=2)}
    assert frequent == {"Alpha"}  # only Alpha appears in >=2 drawers


def test_known_entities_accumulates_and_seeds_from_kg(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1"], ["Dana"], "people", "r")
    assert "Dana" in idx.known_entities()

    # kg_add entities (relational kg_entities, same vault) seed the known set.
    from mempalace.knowledge_graph_postgres import PostgresKnowledgeGraph

    kg = PostgresKnowledgeGraph(backend, team=team)
    kg.add_triple("Zelda", "owns", "ingest-pipeline")

    # Fresh index instance so the TTL cache does not mask the new kg entities.
    known = PostgresEntityIndex(backend, team=team).known_entities()
    assert "Dana" in known
    assert "Zelda" in known
    assert "ingest-pipeline" in known


def test_known_entities_empty_vault_is_empty(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    # _ensure creates the (empty) table; no kg table yet -> empty known set.
    assert idx.known_entities() == frozenset()


def test_write_path_helper_indexes_against_live_db(backend, team, monkeypatch):
    """The mcp_server write-path helper extracts + indexes against real PG.

    Exercises _index_drawer_entities (the function tool_add_drawer calls) end to
    end minus the ONNX embedding — extraction -> per-vault index — confirming the
    incremental stamping wiring works in postgres mode.
    """
    import mempalace.mcp_server as m

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_entity_index_by_team", {})  # isolate the per-vault cache
    assert m._config.backend != "chroma"

    # "Zelda" appears twice -> caught by the frequency tier with an empty vault.
    m._index_drawer_entities(team, ["dX", "dY"], "Zelda met Zelda about the plan.", "people", "today")

    rows = m._get_entity_index(team).drawers_for_entity("Zelda")
    assert {r["drawer_id"] for r in rows} == {"dX", "dY"}

    # And the delete helper clears them.
    m._unindex_drawer_entities(team, ["dX", "dY"])
    assert m._get_entity_index(team).drawers_for_entity("Zelda") == []
