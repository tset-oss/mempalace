"""Integration tests for the per-team PostgresEntityRegistry (disambiguation).

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.

These tests assert the PG registry is a drop-in for the JSON registry: the
disambiguation fields (relationship / DOB / ID / context) round-trip, an
ambiguous name resolves to the SAME result on both backends for the same
operation, vaults are isolated per team, and a missing team fails loud.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.entity_registry import EntityRegistry, get_entity_registry  # noqa: E402
from mempalace.entity_registry_postgres import PostgresEntityRegistry  # noqa: E402


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
    name = "t" + uuid.uuid4().hex[:10]
    yield name
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')


# The shared disambiguation fixture: two people the JSON model keys by
# name, each carrying disambiguation fields (relationship + DOB + national ID +
# context). "Grace" is also a COMMON_ENGLISH_WORDS name, so it exercises the
# context-pattern disambiguation path that must behave identically on both
# backends.
def _seed_people(reg: EntityRegistry) -> None:
    reg.seed(
        mode="personal",
        people=[
            {"name": "Grace", "relationship": "daughter", "context": "personal"},
            {"name": "Devon", "relationship": "colleague", "context": "work"},
        ],
        projects=["MemPalace"],
    )
    # Attach the disambiguation fields the model stores per-person verbatim
    # (the data model is an open dict per name), then persist.
    reg.people["Grace"]["dob"] = "1990-04-01"
    reg.people["Grace"]["national_id"] = "AB-1234"
    reg.people["Devon"]["dob"] = "1985-11-22"
    reg.people["Devon"]["national_id"] = "CD-5678"
    reg.save()


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip parity: disambiguation on PG equals the JSON registry
# ─────────────────────────────────────────────────────────────────────────────


def test_round_trip_parity_with_json_registry(backend, team, tmp_path):
    # JSON reference registry (the chroma path, unchanged behaviour).
    json_reg = EntityRegistry.load(config_dir=tmp_path)
    _seed_people(json_reg)

    # Postgres registry: same seed, persisted to the team vault.
    pg_reg = PostgresEntityRegistry.open(backend, team=team)
    _seed_people(pg_reg)

    # Re-open from storage to prove the document round-trips (not just the
    # in-memory object).
    pg_reloaded = PostgresEntityRegistry.open(backend, team=team)

    # The disambiguation fields survive the round-trip on PG.
    assert pg_reloaded.people["Grace"]["dob"] == "1990-04-01"
    assert pg_reloaded.people["Grace"]["national_id"] == "AB-1234"
    assert pg_reloaded.people["Grace"]["relationship"] == "daughter"
    assert pg_reloaded.people["Devon"]["national_id"] == "CD-5678"
    assert "grace" in pg_reloaded.ambiguous_flags  # ambiguous-name flag persisted

    # Identical disambiguation semantics for the SAME operations on both
    # backends — resolving the ambiguous name "Grace" by context.
    for op_ctx, expected_type in [
        ("I went with Grace today", "person"),  # person context
        ("the grace of God", "concept"),  # concept context
    ]:
        json_result = json_reg.lookup("Grace", context=op_ctx)
        pg_result = pg_reloaded.lookup("Grace", context=op_ctx)
        assert pg_result["type"] == expected_type
        assert pg_result["type"] == json_result["type"]
        assert pg_result["name"] == json_result["name"]
        assert pg_result["confidence"] == json_result["confidence"]

    # Unambiguous lookups + project lookup + query extraction match too.
    assert pg_reloaded.lookup("Devon") == json_reg.lookup("Devon")
    assert pg_reloaded.lookup("MemPalace") == json_reg.lookup("MemPalace")
    assert pg_reloaded.lookup("Xyzzy") == json_reg.lookup("Xyzzy")
    assert sorted(pg_reloaded.extract_people_from_query("Did Devon and Grace meet?")) == sorted(
        json_reg.extract_people_from_query("Did Devon and Grace meet?")
    )


def test_empty_registry_for_new_team(backend, team):
    """A team with no persisted document opens an empty registry (like a missing file)."""
    reg = PostgresEntityRegistry.open(backend, team=team)
    assert reg.people == {}
    assert reg.projects == []
    assert reg.ambiguous_flags == []
    assert reg.mode == "personal"


# ─────────────────────────────────────────────────────────────────────────────
# Per-team isolation: team A's entity is invisible to team B
# ─────────────────────────────────────────────────────────────────────────────


def test_per_team_isolation(backend):
    team_a = "t" + uuid.uuid4().hex[:10]
    team_b = "t" + uuid.uuid4().hex[:10]
    try:
        reg_a = PostgresEntityRegistry.open(backend, team=team_a)
        reg_a.seed(
            mode="personal",
            people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
            projects=["ProjectA"],
        )

        # Team A sees Riley.
        assert "Riley" in reg_a.people
        assert reg_a.lookup("Riley")["type"] == "person"

        # Team B's registry (separate schema) sees nothing of team A's.
        reg_b = PostgresEntityRegistry.open(backend, team=team_b)
        assert reg_b.people == {}
        assert reg_b.lookup("Riley")["type"] == "unknown"
        assert "ProjectA" not in reg_b.projects
    finally:
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_a)}" CASCADE')
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_b)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# Fail loud: get_entity_registry(postgres, team=None) RAISES (no default vault)
# ─────────────────────────────────────────────────────────────────────────────


def test_get_entity_registry_postgres_no_team_raises():
    class _Cfg:
        backend = "postgres"

    with pytest.raises(ValueError, match="no team resolved"):
        get_entity_registry(_Cfg(), team=None)


# ─────────────────────────────────────────────────────────────────────────────
# Chroma path: the seam returns the JSON registry, file behaviour unchanged
# ─────────────────────────────────────────────────────────────────────────────


def test_get_entity_registry_chroma_returns_json_registry(monkeypatch, tmp_path):
    # Point the JSON registry's default path at a temp dir so the test does not
    # touch the real ~/.mempalace.
    monkeypatch.setattr(EntityRegistry, "DEFAULT_PATH", tmp_path / "entity_registry.json")

    class _Cfg:
        backend = "chroma"

    reg = get_entity_registry(_Cfg())
    # It is the JSON registry, NOT the Postgres one.
    assert isinstance(reg, EntityRegistry)
    assert not isinstance(reg, PostgresEntityRegistry)

    # File behaviour is the JSON store's: a save writes the atomic file with no
    # leftover .tmp sidecar (the existing entity-registry tests cover the full
    # atomic-write contract; this confirms the seam preserves it).
    reg.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    assert (tmp_path / "entity_registry.json").exists()
    assert list(tmp_path.glob("entity_registry.json.tmp*")) == []
