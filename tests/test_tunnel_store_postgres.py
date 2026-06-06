"""Integration tests for the per-team PostgresLinkStore (explicit tunnels).

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.

These tests assert the PG store is a drop-in for the JSON store: the canonical
tunnel ids match, the create/list/follow/delete return shapes are identical to
JsonLinkStore's contract, re-creating preserves the dynamics columns, vaults are
isolated per team, the table starts empty (no backfill), and a missing team
fails loud.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.link_store_postgres import PostgresLinkStore  # noqa: E402
from mempalace.palace_graph import _canonical_tunnel_id  # noqa: E402


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


# ─────────────────────────────────────────────────────────────────────────────
# Vault round-trip: create -> list -> follow -> delete, JSON-identical shapes
# ─────────────────────────────────────────────────────────────────────────────


def test_create_list_follow_delete_round_trip(backend, team):
    store = PostgresLinkStore(backend, team=team)

    created = store.create_tunnel(
        "wing_code",
        "auth",
        "wing_people",
        "users",
        label="same concept",
        target_drawer_id="drawer_users_1",
    )

    # Symmetric canonical id matches the JSON store's id for the same endpoints.
    assert created["id"] == _canonical_tunnel_id("wing_code", "auth", "wing_people", "users")
    assert created["kind"] == "explicit"
    assert created["label"] == "same concept"

    # Return-shape contract (matches JsonLinkStore.create_tunnel): id/source/
    # target/label/kind/created_at + the four dynamics fields. updated_at is
    # absent on a brand-new row (JSON store only sets it on re-create).
    assert created["source"] == {"wing": "wing_code", "room": "auth"}
    assert created["target"] == {
        "wing": "wing_people",
        "room": "users",
        "drawer_id": "drawer_users_1",
    }
    assert "drawer_id" not in created["source"]  # no source drawer id supplied
    assert "updated_at" not in created
    assert created["strength"] == 1.0
    assert created["stability"] == 1.0
    assert created["access_count"] == 0

    # Timestamp fields must be ISO strings, not raw datetime objects — matching
    # JsonLinkStore byte-for-byte and safe for json.dumps without default=str.
    assert isinstance(created["created_at"], str)
    assert isinstance(created["last_activated"], str)
    assert "T" in created["created_at"]  # isoformat 'T' separator, not str() space
    assert "T" in created["last_activated"]
    import json

    json.dumps(created)  # must not raise TypeError for any timestamp field

    # list — unfiltered and by either endpoint (symmetric).
    listed = store.list_tunnels()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]
    assert len(store.list_tunnels("wing_people")) == 1
    assert len(store.list_tunnels("wing_code")) == 1
    assert store.list_tunnels("wing_unrelated") == []

    # follow — outgoing from the source endpoint, with a drawer preview.
    col = MagicMock()
    col.get.return_value = {
        "ids": ["drawer_users_1"],
        "documents": ["A" * 400],
        "metadatas": [{}],
    }
    connections = store.follow_tunnels("wing_code", "auth", col=col)
    assert len(connections) == 1
    c = connections[0]
    assert c["direction"] == "outgoing"
    assert c["connected_wing"] == "wing_people"
    assert c["connected_room"] == "users"
    assert c["label"] == "same concept"
    assert c["drawer_id"] == "drawer_users_1"
    assert c["tunnel_id"] == created["id"]
    assert c["drawer_preview"] == "A" * 300  # capped at 300 chars

    # follow — incoming from the target endpoint.
    incoming = store.follow_tunnels("wing_people", "users")
    assert len(incoming) == 1
    assert incoming[0]["direction"] == "incoming"
    assert incoming[0]["connected_wing"] == "wing_code"
    assert incoming[0]["connected_room"] == "auth"

    # delete.
    assert store.delete_tunnel(created["id"]) == {"deleted": created["id"]}
    assert store.list_tunnels() == []


def test_return_shape_keys_match_json_store_contract(backend, team):
    """The PG create dict carries the same keys the JSON store documents."""
    store = PostgresLinkStore(backend, team=team)
    created = store.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="lbl")
    expected_keys = {
        "id",
        "source",
        "target",
        "label",
        "kind",
        "created_at",
        "strength",
        "stability",
        "last_activated",
        "access_count",
    }
    assert set(created.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────────────
# Re-create preserves dynamics columns; updates only label + updated_at
# ─────────────────────────────────────────────────────────────────────────────


def test_recreate_preserves_dynamics_columns(backend, team):
    store = PostgresLinkStore(backend, team=team)
    first = store.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="initial")

    # Mutate the dynamics columns directly to simulate accumulated activity.
    schema = team_schema(team)
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'UPDATE "{schema}"."tunnels" '
                "SET strength = 2.7, access_count = 12, stability = 1.5 WHERE id = %s",
                (first["id"],),
            )

    # Re-create with a NEW label — same canonical id, so this is an upsert.
    second = store.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="updated")

    assert second["id"] == first["id"]
    # label + updated_at change.
    assert second["label"] == "updated"
    assert second.get("updated_at") is not None
    assert isinstance(second["updated_at"], str)  # ISO string, not datetime
    assert "T" in second["updated_at"]
    # The four dynamics columns are PRESERVED (untouched by the ON CONFLICT).
    assert second["strength"] == 2.7
    assert second["access_count"] == 12
    assert second["stability"] == 1.5

    # Still exactly one row — the upsert did not duplicate.
    assert len(store.list_tunnels()) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-team isolation: a tunnel in team A is invisible to team B
# ─────────────────────────────────────────────────────────────────────────────


def test_per_team_isolation(backend):
    team_a = "t" + uuid.uuid4().hex[:10]
    team_b = "t" + uuid.uuid4().hex[:10]
    store_a = PostgresLinkStore(backend, team=team_a)
    store_b = PostgresLinkStore(backend, team=team_b)
    try:
        store_a.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="A's link")

        # Team A sees it.
        assert len(store_a.list_tunnels()) == 1
        # Team B's store (separate schema) sees nothing.
        assert store_b.list_tunnels() == []
        assert store_b.follow_tunnels("wing_a", "room_x") == []
    finally:
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_a)}" CASCADE')
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_b)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# No-backfill: a freshly _ensure-d tunnels table is EMPTY
# ─────────────────────────────────────────────────────────────────────────────


def test_fresh_table_starts_empty(backend, team):
    """_ensure CREATES the table empty and copies nothing from tunnels.json."""
    store = PostgresLinkStore(backend, team=team)
    store._ensure()  # creates schema + table + indexes, copies nothing
    assert store.list_tunnels() == []

    # The table physically exists and is empty.
    schema = team_schema(team)
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{schema}.tunnels",))
            assert cur.fetchone()[0] is not None
            cur.execute(f'SELECT count(*) FROM "{schema}"."tunnels"')
            assert cur.fetchone()[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Fail loud: get_link_store(postgres, team=None) RAISES (no default vault)
# ─────────────────────────────────────────────────────────────────────────────


def test_get_link_store_postgres_no_team_raises():
    import mempalace.link_store as link_store

    class _Cfg:
        backend = "postgres"

    with pytest.raises(ValueError, match="no team resolved"):
        link_store.get_link_store(_Cfg(), team=None)


# ─────────────────────────────────────────────────────────────────────────────
# kind round-trip: explicit / topic / entity persist and read back
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["explicit", "topic", "entity"])
def test_kind_round_trip(backend, team, kind):
    store = PostgresLinkStore(backend, team=team)
    created = store.create_tunnel(
        "wing_a", "room_x", "wing_b", "room_y", label=f"{kind} link", kind=kind
    )
    assert created["kind"] == kind
    listed = store.list_tunnels()
    assert len(listed) == 1
    assert listed[0]["kind"] == kind


# ─────────────────────────────────────────────────────────────────────────────
# Derived hallways are now implemented (server-side derive over entity rows).
# Full behavior lives in tests/test_derived_links_postgres.py; here we only
# assert the methods no longer raise the old placeholder on an empty vault.
# ─────────────────────────────────────────────────────────────────────────────


def test_hallway_methods_return_empty_on_empty_vault(backend, team):
    store = PostgresLinkStore(backend, team=team)
    assert store.compute_hallways_for_wing("wing_a") == []
    assert store.list_hallways() == []
