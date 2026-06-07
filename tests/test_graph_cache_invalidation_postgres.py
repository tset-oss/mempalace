"""Per-vault graph-cache invalidation on the build_graph-input write handlers.

``build_graph`` (palace_graph) caches the palace graph per vault with a 60s TTL.
The graph is built ONLY from drawer metadata (room/wing/hall/date) and is read
ONLY by ``graph_stats`` and ``traverse``. The four handlers that change those
inputs — ``tool_add_drawer``, ``tool_update_drawer``, ``tool_delete_drawer``,
``tool_diary_write`` — must evict the WRITING vault's cache entry on a
successful write, so a graph read in the same request reflects the write
instead of the stale pre-write graph (which would otherwise linger for the TTL).

The invalidation must be PER-VAULT: it evicts only ``_vault_cache_key(col,
config)`` (on postgres: ``schema:team_<slug>``), never the whole cache — a
clear-all would nuke every other team's warm graph. And it must NOT fire on
writes that are not build_graph inputs (``tool_create_tunnel``): tunnels feed
the explicit-link store, not the drawer-metadata graph, so create_tunnel must
leave ``_graph_cache`` untouched (those reads are eventually-consistent and out
of scope here).

These properties are asserted against a LIVE Postgres, driving the real tool
surface in one process with per-request team routing via ``_active_team_var`` —
the same harness shape as ``tests/test_two_team_isolation.py``. The graph reads
go through ``tool_graph_stats`` / ``tool_traverse_graph``, which resolve the
collection via ``_get_collection()`` for the active team and warm
``_graph_cache`` under that team's vault key — exactly the key the write
handlers evict.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

import mempalace.mcp_server as m  # noqa: E402
from mempalace import palace_graph  # noqa: E402
from mempalace.backends import get_backend  # noqa: E402
from mempalace.backends.postgres import team_schema  # noqa: E402
from mempalace.config import MempalaceConfig  # noqa: E402

DIM = 384


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


def _fake_embed(texts):
    """Deterministic content-derived embedding (no ONNX download)."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for i, b in enumerate(hashlib.sha256((t or "").encode()).digest()):
            v[i] = b / 255.0
        out.append(v)
    return out


def _drop(*teams) -> None:
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            for t in teams:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
        conn.commit()


def _pop_caches(*teams) -> None:
    """Drop every per-team cached store/handle so a fresh instance is built."""
    for t in teams:
        m._kg_by_path.pop(f"pgkg::{t}", None)
        m._entity_index_by_team.pop(f"pgentidx::{t}", None)
        m._link_store_by_team.pop(f"pglink::{t}", None)
        m._team_facts_by_team.pop(f"pgfacts::{t}", None)


def _vault_key(team: str) -> str:
    """The cache key the read/write paths use for ``team`` on postgres."""
    return f"schema:{team_schema(team)}"


class _ActiveTeam:
    """Set ``_active_team_var`` for the duration of a block, then reset."""

    def __init__(self, team):
        self._team = team
        self._token = None

    def __enter__(self):
        self._token = m._active_team_var.set(self._team)
        return self

    def __exit__(self, *exc):
        m._active_team_var.reset(self._token)
        return False


@pytest.fixture
def server_pg(monkeypatch):
    """Put the server in postgres mode against the live DB, shared backend.

    Mirrors ``tests/test_two_team_isolation.py``: one shared backend with a
    deterministic embedder, per-request routing via ``_active_team_var``, and a
    fully reset module-global ``_graph_cache`` on entry AND exit so the cache
    state each test observes is its own (the cache is a module global with no
    conftest reset).
    """
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setenv("MEMPALACE_DERIVE_DEBOUNCE_SECONDS", "0")
    token = m._active_team_var.set(None)

    backend = get_backend("postgres")
    prev_embedder = getattr(backend, "_embedder", None)
    backend._embedder = _fake_embed

    monkeypatch.setattr(m, "_config", MempalaceConfig())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)

    prev_deb = m._derived_link_debouncer
    monkeypatch.setattr(m, "_derived_link_debouncer", None)

    # Reset the module-global graph cache so contamination from a prior test
    # cannot mask a stale-read regression here. Clear-all is appropriate in the
    # TEST harness (no other teams' warm caches matter); production code must
    # NEVER clear-all on a write — that is exactly what AC2 guards.
    palace_graph.invalidate_graph_cache()

    yield backend

    palace_graph.invalidate_graph_cache()
    monkeypatch.setattr(m, "_derived_link_debouncer", prev_deb)
    backend._embedder = prev_embedder
    m._active_team_var.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# AC1 — freshness: a write evicts the writing vault's stale graph so a read in
# the same request reflects it (covers add_drawer + delete_drawer; update and
# diary lighter).
# ─────────────────────────────────────────────────────────────────────────────


def _graph_has_room(room: str) -> bool:
    """True iff ``room`` is a node in the active team's graph.

    ``tool_traverse_graph`` reads the SAME cached graph as ``graph_stats`` and
    returns a list (the room is the first hop) when the room exists, or a
    structured ``{"error": ...}`` when it does not — a direct room-membership
    probe against the build_graph output.
    """
    res = m.tool_traverse_graph(room)
    if isinstance(res, dict):  # {"error": ...} -> room not in graph
        return False
    return any(step["room"] == room for step in res)


def test_add_drawer_invalidates_graph_within_request(server_pg):
    """Warm X's graph, add a drawer into a NEW room, re-read X's graph: the new
    room is present. Without invalidation the warm hit would serve the stale
    pre-add graph (missing the new room) for up to the 60s TTL."""
    team = "tg" + uuid.uuid4().hex[:8]
    _pop_caches(team)
    tok = uuid.uuid4().hex[:8]
    new_room = f"room_added_{tok}"
    try:
        with _ActiveTeam(team):
            # Seed one drawer so the palace is non-empty, then WARM the graph.
            assert (
                m.tool_add_drawer(
                    wing=f"wing_seed_{tok}", room=f"room_seed_{tok}", content=f"seed {tok}"
                ).get("success")
                is True
            )
            # The new room does not exist yet — warming the graph here means a
            # later stale hit (without invalidation) would keep reporting its
            # absence.
            assert _graph_has_room(new_room) is False
            # The warm read populated this vault's cache entry (non-vacuous: a
            # later stale read would have to come from here).
            assert _vault_key(team) in palace_graph._graph_cache

            # WRITE: a drawer in a brand-new room/wing changes graph inputs.
            add = m.tool_add_drawer(wing=f"wing_added_{tok}", room=new_room, content=f"added {tok}")
            assert add.get("success") is True
            # The write evicted this vault's entry (per-vault eviction).
            assert _vault_key(team) not in palace_graph._graph_cache

            # READ AGAIN in the same request: graph reflects the add (rebuilt).
            assert _graph_has_room(new_room) is True
    finally:
        _pop_caches(team)
        _drop(team)


def test_delete_drawer_invalidates_graph_within_request(server_pg):
    """Warm X's graph (room present), delete the only drawer in that room,
    re-read: the room is gone. Without invalidation the warm hit would still
    report the deleted room for up to the TTL."""
    team = "tg" + uuid.uuid4().hex[:8]
    _pop_caches(team)
    tok = uuid.uuid4().hex[:8]
    doomed_room = f"room_doomed_{tok}"
    try:
        with _ActiveTeam(team):
            add = m.tool_add_drawer(
                wing=f"wing_del_{tok}", room=doomed_room, content=f"doomed {tok}"
            )
            assert add.get("success") is True
            drawer_id = add["drawer_id"]

            # WARM: the doomed room is present in the cached graph.
            assert _graph_has_room(doomed_room) is True
            assert _vault_key(team) in palace_graph._graph_cache

            # WRITE: delete the only drawer in that room.
            assert m.tool_delete_drawer(drawer_id).get("success") is True
            assert _vault_key(team) not in palace_graph._graph_cache

            # READ AGAIN: the room is gone (graph rebuilt, not stale).
            assert _graph_has_room(doomed_room) is False
    finally:
        _pop_caches(team)
        _drop(team)


def test_update_drawer_invalidates_graph_within_request(server_pg):
    """(lighter) An update that moves a drawer to a new room evicts the vault's
    cache entry; the read after reflects the new room."""
    team = "tg" + uuid.uuid4().hex[:8]
    _pop_caches(team)
    tok = uuid.uuid4().hex[:8]
    moved_room = f"room_moved_{tok}"
    try:
        with _ActiveTeam(team):
            add = m.tool_add_drawer(
                wing=f"wing_upd_{tok}", room=f"room_orig_{tok}", content=f"orig {tok}"
            )
            assert add.get("success") is True
            # Warm: the moved-to room does not exist yet.
            assert _graph_has_room(moved_room) is False
            assert _vault_key(team) in palace_graph._graph_cache

            assert m.tool_update_drawer(add["drawer_id"], room=moved_room).get("success") is True
            assert _vault_key(team) not in palace_graph._graph_cache

            # The graph now knows the new room (rebuilt, not the stale graph).
            assert _graph_has_room(moved_room) is True
    finally:
        _pop_caches(team)
        _drop(team)


def test_diary_write_invalidates_graph_within_request(server_pg):
    """(lighter) A diary write lands a new drawer (room ``diary``) and must evict
    the vault's cache entry."""
    team = "tg" + uuid.uuid4().hex[:8]
    _pop_caches(team)
    tok = uuid.uuid4().hex[:8]
    diary_wing = f"wing_diary_{tok}"
    try:
        with _ActiveTeam(team):
            # Seed + warm so there is a cache entry to evict.
            assert (
                m.tool_add_drawer(
                    wing=f"wing_seed_{tok}", room=f"room_seed_{tok}", content=f"seed {tok}"
                ).get("success")
                is True
            )
            # Warm: the diary wing does not exist yet (rooms_per_wing is keyed
            # by WING).
            warm = m.tool_graph_stats()
            assert diary_wing not in warm["rooms_per_wing"]
            assert _vault_key(team) in palace_graph._graph_cache

            assert (
                m.tool_diary_write(
                    agent_name=f"agent{tok}", entry=f"diary entry {tok}", wing=diary_wing
                ).get("success")
                is True
            )
            assert _vault_key(team) not in palace_graph._graph_cache

            # The diary write created the diary room under the diary wing; the
            # rebuilt graph now knows the wing (not the stale pre-write graph).
            fresh = m.tool_graph_stats()
            assert diary_wing in fresh["rooms_per_wing"], fresh
    finally:
        _pop_caches(team)
        _drop(team)


# ─────────────────────────────────────────────────────────────────────────────
# AC2 — per-vault, no clear-all: a write to X leaves Y's warm cache intact.
# ─────────────────────────────────────────────────────────────────────────────


def test_write_to_x_does_not_clear_y_graph_cache(server_pg):
    """Warm BOTH X and Y, write to X, assert Y's cache entry STILL exists and
    X's was removed. A ``invalidate_graph_cache(col=None)`` clear-all would drop
    Y too and fail this — the explicit no-clear-all guard."""
    team_x = "tx" + uuid.uuid4().hex[:8]
    team_y = "ty" + uuid.uuid4().hex[:8]
    _pop_caches(team_x, team_y)
    tok = uuid.uuid4().hex[:8]
    try:
        with _ActiveTeam(team_x):
            assert (
                m.tool_add_drawer(
                    wing=f"wing_x_{tok}", room=f"room_x_{tok}", content=f"x {tok}"
                ).get("success")
                is True
            )
            m.tool_graph_stats()  # warm X
        with _ActiveTeam(team_y):
            assert (
                m.tool_add_drawer(
                    wing=f"wing_y_{tok}", room=f"room_y_{tok}", content=f"y {tok}"
                ).get("success")
                is True
            )
            m.tool_graph_stats()  # warm Y

        # Both vaults are warm and DISTINCT.
        assert _vault_key(team_x) in palace_graph._graph_cache
        assert _vault_key(team_y) in palace_graph._graph_cache
        assert _vault_key(team_x) != _vault_key(team_y)

        # WRITE to X only.
        with _ActiveTeam(team_x):
            assert (
                m.tool_add_drawer(
                    wing=f"wing_x2_{tok}", room=f"room_x2_{tok}", content=f"x2 {tok}"
                ).get("success")
                is True
            )

        # X's entry was removed; Y's warm entry SURVIVED (per-vault eviction).
        assert _vault_key(team_x) not in palace_graph._graph_cache
        assert _vault_key(team_y) in palace_graph._graph_cache
    finally:
        _pop_caches(team_x, team_y)
        _drop(team_x, team_y)


# ─────────────────────────────────────────────────────────────────────────────
# AC3 / AC4 — no over-invalidation: create_tunnel is NOT a build_graph input and
# must leave the vault's graph cache untouched (and documents that tunnel reads
# are eventually-consistent, out of scope). The per-vault tests above also prove
# no call site passes col=None (a clear-all would have failed AC2).
# ─────────────────────────────────────────────────────────────────────────────


def test_create_tunnel_does_not_invalidate_graph_cache(server_pg):
    """Warm X's graph, create a tunnel for X, assert X's ``_graph_cache`` entry
    is UNTOUCHED. Tunnels feed the explicit-link store, not the drawer-metadata
    graph, so create_tunnel must not evict the graph cache. This also documents
    that tunnel / derived-link reads are eventually-consistent and out of scope
    for graph-cache invalidation."""
    team = "tg" + uuid.uuid4().hex[:8]
    _pop_caches(team)
    tok = uuid.uuid4().hex[:8]
    try:
        with _ActiveTeam(team):
            # Create the per-team tunnels table SYNCHRONOUSLY on this thread
            # first (a read runs the link store's lazy DDL). The seed add below
            # fires a zero-debounce derived-link rebuild on a worker thread that
            # also touches the tunnels table; doing the DDL here once avoids a
            # cross-connection CREATE-TABLE race on the brand-new schema. This is
            # a test-sequencing concern only — in production the table is created
            # once and persists. Mirrors the isolation harness, which drains the
            # debouncer before later tunnel surfaces.
            assert m.tool_list_tunnels() == []
            assert (
                m.tool_add_drawer(
                    wing=f"wing_seed_{tok}", room=f"room_seed_{tok}", content=f"seed {tok}"
                ).get("success")
                is True
            )
            m.tool_graph_stats()  # warm
            assert _vault_key(team) in palace_graph._graph_cache
            entry_before = palace_graph._graph_cache[_vault_key(team)]

            tun = m.tool_create_tunnel(
                f"wing_seed_{tok}",
                f"room_seed_{tok}",
                f"wing_other_{tok}",
                "design",
                label=f"explicit-{tok}",
            )
            assert "error" not in tun, tun

            # The graph-cache entry is the SAME object — create_tunnel did not
            # evict, refresh, or replace it.
            assert _vault_key(team) in palace_graph._graph_cache
            assert palace_graph._graph_cache[_vault_key(team)] is entry_before
    finally:
        _pop_caches(team)
        _drop(team)


# ─────────────────────────────────────────────────────────────────────────────
# AC5 — cache cap (64) and TTL (60s) constants/semantics are unchanged.
# ─────────────────────────────────────────────────────────────────────────────


def test_cache_cap_and_ttl_constants_unchanged():
    """The story forbids touching the cap/TTL; pin them so an accidental edit in
    this story (or a follow-up) is caught."""
    assert palace_graph._GRAPH_CACHE_TTL == 60.0
    assert palace_graph._GRAPH_CACHE_MAX_VAULTS == 64
