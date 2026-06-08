"""Integration tests for the per-team name-resolution MCP tools.

Covers ``mempalace_disambiguate`` (read) and ``mempalace_entity_seed`` (write,
read-merge-write) against a live Postgres (the bundled deploy/docker-compose db
or any instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Live-DB
tests self-skip when psycopg is absent or the DB is unreachable, so chroma-only
CI is unaffected. The chroma-unsupported and fail-loud handler tests need no DB.

These tools expose the entity registry's NAME-RESOLUTION layer (known
people/projects/aliases + ambiguity) per team — a lane distinct from the
knowledge graph (facts/relationships) and team critical-facts (must-know lines).
The asserts below pin the behaviour the lane promises: alias direction resolves
to the canonical name, isolation between teams, a graceful not-found for unseeded
names, no outbound network on any lookup path, fail-loud on a team-less write,
non-clobbering read-merge-write, chroma reporting the feature unavailable,
persistence across a fresh open, and zero leakage into the kg / team-facts lanes.
"""

from __future__ import annotations

import os
import uuid

import pytest

import mempalace.entity_registry as er
import mempalace.mcp_server as m

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402


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


_PG_REACHABLE = _reachable(_dsn())
_needs_pg = pytest.mark.skipif(
    not _PG_REACHABLE,
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)


@pytest.fixture
def backend():
    b = PostgresBackend(dsn=_dsn())
    yield b
    b.close()


@pytest.fixture
def pg_env(monkeypatch, backend):
    """Force postgres backend, route the registry to the live test backend, and
    isolate the per-team caches so a stale cached store is never reused."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_kg_by_path", {})
    monkeypatch.setattr(m, "_team_facts_by_team", {})
    token = m._active_team_var.set(None)
    yield monkeypatch
    m._active_team_var.reset(token)


def _new_team() -> str:
    return "t" + uuid.uuid4().hex[:10]


def _drop(backend, *teams: str) -> None:
    with backend._conn() as conn:
        with conn.cursor() as cur:
            for t in teams:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# AC1: alias direction + per-team isolation
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_alias_resolves_canonical_and_is_team_isolated(pg_env, backend):
    team_x = _new_team()
    team_y = _new_team()
    try:
        m._active_team_var.set(team_x)
        seed = m.tool_entity_seed(
            mode="personal",
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "work"}],
            aliases={"Max": "Maxwell"},
        )
        assert seed["vault"] == team_x

        res = m.tool_disambiguate("Max")
        assert res["found"] is True
        assert res["type"] == "person"
        assert res["name"] == "Maxwell"  # alias direction: Max → canonical Maxwell

        # Team Y, a separate vault, is unaffected by team X's seed.
        m._active_team_var.set(team_y)
        res_y = m.tool_disambiguate("Max")
        assert res_y["found"] is False
        assert res_y["vault"] == team_y
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x, team_y)


# ─────────────────────────────────────────────────────────────────────────────
# AC2: unseeded name → graceful found: false (no exception)
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_disambiguate_unknown_name_returns_found_false(pg_env, backend):
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        res = m.tool_disambiguate("Nobody")
        assert res["found"] is False
        assert res["type"] == "unknown"
        assert res["name"] == "Nobody"
        assert res["vault"] == team_x
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# AC3: NO-NETWORK regression guard — lookup never reaches Wikipedia
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_disambiguate_never_performs_network_lookup(pg_env, backend, monkeypatch):
    team_x = _new_team()

    def _boom(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("disambiguate must never call _wikipedia_lookup")

    # Patch the network primitive itself so ANY path that would reach the wire
    # fails the test loudly, regardless of how the call is threaded.
    monkeypatch.setattr(er, "_wikipedia_lookup", _boom)

    # Spy on research(): if a handler ever routed through it (or threaded
    # allow_network), this records it — research() is the only allow_network user.
    research_calls = []
    real_research = er.EntityRegistry.research

    def _spy_research(self, word, auto_confirm=False, allow_network=False):
        research_calls.append((word, allow_network))
        return real_research(self, word, auto_confirm=auto_confirm, allow_network=allow_network)

    monkeypatch.setattr(er.EntityRegistry, "research", _spy_research)

    try:
        m._active_team_var.set(team_x)
        # "Grace" is in COMMON_ENGLISH_WORDS → exercises the ambiguous path.
        m.tool_entity_seed(
            people=[{"name": "Grace", "relationship": "daughter", "context": "personal"}],
        )
        # (a) known seeded name, (b) ambiguous name WITH context, (c) unknown name.
        m.tool_disambiguate("Grace")
        m.tool_disambiguate("Grace", context="I went with Grace today")
        m.tool_disambiguate("Xyzzy")
        # Network primitive was never hit AND research() was never invoked, so
        # allow_network=True was never threaded.
        assert research_calls == []
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# AC4: fail-loud — entity_seed with no resolvable team RAISES
# ─────────────────────────────────────────────────────────────────────────────


def test_entity_seed_no_team_raises(monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.delenv("MEMPALACE_TEAM", raising=False)
    token = m._active_team_var.set(None)
    try:
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_entity_seed(people=[{"name": "Maxwell", "relationship": "x", "context": "work"}])
    finally:
        m._active_team_var.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# AC5: NO-CLOBBER read-merge-write
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_entity_seed_is_additive_and_idempotent(pg_env, backend):
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        # Projects: seed {A, B} then {C} → union {A, B, C}.
        m.tool_entity_seed(projects=["A", "B"])
        m.tool_entity_seed(projects=["C"])

        from mempalace.entity_registry import get_entity_registry

        reg = get_entity_registry(m._config, team=team_x)
        assert sorted(reg.projects) == ["A", "B", "C"]

        # Person: seed once, then re-seed the SAME person with a NEW context and
        # NEW alias → prior context/alias preserved, new ones added.
        m.tool_entity_seed(
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "work"}],
            aliases={"Max": "Maxwell"},
        )
        m.tool_entity_seed(
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "personal"}],
            aliases={"MW": "Maxwell"},
        )
        reg = get_entity_registry(m._config, team=team_x)
        assert set(reg.people["Maxwell"]["contexts"]) == {"work", "personal"}
        assert set(reg.people["Maxwell"]["aliases"]) == {"Max", "MW"}

        # Idempotent: an identical-payload re-seed grows nothing.
        before_people = len(reg.people)
        before_ctx = sorted(reg.people["Maxwell"]["contexts"])
        before_aliases = sorted(reg.people["Maxwell"]["aliases"])
        before_projects = sorted(reg.projects)
        m.tool_entity_seed(
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "personal"}],
            aliases={"MW": "Maxwell"},
        )
        reg = get_entity_registry(m._config, team=team_x)
        assert len(reg.people) == before_people
        assert sorted(reg.people["Maxwell"]["contexts"]) == before_ctx
        assert sorted(reg.people["Maxwell"]["aliases"]) == before_aliases
        assert sorted(reg.projects) == before_projects
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# AC6: chroma/local backend → available: false gracefully (no crash)
# ─────────────────────────────────────────────────────────────────────────────


def test_both_tools_available_false_on_chroma(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.setenv("HOME", str(tmp_path))  # any host write would land here

    res_read = m.tool_disambiguate("Max")
    assert res_read["available"] is False
    assert res_read["backend"] == "chroma"
    assert "reason" in res_read

    res_write = m.tool_entity_seed(
        people=[{"name": "Maxwell", "relationship": "x", "context": "work"}],
        aliases={"Max": "Maxwell"},
    )
    assert res_write["available"] is False
    assert res_write["backend"] == "chroma"
    assert "reason" in res_write

    # The unsupported path created no host-global registry file.
    assert list(tmp_path.rglob("entity_registry*")) == []


# ─────────────────────────────────────────────────────────────────────────────
# AC7: persistence across a fresh open of the per-team registry
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_seed_persists_across_fresh_registry_open(pg_env, backend):
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        m.tool_entity_seed(
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "work"}],
            aliases={"Max": "Maxwell"},
        )

        # Fresh open straight from storage (not the in-memory object the tool used)
        # proves PostgresEntityRegistry.save round-trips the seeded document.
        from mempalace.entity_registry_postgres import PostgresEntityRegistry

        reopened = PostgresEntityRegistry.open(backend, team=team_x)
        result = reopened.lookup("Max")
        assert result["type"] == "person"
        # The alias entry persisted and points back at its canonical.
        assert reopened.people["Max"]["canonical"] == "Maxwell"
        assert "Max" in reopened.people["Maxwell"]["aliases"]

        # And the tool, reading the same persisted vault fresh, resolves the alias
        # to the canonical name.
        assert m.tool_disambiguate("Max")["name"] == "Maxwell"
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# AC8: lane isolation — entity_seed leaks ZERO rows into kg / team-facts lanes
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_entity_seed_does_not_touch_kg_or_team_facts(pg_env, backend):
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)

        # Snapshot the other two lanes BEFORE the registry write.
        kg_before = m.tool_kg_query("Maxwell")["count"]
        neighbors_before = m.tool_kg_neighbors("Maxwell", depth=2)["count"]
        facts_before = m.tool_team_facts()["count"]

        m.tool_entity_seed(
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "work"}],
            projects=["A", "B"],
            aliases={"Max": "Maxwell"},
        )

        # The name-resolution write must not have created any kg edge or team fact.
        assert m.tool_kg_query("Maxwell")["count"] == kg_before
        assert m.tool_kg_neighbors("Maxwell", depth=2)["count"] == neighbors_before
        assert m.tool_team_facts()["count"] == facts_before

        # And the registry lane itself did record the data (the write landed).
        assert m.tool_disambiguate("Max")["name"] == "Maxwell"
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# Registration: both tools present in the TOOLS dict with the required prefix
# ─────────────────────────────────────────────────────────────────────────────


def test_tools_registered_in_tools_dict():
    for name in ("mempalace_disambiguate", "mempalace_entity_seed"):
        assert name in m.TOOLS, f"{name} not registered in TOOLS"
        entry = m.TOOLS[name]
        assert callable(entry["handler"])
        assert "input_schema" in entry
    assert m.TOOLS["mempalace_disambiguate"]["input_schema"]["required"] == ["name"]
    assert m.TOOLS["mempalace_entity_seed"]["input_schema"]["required"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: mode is set on first seed only; a later seed must not overwrite it
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_mode_set_on_first_seed_only(pg_env, backend):
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        # First seed with mode="work" on an empty registry — mode must be stored.
        m.tool_entity_seed(
            mode="work",
            people=[{"name": "Maxwell", "relationship": "colleague", "context": "work"}],
        )
        from mempalace.entity_registry import get_entity_registry

        reg = get_entity_registry(m._config, team=team_x)
        assert reg.mode == "work"

        # Second seed with a different mode — mode must NOT be overwritten.
        m.tool_entity_seed(
            mode="personal",
            people=[{"name": "Grace", "relationship": "daughter", "context": "personal"}],
        )
        reg = get_entity_registry(m._config, team=team_x)
        assert reg.mode == "work"  # preserved from first seed
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: malformed people item missing `name` is skipped, valid item stored
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_malformed_people_item_missing_name_is_skipped(pg_env, backend):
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        # One valid item, one missing `name` — must not raise.
        res = m.tool_entity_seed(
            people=[
                {"name": "Maxwell", "relationship": "colleague", "context": "work"},
                {"relationship": "unknown", "context": "work"},  # no `name` key
            ],
        )
        assert "error" not in res
        from mempalace.entity_registry import get_entity_registry

        reg = get_entity_registry(m._config, team=team_x)
        assert "Maxwell" in reg.people
        assert len(reg.people) == 1  # malformed entry was not stored
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


# ─────────────────────────────────────────────────────────────────────────────
# kind field: project alias resolves type=project (postgres parity)
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_project_alias_resolves_project_postgres(pg_env, backend):
    # #15/#9 parity — disambiguate opens the registry fresh from Postgres each
    # call, so a passing assertion also proves the kind round-trips through jsonb.
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        m.tool_entity_seed(projects=["mempalace"], aliases={"mempalace-poc": "mempalace"})

        res_alias = m.tool_disambiguate("mempalace-poc")
        assert res_alias["found"] is True
        assert res_alias["type"] == "project"
        assert res_alias["name"] == "mempalace"
        assert res_alias["alias_of"] == "mempalace"

        res_canon = m.tool_disambiguate("mempalace")
        assert res_canon["type"] == "project"

        # The canonical project must not have leaked into the people dict.
        from mempalace.entity_registry import get_entity_registry

        reg = get_entity_registry(m._config, team=team_x)
        assert "mempalace" not in reg.people
        assert reg.people["mempalace-poc"]["kind"] == "project"
        assert reg.check_kind_invariant() == []
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)


@_needs_pg
def test_person_and_project_aliases_coexist_postgres(pg_env, backend):
    # A person alias stays person and a project alias is project, in one vault.
    team_x = _new_team()
    try:
        m._active_team_var.set(team_x)
        m.tool_entity_seed(
            people=[{"name": "Markus Burger"}],
            projects=["mempalace"],
            aliases={"MB": "Markus Burger", "mempalace-poc": "mempalace"},
        )
        person = m.tool_disambiguate("MB")
        assert person["type"] == "person"
        assert person["name"] == "Markus Burger"
        assert person["alias_of"] == "Markus Burger"

        project = m.tool_disambiguate("mempalace-poc")
        assert project["type"] == "project"
        assert project["name"] == "mempalace"
    finally:
        m._active_team_var.set(None)
        _drop(backend, team_x)
