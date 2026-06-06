"""Integration tests for the per-team team critical-facts surface (C2).

The live-DB tests run against a live Postgres (the bundled deploy/docker-compose
db or any instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). They are
skipped when psycopg is absent or the DB is unreachable, so chroma-only CI is
unaffected. The handler-level tests (chroma-unsupported, fail-loud-no-team,
registration) need neither psycopg nor a live database and always run.

What is asserted:
- a fact written in team A is readable by team A and INVISIBLE to team B
  (structural per-team isolation);
- the add/get tools on postgres with no resolvable team RAISE (no default vault);
- on chroma the tools return a structured central-server-only result (no crash,
  no host-global file);
- the length/count caps are enforced with a structured error;
- the two new tools are registered in TOOLS and documented in mcp-tools.md.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pytest

import mempalace.mcp_server as m

# ─────────────────────────────────────────────────────────────────────────────
# Live-DB self-skip plumbing (the per-team Postgres test convention)
# ─────────────────────────────────────────────────────────────────────────────

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.team_facts_postgres import (  # noqa: E402
    MAX_FACT_CHARS,
    PostgresTeamFacts,
)


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
def team(backend):
    name = "t" + uuid.uuid4().hex[:10]
    yield name
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# Store: round-trip + per-team isolation
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_add_then_list_round_trip(backend, team):
    store = PostgresTeamFacts(backend, team=team)
    added = store.add_fact("the prod DB is read-replica only", created_by="alice")
    assert added["fact"] == "the prod DB is read-replica only"
    assert added["created_by"] == "alice"
    assert isinstance(added["id"], int)
    assert isinstance(added["created_at"], str)  # ISO string, JSON-clean

    # Re-open from storage to prove it round-trips (not just the in-memory call).
    reloaded = PostgresTeamFacts(backend, team=team)
    facts = reloaded.list_facts()
    assert [f["fact"] for f in facts] == ["the prod DB is read-replica only"]


@_needs_pg
def test_facts_ordered_oldest_first(backend, team):
    store = PostgresTeamFacts(backend, team=team)
    store.add_fact("first")
    store.add_fact("second")
    store.add_fact("third")
    assert [f["fact"] for f in store.list_facts()] == ["first", "second", "third"]


@_needs_pg
def test_per_team_isolation(backend):
    team_a = "t" + uuid.uuid4().hex[:10]
    team_b = "t" + uuid.uuid4().hex[:10]
    try:
        store_a = PostgresTeamFacts(backend, team=team_a)
        store_a.add_fact("release freeze until Q3")

        # Team A sees its fact.
        assert [f["fact"] for f in store_a.list_facts()] == ["release freeze until Q3"]

        # Team B (separate schema) sees nothing of team A's.
        store_b = PostgresTeamFacts(backend, team=team_b)
        assert store_b.list_facts() == []
    finally:
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_a)}" CASCADE')
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_b)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# Caps: a too-long fact or exceeding the per-team cap is rejected
# ─────────────────────────────────────────────────────────────────────────────


@_needs_pg
def test_fact_too_long_rejected(backend, team):
    store = PostgresTeamFacts(backend, team=team)
    with pytest.raises(ValueError, match="too long"):
        store.add_fact("x" * (MAX_FACT_CHARS + 1))
    # The rejected fact was not stored.
    assert store.list_facts() == []


@_needs_pg
def test_empty_fact_rejected(backend, team):
    store = PostgresTeamFacts(backend, team=team)
    with pytest.raises(ValueError, match="required"):
        store.add_fact("   ")


@_needs_pg
def test_per_team_count_cap_enforced(backend, team, monkeypatch):
    # Shrink the cap so the test does not insert 200 rows.
    monkeypatch.setattr("mempalace.team_facts_postgres.MAX_FACTS_PER_TEAM", 3)
    store = PostgresTeamFacts(backend, team=team)
    for i in range(3):
        store.add_fact(f"fact {i}")
    with pytest.raises(ValueError, match="cap is 3"):
        store.add_fact("one too many")
    assert len(store.list_facts()) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Handler-level: chroma returns the structured central-server-only result
# (no crash, no host-global file). These need no DB.
# ─────────────────────────────────────────────────────────────────────────────


def test_tool_team_fact_add_chroma_unsupported(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.setenv("HOME", str(tmp_path))  # any host write would land here
    res = m.tool_team_fact_add("the prod DB is read-replica only")
    assert res["available"] is False
    assert res["backend"] == "chroma"
    assert "reason" in res
    # No host-global artifact created by the unsupported path.
    assert list(tmp_path.rglob("critical_facts*")) == []


def test_tool_team_facts_chroma_unsupported(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.setenv("HOME", str(tmp_path))
    res = m.tool_team_facts()
    assert res["available"] is False
    assert res["backend"] == "chroma"
    assert "reason" in res
    assert list(tmp_path.rglob("critical_facts*")) == []


# ─────────────────────────────────────────────────────────────────────────────
# Handler-level: fail-loud on postgres with no resolvable team (no default vault)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pg_no_team(monkeypatch):
    """Postgres backend, no explicit team and no active session team."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.delenv("MEMPALACE_TEAM", raising=False)
    token = m._active_team_var.set(None)
    yield monkeypatch
    m._active_team_var.reset(token)


def test_tool_team_fact_add_no_team_raises(pg_no_team):
    with pytest.raises(ValueError, match="no team resolved"):
        m.tool_team_fact_add("the prod DB is read-replica only")


def test_tool_team_facts_no_team_raises(pg_no_team):
    with pytest.raises(ValueError, match="no team resolved"):
        m.tool_team_facts()


# ─────────────────────────────────────────────────────────────────────────────
# Handler-level: end-to-end through the tools on a live DB (add → list; isolation)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pg_team_env(monkeypatch):
    """Force postgres backend + a known primary team via env."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    token = m._active_team_var.set(None)
    # Reset the per-team store cache so a stale cached store is not reused.
    monkeypatch.setattr(m, "_team_facts_by_team", {})
    yield monkeypatch
    m._active_team_var.reset(token)


@_needs_pg
def test_tools_add_then_list_and_isolation(pg_team_env, backend):
    monkeypatch = pg_team_env
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    team_a = "t" + uuid.uuid4().hex[:10]
    team_b = "t" + uuid.uuid4().hex[:10]
    try:
        # Route to team A and add a fact through the tool.
        m._active_team_var.set(team_a)
        add_res = m.tool_team_fact_add("the prod DB is read-replica only", created_by="bot")
        assert add_res["vault"] == team_a
        assert add_res["added"]["fact"] == "the prod DB is read-replica only"

        # Team A reads it back through the tool.
        list_a = m.tool_team_facts()
        assert list_a["vault"] == team_a
        assert [f["fact"] for f in list_a["facts"]] == ["the prod DB is read-replica only"]
        assert list_a["count"] == 1

        # Team B sees nothing.
        m._active_team_var.set(team_b)
        list_b = m.tool_team_facts()
        assert list_b["vault"] == team_b
        assert list_b["facts"] == []
        assert list_b["count"] == 0
    finally:
        m._active_team_var.set(None)
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_a)}" CASCADE')
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_b)}" CASCADE')


@_needs_pg
def test_tool_team_fact_add_too_long_returns_error(pg_team_env, backend):
    monkeypatch = pg_team_env
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    team_x = "t" + uuid.uuid4().hex[:10]
    try:
        m._active_team_var.set(team_x)
        res = m.tool_team_fact_add("x" * (MAX_FACT_CHARS + 1))
        assert "error" in res
        assert "too long" in res["error"]
    finally:
        m._active_team_var.set(None)
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_x)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# Registration + documentation
# ─────────────────────────────────────────────────────────────────────────────


def test_tools_registered_in_tools_dict():
    for name in ("mempalace_team_fact_add", "mempalace_team_facts"):
        assert name in m.TOOLS, f"{name} not registered in TOOLS"
        entry = m.TOOLS[name]
        assert callable(entry["handler"])
        assert "input_schema" in entry
    # The write tool requires `fact`; the read tool takes no required params.
    assert m.TOOLS["mempalace_team_fact_add"]["input_schema"]["required"] == ["fact"]
    assert m.TOOLS["mempalace_team_facts"]["input_schema"]["required"] == []


def test_tools_documented_in_mcp_tools_doc():
    doc_path = Path(__file__).resolve().parent.parent / "website" / "reference" / "mcp-tools.md"
    doc = doc_path.read_text(encoding="utf-8")
    headings = set(re.findall(r"^###\s+`(mempalace_\w+)`", doc, re.MULTILINE))
    assert "mempalace_team_fact_add" in headings
    assert "mempalace_team_facts" in headings
