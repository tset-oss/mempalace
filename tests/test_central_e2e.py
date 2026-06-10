"""End-to-end smoke test for the central, team-vaulted deployment (G005).

Drives the real MCP tool surface (add_drawer -> search -> list_vaults) against a
live Postgres, proving that:

* a machine configured for team `frontend` files and finds memories in the
  `team_frontend` vault, and
* a machine configured for team `backend` cannot see the frontend memories
  (schema-per-team isolation), and
* list_vaults reports both teams.

Skipped when no Postgres is reachable.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends import get_backend  # noqa: E402
from mempalace.backends.postgres import team_schema  # noqa: E402

DIM = 384


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


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * DIM
        for i, b in enumerate(hashlib.sha256((t or "").encode()).digest()):
            v[i] = b / 255.0
        out.append(v)
    return out


def _drop(*teams):
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            for t in teams:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
        conn.commit()


def _use_team(mcp, monkeypatch, team):
    from mempalace.config import MempalaceConfig

    monkeypatch.setenv("MEMPALACE_TEAM", team)
    monkeypatch.setattr(mcp, "_config", MempalaceConfig())
    mcp._kg_by_path.clear()
    # Set the per-request contextvar so strict resolvers see the team.
    # Use monkeypatch.setattr on _active_team_var so monkeypatch teardown
    # restores the original ContextVar object (and thus its original value).
    # On the first call per test we swap in a fresh ContextVar pre-seeded with
    # team; subsequent calls (e.g. switching fe -> be) just set the current var.
    if not getattr(monkeypatch, "_team_var_replaced", False):
        import contextvars

        new_var = contextvars.ContextVar("_active_team_var_test", default=None)
        new_var.set(team)
        monkeypatch.setattr(mcp, "_active_team_var", new_var)
        monkeypatch._team_var_replaced = True
    else:
        mcp._active_team_var.set(team)


def test_central_team_vault_e2e(monkeypatch):
    import mempalace.mcp_server as mcp

    fe = "t" + uuid.uuid4().hex[:10]
    be = "t" + uuid.uuid4().hex[:10]
    content = "frontend widgets api design notes"

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    # Deterministic embedder on the shared registry backend (no ONNX download).
    get_backend("postgres")._embedder = _fake_embed
    try:
        # --- frontend machine files a memory ---
        _use_team(mcp, monkeypatch, fe)
        add = mcp.tool_add_drawer(wing="project", room="api", content=content)
        assert add.get("success") is True, add

        found = mcp.tool_search(content, limit=5)
        found_texts = " ".join(r.get("text", "") for r in found.get("results", []))
        assert content in found_texts, found  # frontend finds its own memory

        # The row physically lives in the frontend team's schema.
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT count(*) FROM "{team_schema(fe)}"."mempalace_drawers"')
                assert cur.fetchone()[0] >= 1

        # --- backend machine cannot see it (isolation) ---
        _use_team(mcp, monkeypatch, be)
        # create the (empty) backend vault so search has a collection to open
        mcp.tool_add_drawer(wing="project", room="api", content="backend unrelated note")
        iso = mcp.tool_search(content, limit=5)
        iso_texts = " ".join(r.get("text", "") for r in iso.get("results", []))
        assert content not in iso_texts, iso  # backend cannot see the frontend memory

        # --- list_vaults reports both teams, primary = backend (current config) ---
        vaults = mcp.tool_list_vaults()
        assert vaults["backend"] == "postgres"
        assert vaults["primary"] == be
        assert fe in vaults["vaults"] and be in vaults["vaults"]
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(fe, be)


def test_check_duplicate_on_empty_central_vault_is_dup_safe(monkeypatch):
    """An empty central vault returns is_duplicate:False WITH a reason — not the
    misleading chroma-shaped "No palace found" error. The vault has had no
    writes, so _get_collection() returns None and we report "nothing to compare
    against" instead of an error that reads like a missing local palace."""
    import mempalace.mcp_server as mcp

    team = "t" + uuid.uuid4().hex[:10]

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    try:
        _use_team(mcp, monkeypatch, team)

        res = mcp.tool_check_duplicate("anything at all", threshold=0.9)

        assert res["is_duplicate"] is False
        assert res["matches"] == []
        assert res["vault"] == team
        assert res.get("empty_vault") is True
        # A reason that names the empty/no-entries state...
        reason = res["reason"].lower()
        assert "empty" in reason or "no entries" in reason
        # ...and explicitly NOT the _no_palace() error shape.
        assert "error" not in res
        assert res.get("error") != "No palace found"
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(team)


def test_empty_vault_dup_false_is_safe_idempotent_refile(monkeypatch):
    """Dup-integrity: even though check_duplicate returns False on the empty
    vault, an exact re-file is caught by add_drawer's content-hash idempotency
    probe — the second identical add returns reason 'already_exists' and does
    NOT create a second row. This is what makes the empty-vault False safe."""
    import mempalace.mcp_server as mcp

    team = "t" + uuid.uuid4().hex[:10]
    content = "verbatim note that we will try to file twice"

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    try:
        _use_team(mcp, monkeypatch, team)

        # The vault is empty -> check_duplicate is False with no error.
        pre = mcp.tool_check_duplicate(content, threshold=0.9)
        assert pre["is_duplicate"] is False
        assert "error" not in pre

        first = mcp.tool_add_drawer(wing="project", room="api", content=content)
        assert first.get("success") is True, first
        assert first.get("reason") != "already_exists"

        # Identical wing/room/content -> idempotency probe short-circuits.
        second = mcp.tool_add_drawer(wing="project", room="api", content=content)
        assert second.get("success") is True, second
        assert second.get("reason") == "already_exists", second

        # No duplicate row was created — exactly one drawer in the vault.
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT count(*) FROM "{team_schema(team)}"."mempalace_drawers"')
                assert cur.fetchone()[0] == 1
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(team)


def test_check_duplicate_flags_near_identical_in_populated_vault(monkeypatch):
    """Once the vault HAS content, the normal similarity path still works: a
    near-identical re-check of filed content is flagged is_duplicate:True with a
    match. The empty-vault branch must not have regressed real detection."""
    import mempalace.mcp_server as mcp

    team = "t" + uuid.uuid4().hex[:10]
    content = "the deployment pipeline runs on gitlab ci with road runner stages"

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    try:
        _use_team(mcp, monkeypatch, team)

        add = mcp.tool_add_drawer(wing="project", room="ci", content=content)
        assert add.get("success") is True, add

        # The deterministic embedder maps identical text to an identical vector,
        # so an exact re-check is a perfect (similarity == 1.0) match.
        res = mcp.tool_check_duplicate(content, threshold=0.9)
        assert res["is_duplicate"] is True, res
        assert len(res["matches"]) >= 1
        assert res["matches"][0]["similarity"] >= 0.9
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(team)


def test_check_duplicate_propagates_backend_error_not_empty_false(monkeypatch):
    """Load-bearing invariant: _get_collection() returns None ONLY for a
    genuinely-empty vault; a transient/backend error RAISES. check_duplicate must
    therefore NOT convert an error into the dup-safe is_duplicate:False empty-vault
    shape — doing so would mask a near-dupe in a populated-but-unreadable vault. If
    a future change broadened _get_collection's except clause to swallow errors
    into None, this test fails."""
    import mempalace.mcp_server as mcp

    team = "t" + uuid.uuid4().hex[:10]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    try:
        _use_team(mcp, monkeypatch, team)

        def _raise_transient(*args, **kwargs):
            raise RuntimeError("simulated backend/connection failure")

        monkeypatch.setattr(mcp, "_get_collection", _raise_transient)
        # The error must propagate, NOT be reported as an empty, dup-safe vault.
        with pytest.raises(RuntimeError):
            mcp.tool_check_duplicate("anything", threshold=0.9)
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(team)


# ---------------------------------------------------------------------------
# T3 — Slice 5: fresh-vault search returns central empty shape, not chroma hint
# ---------------------------------------------------------------------------


def test_fresh_vault_search_returns_central_shape(monkeypatch):
    """T3-POS: searching a never-written postgres vault returns the central
    empty_vault shape — no 'error' key, no chroma init/mine hint."""
    import mempalace.mcp_server as mcp

    team = "t" + uuid.uuid4().hex[:10]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    try:
        _use_team(mcp, monkeypatch, team)

        res = mcp.tool_search("anything")

        # Must use the central empty-vault envelope.
        assert "error" not in res, f"unexpected error key: {res}"
        assert res.get("empty_vault") is True, res
        assert res.get("results") == [], res
        assert res.get("vault") == team, res
        reason = res.get("reason", "")
        assert team in reason or "no entries" in reason or "does not exist" in reason
        # Chroma hint must NOT appear.
        assert "mempalace init" not in res.get("reason", "")
        assert "mempalace init" not in str(res.get("hint", ""))
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(team)


def test_fresh_vault_search_transient_error_surfaces_as_error(monkeypatch):
    """T3-NEG: a transient backend error during search must surface as an error
    dict, NOT be silently swallowed into the empty_vault shape (PM#3 invariant)."""
    import mempalace.mcp_server as mcp
    import mempalace.searcher as searcher_mod

    team = "t" + uuid.uuid4().hex[:10]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    try:
        _use_team(mcp, monkeypatch, team)

        # Patch the get_collection name inside searcher (module-level import) to
        # raise a non-palace exception, simulating a DB outage.
        def _raise_db(*args, **kwargs):
            raise RuntimeError("simulated DB outage")

        monkeypatch.setattr(searcher_mod, "get_collection", _raise_db)

        res = mcp.tool_search("anything")

        # Must surface as an error, not an empty_vault.
        assert "error" in res, f"expected error key, got: {res}"
        assert res.get("empty_vault") is None or res.get("empty_vault") is False
    finally:
        get_backend("postgres")._embedder = None
        mcp._kg_by_path.clear()
        _drop(team)
