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
