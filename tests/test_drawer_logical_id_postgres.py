"""Entity-row consequences of the drawer logical-id consolidation (#1539 Option A).

The chroma-path tests in test_mcp_server.py pin the verbatim ROW mechanics
(full-set delete, re-chunk, no-stale-rows, metadata-only move, chunk-id climb).
These postgres integration tests pin the ENTITY-INDEX side-effects that only
exist on the server backend:

* G002: deleting a multi-chunk drawer by its base id clears its entity rows
  (delete_by_parent sweeps the bare id + every ``{parent}_chunk_*`` row).
* G003: after a content update the entity rows match the NEW chunk-id set, with
  no stranded sibling rows from the prior (larger) chunk count — the core bug
  the consolidation targets, where the old code re-indexed only ``[(parent,
  new_doc)]``.
* G003: a metadata-only wing move updates the entity rows' wing across the whole
  drawer without re-chunking.

Skipped when no Postgres is reachable. Mirrors the fixtures in
test_topic_entity_recall_postgres.py / test_get_drawer_reassembly.py.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

import mempalace.mcp_server as mcp  # noqa: E402
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


def _drop(team):
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
        conn.commit()


def _setup(monkeypatch):
    from mempalace.config import MempalaceConfig

    team = "t" + uuid.uuid4().hex[:10]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setenv("MEMPALACE_TEAM", team)
    get_backend("postgres")._embedder = _fake_embed
    monkeypatch.setattr(mcp, "_config", MempalaceConfig())
    mcp._kg_by_path.clear()
    return team


def _entity_drawer_ids(team, entity):
    return {r["drawer_id"] for r in mcp._get_entity_index(team).drawers_for_entity(entity)}


def _entity_wings(team, entity):
    return {r["wing"] for r in mcp._get_entity_index(team).drawers_for_entity(entity)}


# A proper-noun entity ("Dana") repeated so it lands in every chunk's extracted
# entity set; long filler keeps the content multi-chunk at the 800-char default.
def _dana_content(n_filler_blocks):
    block = "Dana paged Dana about the incident with Dana again. "
    return ("Dana " + block * n_filler_blocks).strip()


def test_delete_multi_chunk_base_id_clears_entity_rows(monkeypatch):
    """G002: deleting a multi-chunk drawer by base id removes its entity rows."""
    team = _setup(monkeypatch)
    try:
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=_dana_content(40))
        assert add.get("chunks", 1) > 1, add
        assert _entity_drawer_ids(team, "Dana"), "precondition: Dana indexed"

        deleted = mcp.tool_delete_drawer(add["drawer_id"])
        assert deleted["success"] is True, deleted
        assert _entity_drawer_ids(team, "Dana") == set(), "entity rows orphaned after delete"
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_delete_chunk_id_clears_entity_rows(monkeypatch):
    """G004: deleting via a chunk id clears the whole drawer's entity rows."""
    team = _setup(monkeypatch)
    try:
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=_dana_content(40))
        assert add.get("chunks", 1) > 1, add
        assert _entity_drawer_ids(team, "Dana")

        deleted = mcp.tool_delete_drawer(add["chunk_ids"][1])
        assert deleted["success"] is True, deleted
        assert _entity_drawer_ids(team, "Dana") == set()
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_update_content_entity_rows_match_new_chunk_set(monkeypatch):
    """G003: after a content update the entity rows are keyed on the NEW chunk
    set — no stranded siblings from the prior (larger) chunk count."""
    team = _setup(monkeypatch)
    try:
        # Start large (many chunks), shrink to fewer chunks.
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=_dana_content(80))
        big = add["chunks"]
        assert big > 2, add

        upd = mcp.tool_update_drawer(add["drawer_id"], content=_dana_content(12))
        assert upd["success"] is True, upd
        assert upd["chunks"] < big, upd

        # The entity rows for Dana must be EXACTLY the new physical chunk ids.
        new_ids = set(upd.get("chunk_ids") or [upd["drawer_id"]])
        assert _entity_drawer_ids(team, "Dana") == new_ids, "stranded sibling entity rows remain"
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_update_metadata_only_move_updates_entity_wing(monkeypatch):
    """G003: a content=None wing move re-stamps the entity rows' wing across the
    whole drawer (no re-chunk)."""
    team = _setup(monkeypatch)
    try:
        add = mcp.tool_add_drawer(wing="oldwing", room="r1", content=_dana_content(40))
        assert add.get("chunks", 1) > 1, add
        assert _entity_wings(team, "Dana") == {"oldwing"}, _entity_wings(team, "Dana")
        before_ids = _entity_drawer_ids(team, "Dana")

        upd = mcp.tool_update_drawer(add["drawer_id"], wing="newwing")
        assert upd["success"] is True and upd["wing"] == "newwing", upd

        assert _entity_wings(team, "Dana") == {"newwing"}, _entity_wings(team, "Dana")
        # Same physical ids (no re-chunk), just re-stamped wing.
        assert _entity_drawer_ids(team, "Dana") == before_ids
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_update_preserves_topic_labels_across_rechunk(monkeypatch):
    """G003: caller topic labels survive a content re-chunk (re-stamped on the
    new chunk set), keyed under the new ids."""
    team = _setup(monkeypatch)
    try:
        add = mcp.tool_add_drawer(
            wing="ops", room="r1", content=_dana_content(40), topics=["infra.eden"]
        )
        assert add.get("chunks", 1) > 1, add
        assert _entity_drawer_ids(team, "infra.eden"), "precondition: topic indexed"

        upd = mcp.tool_update_drawer(add["drawer_id"], content=_dana_content(12))
        assert upd["success"] is True, upd

        new_ids = set(upd.get("chunk_ids") or [upd["drawer_id"]])
        topic_ids = _entity_drawer_ids(team, "infra.eden")
        assert topic_ids, "topic recall lost across re-chunk"
        assert topic_ids <= new_ids, (topic_ids, new_ids)
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)
