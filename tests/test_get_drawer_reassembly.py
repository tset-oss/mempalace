"""get_drawer dereferences the LOGICAL id of a chunked drawer (verbatim reassembly).

Oversized content is stored as physical ``{id}_chunk_NNNNNN`` rows with NO row
under the logical ``drawer_id`` add_drawer returns, so ``get_drawer(logical_id)``
used to report "not found" — the id the caller was handed was undereferenceable.
``tool_get_drawer`` now falls back to the chunks carrying ``parent_drawer_id`` and
rejoins them, byte-exact, in ``chunk_index`` order.

Two layers of coverage:
* Hermetic unit tests drive ``_reassemble_chunked_drawer`` with a fake collection
  to pin the load-bearing ordering/error behaviour — including the chroma case
  where ``get(where=)`` returns chunks UNORDERED (the explicit sort is the real
  guard there), and the refuse-rather-than-scramble paths.
* Postgres integration tests drive the real add -> get round-trip, including a
  drawer with more chunks than the default page size (no silent truncation) and a
  chunked diary entry.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

import mempalace.mcp_server as mcp
from mempalace.mcp_server import _reassemble_chunked_drawer, _resolve_drawer_physical_ids

# ── Hermetic: _reassemble_chunked_drawer over a fake collection ──────────────


class _FakeCol:
    """Minimal collection stub: only the ``get(where={parent_drawer_id})`` path."""

    def __init__(self, rows):
        # rows: list of (id, document, metadata)
        self._rows = rows

    def get(self, where=None, include=None, ids=None, limit=None):
        # The reassembly must pass NO limit (a cap would truncate -> verbatim
        # violation); fail loudly here if a future change starts passing one.
        assert limit is None, "reassembly must not paginate the chunk fetch"
        pid = (where or {}).get("parent_drawer_id")
        sel = [r for r in self._rows if (r[2] or {}).get("parent_drawer_id") == pid]
        return {
            "ids": [r[0] for r in sel],
            "documents": [r[1] for r in sel],
            "metadatas": [r[2] for r in sel],
        }


def _chunk(pid, idx, text, *, with_index=True, cid=None):
    meta = {"parent_drawer_id": pid, "wing": "ops", "room": "r1"}
    if with_index:
        meta["chunk_index"] = idx
    return (cid or f"{pid}_chunk_{idx:06d}", text, meta)


def test_reassembles_unordered_chunks_in_chunk_index_order():
    """Chroma's get(where=) is unordered, so the chunk_index sort is load-bearing."""
    pid = "drawer_x"
    # Supplied in REVERSE order on purpose — the result must still be in order.
    rows = [_chunk(pid, 2, "ccc"), _chunk(pid, 0, "aaa"), _chunk(pid, 1, "bbb")]
    out = _reassemble_chunked_drawer(_FakeCol(rows), pid)
    assert out["content"] == "aaabbbccc", out
    assert out["chunks"] == 3
    assert out["chunk_ids"] == [f"{pid}_chunk_{i:06d}" for i in range(3)]
    assert "chunk_index" not in out["metadata"]  # chunk-local key stripped


def test_reassembles_via_id_suffix_when_chunk_index_metadata_absent():
    """A chunk missing chunk_index metadata still orders via the _chunk_NNNNNN id."""
    pid = "drawer_y"
    rows = [
        _chunk(pid, 1, "world", with_index=False),
        _chunk(pid, 0, "hello", with_index=False),
    ]
    out = _reassemble_chunked_drawer(_FakeCol(rows), pid)
    assert out["content"] == "helloworld", out


def test_unresolvable_chunk_index_returns_error_not_scramble():
    """No chunk_index metadata AND no _chunk_NNNNNN suffix -> explicit error."""
    pid = "drawer_z"
    rows = [
        _chunk(pid, 0, "aaa", with_index=False, cid="weird_id_no_suffix_1"),
        _chunk(pid, 1, "bbb", with_index=False, cid="weird_id_no_suffix_2"),
    ]
    out = _reassemble_chunked_drawer(_FakeCol(rows), pid)
    assert "error" in out and "resolvable chunk_index" in out["error"], out


def test_duplicate_chunk_index_returns_error_not_scramble():
    """Two chunks claiming the same index -> refuse to concatenate."""
    pid = "drawer_dup"
    rows = [_chunk(pid, 0, "aaa"), _chunk(pid, 0, "bbb", cid=f"{pid}_chunk_000099")]
    out = _reassemble_chunked_drawer(_FakeCol(rows), pid)
    assert "error" in out and "duplicate chunk_index" in out["error"], out


def test_missing_chunk_in_sequence_returns_error_not_truncation():
    """A gap in the chunk_index sequence (a chunk is MISSING, e.g. partial delete)
    must error rather than silently concatenate the survivors — verbatim always."""
    pid = "drawer_gap"
    rows = [_chunk(pid, 0, "aaa"), _chunk(pid, 2, "ccc")]  # chunk 1 missing
    out = _reassemble_chunked_drawer(_FakeCol(rows), pid)
    assert "error" in out and "not contiguous" in out["error"], out


def test_no_chunks_returns_none():
    """A bare id that is neither a row nor a parent -> None (caller -> not found)."""
    assert _reassemble_chunked_drawer(_FakeCol([]), "nope") is None


# ── Hermetic: _resolve_drawer_physical_ids (the addressing authority) ─────────


class _IdCol:
    """Collection stub serving BOTH get(ids=[...]) and get(where={parent}).

    ``_FakeCol`` only models the parent-fetch path; ``_resolve_drawer_physical_ids``
    also does a literal id lookup first, so it needs an ids= path too.
    """

    def __init__(self, rows):
        # rows: list of (id, document, metadata)
        self._rows = rows

    def get(self, where=None, include=None, ids=None, limit=None):
        if ids is not None:
            sel = [r for r in self._rows if r[0] in ids]
        else:
            pid = (where or {}).get("parent_drawer_id")
            sel = [r for r in self._rows if (r[2] or {}).get("parent_drawer_id") == pid]
        return {
            "ids": [r[0] for r in sel],
            "documents": [r[1] for r in sel],
            "metadatas": [r[2] for r in sel],
        }


def test_resolve_single_chunk_base_id_returns_base_only():
    """A single-chunk drawer: one row under the base id -> (base_id, [base_id])."""
    base = "drawer_ops_r1_singledeadbeef"
    rows = [(base, "small note", {"wing": "ops", "room": "r1", "chunk_index": 0})]
    parent, ids = _resolve_drawer_physical_ids(_IdCol(rows), base)
    assert parent == base
    assert ids == [base]


def test_resolve_multi_chunk_base_id_returns_all_chunk_ids():
    """A logical (chunked) handle: no base row -> (base_id, [all chunk ids sorted])."""
    base = "drawer_ops_r1_chunkydeadbeef"
    rows = [_chunk(base, 0, "aaa"), _chunk(base, 1, "bbb"), _chunk(base, 2, "ccc")]
    parent, ids = _resolve_drawer_physical_ids(_IdCol(rows), base)
    assert parent == base
    assert ids == [f"{base}_chunk_{i:06d}" for i in range(3)]


def test_resolve_chunk_id_climbs_to_full_parent_set():
    """Passing a physical CHUNK id climbs to its parent and returns the whole set."""
    base = "drawer_ops_r1_chunkydeadbeef"
    rows = [_chunk(base, 0, "aaa"), _chunk(base, 1, "bbb"), _chunk(base, 2, "ccc")]
    parent, ids = _resolve_drawer_physical_ids(_IdCol(rows), f"{base}_chunk_000001")
    assert parent == base
    assert ids == [f"{base}_chunk_{i:06d}" for i in range(3)]


def test_resolve_unknown_id_returns_empty():
    """An id that is neither a row nor a parent -> ('drawer_nope', [])."""
    parent, ids = _resolve_drawer_physical_ids(_IdCol([]), "drawer_nope")
    assert parent == "drawer_nope"
    assert ids == []


def test_resolve_tolerates_none_metadata_cell():
    """A None metadata cell on the literal row must not crash (defensive _safe_meta)."""
    base = "drawer_ops_r1_nullmetadeadbeef"
    rows = [(base, "note", None)]  # single-row base with no metadata
    parent, ids = _resolve_drawer_physical_ids(_IdCol(rows), base)
    assert parent == base
    assert ids == [base]


# ── Postgres integration: real add -> get round-trip ─────────────────────────

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


_PG = _reachable()
pg_only = pytest.mark.skipif(not _PG, reason="no reachable Postgres")


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


@pg_only
def test_get_logical_id_reassembles_chunked_drawer(monkeypatch):
    team = _setup(monkeypatch)
    try:
        content = "The renovate operator rollout note. " * 60  # > chunk_size
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=content)
        assert add["chunks"] > 1, add
        got = mcp.tool_get_drawer(add["drawer_id"])  # the LOGICAL id
        assert got.get("error") is None, got
        assert got["content"] == content  # byte-exact
        assert got["chunks"] == add["chunks"]
        # A physical chunk id now CLIMBS to the whole drawer (#1539 Option A:
        # logical-id everywhere), not just that chunk's slice.
        one = mcp.tool_get_drawer(add["chunk_ids"][1])
        assert one.get("error") is None, one
        assert one["content"] == content  # whole drawer, byte-exact
        assert one["chunks"] == add["chunks"]
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


@pg_only
def test_get_logical_id_reassembles_more_chunks_than_page_size(monkeypatch):
    """No silent truncation: a drawer with > default page size chunks reassembles whole."""
    team = _setup(monkeypatch)
    try:
        content = "x" * 81000  # 800-char chunks -> ~102 chunks, exceeds _MAX_RESULTS=100
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=content)
        assert add["chunks"] > mcp._MAX_RESULTS, add
        got = mcp.tool_get_drawer(add["drawer_id"])
        assert got.get("error") is None, got
        assert len(got["content"]) == len(content)
        assert got["content"] == content
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


@pg_only
def test_get_nonexistent_and_single_chunk(monkeypatch):
    team = _setup(monkeypatch)
    try:
        assert mcp.tool_get_drawer("drawer_does_not_exist").get("error")
        add = mcp.tool_add_drawer(wing="ops", room="r1", content="a small note")
        got = mcp.tool_get_drawer(add["drawer_id"])
        assert got["content"] == "a small note"
        assert "chunks" not in got  # single-row path unchanged
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


@pg_only
def test_get_logical_id_reassembles_chunked_diary_entry(monkeypatch):
    """tool_diary_write also stamps parent_drawer_id, so get_drawer reassembles diary too."""
    team = _setup(monkeypatch)
    try:
        body = "diary entry body line. " * 60  # > chunk_size -> chunked
        token = mcp._active_team_var.set(team)
        try:
            d = mcp.tool_diary_write(agent_name="tester", entry=body, topic="rollout")
        finally:
            mcp._active_team_var.reset(token)
        if d.get("chunks", 1) <= 1:
            pytest.skip("diary entry did not chunk in this config")
        got = mcp.tool_get_drawer(d["entry_id"])
        assert got.get("error") is None, got
        assert got["content"] == body  # byte-exact
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)
