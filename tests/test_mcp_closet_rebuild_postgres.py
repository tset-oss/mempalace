"""Live postgres e2e for the server-side closet rebuild (G005).

Proves the central, miner-less write path actually builds closets and that the
search closet boost FIRES for those drawers — including the empty-``source_file``
agent-curated case that is silently inert without this story.

The flow per test:
  1. ``tool_add_drawer`` (>= 2 drawers, same wing/room, same source) through the
     real MCP write path in postgres mode.
  2. Flush the debounce coalescer deterministically (window pinned to 0s; no
     sleeps) so the rebuild has run.
  3. ``search_memories`` for the content and assert the drawer comes back with
     ``matched_via == "drawer+closet"`` — the boost actually fired.

Self-skips with a clear reason when no Postgres is reachable, mirroring
``tests/test_search_parity_postgres.py``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")


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


_LIVE = _reachable(_dsn())

pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)

DIM = 384
PINNED_MODEL = "embeddinggemma"


def _real_hf_cache_home() -> str | None:
    for var in ("HF_HOME", "HF_HUB_CACHE"):
        val = os.environ.get(var)
        if val:
            return str(Path(val).parent if var == "HF_HUB_CACHE" else val)
    home = os.environ.get("MEMPALACE_REAL_HOME")
    if home and (Path(home) / ".cache" / "huggingface").exists():
        return str(Path(home) / ".cache" / "huggingface")
    return None


@pytest.fixture
def pg_closet_env(monkeypatch):
    """Postgres backend + pinned embeddinggemma + a fresh throwaway team vault.

    Pins the embedder process-wide (the M2 trap: env var alone is not enough —
    the cached singletons must be reset) and a 0s debounce window so flushes are
    deterministic. Tears the team schema down afterwards.
    """
    import mempalace.backends.postgres as pg_module
    import mempalace.embedding as embedding
    import mempalace.mcp_server as m
    from mempalace.backends.postgres import team_schema

    dsn = _dsn()
    team = "g005c" + uuid.uuid4().hex[:10]

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", team)
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", dsn)
    monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", PINNED_MODEL)
    monkeypatch.setenv("MEMPALACE_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("MEMPALACE_CLOSET_DEBOUNCE_SECONDS", "0")
    hf_home = _real_hf_cache_home()
    if hf_home:
        monkeypatch.setenv("HF_HOME", hf_home)
        monkeypatch.setenv("HF_HUB_CACHE", str(Path(hf_home) / "hub"))

    # Reset cached embedder singletons so the pin actually takes.
    embedding._EF_CACHE.clear()
    pg_module._embedder = None

    # Force a fresh debouncer bound to this env (0s window).
    m._closet_debouncer = None
    m._closet_reconcile_started = False
    # Set the per-request contextvar so strict resolvers see the team.
    token = m._active_team_var.set(team)

    ef = embedding.get_embedding_function()
    assert type(ef).__name__ == "EmbeddinggemmaONNX", (
        f"embedder pin failed: got {type(ef).__name__}; the _EF_CACHE/_embedder reset did not take"
    )

    try:
        yield {"dsn": dsn, "team": team, "module": m}
    finally:
        m._active_team_var.reset(token)
        try:
            deb = m._closet_debouncer
            if deb is not None:
                deb.flush(timeout=5.0)
        except Exception:
            pass
        m._closet_debouncer = None
        try:
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
                conn.commit()
        except Exception:
            pass
        embedding._EF_CACHE.clear()
        pg_module._embedder = None


def _search(team, query, n_results=5):
    from mempalace.searcher import search_memories

    return search_memories(
        query,
        palace_path="/pg-closet",
        n_results=n_results,
        team=team,
        candidate_strategy="vector",
    )


def test_empty_source_drawers_get_closet_boost(pg_closet_env):
    """Two empty-source drawers in the SAME (wing, room) -> closet under the
    fallback grouping key -> search returns one with matched_via drawer+closet.

    This is the exact path that is silently inert today: an empty source_file
    never anchors a closet, so the boost never fires.
    """
    m = pg_closet_env["module"]
    team = pg_closet_env["team"]

    # >= 2 empty-source drawers in the same wing/room so the (wing, room)
    # aggregation/collapse behaviour is actually exercised.
    r1 = m.tool_add_drawer(
        wing="people",
        room="alice",
        content="Alice migrated the billing service to the new quotient ledger pipeline.",
        source_file="",
    )
    r2 = m.tool_add_drawer(
        wing="people",
        room="alice",
        content="Alice also reviewed the quotient ledger rollout and approved the cutover plan.",
        source_file="",
    )
    assert r1.get("success") and r2.get("success"), (r1, r2)

    # Deterministically wait for the debounced rebuild (0s window + flush).
    assert m._get_closet_debouncer().flush(timeout=30.0)

    # A closet was actually written under the (wing, room) fallback key.
    from mempalace.closet_rebuild import closet_grouping_key
    from mempalace.palace import get_closets_collection

    grouping_key = closet_grouping_key("", "people", "alice")
    closets = get_closets_collection("/pg-closet", create=False, team=team)
    rows = closets.get(where={"source_file": grouping_key}, include=["metadatas"])
    assert rows.ids, "no closet was built under the empty-source fallback grouping key"

    res = _search(team, "quotient ledger pipeline rollout")
    assert "results" in res, res
    boosted = [h for h in res["results"] if h.get("matched_via") == "drawer+closet"]
    assert boosted, (
        "no empty-source drawer received the closet boost; "
        f"matched_via values: {[h.get('matched_via') for h in res['results']]}"
    )


def test_non_empty_source_drawers_get_closet_boost(pg_closet_env):
    """A non-empty source_file add -> closet under that source -> boost fires."""
    m = pg_closet_env["module"]
    team = pg_closet_env["team"]
    src = "specs/payments.md"

    r1 = m.tool_add_drawer(
        wing="projects",
        room="payments",
        content="The settlement reconciler batches transactions every fifteen minutes.",
        source_file=src,
    )
    r2 = m.tool_add_drawer(
        wing="projects",
        room="payments",
        content="Settlement reconciler retries use exponential backoff on gateway timeouts.",
        source_file=src,
    )
    assert r1.get("success") and r2.get("success"), (r1, r2)

    assert m._get_closet_debouncer().flush(timeout=30.0)

    from mempalace.palace import get_closets_collection

    closets = get_closets_collection("/pg-closet", create=False, team=team)
    rows = closets.get(where={"source_file": src}, include=["metadatas"])
    assert rows.ids, "no closet was built under the non-empty source_file"

    res = _search(team, "settlement reconciler batches transactions")
    assert "results" in res, res
    boosted = [h for h in res["results"] if h.get("matched_via") == "drawer+closet"]
    assert boosted, (
        "non-empty-source drawer did not receive the closet boost; "
        f"matched_via values: {[h.get('matched_via') for h in res['results']]}"
    )


def test_closet_points_at_real_stored_drawer_ids(pg_closet_env):
    """The rebuilt closet's drawer pointers are the ACTUAL stored ids, not
    miner-style reconstructed ids (which would be dead pointers)."""
    m = pg_closet_env["module"]
    team = pg_closet_env["team"]
    src = "specs/ids.md"

    res_add = m.tool_add_drawer(
        wing="projects",
        room="idcheck",
        content="The identity audit logged every credential rotation event.",
        source_file=src,
    )
    assert res_add.get("success"), res_add
    real_id = res_add["drawer_id"]

    assert m._get_closet_debouncer().flush(timeout=30.0)

    from mempalace.palace import get_closets_collection
    from mempalace.searcher import _extract_drawer_ids_from_closet

    closets = get_closets_collection("/pg-closet", create=False, team=team)
    rows = closets.get(where={"source_file": src}, include=["documents"])
    assert rows.ids, "no closet written"
    pointed_ids = set()
    for doc in rows.documents:
        pointed_ids.update(_extract_drawer_ids_from_closet(doc))
    assert real_id in pointed_ids, (
        f"closet does not point at the real stored drawer id {real_id}; pointed at {pointed_ids}"
    )

    # And the pointed id is actually readable in the drawers collection.
    from mempalace.palace import get_collection

    drawers = get_collection("/pg-closet", create=False, team=team)
    got = drawers.get(ids=[real_id], include=[])
    assert got.ids == [real_id], "closet points at a non-existent drawer id"
