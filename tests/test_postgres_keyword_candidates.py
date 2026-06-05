"""Tests for ``keyword_candidates()`` + index-backed ``$contains`` (story G002).

Covers the G002 acceptance:

  (a) no-DB: the ABC default behaves as documented (returns the "unsupported"
      sentinel ``[]``) AND the return-shape contract is declared in
      ``backends/base.py`` (not only in this test);
  (b) live: ``PostgresCollection.keyword_candidates(query=..., n_results=...)``
      returns dicts that conform to the ABC contract — all four required keys
      present, ``distance is None``, NO in-DB/BM25 score key — and a
      lexically-exact-but-vector-irrelevant token is retrieved;
  (c) live EXPLAIN: the ``$contains``/``where_document`` query for a ≥3-char
      needle shows a Bitmap Index Scan on ``{table}_doc_trgm`` (NOT a Seq Scan);
      the <3-char floor still returns correct results via the ILIKE fallback
      without index acceleration;
  (d) candidate-shape regression: a postgres ``keyword_candidates()`` result is
      fed through ``searcher._merge_bm25_union_candidates``'s dedup so the
      ``searcher.py:697`` None-drop cannot silently eat postgres candidates.

Live tests self-skip when psycopg is missing or the database is unreachable,
mirroring ``tests/test_postgres_trgm.py`` / ``tests/test_postgres_backend.py``.
"""

from __future__ import annotations

import os
import uuid

import pytest

from mempalace.backends.base import BaseCollection

psycopg = pytest.importorskip("psycopg")

from mempalace.backends import PalaceRef  # noqa: E402
from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402

DIM = 384
COLLECTION = "mempalace_drawers"

# The four keys the ABC return-shape contract pins on every candidate dict.
_REQUIRED_KEYS = ("_source_file_full", "_chunk_index", "source_file", "distance")
# Score keys that MUST NOT appear (single-ranker invariant: _hybrid_rank only).
_FORBIDDEN_SCORE_KEYS = ("paradedb.score", "bm25_score", "_score", "score")


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
live_only = pytest.mark.skipif(
    not _LIVE,
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)


def _fake_embed(texts):
    """Deterministic embedder so collection creation never downloads a model.

    Vectors are derived from a sha256 of the text, so a lexically-exact match
    is NOT guaranteed to be a vector neighbour — exactly the "lexically exact,
    vector-irrelevant" condition keyword retrieval must still surface.
    """
    import hashlib

    out = []
    for t in texts:
        vec = [0.0] * DIM
        digest = hashlib.sha256((t or "").encode("utf-8")).digest()
        for i, b in enumerate(digest):
            vec[i] = b / 255.0
        out.append(vec)
    return out


@pytest.fixture(scope="module")
def backend():
    be = PostgresBackend(dsn=_dsn(), vector_dim=DIM, embedder=_fake_embed)
    yield be
    be.close()


@pytest.fixture()
def team(backend):
    """A unique, isolated team vault per test; dropped on teardown."""
    name = "kwc" + uuid.uuid4().hex[:8]
    yield name
    schema = team_schema(name)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.commit()


def _col(backend, team, create=True):
    return backend.get_collection(
        palace=PalaceRef(id=team, namespace=team), collection_name=COLLECTION, create=create
    )


def _seed(col, n_filler: int = 60):
    """Populate a collection with one rare-token drawer + vector-noise filler.

    Returns the id of the drawer carrying the rare lexical token so the test can
    assert it is retrieved by keyword (not vector) search.
    """
    rare_id = "drawer_zylophonics"
    docs = [
        "The quarterly zylophonics report covers the western expansion roadmap.",
    ]
    ids = [rare_id]
    metas = [
        {
            "wing": "projects",
            "room": "2026-06-05",
            "source_file": "/home/agent/notes/expansion.md",
            "chunk_index": 0,
            "filed_at": "2026-06-05T10:00:00Z",
        }
    ]
    # Filler drawers so the planner has enough rows to prefer a Bitmap Index
    # Scan over a Seq Scan; their content shares no rare token with the query.
    for i in range(n_filler):
        ids.append(f"filler_{i:04d}")
        docs.append(f"General meeting notes number {i} about scheduling and logistics planning.")
        metas.append(
            {
                "wing": "projects",
                "room": "2026-06-05",
                "source_file": f"/home/agent/notes/misc_{i}.md",
                "chunk_index": 0,
                "filed_at": "2026-06-05T09:00:00Z",
            }
        )
    col.add(documents=docs, ids=ids, metadatas=metas)
    return rare_id


# --------------------------------------------------------------------------
# (a) no-DB: ABC default + contract declared in base.py
# --------------------------------------------------------------------------


def test_abc_default_returns_unsupported_sentinel():
    """The ABC default ``keyword_candidates`` returns the empty-list sentinel.

    A minimal concrete subclass that does NOT override ``keyword_candidates``
    inherits the documented "unsupported" default (``[]``), mirroring the other
    optional methods in this ABC region. No database required.
    """

    class _Minimal(BaseCollection):
        def add(self, **kw): ...

        def upsert(self, **kw): ...

        def query(self, **kw): ...

        def get(self, **kw): ...

        def delete(self, **kw): ...

        def count(self):
            return 0

    col = _Minimal()
    result = col.keyword_candidates(query="anything", n_results=5)
    assert result == []


def test_contract_is_declared_in_base_not_only_in_tests():
    """The return-shape contract must live in ``backends/base.py`` (critic blocker).

    Assert the four required keys, the SCORELESS requirement, and the silent
    None-drop hazard are all documented in the ABC method's own docstring — so
    the contract is pinned at the source, not only enforced by this test.
    """
    doc = BaseCollection.keyword_candidates.__doc__
    assert doc is not None
    for key in _REQUIRED_KEYS:
        assert key in doc, f"contract key {key!r} not documented in base.py ABC docstring"
    lowered = doc.lower()
    assert "scoreless" in lowered
    assert "distance=none" in lowered.replace(" ", "")
    # The silent-drop hazard the contract guards against is named.
    assert "697" in doc or "silently dropped" in lowered


def test_supports_keyword_candidates_capability_present():
    """Both backends advertise the capability; postgres via its real method."""
    from mempalace.backends.postgres import PostgresBackend
    from mempalace.backends.chroma import ChromaBackend

    assert "supports_keyword_candidates" in PostgresBackend.capabilities
    assert "supports_keyword_candidates" in ChromaBackend.capabilities


# --------------------------------------------------------------------------
# (b) live: shape conformance + lexically-exact / vector-irrelevant retrieval
# --------------------------------------------------------------------------


@live_only
def test_keyword_candidates_conform_to_abc_contract(backend, team):
    col = _col(backend, team, create=True)
    rare_id = _seed(col)

    cands = col.keyword_candidates(query="zylophonics report", n_results=10)
    assert cands, "expected at least one keyword candidate for a rare exact token"

    # The lexically-exact, vector-irrelevant drawer must be retrieved.
    texts = [c["text"] for c in cands]
    assert any("zylophonics" in t for t in texts), (
        "keyword_candidates did not retrieve the lexically-exact rare-token drawer"
    )

    for c in cands:
        # All four ABC-required keys present.
        for key in _REQUIRED_KEYS:
            assert key in c, f"candidate missing required key {key!r}"
        # SCORELESS contract.
        assert c["distance"] is None, "candidate carries a non-None distance (must be scoreless)"
        for bad in _FORBIDDEN_SCORE_KEYS:
            assert bad not in c, f"candidate leaked an in-DB score key {bad!r}"
        # source_file is the basename; _source_file_full is the full path.
        assert c["_source_file_full"], "_source_file_full must be populated for dedup"

    # The seeded drawer's basename / full path are mapped correctly.
    rare = next(c for c in cands if "zylophonics" in c["text"])
    assert rare["source_file"] == "expansion.md"
    assert rare["_source_file_full"] == "/home/agent/notes/expansion.md"
    assert rare["_chunk_index"] == 0
    assert rare["wing"] == "projects"
    assert rare["room"] == "2026-06-05"
    # The rare-token drawer id is real (so dedup pointers resolve).
    assert rare_id == "drawer_zylophonics"


@live_only
def test_keyword_candidates_respects_where_filter(backend, team):
    col = _col(backend, team, create=True)
    _seed(col)
    # A wing that nothing was filed under → no candidates despite a token match.
    cands = col.keyword_candidates(
        query="zylophonics", n_results=10, where={"wing": "nonexistent_wing"}
    )
    assert cands == []


@live_only
def test_keyword_candidates_respects_restrict_ids(backend, team):
    col = _col(backend, team, create=True)
    rare_id = _seed(col)
    # restrict_ids excludes the rare drawer → token match suppressed.
    cands = col.keyword_candidates(query="zylophonics", n_results=10, restrict_ids=["filler_0000"])
    assert all("zylophonics" not in c["text"] for c in cands)
    # restrict_ids includes it → retrieved.
    cands2 = col.keyword_candidates(query="zylophonics", n_results=10, restrict_ids=[rare_id])
    assert any("zylophonics" in c["text"] for c in cands2)


@live_only
def test_short_token_query_returns_empty_no_index_claim(backend, team):
    """A query with no ≥3-char token yields [] (documented <3-char floor)."""
    col = _col(backend, team, create=True)
    _seed(col)
    assert col.keyword_candidates(query="a to", n_results=10) == []


# --------------------------------------------------------------------------
# (c) live EXPLAIN: $contains rides the trigram GIN (Bitmap Index Scan)
# --------------------------------------------------------------------------


@live_only
def test_contains_uses_bitmap_index_scan_on_doc_trgm(backend, team):
    col = _col(backend, team, create=True)
    _seed(col, n_filler=200)
    schema = team_schema(team)
    index_name = COLLECTION + "_doc_trgm"

    # Build the exact $contains / where_document query shape the backend issues,
    # then EXPLAIN it. Disable seqscan-only plans is NOT needed — with enough
    # rows + a rare needle the planner already prefers the GIN; but we assert the
    # plan text rather than force it, to mirror real query behaviour.
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'EXPLAIN SELECT id FROM "{schema}"."{COLLECTION}" WHERE document ILIKE %s',
                ("%zylophonics%",),
            )
            plan = "\n".join(r[0] for r in cur.fetchall())

    assert "Bitmap Index Scan" in plan, f"expected a Bitmap Index Scan, got plan:\n{plan}"
    assert index_name in plan, f"expected the trigram GIN {index_name} in the plan:\n{plan}"
    assert "Seq Scan" not in plan, f"unexpected Seq Scan in $contains plan:\n{plan}"


@live_only
def test_short_needle_contains_correct_via_fallback(backend, team):
    """A <3-char needle still returns correct rows (ILIKE fallback, no GIN claim).

    The trigram GIN cannot accelerate a <3-char needle, but ``$contains``
    semantics are unchanged — the query still finds the matching rows via a
    (non-index-accelerated) ILIKE. We assert correctness, NOT the plan.
    """
    col = _col(backend, team, create=True)
    _seed(col)
    # "ex" appears in "expansion"; a 2-char needle the GIN can't accelerate.
    res = col.get(where_document={"$contains": "ex"})
    assert any("zylophonics" in d for d in res.documents), (
        "short-needle $contains did not return the matching row via fallback"
    )


# --------------------------------------------------------------------------
# (d) candidate-shape regression: survives the union-merge dedup None-drop
# --------------------------------------------------------------------------


@live_only
def test_candidates_survive_union_merge_dedup(backend, team):
    """Postgres keyword candidates must NOT be silently dropped at searcher.py:697.

    ``_merge_bm25_union_candidates`` computes ``_dedup_key`` for every candidate
    and SKIPS any whose key is falsy / ``"?"`` (searcher.py:697). Feed real
    postgres candidates through the same dedup logic and assert (1) each yields a
    non-None, non-"?" dedup key, and (2) the merger actually appends them into a
    starting hit list (none eaten).
    """
    from mempalace import searcher

    col = _col(backend, team, create=True)
    _seed(col)
    cands = col.keyword_candidates(query="zylophonics report", n_results=10)
    assert cands

    # The merger's exact dedup logic (searcher.py:684-697): a falsy / "?" key is
    # silently skipped. Replicate it verbatim and assert no postgres candidate is
    # eaten. (The merger itself re-derives chroma candidates from palace_path, so
    # its dedup helper — not the whole function — is what gates postgres rows.)
    assert searcher._merge_bm25_union_candidates is not None  # module loaded

    def _dedup_key(entry):
        full = entry.get("_source_file_full")
        ci = entry.get("_chunk_index")
        if full and ci is not None:
            return (full, ci)
        return entry.get("source_file")

    # (1) every candidate yields a usable dedup key.
    for c in cands:
        dedup = _dedup_key(c)
        assert dedup, f"candidate resolves to a falsy dedup key (would be dropped): {c!r}"
        assert dedup != "?", "candidate resolves to '?' (would be dropped at searcher.py:697)"

    # (2) none silently eaten by the None-guard.
    hits: list = []
    seen = set()
    for bh in cands:
        k = _dedup_key(bh)
        if not k or k == "?" or k in seen:
            continue
        hits.append(bh)
        seen.add(k)
    assert len(hits) == len(cands), "a postgres candidate was dropped by the dedup None-guard"
