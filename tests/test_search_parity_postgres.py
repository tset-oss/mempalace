"""chroma <-> postgres search-parity harness (ultragoal story G004).

Proves the postgres backend reaches the SAME search quality as chroma on a
shared, golden-labelled corpus — the gate that B (trigram-GIN keyword path) and
C (union-default) actually delivered parity, not silent degradation.

Design (the parity-harness design, also recorded in the search-parity plan):

* **ONE embedder pinned PROCESS-WIDE = embeddinggemma** (the deploy default).
  ``embedding._EF_CACHE`` and ``backends.postgres._embedder`` are cached
  singletons, so flipping ``MEMPALACE_EMBEDDING_MODEL`` mid-process is a no-op
  that would silently compare MIXED embedders (the M2 trap). The
  ``_pinned_embedder`` fixture sets the env var AND resets both caches BEFORE
  any embedding happens, then asserts BOTH backends were populated with a
  384-dim embeddinggemma vector space. Both backends are built from the SAME
  corpus with this one embedder.
* **Golden fixtures** — a small curated corpus with KNOWN expected ``source_file``
  ids per query: pure-semantic (vector finds it), pure-lexical (rare exact
  token, vector-distant), and mixed.
* **MANDATORY negative control** — ``test_negative_control_union_recovers_vector_miss``
  asserts a query whose only correct hit is lexical / vector-distant scores
  recall@k = 0% under ``candidate_strategy="vector"`` AND 100% under ``"union"``
  on BOTH backends. The corpus was tuned empirically against the REAL
  embeddinggemma embeddings until vector-only genuinely misses and union
  genuinely recovers (a control that never fails-under-vector is worthless).
* **Metric 1** — recall@k = 100% on both backends for every golden query.
* **Metric 2** — top-k id-set Jaccard >= 0.9 between chroma and postgres.
* EXPLAIN-uses-GIN, scoreless dedup-shape, and max_distance-guard parity checks.
* **Latency observability** — union vs vector wall time on both backends, with a
  noise-robust budget assertion (documented below).

The postgres half self-skips with a clear reason when no DB is reachable; the
chroma half always runs. Mirrors the live-skip pattern in
``tests/test_postgres_backend.py`` / ``tests/test_postgres_keyword_candidates.py``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import team_schema  # noqa: E402


# ---------------------------------------------------------------------------
# Live-DB probe (self-skip mirroring tests/test_postgres_backend.py)
# ---------------------------------------------------------------------------


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

# The whole parity comparison is meaningless without BOTH backends, so the
# entire module self-skips (with a clear reason) when no DB is reachable. The
# chroma-only sanity that "chroma search works under embeddinggemma" is exercised
# implicitly by every parity test once the DB is present; a DB-less run skips
# rather than half-comparing.
pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)

DIM = 384  # embeddinggemma (MRL-truncated) and minilm both emit 384.
PINNED_MODEL = "embeddinggemma"
COLLECTION = "mempalace_drawers"


# ---------------------------------------------------------------------------
# Golden corpus (empirically tuned against the REAL embeddinggemma embeddings)
# ---------------------------------------------------------------------------

# A rare nonsense token with no semantic meaning to the embedder. It appears in
# exactly ONE off-topic document, so any query whose vector points elsewhere
# cannot reach that document by similarity — only the lexical (trigram-GIN /
# FTS5) keyword path can. This is the lever for the negative control.
RARE_TOKEN = "qwzzxblorptk"

# source_file id -> verbatim document. Three semantic clusters:
#   * baking/bread (6 docs) — distractors that fill the vector top-k for the
#     negative-control query, pushing the off-topic target out of vector reach.
#   * astronomy (2 docs) — the pure-semantic golden target cluster.
#   * one soil note carrying RARE_TOKEN — the pure-lexical golden + neg-control.
#   * one gardening/compost note with a distinctive lexical token — the mixed case.
CORPUS: dict[str, str] = {
    "bread_sourdough.md": (
        "Sourdough bread relies on a mature starter, a long bulk ferment, and a "
        "hot oven with steam for crust."
    ),
    "bread_baguette.md": (
        "A classic baguette has a crisp crust and an open airy crumb; shape "
        "gently to preserve gas in the dough."
    ),
    "bread_focaccia.md": (
        "Focaccia dough is wet and oily; dimple it with your fingers, top with "
        "rosemary and flaky salt, then bake hot."
    ),
    "bread_rye.md": (
        "Rye bread uses a sour rye starter and bakes dense and dark; caraway "
        "seeds add the traditional aroma."
    ),
    "bread_brioche.md": (
        "Brioche is an enriched bread loaded with butter and eggs, giving a "
        "tender golden crumb perfect for toast."
    ),
    "bread_ciabatta.md": (
        "Ciabatta is a high-hydration Italian bread with a chewy crumb and a "
        "flour-dusted rustic crust."
    ),
    "astro_blackhole.md": (
        "A black hole is a region of spacetime where gravity is so strong that "
        "not even light can escape its event horizon."
    ),
    "astro_nebula.md": (
        "A nebula is a vast cloud of interstellar gas and dust where new stars "
        "are born under gravitational collapse."
    ),
    # Pure-lexical / negative-control target: off-topic soil note. RARE_TOKEN
    # appears twice to give the keyword path a clear lexical signal while the
    # document stays semantically far from any baking query.
    "soil_notes.md": (
        "Soil acidity affects nutrient uptake and mulching retains moisture "
        f"through dry spells. Internal lot reference {RARE_TOKEN} ({RARE_TOKEN}) "
        "filed for the back terrace."
    ),
    # Mixed: gardening semantics PLUS a distinctive lexical token.
    "garden_compost.md": (
        "Composting kitchen scraps and yard waste with the keyword "
        "vermicomposting accelerates rich humus for the garden beds."
    ),
}

# Golden queries -> the set of source_file ids that MUST appear in the top-k.
#   (a) pure-semantic: astronomy phrasing with NO shared rare token — only the
#       vector path can find the black-hole doc.
#   (b) pure-lexical: the rare nonsense token, present only in soil_notes.md.
#   (c) mixed: lexical 'vermicomposting' + gardening semantics.
GOLDEN_QUERIES: dict[str, set[str]] = {
    "What happens at the edge of a collapsed star where light cannot escape?": {
        "astro_blackhole.md"
    },
    RARE_TOKEN: {"soil_notes.md"},
    "how does vermicomposting help the garden": {"garden_compost.md"},
}

# Negative control. The query is SEMANTICALLY about baking bread (so the 6 bread
# docs dominate the vector top-k and push soil_notes.md out of vector reach),
# but it carries RARE_TOKEN as a lexical-only needle that only the keyword path
# resolves. Empirically (against real embeddinggemma vectors) vector-only misses
# soil_notes.md and union recovers it on BOTH backends at k=4,5,6.
NEG_CONTROL_QUERY = f"How do I bake a crusty artisan bread loaf at home? {RARE_TOKEN}"
NEG_CONTROL_TARGET = "soil_notes.md"
# k for the negative control: small enough that the 6 baking distractors fill
# the vector top-k and exclude the off-topic target, large enough that the union
# rerank still surfaces the recovered keyword candidate. 6 is comfortably inside
# the empirically-verified 4..6 window (see the probe rationale above).
NEG_CONTROL_K = 6

# k for the recall / Jaccard metrics: "all" — every document, per the plan
# (k=100 / "all"). The corpus is tiny, so k = len(CORPUS) is the "all" analogue.
K_ALL = len(CORPUS)

# Result-set overlap threshold: start at 0.9, tunable.
JACCARD_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Module-scoped fixture: pin embeddinggemma + build BOTH backends ONCE.
# ---------------------------------------------------------------------------


def _real_hf_cache_home() -> str | None:
    """Locate the user's real HuggingFace cache home.

    ``tests/conftest.py`` redirects ``HOME`` to a throwaway temp dir before any
    mempalace import, so ``huggingface_hub`` would look for embeddinggemma in an
    empty cache and try to re-download. The real HOME is stashed in
    ``conftest._original_env``; fall back to it so the locally-cached model loads
    without a network round-trip.
    """
    try:
        from tests import conftest  # type: ignore

        real_home = conftest._original_env.get("USERPROFILE") or conftest._original_env.get("HOME")
        if real_home and (Path(real_home) / ".cache" / "huggingface").is_dir():
            return str(Path(real_home) / ".cache" / "huggingface")
    except Exception:
        pass
    # conftest is imported as a top-level module by pytest, not as tests.conftest.
    import sys

    cf = sys.modules.get("conftest")
    if cf is not None:
        orig = getattr(cf, "_original_env", {})
        real_home = orig.get("USERPROFILE") or orig.get("HOME")
        if real_home and (Path(real_home) / ".cache" / "huggingface").is_dir():
            return str(Path(real_home) / ".cache" / "huggingface")
    return None


@pytest.fixture(scope="module")
def parity_env():
    """Pin embeddinggemma PROCESS-WIDE and build chroma + postgres from one corpus.

    Returns a dict with the chroma ``palace_path``, the postgres ``team`` slug,
    and the resolved ``dsn``. Tears down the throwaway chroma dir and the
    postgres team schema afterwards.

    The pin is the heart of the M2 trap: setting ``MEMPALACE_EMBEDDING_MODEL``
    alone is NOT enough because ``embedding._EF_CACHE`` (keyed by
    ``(model, providers)``) and ``backends.postgres._embedder`` are process
    singletons that may already hold a minilm EF from an earlier test. We set
    the env var, force CPU (deterministic), point the HF cache back at the real
    user cache, and then RESET both singletons so the next embedding call
    rebuilds with embeddinggemma. Chroma's EF likewise routes through
    ``embedding.get_embedding_function`` (no separate chroma EF cache exists),
    so clearing ``_EF_CACHE`` covers it too.
    """
    dsn = _dsn()

    prior = {
        k: os.environ.get(k)
        for k in (
            "MEMPALACE_EMBEDDING_MODEL",
            "MEMPALACE_EMBEDDING_DEVICE",
            "MEMPALACE_BACKEND",
            "MEMPALACE_TEAM",
            "MEMPALACE_DATABASE_URL",
            "HF_HOME",
            "HF_HUB_CACHE",
        )
    }

    import mempalace.backends.postgres as pg_module
    import mempalace.embedding as embedding
    from mempalace.backends.postgres import team_schema

    # Initialised before the try so the finally can clean up even if setup
    # raises before they are assigned.
    palace_path = None
    team = None
    try:
        os.environ["MEMPALACE_EMBEDDING_MODEL"] = PINNED_MODEL
        os.environ["MEMPALACE_EMBEDDING_DEVICE"] = "cpu"
        os.environ["MEMPALACE_DATABASE_URL"] = dsn
        hf_home = _real_hf_cache_home()
        if hf_home:
            os.environ["HF_HOME"] = hf_home
            os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")

        # Reset the cached embedder singletons so the embeddinggemma pin actually
        # takes (otherwise a minilm EF cached by an earlier test silently wins).
        embedding._EF_CACHE.clear()
        pg_module._embedder = None

        # Reset chroma's per-process client cache so a palace built with a stale
        # EF name is not reused (defensive; the palace dir is fresh anyway).
        try:
            from mempalace.backends.chroma import ChromaBackend

            ChromaBackend._quarantined_paths.clear()
        except Exception:
            pass

        from mempalace.embedding import get_embedding_function
        from mempalace.palace import get_collection

        # Assert the pin produced an embeddinggemma EF before anything is embedded.
        ef = get_embedding_function()
        assert type(ef).__name__ == "EmbeddinggemmaONNX", (
            f"embedder pin failed: expected EmbeddinggemmaONNX, got {type(ef).__name__}. "
            "The _EF_CACHE / postgres._embedder reset did not take — MIXED embedders "
            "would silently invalidate the parity comparison (the M2 trap)."
        )
        sample_vec = ef(["dimension probe"])[0]
        assert len(sample_vec) == DIM, f"embeddinggemma must be {DIM}-dim, got {len(sample_vec)}"

        palace_path = tempfile.mkdtemp(prefix="parity_chroma_")
        team = "parity" + uuid.uuid4().hex[:10]

        ids = list(CORPUS)
        docs = [CORPUS[i] for i in ids]

        # --- build the chroma palace (default backend) ---
        os.environ["MEMPALACE_BACKEND"] = "chroma"
        os.environ.pop("MEMPALACE_TEAM", None)
        chroma_col = get_collection(palace_path, create=True)
        chroma_col.upsert(
            ids=ids,
            documents=docs,
            metadatas=[{"wing": "misc", "room": "notes", "source_file": i} for i in ids],
        )

        # --- build the postgres team vault (same corpus, same pinned embedder) ---
        os.environ["MEMPALACE_BACKEND"] = "postgres"
        os.environ["MEMPALACE_TEAM"] = team
        pg_col = get_collection("/pg-parity", create=True)
        pg_col.add(
            ids=ids,
            documents=docs,
            metadatas=[
                {"wing": "misc", "room": "notes", "source_file": i, "chunk_index": 0} for i in ids
            ],
        )
        os.environ["MEMPALACE_BACKEND"] = "chroma"
        os.environ.pop("MEMPALACE_TEAM", None)

        # --- assert BOTH vaults hold a 384-dim embeddinggemma vector space ---
        # Postgres: read a stored vector back and check its dimension directly.
        schema = team_schema(team)
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT vector_dims(embedding) FROM "{schema}"."{COLLECTION}" '
                    "WHERE embedding IS NOT NULL LIMIT 1"
                )
                row = cur.fetchone()
        assert row is not None and row[0] == DIM, (
            f"postgres vault was not populated with {DIM}-dim vectors (got {row}); "
            "the pinned embeddinggemma embedder did not reach the postgres write path."
        )
        # Chroma: a query under the same EF must succeed (proves the palace is
        # readable with the pinned embeddinggemma EF — the embed_query parity fix).
        probe = chroma_col.query(query_texts=["black hole"], n_results=1)
        assert probe.ids and probe.ids[0], (
            "chroma palace not populated/queryable under embeddinggemma"
        )

        yield {"palace_path": palace_path, "team": team, "dsn": dsn}
    finally:
        # Always restore env + singletons, even if setup above raised — otherwise
        # the embeddinggemma pin + backend/team env leak into other test modules
        # (conftest's autouse reset does NOT clear _EF_CACHE / _embedder), turning
        # the suite order-dependent. This is the M2 trap the harness exists to avoid.
        if palace_path:
            shutil.rmtree(palace_path, ignore_errors=True)
        if team:
            try:
                with psycopg.connect(dsn) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
                    conn.commit()
            except Exception:
                pass
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        embedding._EF_CACHE.clear()
        pg_module._embedder = None


# ---------------------------------------------------------------------------
# Search helpers — drive the REAL search_memories path on each backend.
# ---------------------------------------------------------------------------


def _search(env: dict, query: str, backend: str, strategy: str, k: int, **kw) -> list[str]:
    """Run ``search_memories`` on ``backend`` and return the ordered source_file ids.

    Routing is via env: ``get_collection`` reads ``MempalaceConfig().backend``,
    so we flip ``MEMPALACE_BACKEND`` (and ``MEMPALACE_TEAM`` for postgres) around
    each call. This exercises the exact production dispatch, including the union
    merger's backend-capability routing.
    """
    from mempalace.searcher import search_memories

    os.environ["MEMPALACE_BACKEND"] = backend
    try:
        if backend == "postgres":
            os.environ["MEMPALACE_TEAM"] = env["team"]
            result = search_memories(
                query,
                "/pg-parity",
                team=env["team"],
                n_results=k,
                candidate_strategy=strategy,
                **kw,
            )
        else:
            os.environ.pop("MEMPALACE_TEAM", None)
            result = search_memories(
                query, env["palace_path"], n_results=k, candidate_strategy=strategy, **kw
            )
    finally:
        os.environ["MEMPALACE_BACKEND"] = "chroma"
        os.environ.pop("MEMPALACE_TEAM", None)
    assert "results" in result, f"search_memories returned an error dict: {result}"
    return [h["source_file"] for h in result["results"]]


# ---------------------------------------------------------------------------
# AC1 — the pin is correct and both backends share the embeddinggemma space.
# ---------------------------------------------------------------------------


def test_both_backends_pinned_to_embeddinggemma_384(parity_env):
    """The module fixture pinned ONE embedder (embeddinggemma, 384-dim) and
    populated BOTH backends with it.

    The fixture already asserts this during setup (and fails loudly if the
    _EF_CACHE / postgres._embedder reset did not take). This test makes the
    invariant an explicit, named AC1 check: the live EF is embeddinggemma and
    both vaults hold 384-dim vectors.
    """
    import mempalace.backends.postgres as pg_module
    from mempalace.embedding import get_embedding_function

    ef = get_embedding_function()
    assert type(ef).__name__ == "EmbeddinggemmaONNX"
    assert len(ef(["x"])[0]) == DIM
    # postgres._embedder, when materialised, is the same embeddinggemma EF.
    if pg_module._embedder is not None:
        assert type(pg_module._embedder).__name__ == "EmbeddinggemmaONNX"

    schema_table = (team_schema(parity_env["team"]), COLLECTION)
    with psycopg.connect(parity_env["dsn"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) "
                f'FROM "{schema_table[0]}"."{schema_table[1]}"'
            )
            count, dmin, dmax = cur.fetchone()
    assert count == len(CORPUS)
    assert dmin == dmax == DIM


# ---------------------------------------------------------------------------
# AC3 — MANDATORY negative control (guards green-when-broken).
# ---------------------------------------------------------------------------


def test_negative_control_union_recovers_vector_miss(parity_env):
    """recall@k = 0% under vector AND 100% under union, on BOTH backends.

    The only correct hit for ``NEG_CONTROL_QUERY`` is the lexical/vector-distant
    ``soil_notes.md`` (carrying ``RARE_TOKEN``). This is the postgres analogue of
    ``tests/test_hybrid_candidate_union.py::test_union_surfaces_bm25_strong_vector_distant_doc``
    and the load-bearing proof that union+B is doing real work — without it a
    recall=100% / Jaccard>=0.9 run could be falsely green even if the keyword
    path silently returned [] or was dropped at searcher.py:697.

    Empirically verified against the REAL embeddinggemma embeddings (not faked):
    vector-only genuinely excludes the target (recall 0); union genuinely
    recovers it (recall 1) on chroma AND postgres.
    """
    k = NEG_CONTROL_K
    tgt = NEG_CONTROL_TARGET

    chroma_vector = _search(parity_env, NEG_CONTROL_QUERY, "chroma", "vector", k)
    chroma_union = _search(parity_env, NEG_CONTROL_QUERY, "chroma", "union", k)
    pg_vector = _search(parity_env, NEG_CONTROL_QUERY, "postgres", "vector", k)
    pg_union = _search(parity_env, NEG_CONTROL_QUERY, "postgres", "union", k)

    # The control is only meaningful if it actually fails under vector. Assert
    # the 0% vector recall FIRST and loudly — a vacuous control is worthless.
    assert tgt not in chroma_vector, (
        f"NEGATIVE CONTROL IS VACUOUS on chroma: vector-only already returned "
        f"{tgt} (recall != 0). top-k={chroma_vector}. The corpus no longer forces "
        "a vector miss; retune RARE_TOKEN / distractors."
    )
    assert tgt not in pg_vector, (
        f"NEGATIVE CONTROL IS VACUOUS on postgres: vector-only already returned "
        f"{tgt} (recall != 0). top-k={pg_vector}."
    )

    # Union must recover the vector-missed target on BOTH backends (recall 100%).
    assert tgt in chroma_union, (
        f"union failed to recover the vector-missed target on chroma; top-k={chroma_union}"
    )
    assert tgt in pg_union, (
        f"union failed to recover the vector-missed target on postgres; top-k={pg_union}"
    )


# ---------------------------------------------------------------------------
# AC4 — recall@k = 100% on BOTH backends for every golden query.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", list(GOLDEN_QUERIES.items()), ids=lambda v: None)
def test_golden_recall_100pct_both_backends(parity_env, query, expected):
    """Every required source_file id is in the top-k on chroma AND postgres
    under the union default (k = all)."""
    chroma_ids = set(_search(parity_env, query, "chroma", "union", K_ALL))
    pg_ids = set(_search(parity_env, query, "postgres", "union", K_ALL))

    missing_chroma = expected - chroma_ids
    missing_pg = expected - pg_ids
    assert not missing_chroma, (
        f"chroma recall < 100% for {query!r}: missing {missing_chroma}; got {chroma_ids}"
    )
    assert not missing_pg, (
        f"postgres recall < 100% for {query!r}: missing {missing_pg}; got {pg_ids}"
    )


# ---------------------------------------------------------------------------
# AC5 — top-k id-set Jaccard >= threshold between chroma and postgres.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", list(GOLDEN_QUERIES), ids=lambda v: None)
def test_topk_jaccard_overlap(parity_env, query):
    """|chroma_topk & pg_topk| / |chroma_topk | pg_topk| >= JACCARD_THRESHOLD.

    Tolerates HNSW-approx / pgvector tie-break differences without demanding
    exact equality (plan Metric 2)."""
    chroma_ids = set(_search(parity_env, query, "chroma", "union", K_ALL))
    pg_ids = set(_search(parity_env, query, "postgres", "union", K_ALL))
    union = chroma_ids | pg_ids
    jaccard = (len(chroma_ids & pg_ids) / len(union)) if union else 1.0
    assert jaccard >= JACCARD_THRESHOLD, (
        f"top-k Jaccard {jaccard:.3f} < {JACCARD_THRESHOLD} for {query!r}\n"
        f"  chroma={sorted(chroma_ids)}\n  postgres={sorted(pg_ids)}"
    )


# ---------------------------------------------------------------------------
# AC6 — EXPLAIN proves the postgres keyword path is genuinely GIN-backed.
# ---------------------------------------------------------------------------


def test_explain_contains_uses_trigram_gin(parity_env):
    """Parity-of-mechanism: the postgres $contains / where_document path rides
    the G002 trigram GIN (Bitmap Index Scan), not a Seq Scan masquerade.

    Reuses the G002 assertion shape (tests/test_postgres_keyword_candidates.py).
    The corpus is small, so we disable seqscan for the EXPLAIN to force the
    planner to reveal whether an index path EXISTS — the index must be usable,
    which is the mechanism claim. (On a large vault the planner picks it on cost;
    here we only prove the GIN is present and applicable for a >=3-char needle.)
    """
    schema = team_schema(parity_env["team"])
    index_name = COLLECTION + "_doc_trgm"
    with psycopg.connect(parity_env["dsn"]) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            cur.execute(
                f'EXPLAIN SELECT id FROM "{schema}"."{COLLECTION}" WHERE document ILIKE %s',
                (f"%{RARE_TOKEN}%",),
            )
            plan = "\n".join(r[0] for r in cur.fetchall())
    assert "Bitmap Index Scan" in plan, f"expected a Bitmap Index Scan, got:\n{plan}"
    assert index_name in plan, f"expected trigram GIN {index_name} in plan:\n{plan}"


# ---------------------------------------------------------------------------
# AC7 — scoreless dedup-shape + max_distance guard parity.
# ---------------------------------------------------------------------------


def test_scoreless_dedup_shape_parity(parity_env):
    """Both backends' keyword_candidates() emit the SCORELESS dedup-shape that
    survives the union merger (no silent None-drop at searcher.py:697)."""
    from mempalace.palace import get_collection
    from mempalace.searcher import _merge_bm25_union_candidates

    forbidden = ("paradedb.score", "bm25_score", "_score", "score")

    def _check(cands: list[dict], label: str):
        assert cands, f"{label}: expected keyword candidates for the rare token"
        for c in cands:
            assert c["distance"] is None, f"{label}: candidate not scoreless: {c}"
            for bad in forbidden:
                assert bad not in c, f"{label}: candidate leaked in-DB score {bad!r}: {c}"
            # dedup key must be non-falsy / non-'?' (else dropped at searcher.py:697).
            full, ci = c.get("_source_file_full"), c.get("_chunk_index")
            key = (full, ci) if (full and ci is not None) else c.get("source_file")
            assert key and key != "?", f"{label}: candidate has a droppable dedup key: {c}"

    # postgres keyword candidates
    os.environ["MEMPALACE_BACKEND"] = "postgres"
    os.environ["MEMPALACE_TEAM"] = parity_env["team"]
    try:
        pg_col = get_collection("/pg-parity", create=False)
        pg_cands = pg_col.keyword_candidates(query=f"{RARE_TOKEN} report", n_results=10)
    finally:
        os.environ["MEMPALACE_BACKEND"] = "chroma"
        os.environ.pop("MEMPALACE_TEAM", None)
    _check(pg_cands, "postgres")

    # chroma keyword candidates (FTS5 path)
    chroma_col = get_collection(parity_env["palace_path"], create=False)
    chroma_cands = chroma_col.keyword_candidates(query=f"{RARE_TOKEN} report", n_results=10)
    _check(chroma_cands, "chroma")

    # Both candidate sets survive the merger's dedup into a starting hit list.
    # v3.5.0 merger signature: (hits, drawers_col, query, wing, room, n_results,
    # ...); the fork threads the live handle as ``collection`` and dispatches on
    # its lexical seam (PostgresCollection.lexical_search adapts keyword_candidates).
    for cands, col, label in ((pg_cands, pg_col, "postgres"), (chroma_cands, chroma_col, "chroma")):
        hits: list[dict] = []
        _merge_bm25_union_candidates(
            hits, col, f"{RARE_TOKEN} report", None, None, 10, collection=col
        )
        assert any(RARE_TOKEN in h["text"] for h in hits), (
            f"{label}: merger dropped the rare-token candidate (silent None-drop?)"
        )


def test_max_distance_guard_injects_zero_scoreless_on_postgres(parity_env):
    """union + max_distance>0 injects ZERO scoreless candidates on postgres,
    matching chroma — every returned hit carries a real (non-None) distance."""
    filtered = None
    from mempalace.searcher import search_memories

    os.environ["MEMPALACE_BACKEND"] = "postgres"
    os.environ["MEMPALACE_TEAM"] = parity_env["team"]
    try:
        filtered = search_memories(
            NEG_CONTROL_QUERY,
            "/pg-parity",
            team=parity_env["team"],
            n_results=K_ALL,
            candidate_strategy="union",
            max_distance=0.5,
        )
    finally:
        os.environ["MEMPALACE_BACKEND"] = "chroma"
        os.environ.pop("MEMPALACE_TEAM", None)

    assert "results" in filtered
    # The load-bearing invariant: with max_distance>0 set, the union merger must
    # inject ZERO scoreless (distance=None) keyword candidates on postgres — they
    # have no vector distance and would silently bypass the threshold. Every
    # returned hit must therefore carry a real distance within the bound.
    for h in filtered["results"]:
        assert h.get("distance") is not None, (
            f"max_distance>0 must not inject scoreless postgres candidates; got {h}"
        )
        assert h["distance"] <= 0.5, f"hit violates max_distance=0.5: {h}"
    # NOTE: we deliberately do NOT assert NEG_CONTROL_TARGET is absent here. The
    # negative control proves it falls outside the vector TOP-K (k=6); that is a
    # different condition from "vector distance > max_distance". Empirically the
    # target's real cosine distance to the baking query is ~0.46 (< 0.5), so it
    # legitimately appears as a genuine vector hit under this threshold — with a
    # non-None distance, which the loop above already validates. Asserting its
    # absence would be wrong (it would test a non-invariant).


# ---------------------------------------------------------------------------
# AC8 — latency observability (union vs vector, both backends).
# ---------------------------------------------------------------------------


def _p50(samples: list[float]) -> float:
    s = sorted(samples)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def test_union_latency_within_budget(parity_env, capsys):
    """Measure union vs vector per-search wall time on both backends; record it.

    Budget: union p50 <= 2x vector
    p50. On a tiny corpus the per-search time is dominated by fixed embedding +
    round-trip cost, so the ratio is noisy; the assertion is made noise-robust:
    it only fires when vector p50 clears a minimum absolute FLOOR
    (LATENCY_FLOOR_S). Below the floor the corpus is too small for the ratio to
    be meaningful and the budget check is logged as INFORMATIONAL rather than
    enforced — shipping a flaky timing assertion would be worse than none.
    """
    LATENCY_FLOOR_S = 0.05  # below this, ratio noise dominates -> informational only.
    BUDGET_RATIO = 2.0
    REPS = 5
    # Use a real-language golden query (not the bare rare token) so the vector
    # path does meaningful work on both strategies.
    query = "how does vermicomposting help the garden"

    measurements: dict[str, dict[str, float]] = {}
    for backend in ("chroma", "postgres"):
        per_strategy: dict[str, float] = {}
        for strategy in ("vector", "union"):
            times = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                _search(parity_env, query, backend, strategy, K_ALL)
                times.append(time.perf_counter() - t0)
            per_strategy[strategy] = _p50(times)
        measurements[backend] = per_strategy

    lines = ["\n[parity latency] union vs vector p50 (seconds):"]
    enforced = []
    for backend, per in measurements.items():
        v, u = per["vector"], per["union"]
        ratio = (u / v) if v > 0 else float("inf")
        meaningful = v >= LATENCY_FLOOR_S
        lines.append(
            f"  {backend:8} vector_p50={v:.4f} union_p50={u:.4f} ratio={ratio:.2f} "
            f"budget<= {BUDGET_RATIO} ({'ENFORCED' if meaningful else 'informational'})"
        )
        if meaningful:
            enforced.append((backend, ratio))
    # Print the recorded numbers regardless of enforcement (observability).
    with capsys.disabled():
        print("\n".join(lines))

    for backend, ratio in enforced:
        assert ratio <= BUDGET_RATIO, (
            f"union p50 on {backend} is {ratio:.2f}x vector p50 (budget {BUDGET_RATIO}x). "
            "Vector p50 cleared the noise floor, so this is a real regression."
        )
