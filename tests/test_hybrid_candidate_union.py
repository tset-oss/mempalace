"""Tests for ``candidate_strategy`` in ``search_memories``.

As of G003 the DEFAULT strategy is ``"union"``: candidates come from BOTH the
vector index AND the live collection's own scoreless keyword route
(``collection.keyword_candidates`` — chroma's FTS5 path, postgres' trigram
GIN), merged into the rerank pool. Docs with strong keyword signal but vector
embeddings far from the query — terminology guides looked up by
narrative-shaped queries are the canonical case — are surfaced by default.

The opt-out ``"vector"`` strategy gathers candidates from the vector index
only (the historical narrow behavior); no keyword candidates are injected.
``MEMPALACE_CANDIDATE_STRATEGY`` is the per-deployment env override; an
explicit ``candidate_strategy`` argument always wins over it.
"""

import os

from mempalace.palace import get_collection
from mempalace.searcher import search_memories


def _seed_drawers(palace_path):
    """Seed a corpus where the right doc for one query is BM25-strong but
    vector-distant.

    D1-D3 are short narrative tickets that semantically cluster around
    "customer support / order / shipped" vocabulary. D4 is a meta-document
    of bullet rules ("brand voice") that contains rare keywords like
    "Absolutely" and "apologize" the query repeats verbatim — strong BM25
    signal but stylistically far from the narrative tickets.
    """
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["D1", "D2", "D3", "D4"],
        documents=[
            "Customer wrote in asking why their order shipped without "
            "the promo sticker. Standard reply explaining the threshold.",
            "Order delivery delayed three days; customer requested a "
            "refund. Support agent processed return via ticket queue.",
            "Customer asked about the missing freebie; the reply "
            "explained the campaign mechanics and shipped status.",
            "Brand voice rules: dry, sturdy, never effusive. "
            "Never 'Absolutely!' Never apologize for policy — explain it. "
            "Avoid premium / curated / elevated vocabulary.",
        ],
        metadatas=[
            {"wing": "shop", "room": "support", "source_file": "ticket_D1.md"},
            {"wing": "shop", "room": "support", "source_file": "ticket_D2.md"},
            {"wing": "shop", "room": "support", "source_file": "ticket_D3.md"},
            {"wing": "shop", "room": "guides", "source_file": "brand_voice_D4.md"},
        ],
    )


_NARRATIVE_QUERY = (
    "A support agent is drafting a reply to a customer asking why their "
    "order shipped without a free sticker. Draft the reply, but never say "
    "'Absolutely!' and do not apologize for policy."
)


class TestCandidateUnion:
    def test_default_is_union(self, tmp_path):
        """As of G003 the default strategy is ``"union"``: omitting the
        argument must be identical to passing ``candidate_strategy="union"``.
        """
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Ensure no env override skews the default-resolution path.
        prior = os.environ.pop("MEMPALACE_CANDIDATE_STRATEGY", None)
        try:
            without = search_memories(_NARRATIVE_QUERY, palace, n_results=5)
        finally:
            if prior is not None:
                os.environ["MEMPALACE_CANDIDATE_STRATEGY"] = prior
        with_union = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
        )
        ids_a = [h["source_file"] for h in without["results"]]
        ids_b = [h["source_file"] for h in with_union["results"]]
        assert ids_a == ids_b, "default (no arg) must match explicit candidate_strategy='union'"
        # The default must actually do union work: the BM25-strong/vector-distant
        # doc is surfaced (proves the flip is not a silent no-op).
        assert "brand_voice_D4.md" in ids_a, (
            f"default strategy must surface the keyword-strong doc; got {ids_a}"
        )

    def test_explicit_vector_suppresses_keyword_candidates(self, tmp_path):
        """Explicit ``candidate_strategy="vector"`` must NOT silently union.

        With ``n_results=2`` the vector index only surfaces the 2 closest
        narrative tickets; the keyword-strong/vector-distant brand-voice doc is
        out of reach without keyword injection. Union surfaces it; vector-only
        does not. (The all-4-docs/``n_results=5`` case can't distinguish the
        strategies because both return everything — see ``:73``.)
        """
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        vector = search_memories(
            _NARRATIVE_QUERY, palace, n_results=2, candidate_strategy="vector"
        )
        vec_ids = {h["source_file"] for h in vector["results"]}
        assert "brand_voice_D4.md" not in vec_ids, (
            "explicit candidate_strategy='vector' must NOT inject keyword "
            f"candidates (no silent union); got {vec_ids}"
        )
        # Sanity: union over the same corpus DOES reach the keyword-strong doc
        # (proving the suppression above is real, not a corpus artifact).
        union = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
        )
        union_ids = {h["source_file"] for h in union["results"]}
        assert "brand_voice_D4.md" in union_ids, (
            f"union must surface the keyword-strong doc; got {union_ids}"
        )

    def test_union_respects_both_wing_and_room_scope(self, tmp_path):
        """Regression (G003 chroma ``$and`` leak): a scoped search passing BOTH
        wing AND room must NOT leak keyword candidates from another room.

        ``build_where_filter(wing, room)`` emits ``{"$and": [{"wing": w},
        {"room": r}]}`` when both are set. The chroma keyword wrapper must
        flatten that shape; otherwise ``_bm25_only_via_sqlite`` runs unscoped and
        the union default leaks out-of-room docs into a scoped search.
        ``brand_voice_D4.md`` lives in ``room="guides"`` and is the
        BM25-strong/vector-distant match for the query — a search scoped to
        ``room="support"`` must never return it.
        """
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        scoped = search_memories(
            _NARRATIVE_QUERY,
            palace,
            n_results=5,
            wing="shop",
            room="support",
            candidate_strategy="union",
        )
        ids = {h["source_file"] for h in scoped["results"]}
        assert "brand_voice_D4.md" not in ids, (
            "union must not leak the room='guides' doc into a room='support' "
            f"scoped search (chroma $and wing/room filter must be honored); got {ids}"
        )
        # Sanity: the in-scope tickets are still returned (the scope did not
        # over-filter to nothing).
        assert ids, "scoped union must still return in-scope results"
        assert ids <= {"ticket_D1.md", "ticket_D2.md", "ticket_D3.md"}, (
            f"scoped union returned out-of-scope docs: {ids}"
        )

    def test_union_surfaces_bm25_strong_vector_distant_doc(self, tmp_path):
        """The brand-voice doc has strong BM25 signal for the query but is
        stylistically far from the narrative tickets. Union mode must
        retrieve it; vector-only mode is allowed to miss it."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        ids = [h["source_file"] for h in result["results"]]
        assert "brand_voice_D4.md" in ids, (
            f"union mode must surface BM25-strong docs even when vector signal is weak; got {ids}"
        )

    def test_union_preserves_vector_hits(self, tmp_path):
        """Union mode must not drop docs that vector-only mode finds —
        the rerank pool grows, it doesn't shrink."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        vector = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="vector")
        union = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        vec_ids = {h["source_file"] for h in vector["results"]}
        union_ids = {h["source_file"] for h in union["results"]}
        # In a 4-doc corpus with n_results=5, both should return all 4.
        # The invariant is: union should not lose anything vector found.
        missing = vec_ids - union_ids
        assert not missing, f"union dropped docs that vector found: {missing}"

    def test_union_handles_empty_palace(self, tmp_path):
        """No drawers — union mode should return empty results, not crash."""
        palace = str(tmp_path / "palace")
        get_collection(palace, create=True)  # create empty collection
        result = search_memories("anything", palace, n_results=5, candidate_strategy="union")
        assert result.get("results", []) == []

    def test_invalid_candidate_strategy_raises(self, tmp_path):
        """Bad arg should raise rather than silently fall back."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        import pytest

        with pytest.raises(ValueError, match="candidate_strategy"):
            search_memories("anything", palace, n_results=5, candidate_strategy="bogus")

    def test_invalid_strategy_raises_even_when_vector_disabled(self, tmp_path):
        """Validation must happen before the ``vector_disabled`` early return —
        invalid values must fail consistently regardless of routing."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        import pytest

        with pytest.raises(ValueError, match="candidate_strategy"):
            search_memories(
                "anything",
                palace,
                n_results=5,
                vector_disabled=True,
                candidate_strategy="bogus",
            )

    def test_union_respects_n_results_limit(self, tmp_path):
        """When the merged candidate set is larger than ``n_results``, the
        result must be trimmed back to the requested size — the MCP
        ``limit`` contract depends on this invariant."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # 4-doc corpus, n_results=2 → union pool can grow to ~8 candidates,
        # rerank reorders them, but final list must respect the cap.
        result = search_memories(_NARRATIVE_QUERY, palace, n_results=2, candidate_strategy="union")
        assert len(result["results"]) <= 2, (
            f"union must trim to n_results=2; got {len(result['results'])} results"
        )

    def test_union_skipped_when_max_distance_set(self, tmp_path):
        """``max_distance`` is a vector-distance threshold; BM25-only
        candidates have ``distance=None`` and cannot satisfy it. Union
        must not silently inject them when a strict threshold is set,
        otherwise the existing ``max_distance`` guarantee regresses."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Sanity: without max_distance, union surfaces the BM25-strong doc.
        unfiltered = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
        )
        assert "brand_voice_D4.md" in {h["source_file"] for h in unfiltered["results"]}

        # With a tight max_distance, union must NOT inject BM25-only hits —
        # every returned hit must have a real (non-None) distance.
        filtered = search_memories(
            _NARRATIVE_QUERY,
            palace,
            n_results=5,
            candidate_strategy="union",
            max_distance=0.5,
        )
        for h in filtered["results"]:
            assert h.get("distance") is not None, (
                f"union under max_distance must not inject BM25-only "
                f"(distance=None) candidates; offending hit: {h}"
            )
            assert h["distance"] <= 0.5, f"hit violates max_distance=0.5: distance={h['distance']}"

    def test_union_dedup_is_chunk_precise_not_basename(self, tmp_path):
        """Two files with the same basename in different directories must
        not collide — union must dedup on full path (or chunk-level key),
        not on basename alone. Otherwise a BM25-strong README from one
        directory silently shadows a BM25-strong README from another.
        """
        palace = str(tmp_path / "palace")
        col = get_collection(palace, create=True)
        col.upsert(
            ids=["A_README", "B_README", "narrative"],
            documents=[
                # Both README files share the basename README.md but live
                # in different directories. Each contains distinctive
                # terminology a query might surface via BM25.
                "PROJECT ALPHA: configuration for the Frobnitz subsystem. "
                "Set FROBNITZ_TIMEOUT=30 to enable widget rotation.",
                "PROJECT BETA: configuration for the Wibble subsystem. "
                "Set WIBBLE_THRESHOLD=0.5 to enable signal smoothing.",
                "Engineers occasionally chat about how the legacy "
                "subsystems all need their config knobs tweaked.",
            ],
            metadatas=[
                {"wing": "code", "room": "docs", "source_file": "alpha/README.md"},
                {"wing": "code", "room": "docs", "source_file": "beta/README.md"},
                {"wing": "code", "room": "docs", "source_file": "chat.md"},
            ],
        )
        # Query that hits BM25 for BOTH READMEs (distinct vocab from each).
        # Vector-only might pick the chat doc as semantically "closest";
        # union must surface both READMEs without basename collision.
        result = search_memories(
            "FROBNITZ_TIMEOUT WIBBLE_THRESHOLD configuration",
            palace,
            n_results=5,
            candidate_strategy="union",
        )
        sources = [h["source_file"] for h in result["results"]]
        readme_count = sum(1 for s in sources if s == "README.md")
        assert readme_count >= 2, (
            f"union must surface both README.md files from different dirs "
            f"(basename collision would drop one); got sources={sources}"
        )


class TestCandidateStrategyEnvOverride:
    """``MEMPALACE_CANDIDATE_STRATEGY`` is the per-deployment opt-out; an
    explicit ``candidate_strategy`` argument always wins over it."""

    def _restore_env(self, prior):
        if prior is None:
            os.environ.pop("MEMPALACE_CANDIDATE_STRATEGY", None)
        else:
            os.environ["MEMPALACE_CANDIDATE_STRATEGY"] = prior

    def test_env_vector_suppresses_keyword_candidates(self, tmp_path):
        """Env set to ``"vector"`` with NO explicit arg → no keyword candidates
        (the latency escape hatch forces vector-only)."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        prior = os.environ.get("MEMPALACE_CANDIDATE_STRATEGY")
        os.environ["MEMPALACE_CANDIDATE_STRATEGY"] = "vector"
        try:
            result = search_memories(_NARRATIVE_QUERY, palace, n_results=2)
        finally:
            self._restore_env(prior)
        ids = {h["source_file"] for h in result["results"]}
        assert "brand_voice_D4.md" not in ids, (
            f"env=vector must suppress keyword candidates; got {ids}"
        )

    def test_env_unset_defaults_to_union(self, tmp_path):
        """Env unset, no explicit arg → union (keyword-strong doc surfaced)."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        prior = os.environ.get("MEMPALACE_CANDIDATE_STRATEGY")
        os.environ.pop("MEMPALACE_CANDIDATE_STRATEGY", None)
        try:
            result = search_memories(_NARRATIVE_QUERY, palace, n_results=5)
        finally:
            self._restore_env(prior)
        ids = {h["source_file"] for h in result["results"]}
        assert "brand_voice_D4.md" in ids, (
            f"env unset must default to union; got {ids}"
        )

    def test_explicit_arg_wins_over_env(self, tmp_path):
        """Explicit ``candidate_strategy="union"`` overrides env=vector."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        prior = os.environ.get("MEMPALACE_CANDIDATE_STRATEGY")
        os.environ["MEMPALACE_CANDIDATE_STRATEGY"] = "vector"
        try:
            result = search_memories(
                _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
            )
        finally:
            self._restore_env(prior)
        ids = {h["source_file"] for h in result["results"]}
        assert "brand_voice_D4.md" in ids, (
            "explicit candidate_strategy='union' must win over env=vector; "
            f"got {ids}"
        )

    def test_invalid_env_value_raises(self, tmp_path):
        """An invalid env value fails the same way an invalid argument does."""
        import pytest

        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        prior = os.environ.get("MEMPALACE_CANDIDATE_STRATEGY")
        os.environ["MEMPALACE_CANDIDATE_STRATEGY"] = "bogus"
        try:
            with pytest.raises(ValueError, match="candidate_strategy"):
                search_memories(_NARRATIVE_QUERY, palace, n_results=5)
        finally:
            self._restore_env(prior)


class TestUnionMergerBackendDispatch:
    """``_merge_bm25_union_candidates`` dispatches keyword retrieval on the live
    collection handle's capability — exactly ONE route per backend, no legacy
    ``_bm25_only_via_sqlite(palace_path, ...)`` fallback inside the merger."""

    def test_supporting_collection_routes_to_keyword_candidates(self):
        from mempalace.searcher import _merge_bm25_union_candidates

        calls = {"keyword": 0}

        class _FakeCollection:
            def supports_keyword_candidates(self):
                return True

            def keyword_candidates(self, *, query, n_results, where=None, restrict_ids=None):
                calls["keyword"] += 1
                return [
                    {
                        "text": "rare zylophonics term doc",
                        "wing": "w",
                        "room": "r",
                        "source_file": "kw.md",
                        "distance": None,
                        "_source_file_full": "dir/kw.md",
                        "_chunk_index": 0,
                    }
                ]

        hits = [{"text": "vector hit", "distance": 0.2, "source_file": "v.md",
                 "_source_file_full": "dir/v.md", "_chunk_index": 0}]
        _merge_bm25_union_candidates(
            hits, "zylophonics", "/ignored", None, None, 5, collection=_FakeCollection()
        )
        assert calls["keyword"] == 1, "must call the collection's keyword_candidates exactly once"
        sources = {h["source_file"] for h in hits}
        assert "kw.md" in sources, "keyword candidate must be merged into hits"
        # The merged candidate is tagged scoreless.
        kw_hit = next(h for h in hits if h["source_file"] == "kw.md")
        assert kw_hit["distance"] is None
        assert kw_hit["effective_distance"] is None
        assert kw_hit["closet_boost"] == 0.0

    def test_non_supporting_collection_injects_nothing(self):
        from mempalace.searcher import _merge_bm25_union_candidates

        class _NoKeywordCollection:
            def supports_keyword_candidates(self):
                return False

            def keyword_candidates(self, **kwargs):  # pragma: no cover - must not be called
                raise AssertionError("keyword_candidates must not be called when unsupported")

        hits = [{"text": "vector hit", "distance": 0.2, "source_file": "v.md",
                 "_source_file_full": "dir/v.md", "_chunk_index": 0}]
        before = list(hits)
        _merge_bm25_union_candidates(
            hits, "anything", "/ignored", None, None, 5, collection=_NoKeywordCollection()
        )
        assert hits == before, "no candidates may be injected when capability is absent"

    def test_collection_none_is_byte_identical_noop(self):
        from mempalace.searcher import _merge_bm25_union_candidates

        hits = [{"text": "vector hit", "distance": 0.2, "source_file": "v.md"}]
        before = list(hits)
        _merge_bm25_union_candidates(hits, "anything", "/ignored", None, None, 5)
        assert hits == before, "collection=None must be a no-op (legacy callers unaffected)"

    def test_max_distance_guard_skips_keyword_candidates(self):
        from mempalace.searcher import _merge_bm25_union_candidates

        class _FakeCollection:
            def supports_keyword_candidates(self):  # pragma: no cover - guarded before this
                return True

            def keyword_candidates(self, **kwargs):  # pragma: no cover - must not be reached
                raise AssertionError("keyword_candidates must not run under max_distance>0")

        hits = [{"text": "vector hit", "distance": 0.2, "source_file": "v.md"}]
        before = list(hits)
        _merge_bm25_union_candidates(
            hits, "anything", "/ignored", None, None, 5,
            max_distance=0.5, collection=_FakeCollection(),
        )
        assert hits == before, "max_distance>0 must skip keyword candidates on every route"


class TestHybridRankTolerantOfMissingDistance:
    """``_hybrid_rank`` accepts ``distance=None`` — required for BM25-only
    candidates injected by union mode."""

    def test_distance_none_scored_as_zero_vector_sim(self):
        from mempalace.searcher import _hybrid_rank

        results = [
            {"text": "alpha beta gamma", "distance": 0.2},  # close vector match
            {"text": "alpha alpha alpha", "distance": None},  # BM25-only — heavy term repetition
        ]
        # Query matches "alpha" heavily; the BM25-only candidate with no
        # vector signal should still rank competitively on BM25 alone.
        ranked = _hybrid_rank(results, "alpha")
        assert all("bm25_score" in r for r in ranked), "rerank should add bm25_score"
        # Both must survive — neither should crash on distance=None.
        assert len(ranked) == 2
