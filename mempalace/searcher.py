#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Hybrid search: BM25 keyword matching + vector semantic similarity. The
drawer query is the floor — always runs — and closet hits add a rank-based
boost when they agree. Closets are a ranking *signal*, never a gate, so
weak closets (regex extraction on narrative content) can only help, never
hide drawers the direct path would have found.
"""

import logging
import math
import os
import re
import sqlite3
from pathlib import Path

from .backends import (
    BackendError,
    BackendMismatchError,
    CollectionNotInitializedError,
    PalaceNotFoundError,
    UnsupportedCapabilityError,
)
from .closet_rebuild import closet_grouping_key
from .config import sqlite_read_uri
from .palace import (
    _open_collection_or_explain,
    get_closets_collection,
    get_collection,
    resolve_backend_name,
)

# Closet pointer line format: "topic|entities|→drawer_id_a,drawer_id_b"
# Multiple lines may join with newlines inside one closet document.
_CLOSET_DRAWER_REF_RE = re.compile(r"→([\w,]+)")

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _first_or_empty(results, key: str) -> list:
    """Return the first inner list of a query result field, or [].

    Accepts both the typed :class:`QueryResult` (attribute access) and the
    pre-typed chroma dict shape; this polymorphism is retained so test mocks
    still work and callers mid-migration do not crash. Preserves the empty-
    collection semantics from issue #195: when no queries returned hits, the
    outer list may be empty and indexing ``[0]`` would raise.
    """
    outer = getattr(results, key, None) if not isinstance(results, dict) else results.get(key)
    if not outer:
        return []
    return outer[0] or []


def _tokenize(text: str) -> list:
    """Lowercase + strip to alphanumeric tokens of length ≥ 2.

    Tolerates ``None`` documents — Chroma can return ``None`` in the
    ``documents`` field for drawers without text content, which would
    otherwise raise ``AttributeError`` mid-rerank.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(
    query: str,
    documents: list,
    k1: float = 1.5,
    b: float = 0.75,
) -> list:
    """Compute Okapi-BM25 scores for ``query`` against each document.

    IDF is computed over the *provided corpus* using the Lucene/BM25+
    smoothed formula ``log((N - df + 0.5) / (df + 0.5) + 1)``, which is
    always non-negative. This is well-defined for re-ranking a small
    candidate set returned by vector retrieval — IDF then reflects how
    discriminative each query term is *within the candidates*, exactly
    what's needed to reorder them.

    Parameters mirror Okapi-BM25 conventions:
        k1 — term-frequency saturation (1.2-2.0 typical, 1.5 default)
        b  — length normalization (0.0 = none, 1.0 = full, 0.75 default)

    Returns a list of scores in the same order as ``documents``.
    """
    n_docs = len(documents)
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    # Document frequency: how many docs contain each query term?
    df = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in query_terms}

    scores = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _distance_to_similarity(distance, metric: str = "cosine") -> float:
    """Map a backend-reported ``distance`` to a [0, 1]-ish similarity.

    The backend contract for the ``distances`` field is *lower = closer*
    regardless of metric (RFC 001, backend metric declaration), so every
    mapping here is monotonic decreasing in ``distance``. The output stays
    bounded so it is
    commensurable with the min-max-normalized BM25 term in
    :func:`_hybrid_rank`.

    * ``cosine`` — distance ∈ [0, 2], 0 = identical: ``max(0, 1 - d)``.
    * ``l2`` — Euclidean ∈ [0, ∞): ``1 / (1 + d)`` (1 at d=0, →0 as d→∞).
    * ``ip`` — inner-product distance (e.g. pgvector ``<#>`` = -dot, lower =
      closer), unbounded and signed: logistic squash ``1 / (1 + e^d)``.
      Provisional until a real ip backend exercises it; no in-tree backend
      uses ip today.

    ``distance is None`` (vector-unknown, e.g. a BM25-only candidate) maps to
    0.0 so the candidate scores on its BM25 contribution alone.
    """
    if distance is None:
        return 0.0
    m = (metric or "cosine").lower()
    if m == "l2":
        return 1.0 / (1.0 + max(0.0, distance))
    if m == "ip":
        # Clamp the exponent so a large positive distance can't overflow.
        return 1.0 / (1.0 + math.exp(min(60.0, distance)))
    # cosine (default)
    return max(0.0, 1.0 - distance)


def _metric_for_collection(col) -> str:
    """Resolve a collection's declared distance metric, defaulting to cosine.

    Reads the ``distance_metric`` exposed by the backend collection (the
    RFC 001 backend metric declaration). ``EmbeddingCollection`` delegates the
    attribute to its inner collection; legacy Chroma palaces report their
    actual ``hnsw:space``.
    Any failure falls back to ``"cosine"`` — the value all in-tree backends
    use and the only metric MemPalace created palaces with historically.
    """
    try:
        metric = getattr(col, "distance_metric", "cosine")
    except Exception:
        return "cosine"
    metric = str(metric or "cosine").lower()
    return metric if metric in ("cosine", "l2", "ip") else "cosine"


def _hybrid_rank(
    results: list,
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    metric: str = "cosine",
) -> list:
    """Re-rank ``results`` by a convex combination of vector similarity and BM25.

    * Vector similarity is derived from each candidate's backend-reported
      ``distance`` via :func:`_distance_to_similarity`, interpreted in the
      collection's declared ``metric`` (per RFC 001) rather than assuming
      cosine. Absolute (not relative-to-max) means adding/removing a
      candidate can't reshuffle the others.
    * BM25 is real Okapi-BM25 with corpus-relative IDF over the candidates
      themselves. Since the absolute scale is unbounded, BM25 is min-max
      normalized within the candidate set so weights are commensurable.

    Candidates with ``distance=None`` are treated as vector-unknown
    (no vector signal available) and scored on BM25 contribution alone.
    Used by candidate-union mode to merge BM25-only candidates that the
    vector index didn't surface.

    Mutates each result dict to add ``bm25_score`` and reorders the list
    in place. Returns the same list for convenience.
    """
    if not results:
        return results

    docs = [r.get("text", "") for r in results]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        vec_sim = _distance_to_similarity(r.get("distance"), metric)
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results[:] = [r for _, r in scored]
    return results


def build_where_filter(wing: str = None, room: str = None, source_file: str = None) -> dict:
    """Build a ChromaDB where filter from optional wing/room/source_file.

    ChromaDB needs a ``$and`` only when ≥2 clauses are present; a single
    clause is returned bare and zero clauses yield an empty filter (#1815).
    """
    clauses = []
    if wing:
        clauses.append({"wing": wing})
    if room:
        clauses.append({"room": room})
    if source_file:
        clauses.append({"source_file": source_file})
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _extract_drawer_ids_from_closet(closet_doc: str) -> list:
    """Parse all `→drawer_id_a,drawer_id_b` pointers out of a closet document.

    Preserves order and dedupes.
    """
    seen: dict = {}
    for match in _CLOSET_DRAWER_REF_RE.findall(closet_doc):
        for did in match.split(","):
            did = did.strip()
            if did and did not in seen:
                seen[did] = None
    return list(seen.keys())


def _scoped_source_filter(source_file: str, parent_drawer_id=None) -> dict:
    """Build a Chroma ``where`` clause that scopes a query to ``source_file``,
    additionally constrained by ``parent_drawer_id`` when one is supplied.

    Two unrelated oversized ``tool_add_drawer`` writes (chunked path from
    #1539) can pass the same ``source_file`` (e.g. two pastes tagged
    ``"chat.log"``); each call stores its own ``parent_drawer_id`` group
    of chunks but the bare ``source_file`` filter pulls chunks from both
    groups as if they were siblings (#1580). When the matched chunk
    carries a ``parent_drawer_id`` the filter narrows to that logical
    group. Otherwise (pre-#1539 drawers, single-chunk writes, and
    ``diary_ingest`` drawers grouped by real file path) the original
    file-global shape is preserved. Mirrors the conditional-``$and``
    precedent in ``build_where_filter``.
    """
    if parent_drawer_id:
        return {
            "$and": [
                {"source_file": source_file},
                {"parent_drawer_id": parent_drawer_id},
            ]
        }
    return {"source_file": source_file}


def _expand_with_neighbors(drawers_col, matched_doc: str, matched_meta: dict, radius: int = 1):
    """Expand a matched drawer with its ±radius sibling chunks in the same source file.

    Motivation — "drawer-grep context" feature: a closet hit returns one
    drawer, but the chunk boundary may clip mid-thought (e.g., the matched
    chunk says "here's a breakdown:" and the actual breakdown lives in the
    next chunk). Fetching the small neighborhood around the match gives
    callers enough context without forcing a follow-up ``get_drawer`` call.

    Returns a dict with:
        ``text``            combined chunks in chunk_index order
        ``drawer_index``    the matched chunk's index in the source file
        ``total_drawers``   total drawer count for the source file (or None)

    On any ChromaDB failure or missing metadata, falls back to returning the
    matched drawer alone so search never breaks because neighbor expansion
    failed.
    """
    src = matched_meta.get("source_file")
    chunk_idx = matched_meta.get("chunk_index")
    if not src or not isinstance(chunk_idx, int):
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    # Narrow by ``parent_drawer_id`` when present so chunks from unrelated
    # logical drawers sharing ``source_file`` do not stitch (#1580). See
    # ``_scoped_source_filter`` for the contract.
    parent_id = matched_meta.get("parent_drawer_id")
    target_indexes = [chunk_idx + offset for offset in range(-radius, radius + 1)]
    neighbor_clauses = [
        {"source_file": src},
        {"chunk_index": {"$in": target_indexes}},
    ]
    if parent_id:
        neighbor_clauses.append({"parent_drawer_id": parent_id})
    try:
        neighbors = drawers_col.get(
            where={"$and": neighbor_clauses},
            include=["documents", "metadatas"],
        )
    except Exception:
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    indexed_docs = []
    for doc, meta in zip(neighbors.documents, neighbors.metadatas):
        ci = meta.get("chunk_index")
        if isinstance(ci, int):
            indexed_docs.append((ci, doc))
    indexed_docs.sort(key=lambda pair: pair[0])

    if not indexed_docs:
        combined_text = matched_doc
    else:
        combined_text = "\n\n".join(doc for _, doc in indexed_docs)

    # Cheap total_drawers lookup. When ``parent_drawer_id`` is present the
    # count is scoped to that group so the returned number matches the
    # text the caller gets back. Without a parent id, the legacy
    # file-global count is preserved.
    total_drawers = None
    try:
        all_meta = drawers_col.get(
            where=_scoped_source_filter(src, parent_id),
            include=["metadatas"],
        )
        total_drawers = len(all_meta.ids) if all_meta.ids else None
    except Exception:
        logger.debug("total_drawers lookup failed for %s", src, exc_info=True)

    return {
        "text": combined_text,
        "drawer_index": chunk_idx,
        "total_drawers": total_drawers,
    }


def _warn_if_legacy_metric(col) -> None:
    """Print a one-line notice if the palace was created without
    ``hnsw:space=cosine``.

    ChromaDB's default is L2 (Euclidean), under which cosine-based
    similarity interpretation falls apart — distances routinely exceed
    1.0 and the display ``max(0, 1 - dist)`` floors every result to 0.
    Legacy palaces (mined before this metadata was consistently set)
    need ``mempalace repair`` to rebuild with the correct metric.

    The warning fires only for palaces that clearly have the wrong
    metric; palaces with no metadata table at all (empty dict) also
    fall under this check since that is the signal of a pre-metadata
    palace.
    """
    try:
        meta = getattr(col, "metadata", None)
    except Exception:
        return
    if not isinstance(meta, dict):
        return
    space = meta.get("hnsw:space")
    if space == "cosine":
        return
    # Either missing or set to something else — both are suspect.
    import sys as _sys

    detail = f"hnsw:space={space!r}" if space else "no hnsw:space metadata"
    print(
        f"\n  NOTICE: this palace was created without cosine distance ({detail}).\n"
        "          Semantic similarity scores will not be meaningful.\n"
        "          Run `mempalace repair` to rebuild the index with the correct metric.",
        file=_sys.stderr,
    )


def search(query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    col = _open_collection_or_explain(palace_path, opener=get_collection)
    if col is None:
        if not os.path.isdir(palace_path):
            raise SearchError(f"No palace found at {palace_path}")
        raise SearchError(f"No palace database at {palace_path}")

    # Alert the user if this palace predates hnsw:space=cosine being set on
    # creation — their similarity scores will be junk until they run repair.
    _warn_if_legacy_metric(col)

    where = build_where_filter(wing, room)

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = _first_or_empty(results, "documents")
    metas = _first_or_empty(results, "metadatas")
    dists = _first_or_empty(results, "distances")

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    # Pure-cosine retrieval on the CLI path was missing lexical matches:
    # a drawer whose text contains every query term can still score distance
    # >= 1.0 against the natural-language query when the drawer is a
    # mechanical artifact (directory listing, diff, log fragment) that
    # embeds as file-tree noise rather than as prose about its subject.
    # The MCP tool path already hybridizes BM25 with vector sim via
    # `_hybrid_rank`; do the same here so CLI results match what agents
    # see via `mempalace_search`.
    metric = _metric_for_collection(col)
    hits = [
        {"text": doc or "", "distance": float(dist), "metadata": meta or {}}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
    hits = _hybrid_rank(hits, query, metric=metric)

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, hit in enumerate(hits, 1):
        vec_sim = round(_distance_to_similarity(hit["distance"], metric), 3)
        bm25 = hit.get("bm25_score", 0.0)
        meta = hit["metadata"]
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  {metric}_sim={vec_sim}  bm25={bm25}")
        print()
        # Print the verbatim text, indented
        for line in hit["text"].strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def _bm25_only_via_sqlite(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    source_file: str = None,
    n_results: int = 5,
    max_candidates: int = 500,
    _include_internal: bool = False,
    collection_name: str = None,
) -> dict:
    """BM25-only search reading drawers directly from chroma.sqlite3.

    Used when HNSW is diverged or unloadable (#1222). Bypasses chromadb's
    Python client entirely so a corrupt vector segment can't segfault the
    MCP server. Routes through chromadb's own FTS5 trigram index
    (``embedding_fulltext_search``) for candidate selection, then re-ranks
    with the same Okapi-BM25 used in :func:`_hybrid_rank` so the result
    shape matches the vector path.

    The query is split into ≥3-char trigram-tokens and OR-joined for the
    FTS5 MATCH — chromadb writes the index with ``tokenize='trigram'``,
    so single-character tokens never match. When no usable token survives
    (e.g. "is a"), candidate selection falls back to the most-recent
    ``max_candidates`` rows so we still return *something* rather than
    nothing.
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }
    if collection_name is None:
        from .config import get_configured_collection_name

        collection_name = get_configured_collection_name()

    def _metadata_filter_sql(row_id_expr: str) -> tuple[str, list[str]]:
        clauses = []
        params = []
        for key, value in (("wing", wing), ("room", room), ("source_file", source_file)):
            if not value:
                continue
            clauses.append(
                f"""
                AND EXISTS (
                    SELECT 1
                    FROM embedding_metadata mf
                    WHERE mf.id = {row_id_expr}
                      AND mf.key = ?
                      AND COALESCE(
                        mf.string_value,
                        CAST(mf.int_value AS TEXT),
                        CAST(mf.float_value AS TEXT),
                        CAST(mf.bool_value AS TEXT)
                      ) = ?
                )
                """
            )
            params.extend([key, value])
        return "".join(clauses), params

    try:
        conn = sqlite3.connect(sqlite_read_uri(db_path), uri=True)
    except sqlite3.Error as e:
        return {"error": f"sqlite open failed: {e}"}

    try:
        # FTS5 MATCH expects whitespace-separated tokens. Drop tokens
        # shorter than 3 chars (trigram tokenizer can't match them).
        tokens = [t for t in _tokenize(query) if len(t) >= 3]
        candidate_ids: list[int] = []
        use_recency_fallback = not tokens
        if tokens:
            fts_query = " OR ".join(tokens)
            filter_sql, filter_params = _metadata_filter_sql("embedding_fulltext_search.rowid")
            try:
                rows = conn.execute(
                    f"""
                    SELECT embedding_fulltext_search.rowid
                    FROM embedding_fulltext_search
                    JOIN embeddings e ON e.id = embedding_fulltext_search.rowid
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE embedding_fulltext_search MATCH ?
                      AND c.name = ?
                    {filter_sql}
                    LIMIT ?
                    """,
                    (fts_query, collection_name, *filter_params, max_candidates),
                ).fetchall()
                candidate_ids = [r[0] for r in rows]
            except sqlite3.Error:
                # FTS5 tokenizer mismatch or syntax error — fall through
                # to the recency-window selector below.
                logger.debug("FTS5 MATCH failed; using recency fallback", exc_info=True)
                use_recency_fallback = True

        if not candidate_ids and use_recency_fallback:
            # No usable FTS tokens, or FTS itself failed — pull the most
            # recent rows for the drawers segment so we can BM25-rank
            # something rather than return empty-handed. A clean FTS miss
            # must stay empty, especially after wing/room filtering, because
            # recency fallback would return unrelated scoped drawers.
            # Wrapped in try/except because the schema may differ on legacy
            # palaces (older chromadb without ``created_at``, missing
            # ``segments`` rows after partial restore, etc.); on schema
            # mismatch we fall back to ordering by primary-key id and finally
            # to an empty result rather than letting search raise.
            try:
                filter_sql, filter_params = _metadata_filter_sql("e.id")
                rows = conn.execute(
                    f"""
                    SELECT e.id
                    FROM embeddings e
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE c.name = ?
                    {filter_sql}
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (collection_name, *filter_params, max_candidates),
                ).fetchall()
                candidate_ids = [r[0] for r in rows]
            except sqlite3.Error:
                logger.debug(
                    "recency-window query failed; trying id-ordered fallback",
                    exc_info=True,
                )
                try:
                    filter_sql, filter_params = _metadata_filter_sql("e.id")
                    rows = conn.execute(
                        f"""
                        SELECT e.id
                        FROM embeddings e
                        JOIN segments s ON e.segment_id = s.id
                        JOIN collections c ON s.collection = c.id
                        WHERE c.name = ?
                        {filter_sql}
                        ORDER BY e.id DESC
                        LIMIT ?
                        """,
                        (collection_name, *filter_params, max_candidates),
                    ).fetchall()
                    candidate_ids = [r[0] for r in rows]
                except sqlite3.Error:
                    logger.debug("id-ordered fallback also failed", exc_info=True)
                    candidate_ids = []

        if not candidate_ids:
            return {
                "query": query,
                "filters": {"wing": wing, "room": room, "source_file": source_file},
                "total_before_filter": 0,
                "results": [],
                "fallback": "bm25_only_via_sqlite",
            }

        placeholders = ",".join(["?"] * len(candidate_ids))
        meta_rows = conn.execute(
            f"""
            SELECT id, key, string_value, int_value
            FROM embedding_metadata
            WHERE id IN ({placeholders})
            """,
            candidate_ids,
        ).fetchall()
    finally:
        conn.close()

    # Group metadata rows into per-drawer dicts.
    drawers: dict[int, dict] = {}
    for emb_id, key, sval, ival in meta_rows:
        d = drawers.setdefault(emb_id, {"_id": emb_id, "metadata": {}, "text": ""})
        if key == "chroma:document":
            d["text"] = sval or ""
        else:
            d["metadata"][key] = sval if sval is not None else ival

    # Apply wing/room filters in Python (FTS5 candidates may include
    # entries from other wings).
    candidates = []
    for d in drawers.values():
        meta = d["metadata"]
        if wing and meta.get("wing") != wing:
            continue
        if room and meta.get("room") != room:
            continue
        if source_file and meta.get("source_file") != source_file:
            continue
        full_source = meta.get("source_file", "") or ""
        candidates.append(
            {
                "text": d["text"],
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(full_source).name if full_source else "?",
                "source_path": full_source,
                "created_at": meta.get("filed_at", "unknown"),
                # No vector distance available in BM25-only mode.
                "similarity": None,
                "distance": None,
                "matched_via": "bm25_sqlite",
                # Internal: full path + chunk_index let callers (notably
                # candidate_strategy="union") dedupe at chunk granularity
                # rather than basename — two files in different directories
                # may share a basename, and one source_file is split across
                # multiple chunks. Stripped before this helper returns.
                "_source_file_full": full_source,
                "_chunk_index": meta.get("chunk_index"),
            }
        )

    # Local BM25 over the candidate set.
    docs = [c["text"] for c in candidates]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    for c, raw in zip(candidates, bm25_raw):
        c["bm25_score"] = round(raw, 3)
        c["_score"] = (raw / max_bm25) if max_bm25 > 0 else 0.0
    candidates.sort(key=lambda c: c["_score"], reverse=True)
    hits = candidates[:n_results]
    for h in hits:
        h.pop("_score", None)
        # Strip internal fields by default so the public BM25-only fallback
        # response stays clean. Callers that need chunk-precise dedup
        # (notably the union-merge path) opt in via _include_internal.
        if not _include_internal:
            h.pop("_source_file_full", None)
            h.pop("_chunk_index", None)

    return {
        "query": query,
        "filters": {"wing": wing, "room": room, "source_file": source_file},
        "total_before_filter": len(candidates),
        "results": hits,
        "fallback": "bm25_only_via_sqlite",
        "fallback_reason": "vector_search_disabled",
    }


def _supports_keyword_candidates(collection) -> bool:
    """True when ``collection`` advertises the fork's scoreless keyword seam.

    The Postgres peer backend exposes a ``lexical_search`` that accepts the
    fork's ``restrict_ids`` candidate-scoping argument and advertises it via
    ``supports_keyword_candidates``; chroma's ABC-shaped ``lexical_search`` does
    not. The probe is defensive so a backend without the predicate (or one that
    raises) simply reports no support.
    """
    probe = getattr(collection, "supports_keyword_candidates", None)
    if probe is None:
        return False
    try:
        return probe() is True
    except Exception:
        logger.debug("candidate_strategy=union: capability probe failed", exc_info=True)
        return False


def _fetch_union_candidates(
    lexical_col,
    *,
    query: str,
    n_results: int,
    where: dict | None,
    restrict_ids: list | None,
) -> "list[dict] | None":
    """Pull scoreless keyword candidates from ``lexical_col`` for the union merge.

    Reconciles the fork's ``keyword_candidates`` seam with the merged-in upstream
    ``lexical_search`` contract by dispatching on the collection's advertised
    capability:

    * ``supports_keyword_candidates()`` is the fork's primary route — the
      backend (postgres trigram GIN, chroma FTS5) returns dicts already in
      candidate shape with their own ``matched_via`` (``keyword_postgres`` /
      ``keyword_chroma``), which we leave intact.
    * Otherwise we fall back to the RFC-001 ``lexical_search()`` seam (backends
      such as sqlite_exact / qdrant / pgvector that implement only the typed
      ``LexicalResult`` surface). Those hits are adapted to candidate shape and
      tagged ``matched_via="bm25_backend"``.

    Returns the candidate list (possibly empty), or ``None`` when the handle
    advertises neither route — the caller then skips the merge entirely (the
    pre-G003 no-op).
    """
    over_fetch = n_results * 3

    if _supports_keyword_candidates(lexical_col):
        try:
            return lexical_col.keyword_candidates(
                query=query,
                n_results=over_fetch,
                where=where or None,
                restrict_ids=restrict_ids,
            )
        except Exception:
            # Recall is the design requirement, so a persistently-broken keyword
            # path (missing trigram GIN, schema drift, backend error) must be
            # visible rather than silently degrading union to vector-only.
            logger.warning("candidate_strategy=union: keyword fetch failed", exc_info=True)
            return None

    # Upstream RFC-001 lexical seam: backends that expose a typed
    # ``lexical_search() -> LexicalResult`` but not the fork's
    # ``keyword_candidates`` (sqlite_exact, qdrant, pgvector). Their hits are
    # adapted to candidate shape here and tagged ``bm25_backend``.
    if getattr(lexical_col, "lexical_search", None) is None:
        return None
    try:
        lexical = lexical_col.lexical_search(query=query, n_results=over_fetch, where=where or None)
    except UnsupportedCapabilityError:
        raise
    except Exception:
        logger.warning("candidate_strategy=union: lexical fetch failed", exc_info=True)
        return None

    candidates: list[dict] = []
    for hit in lexical.hits:
        meta = hit.metadata or {}
        full_source = meta.get("source_file", "") or ""
        candidates.append(
            {
                "text": hit.document or "",
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(full_source).name if full_source else "?",
                "source_path": full_source,
                "created_at": meta.get("filed_at", "unknown"),
                "similarity": None,
                "distance": None,
                "matched_via": "bm25_backend",
                "_source_file_full": full_source,
                "_chunk_index": meta.get("chunk_index"),
            }
        )
    return candidates


def _merge_bm25_union_candidates(
    hits: list,
    drawers_col,
    query: str,
    wing: str,
    room: str,
    n_results: int,
    max_distance: float = 0.0,
    collection=None,
    restrict_ids: list | None = None,
    source_file: str = None,
) -> None:
    """Append top-K backend lexical candidates into ``hits`` in place.


    Used by ``search_memories(..., candidate_strategy="union")`` to widen
    the rerank pool's *source* (not just its size) — vector-only candidate
    selection skips docs whose embeddings are far from the query even when
    keyword signal is strong.

    Candidate source is dispatched on the live ``collection`` handle (falling
    back to ``drawers_col`` when no dedicated handle is threaded), reconciling
    the fork's scoreless ``keyword_candidates`` seam with the merged-in upstream
    ``lexical_search`` contract:

    * When the handle advertises
      :meth:`BaseCollection.supports_keyword_candidates` the candidates come
      from ``collection.keyword_candidates(...)`` — the backend's own scoreless
      route (postgres' trigram GIN, chroma's FTS5 wrapper). These dicts arrive
      already in candidate shape and carry their own backend-specific
      ``matched_via`` (``keyword_postgres`` / ``keyword_chroma``), which is
      preserved.
    * Otherwise, when the handle exposes the RFC-001
      :meth:`BaseCollection.lexical_search` seam (e.g. the sqlite_exact / qdrant
      / pgvector backends, which do NOT implement ``keyword_candidates``), the
      candidates come from ``lexical_search() -> LexicalResult`` and are tagged
      ``matched_via="bm25_backend"``.

    A handle advertising neither (or ``collection=None`` with a ``drawers_col``
    that exposes neither) yields zero extra candidates — byte-identical to the
    pre-G003 no-op for any caller that doesn't thread a capable handle.

    Dedup is chunk-precise: the key is ``(_source_file_full, _chunk_index)``
    so two files sharing a basename in different directories don't collide,
    and a vector hit on chunk N of a file doesn't block keyword candidates
    from contributing chunk M of the same file. Falls back to ``source_file``
    only when full-path/chunk metadata is absent.

    Keyword-only additions carry ``distance=None`` so ``_hybrid_rank`` scores
    them on BM25 contribution alone.

    When ``max_distance > 0.0`` (a strict vector-distance threshold is
    set), keyword-only candidates are skipped entirely on BOTH backends —
    they have no vector distance to satisfy the threshold, and silently
    injecting them would break the existing ``max_distance`` guarantee that
    hybrid results lie within the requested vector-distance bound.
    """
    if max_distance > 0.0:
        return

    # The fork threads a dedicated ``collection`` handle (the team-routed backend
    # collection); when present it owns the candidate route, otherwise the
    # single-vault ``drawers_col`` does.
    lexical_col = collection if collection is not None else drawers_col
    if lexical_col is None:
        return

    where = build_where_filter(wing, room, source_file)
    bm25_extra = _fetch_union_candidates(
        lexical_col, query=query, n_results=n_results, where=where, restrict_ids=restrict_ids
    )
    if bm25_extra is None:
        return

    def _dedup_key(entry: dict):
        full = entry.get("_source_file_full")
        ci = entry.get("_chunk_index")
        if full and ci is not None:
            return (full, ci)
        # Empty/missing source_file — the agent-curated add_drawer write path
        # stores source_file="". The old basename fallback returned "?" here, and
        # the merge loop below skips key=="?", so keyword-strong/vector-distant
        # EMPTY-SOURCE drawers (the dominant central-deployment case) never
        # widened the union pool. Key them on (wing, room, chunk_index, text):
        # distinct empty-source drawers stay distinct, while the SAME drawer's
        # vector hit and keyword candidate carry identical fields and still dedup
        # to a single entry (no double-add). Uses only fields already present on
        # both candidate shapes — no contract/backend change.
        if not full:
            return ("empty_source", entry.get("wing"), entry.get("room"), ci, entry.get("text", ""))
        # Non-empty source but no chunk_index: basename dedup (legacy data).
        return entry.get("source_file")

    seen = {_dedup_key(h) for h in hits}
    for bh in bm25_extra:
        key = _dedup_key(bh)
        if not key or key == "?" or key in seen:
            continue
        bh["distance"] = None
        bh["effective_distance"] = None
        bh["closet_boost"] = 0.0
        hits.append(bh)
        seen.add(key)


# Strategy dispatch — keeps search_memories' branch count under the
# project's complexity ceiling (C901 max-complexity=25). New strategies
# register here.
_CANDIDATE_MERGERS = {
    "vector": None,  # default no-op
    "union": _merge_bm25_union_candidates,
}


def _validate_candidate_strategy(strategy: str) -> None:
    """Raise ``ValueError`` for unknown strategies.

    Called eagerly at the top of ``search_memories`` so invalid values
    fail consistently regardless of whether the call routes through the
    vector path, the BM25-only fallback, or returns an early error dict.
    """
    if strategy not in _CANDIDATE_MERGERS:
        raise ValueError(
            f"candidate_strategy must be one of {tuple(_CANDIDATE_MERGERS)}, got {strategy!r}"
        )


# Sentinel for ``search_memories(candidate_strategy=...)``: distinguishes
# "caller passed nothing" (consult the env, then default) from an explicit
# value. The default candidate strategy is ``"union"`` (G003) — keyword
# candidates are merged into the rerank pool on both backends.
_DEFAULT_CANDIDATE_STRATEGY = "union"
_CANDIDATE_STRATEGY_UNSET = object()


def _resolve_candidate_strategy(explicit) -> str:
    """Resolve the effective candidate strategy.

    Precedence: an explicit ``candidate_strategy`` argument always wins; when
    the caller passes nothing, the ``MEMPALACE_CANDIDATE_STRATEGY`` env var is
    consulted (the per-deployment latency escape hatch — e.g. force ``"vector"``
    to skip the extra keyword query); when neither is set the default is
    ``"union"``. The resolved value is validated against the allowed set so an
    invalid env value fails the same way an invalid argument does.
    """
    if explicit is not _CANDIDATE_STRATEGY_UNSET:
        return explicit
    env_value = os.environ.get("MEMPALACE_CANDIDATE_STRATEGY")
    if env_value:
        _validate_candidate_strategy(env_value)
        return env_value
    return _DEFAULT_CANDIDATE_STRATEGY


def _apply_candidate_strategy(
    strategy: str,
    hits: list,
    drawers_col,
    query: str,
    wing: str,
    room: str,
    n_results: int,
    max_distance: float = 0.0,
    collection=None,
    restrict_ids: list | None = None,
    source_file: str = None,
) -> None:
    """Dispatch to the registered merger for ``strategy``.

    Strategy validity is assumed (``_validate_candidate_strategy`` runs
    earlier); ``"vector"`` is a no-op. The live ``collection`` handle (the
    fork's team-routed backend collection) is threaded through so the union
    merger can dispatch keyword retrieval on the backend's capability, along
    with the fork's ``restrict_ids`` candidate scope (see
    ``_merge_bm25_union_candidates``).
    """
    merger = _CANDIDATE_MERGERS[strategy]
    if merger is not None:
        merger(
            hits,
            drawers_col,
            query,
            wing,
            room,
            n_results,
            max_distance=max_distance,
            collection=collection,
            restrict_ids=restrict_ids,
            source_file=source_file,
        )


def _open_drawers_collection(palace_path, collection_name, team, query):
    """Fetch the drawers collection, returning the collection object or an error dict.

    Separates the exception-handling branches from ``search_memories`` to keep
    that function's cyclomatic complexity within the ruff C901 budget.
    """
    try:
        return get_collection(palace_path, collection_name=collection_name, create=False, team=team)
    except (PalaceNotFoundError, CollectionNotInitializedError):
        if team is not None:
            # Central / postgres path: the vault simply has no entries yet.
            # Return the empty-vault shape so callers can distinguish "nothing
            # filed yet" from a real backend error. Other exceptions propagate
            # (PM#3: a DB outage must NOT look like an empty vault).
            return {
                "query": query,
                "results": [],
                "vault": team,
                "empty_vault": True,
                "reason": (
                    f"vault {team!r} has no entries yet or does not exist; "
                    "check the team name or write first"
                ),
            }
        # Chroma / local path: the palace genuinely needs init+mine.
        logger.error("No palace found at %s", palace_path)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }
    except BackendMismatchError as e:
        return _backend_mismatch_result(e)
    except BackendError as e:
        # A real backend failure (DB outage, connection refused, remote
        # unavailable) is surfaced DISTINCTLY from a missing palace (RFC-001):
        # callers must be able to tell "the vault is empty / not yet built"
        # apart from "the storage layer is down". Mirrors the shape returned by
        # ``_open_search_collection`` so both read paths report identically.
        logger.error("Backend error opening palace at %s: %s", palace_path, e)
        return {
            "error": "Backend error",
            "details": str(e),
            "hint": "Check the selected backend configuration and availability.",
        }
    except Exception as e:
        # Any other exception surfaces as an error — never silently converted to
        # the empty-vault shape.
        logger.error("Palace read error at %s: %s", palace_path, e)
        return {"error": f"palace read error: {e}"}


def _finalize_candidate_hits(
    *,
    candidate_strategy: str,
    hits: list,
    drawers_col,
    query: str,
    wing: str,
    room: str,
    n_results: int,
    max_distance: float,
    collection=None,
    restrict_ids: list | None = None,
    source_file: str = None,
) -> tuple:
    try:
        _apply_candidate_strategy(
            candidate_strategy,
            hits,
            drawers_col,
            query,
            wing,
            room,
            n_results,
            max_distance=max_distance,
            collection=collection,
            restrict_ids=restrict_ids,
            source_file=source_file,
        )
    except UnsupportedCapabilityError:
        return [], {
            "error": "candidate_strategy='union' requires a backend with lexical_search support",
            "unsupported_capability": "supports_lexical_search",
            "hint": "Use candidate_strategy='vector' or select a backend that supports lexical search.",
        }

    hits = _hybrid_rank(hits, query, metric=_metric_for_collection(drawers_col))[:n_results]
    for h in hits:
        h.pop("_sort_key", None)
        h.pop("_source_file_full", None)
        h.pop("_chunk_index", None)
        h.pop("_parent_drawer_id", None)
    return hits, None


def _backend_mismatch_result(error: BackendMismatchError) -> dict:
    return {
        "error": "Backend mismatch",
        "details": str(error),
        "hint": "Select the matching backend or use a fresh palace directory.",
    }


def _unknown_backend_result(error: KeyError) -> dict:
    return {
        "error": "Unknown backend",
        "details": str(error),
        "hint": "Check MEMPALACE_BACKEND or the configured backend name.",
    }


def _vector_disabled_search(
    *,
    query: str,
    palace_path: str,
    wing: str,
    room: str,
    n_results: int,
    collection_name: str,
    source_file: str = None,
) -> dict:
    try:
        backend_name = resolve_backend_name(palace_path)
    except BackendMismatchError as e:
        return _backend_mismatch_result(e)
    except KeyError as e:
        return _unknown_backend_result(e)
    if backend_name != "chroma":
        return {
            "error": "vector_disabled fallback is Chroma-only",
            "unsupported_capability": "chroma_hnsw_fallback",
            "backend": backend_name,
            "hint": "Disable vector_disabled for non-Chroma backends.",
        }
    return _bm25_only_via_sqlite(
        query,
        palace_path,
        wing=wing,
        room=room,
        source_file=source_file,
        n_results=n_results,
        collection_name=collection_name,
    )


def _open_search_collection(palace_path: str, collection_name: str):
    try:
        return get_collection(palace_path, collection_name=collection_name, create=False), None
    except BackendMismatchError as e:
        return None, _backend_mismatch_result(e)
    except KeyError as e:
        return None, _unknown_backend_result(e)
    except (CollectionNotInitializedError, PalaceNotFoundError) as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return None, {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }
    except BackendError as e:
        logger.error("Backend error opening palace at %s: %s", palace_path, e)
        return None, {
            "error": "Backend error",
            "details": str(e),
            "hint": "Check the selected backend configuration and availability.",
        }
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return None, {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }


def _query_drawers_with_filter_fallback(
    drawers_col, dkwargs, query, n_results, wing, room, source_file=None
):
    """Run the filtered drawer query, falling back to an unfiltered query plus a
    Python-side post-filter when ChromaDB raises on the filtered query.

    A ChromaDB HNSW/SQLite index mismatch makes filtered queries fail with
    "Error finding id" even when unfiltered search works fine — it happens when
    drawers are ingested via two different paths (e.g. bulk import vs MCP tool
    calls), leaving the vector index inconsistent with the metadata store. We
    retry unfiltered (over-fetching) and re-apply the wing/room/source_file filter in Python.
    See #1245 / #1035.
    """
    where = dkwargs.get("where")
    try:
        return drawers_col.query(**dkwargs)
    except Exception as filter_err:
        if not where:
            raise
        logger.warning(
            "Filtered search failed (%s); falling back to unfiltered + post-filter",
            filter_err,
        )
        raw = drawers_col.query(
            query_texts=[query],
            n_results=min(n_results * 15, 500),
            include=["documents", "metadatas", "distances"],
        )
        fdocs, fmetas, fdists = [], [], []
        for doc, meta, dist in zip(
            _first_or_empty(raw, "documents"),
            _first_or_empty(raw, "metadatas"),
            _first_or_empty(raw, "distances"),
        ):
            meta = meta or {}
            if wing and meta.get("wing") != wing:
                continue
            if room and meta.get("room") != room:
                continue
            if source_file and meta.get("source_file") != source_file:
                continue
            fdocs.append(doc)
            fmetas.append(meta)
            fdists.append(dist)
        return {"documents": [fdocs], "metadatas": [fmetas], "distances": [fdists]}


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    source_file: str = None,
    n_results: int = 5,
    max_distance: float = 0.0,
    vector_disabled: bool = False,
    candidate_strategy=_CANDIDATE_STRATEGY_UNSET,
    collection_name: str = None,
    team: str = None,
    restrict_ids: list | None = None,
) -> dict:
    """Programmatic search — returns a dict instead of printing.

    Used by the MCP server and other callers that need data.

    Args:
        query: Natural language search query.
        palace_path: Path to the ChromaDB palace directory.
        wing: Optional wing filter.
        room: Optional room filter.
        source_file: Optional exact source_file filter. Matches the full
            stored source_file value verbatim (#1815).
        n_results: Max results to return.
        max_distance: Max cosine distance threshold. The palace collection uses
            cosine distance (hnsw:space=cosine) — 0 = identical, 2 = opposite.
            Results with distance > this value are filtered out. A value of
            0.0 disables filtering. Typical useful range: 0.3–1.0.
        vector_disabled: When True, route to the sqlite-only BM25 fallback
            (#1222). Set by the MCP server when the HNSW capacity probe
            detects a divergence that would segfault chromadb on segment
            load.
        candidate_strategy: How candidates for the hybrid re-rank are gathered.
            When the argument is omitted, the value is resolved from the
            ``MEMPALACE_CANDIDATE_STRATEGY`` env var if set, otherwise it
            defaults to ``"union"`` (G003). An explicit argument always wins
            over the env var.

            * ``"union"`` (default) — pull top ``n_results * 3`` scoreless
              keyword candidates from the live collection's own lexical route
              (``collection.lexical_search`` — postgres' trigram GIN adapter,
              chroma's FTS5 path) and merge them into the rerank pool (deduped
              chunk-precisely). Catches docs with strong keyword signal that
              are vector-distant from the query (e.g. terminology guides looked
              up by narrative-shaped queries; policy clauses surfaced by
              scenario descriptions). Adds one indexed keyword query per search
              on both backends.

              When ``max_distance > 0.0`` is also set, keyword-only candidates
              are skipped — they have no vector distance and would silently
              violate the requested distance threshold.
            * ``"vector"`` — preserves the historical narrow behavior: only the
              top ``n_results * 3`` rows from the vector index form the rerank
              pool; no keyword candidates are injected. The per-process latency
              escape hatch (set ``MEMPALACE_CANDIDATE_STRATEGY=vector``, or pass
              the argument explicitly) when the extra keyword query is too
              costly for a deployment.
        restrict_ids: Optional list of drawer ids to restrict the candidate set
            to before ranking (entity-scoped recall). Applied to the drawer
            floor query only; ``None`` (default) leaves search unscoped. Honored
            by the Postgres backend; the chroma path does not set it.
    """
    # Resolve the effective strategy: explicit arg > MEMPALACE_CANDIDATE_STRATEGY
    # env > "union" default. Validate eagerly so invalid values (arg or env)
    # fail the same way regardless of whether the call routes through the
    # vector path or the BM25-only fallback below.
    candidate_strategy = _resolve_candidate_strategy(candidate_strategy)
    _validate_candidate_strategy(candidate_strategy)

    if vector_disabled:
        return _vector_disabled_search(
            query=query,
            palace_path=palace_path,
            wing=wing,
            room=room,
            n_results=n_results,
            collection_name=collection_name,
            source_file=source_file,
        )

    drawers_col_or_err = _open_drawers_collection(palace_path, collection_name, team, query)
    if isinstance(drawers_col_or_err, dict):
        return drawers_col_or_err
    drawers_col = drawers_col_or_err

    metric = _metric_for_collection(drawers_col)
    where = build_where_filter(wing, room, source_file)

    # Hybrid retrieval: always query drawers directly (the floor), then use
    # closet hits to boost rankings. Closets are a ranking SIGNAL, never a
    # GATE — direct drawer search is always the baseline.
    #
    # This avoids the "weak-closets regression" where narrative content
    # produces low-signal closets (regex extraction matches few topics)
    # and closet-first routing hides drawers that direct search would find.
    try:
        dkwargs = {
            "query_texts": [query],
            "n_results": n_results * 3,  # over-fetch for re-ranking
            # Entity-scoped recall: restrict the drawer-floor candidate set to
            # these ids before the vector rank (None = unscoped; closets stay an
            # unrestricted ranking signal). Postgres honors it; chroma accepts
            # and ignores it (entity= is postgres-only).
            "restrict_ids": restrict_ids,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            dkwargs["where"] = where
        drawer_results = _query_drawers_with_filter_fallback(
            drawers_col, dkwargs, query, n_results, wing, room, source_file
        )
    except Exception as e:
        return {"error": f"Search error: {e}"}

    # Gather closet hits (best-per-source) to build a boost lookup.
    closet_boost_by_source: dict = {}  # source_file -> (rank, closet_dist, preview)
    try:
        closets_col = get_closets_collection(palace_path, create=False, team=team)
        ckwargs = {
            "query_texts": [query],
            "n_results": n_results * 2,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
        for rank, (cdoc, cmeta, cdist) in enumerate(
            zip(
                _first_or_empty(closet_results, "documents"),
                _first_or_empty(closet_results, "metadatas"),
                _first_or_empty(closet_results, "distances"),
            )
        ):
            cmeta = cmeta or {}
            source = cmeta.get("source_file", "")
            if source and source not in closet_boost_by_source:
                closet_boost_by_source[source] = (rank, cdist, cdoc[:200])
    except Exception:
        # No closets yet — hybrid degrades to pure drawer search.
        logger.debug("Closet collection unavailable; using drawer-only search", exc_info=True)

    # Rank-based boost. The ordinal signal ("which closet matched best") is
    # more reliable than absolute distance on narrative content, where
    # closet distances cluster in 1.2-1.5 range regardless of match quality.
    CLOSET_RANK_BOOSTS = [0.40, 0.25, 0.15, 0.08, 0.04]
    CLOSET_DISTANCE_CAP = 1.5  # cosine dist > 1.5 = too weak to use as signal

    scored: list = []
    for doc, meta, dist in zip(
        _first_or_empty(drawer_results, "documents"),
        _first_or_empty(drawer_results, "metadatas"),
        _first_or_empty(drawer_results, "distances"),
    ):
        meta = meta or {}
        doc = doc or ""
        # Filter on raw distance before rounding to avoid precision loss.
        if max_distance > 0.0 and dist > max_distance:
            continue

        meta = meta or {}
        source = meta.get("source_file", "") or ""
        # Effective closet-boost key: a real source_file keys on itself; an
        # empty source_file (an agent-curated add_drawer write) keys on the same
        # deterministic (wing, room) fallback the server-side rebuild used, so
        # the closet built for empty-source drawers actually boosts them. Without
        # this, empty-source drawers never match a closet and the boost is inert.
        boost_source = closet_grouping_key(source, meta.get("wing", ""), meta.get("room", ""))
        boost = 0.0
        matched_via = "drawer"
        closet_preview = None
        if boost_source in closet_boost_by_source:
            c_rank, c_dist, c_preview = closet_boost_by_source[boost_source]
            if c_dist <= CLOSET_DISTANCE_CAP and c_rank < len(CLOSET_RANK_BOOSTS):
                boost = CLOSET_RANK_BOOSTS[c_rank]
                matched_via = "drawer+closet"
                closet_preview = c_preview

        # Clamp to the valid cosine-distance range [0, 2]. When a strong
        # closet boost (up to 0.40) exceeds the raw distance, the subtraction
        # can go negative — which (a) yields ``similarity > 1.0`` downstream
        # and (b) makes the sort key land *below* ordinary positive distances,
        # inverting the ranking so the best hybrid matches sort last.
        effective_dist = max(0.0, min(2.0, dist - boost))
        entry = {
            "text": doc,
            "wing": meta.get("wing", "unknown"),
            "room": meta.get("room", "unknown"),
            # source_file is the basename (display); source_path is the full
            # stored value, the round-trippable key for the source_file filter.
            "source_file": Path(source).name if source else "?",
            "source_path": source,
            "created_at": meta.get("filed_at", "unknown"),
            "similarity": round(_distance_to_similarity(effective_dist, metric), 3),
            "distance": round(dist, 4),
            "effective_distance": round(effective_dist, 4),
            "closet_boost": round(boost, 3),
            "matched_via": matched_via,
            # Internal: retain the full source_file path + chunk_index so the
            # enrichment step below doesn't have to reverse-lookup via
            # basename-suffix matching (which silently collides when two
            # files share a basename across different directories).
            "_sort_key": effective_dist,
            "_source_file_full": source,
            "_chunk_index": meta.get("chunk_index"),
            "_parent_drawer_id": meta.get("parent_drawer_id"),
        }
        if closet_preview:
            entry["closet_preview"] = closet_preview
        scored.append(entry)

    scored.sort(key=lambda h: h["_sort_key"])
    hits = scored[:n_results]

    # Drawer-grep enrichment: for closet-boosted hits whose source has
    # multiple drawers, return the keyword-best chunk + its immediate
    # neighbors instead of just the drawer vector search landed on. The
    # closet said "this source is relevant"; vector may have picked the
    # wrong chunk within it; grep picks the right one.
    MAX_HYDRATION_CHARS = 10000
    for h in hits:
        if h["matched_via"] == "drawer":
            continue
        full_source = h.get("_source_file_full") or ""
        if not full_source:
            continue
        # Narrow by ``parent_drawer_id`` when present so unrelated
        # chunked drawers sharing ``source_file`` do not stitch (#1580).
        try:
            source_drawers = drawers_col.get(
                where=_scoped_source_filter(full_source, h.get("_parent_drawer_id")),
                include=["documents", "metadatas"],
            )
        except Exception:
            logger.debug("Neighbor fetch failed for %s", full_source, exc_info=True)
            continue
        docs = source_drawers.documents
        metas_ = source_drawers.metadatas
        if len(docs) <= 1:
            continue

        # Sort by chunk_index so best_idx + neighbors are positional.
        indexed = []
        for idx, (d, m) in enumerate(zip(docs, metas_)):
            ci = m.get("chunk_index", idx) if isinstance(m, dict) else idx
            if not isinstance(ci, int):
                ci = idx
            indexed.append((ci, d))
        indexed.sort(key=lambda p: p[0])
        ordered_docs = [d for _, d in indexed]

        query_terms = set(_tokenize(query))
        best_idx, best_score = 0, -1
        for idx, d in enumerate(ordered_docs):
            d_lower = d.lower()
            s = sum(1 for t in query_terms if t in d_lower)
            if s > best_score:
                best_score, best_idx = s, idx

        start = max(0, best_idx - 1)
        end = min(len(ordered_docs), best_idx + 2)
        expanded = "\n\n".join(ordered_docs[start:end])
        if len(expanded) > MAX_HYDRATION_CHARS:
            expanded = (
                expanded[:MAX_HYDRATION_CHARS]
                + f"\n\n[...truncated. {len(ordered_docs)} total drawers. "
                "Use mempalace_get_drawer for full content.]"
            )
        h["text"] = expanded
        h["drawer_index"] = best_idx
        h["total_drawers"] = len(ordered_docs)

    # Candidate strategy hook: optionally widen the rerank pool's *source*
    # before ranking. Default ("vector") is a no-op; "union" merges top-K
    # backend lexical candidates. See `_apply_candidate_strategy`.
    # ``max_distance`` is forwarded so union mode can refuse to inject
    # BM25-only (distance=None) candidates that would silently bypass the
    # caller's strict distance threshold.
    # The helper also runs the final BM25 hybrid re-rank and strips internal
    # dedup fields before returning.
    hits, strategy_error = _finalize_candidate_hits(
        candidate_strategy=candidate_strategy,
        hits=hits,
        drawers_col=drawers_col,
        query=query,
        wing=wing,
        room=room,
        n_results=n_results,
        max_distance=max_distance,
        collection=drawers_col,
        restrict_ids=restrict_ids,
        source_file=source_file,
    )
    if strategy_error:
        return strategy_error

    return {
        "query": query,
        "filters": {"wing": wing, "room": room, "source_file": source_file},
        "total_before_filter": len(_first_or_empty(drawer_results, "documents")),
        "results": hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Virtual line numbering — read-time grid for drawers (3.3.6).
#
# Drawers are stored verbatim on disk. The reader applies a line-number grid
# at read time so any drawer — numbered or not — can be sectioned by a closet
# pointer like ``→2026-01-18:L55-L72`` without rewriting the corpus. Pure
# functions, no I/O. Source drawer text is never mutated.
# See docs/virtual-line-numbering.md for the full design rationale.
# ─────────────────────────────────────────────────────────────────────────────


# A line is "already numbered" iff it starts with [<digits>].
_ALREADY_NUMBERED_RE = re.compile(r"^\[\d+\]")


def render_with_line_numbers(text: "str | None", start_line: int = 1) -> str:
    """Prefix each line of ``text`` with ``[N] `` for read-time grid display.

    Lines that already begin with ``[<digits>]`` pass through unchanged,
    but the counter still advances on them so callers can rely on positional
    alignment with the original line indices.

    ``None`` is treated as empty string. Pure function.
    """
    if not text:
        return ""
    out = []
    for i, line in enumerate(text.split("\n"), start=start_line):
        if _ALREADY_NUMBERED_RE.match(line):
            out.append(line)
        else:
            out.append(f"[{i}] {line}")
    return "\n".join(out)


def extract_line_range(text: str, line_start: int, line_end: int) -> str:
    """Return the 1-indexed inclusive slice ``[line_start, line_end]`` rendered with line numbers.

    This is the closet-pointer read path. A pointer like ``→2026-01-18:L55-L72``
    resolves by opening the day-drawer and calling ``extract_line_range(drawer_text, 55, 72)``.
    Out-of-bounds ranges are clamped. Invalid ranges return ``""``.
    """
    if not text:
        return ""
    if line_end < line_start:
        return ""

    lines = text.split("\n")
    effective_start = max(1, line_start)
    effective_end = min(len(lines), line_end)

    if effective_start > effective_end:
        return ""

    section = "\n".join(lines[effective_start - 1 : effective_end])
    return render_with_line_numbers(section, start_line=effective_start)
