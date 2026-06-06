"""PostgreSQL-backed MemPalace storage backend (RFC 001).

Central, team-vaulted storage for a company-hosted MemPalace. Implements the
:class:`BaseBackend` / :class:`BaseCollection` contract so it is a drop-in for
the default Chroma backend.

Design:

* **Schema-per-team vaults.** ``PalaceRef.namespace`` (the team, e.g.
  ``frontend``) maps to a Postgres schema ``team_<namespace>``. Each logical
  collection (``mempalace_drawers`` / ``mempalace_closets``) is a table inside
  that schema, so isolation, per-team backup, and ``list_vaults`` are
  structural rather than filter-based.
* **pgvector** stores the 384-dim embeddings (``vector(N)``) and serves cosine
  nearest-neighbour search via the ``<=>`` operator + an HNSW index. Distances
  are returned in pgvector's native cosine range ``[0, 2]`` — identical to
  Chroma's ``hnsw:space=cosine``, so ``searcher.py`` needs no changes.
* **JSONB metadata** with the Chroma where-clause algebra translated to SQL.
* **Keyword recall** rides a ``pg_trgm`` trigram GIN on ``document``: both the
  ``$contains`` filter (for ≥3-char needles) and ``keyword_candidates()`` use it
  to fetch SCORELESS rows (``distance=None``) that the single shared Python
  Okapi-BM25 reranker (``searcher._hybrid_rank``) then ranks — the same ranker
  the Chroma path uses. No in-DB score (no ``paradedb.score``) ever enters the
  pipeline; the ``pg_search`` extension is provisioned (``_ensure_bootstrap``)
  but the query path does not use it for retrieval.
* **Embeddings are computed client-side** (the same embedding function Chroma
  uses) whenever the caller does not pass precomputed vectors, so existing
  miner/searcher call sites keep working unchanged.

Connection work is deferred to first use; a single ``psycopg_pool`` connection
pool is shared per DSN. No authentication layer beyond Postgres itself — this
is internal/engineering-only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Callable, Optional

from .base import (
    BaseBackend,
    BaseCollection,
    BackendClosedError,
    CollectionNotInitializedError,
    DimensionMismatchError,
    GetResult,
    HealthStatus,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)

logger = logging.getLogger(__name__)

# Where-clause operators — identical contract to the Chroma backend.
_REQUIRED_OPERATORS = frozenset({"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains"})
_OPTIONAL_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
_SUPPORTED_OPERATORS = _REQUIRED_OPERATORS | _OPTIONAL_OPERATORS
_COMPARISON_OPS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}

# Default embedding dimension (minilm / embeddinggemma-300m both emit 384).
_DEFAULT_VECTOR_DIM = 384

# Apache AGE graph name (provisioned best-effort; used by palace traversal / KG).
_AGE_GRAPH = "mempalace"

# Identifier safety: schemas and tables are interpolated into DDL/DML, so they
# must match a strict allowlist before being double-quoted.
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_TEAM_RE = re.compile(r"^[a-z0-9_]+$")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _require_psycopg():
    """Return ``psycopg_pool.ConnectionPool``, erroring if the extra is missing."""
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "The Postgres backend requires psycopg[binary] and psycopg_pool. "
            "Install the extra: pip install 'mempalace[postgres]'"
        ) from e
    return ConnectionPool


def dsn_from_env() -> str:
    """Build a libpq DSN/URL from MEMPALACE_* env vars.

    ``MEMPALACE_DATABASE_URL`` wins if set; otherwise the discrete
    ``MEMPALACE_PG_*`` parts are assembled. Defaults target the bundled
    docker-compose database.
    """
    url = os.environ.get("MEMPALACE_DATABASE_URL")
    if url:
        return url
    host = os.environ.get("MEMPALACE_PG_HOST", "localhost")
    port = os.environ.get("MEMPALACE_PG_PORT", "5432")
    db = os.environ.get("MEMPALACE_PG_DB", "mempalace")
    user = os.environ.get("MEMPALACE_PG_USER", "mempalace")
    password = os.environ.get("MEMPALACE_PG_PASSWORD", "mempalace")
    return f"host={host} port={port} dbname={db} user={user} password={password}"


def sanitize_team(namespace: Optional[str]) -> str:
    """Normalise a team/namespace into a schema-safe slug, default ``default``.

    Capped at 40 chars to match the override-path validator (mcp_server's
    ``_TEAM_SLUG_RE`` / ``_canonical_default_team``), so the physical schema slug
    derived here can never exceed — and therefore never diverge from — the slug
    the server reports back to the caller.
    """
    raw = (namespace or os.environ.get("MEMPALACE_TEAM") or "default").strip().lower()
    raw = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")[:40]
    if not raw:
        raw = "default"
    if not _TEAM_RE.match(raw):  # pragma: no cover - defensive
        raise ValueError(f"invalid team namespace: {namespace!r}")
    return raw


def team_schema(namespace: Optional[str]) -> str:
    """Return the Postgres schema name for a team vault."""
    return f"team_{sanitize_team(namespace)}"


# Central write-ahead audit log (G004). One table for the whole org; the per-row
# ``team`` column records which vault each write targeted. Kept in its own schema
# (not a ``team_<slug>`` vault schema) so the audit trail is queryable across all
# teams from one place.
_WAL_AUDIT_SCHEMA = "mempalace_audit"
_WAL_AUDIT_TABLE = "write_log"


def _qi(ident: str) -> str:
    """Validate + double-quote an SQL identifier."""
    if not _IDENT_RE.match(ident):
        raise ValueError(f"unsafe SQL identifier: {ident!r}")
    return f'"{ident}"'


def _vec_literal(embedding) -> str:
    """Format a float sequence as a pgvector literal ``[a,b,c]``."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _validate_where(where: Optional[dict]) -> None:
    """Reject unknown operators (RFC 001 §1.4: silent drops forbidden)."""
    if not where:
        return
    stack = [where]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if k.startswith("$") and k not in _SUPPORTED_OPERATORS:
                raise UnsupportedFilterError(f"operator {k!r} not supported by postgres backend")
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))


# ---------------------------------------------------------------------------
# Where-clause translation (Chroma dict algebra -> SQL over a jsonb column)
# ---------------------------------------------------------------------------


class _WhereTranslator:
    """Translate a Chroma ``where`` dict into a parameterised SQL fragment.

    Metadata is stored in a ``metadata`` JSONB column. Scalar comparisons read
    ``metadata->>'field'`` (text) and cast for numeric operators.
    """

    def __init__(self, meta_col: str = "metadata"):
        self.meta_col = meta_col
        self.params: list[Any] = []

    def translate(self, where: Optional[dict]) -> Optional[str]:
        if not where:
            return None
        clauses = [self._node(k, v) for k, v in where.items()]
        clauses = [c for c in clauses if c]
        if not clauses:
            return None
        return " AND ".join(f"({c})" for c in clauses)

    def _node(self, key: str, value) -> Optional[str]:
        if key == "$and":
            parts = [self._dict(sub) for sub in value]
            parts = [p for p in parts if p]
            return " AND ".join(f"({p})" for p in parts) if parts else None
        if key == "$or":
            parts = [self._dict(sub) for sub in value]
            parts = [p for p in parts if p]
            return " OR ".join(f"({p})" for p in parts) if parts else None
        if key.startswith("$"):
            raise UnsupportedFilterError(f"operator {key!r} not valid at this position")
        # key is a metadata field; value is a scalar (=> $eq) or an operator dict.
        if isinstance(value, dict):
            return self._field_ops(key, value)
        return self._cmp(key, "$eq", value)

    def _dict(self, sub: dict) -> Optional[str]:
        clauses = [self._node(k, v) for k, v in sub.items()]
        clauses = [c for c in clauses if c]
        return " AND ".join(f"({c})" for c in clauses) if clauses else None

    def _field_ops(self, field: str, ops: dict) -> Optional[str]:
        clauses = []
        for op, operand in ops.items():
            clauses.append(self._cmp(field, op, operand))
        return " AND ".join(f"({c})" for c in clauses) if clauses else None

    def _accessor(self, field: str) -> str:
        # ``->>`` returns text; safe for psycopg to bind the key as a literal
        # since field names come from trusted callers, but we still bind it.
        self.params.append(field)
        return f"{self.meta_col}->>%s"

    def _cmp(self, field: str, op: str, operand) -> str:
        acc = self._accessor(field)
        if op == "$eq":
            self.params.append(_as_text(operand))
            return f"{acc} = %s"
        if op == "$ne":
            self.params.append(_as_text(operand))
            return f"{acc} IS DISTINCT FROM %s"
        if op == "$in":
            self.params.append([_as_text(x) for x in operand])
            return f"{acc} = ANY(%s)"
        if op == "$nin":
            # ``acc`` already consumed one field param; the array param comes
            # next, then a second field accessor for the IS NULL branch (so
            # rows missing the key are excluded, matching Chroma's $nin).
            self.params.append([_as_text(x) for x in operand])
            self.params.append(field)
            return f"({acc} <> ALL(%s) OR {self.meta_col}->>%s IS NULL)"
        if op in _COMPARISON_OPS:
            self.params.append(float(operand))
            # Cast text accessor to numeric for ordered comparison.
            return f"({acc})::float8 {_COMPARISON_OPS[op]} %s"
        if op == "$contains":
            raise UnsupportedFilterError("$contains is only supported in where_document")
        raise UnsupportedFilterError(f"operator {op!r} not supported by postgres backend")


def _as_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# Token splitter for keyword-candidate selection. Mirrors ``searcher._tokenize``
# (lowercase, ``\w{2,}`` unicode word runs) so the postgres keyword path and the
# chroma FTS path tokenise identically; re-implemented locally to avoid importing
# ``searcher`` (which imports the backends — a circular import). Callers apply the
# ≥3-char trigram floor on top, matching ``_bm25_only_via_sqlite``.
_KEYWORD_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _tokenize_query(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return _KEYWORD_TOKEN_RE.findall(text.lower())


def _basename(path: str) -> str:
    """Return the trailing path component (handles both '/' and '\\')."""
    return re.split(r"[\\/]", path)[-1]


def _where_document_clause(
    where_document: Optional[dict], params: list[Any], doc_col: str = "document"
) -> Optional[str]:
    """Translate a where_document ``{"$contains": "x"}`` into an ILIKE clause."""
    if not where_document:
        return None
    clauses = []
    for op, operand in where_document.items():
        if op == "$contains":
            params.append(f"%{operand}%")
            clauses.append(f"{doc_col} ILIKE %s")
        elif op in ("$and", "$or"):
            joiner = " AND " if op == "$and" else " OR "
            subs = [_where_document_clause(sub, params, doc_col) for sub in operand]
            subs = [s for s in subs if s]
            if subs:
                clauses.append(joiner.join(f"({s})" for s in subs))
        else:
            raise UnsupportedFilterError(f"where_document operator {op!r} not supported")
    return " AND ".join(f"({c})" for c in clauses) if clauses else None


# ---------------------------------------------------------------------------
# Embedding (client-side, same EF as Chroma)
# ---------------------------------------------------------------------------

_embedder: Optional[Callable] = None
_embedder_lock = threading.Lock()


def _default_embedder() -> Optional[Callable]:
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        try:
            from ..embedding import get_embedding_function

            _embedder = get_embedding_function()
        except Exception:
            logger.exception("Failed to build embedding function for postgres backend")
            _embedder = None
        return _embedder


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class PostgresCollection(BaseCollection):
    """A table inside a team schema, exposing the BaseCollection contract."""

    def __init__(self, backend: "PostgresBackend", schema: str, table: str, dim: int):
        self._backend = backend
        self._schema = schema
        self._table = table
        self._dim = dim

    @property
    def _fqtn(self) -> str:
        return f"{_qi(self._schema)}.{_qi(self._table)}"

    # -- embedding --------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        ef = self._backend._embedder or _default_embedder()
        if ef is None:
            raise RuntimeError(
                "No embedding function available; pass embeddings explicitly "
                "or install the embedding model dependencies."
            )
        result = ef(texts)
        return [list(v) for v in result]

    def _check_dim(self, embeddings) -> None:
        for vec in embeddings:
            if vec is not None and len(vec) != self._dim:
                raise DimensionMismatchError(
                    f"embedding dim {len(vec)} != collection dim {self._dim}"
                )

    # -- writes -----------------------------------------------------------
    def add(self, *, documents, ids, metadatas=None, embeddings=None) -> None:
        self._write(documents, ids, metadatas, embeddings, upsert=False)

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None) -> None:
        self._write(documents, ids, metadatas, embeddings, upsert=True)

    def _write(self, documents, ids, metadatas, embeddings, *, upsert: bool) -> None:
        n = len(ids)
        if documents is None:
            documents = ["" for _ in ids]
        if embeddings is None:
            embeddings = self._embed([d or "" for d in documents])
        self._check_dim(embeddings)
        if metadatas is None:
            metadatas = [{} for _ in ids]

        rows = []
        for i in range(n):
            meta = metadatas[i] if metadatas[i] else {}
            emb = embeddings[i] if i < len(embeddings) else None
            rows.append(
                (
                    ids[i],
                    documents[i] if documents[i] is not None else "",
                    json.dumps(meta),
                    _vec_literal(emb) if emb is not None else None,
                )
            )
        conflict = (
            "ON CONFLICT (id) DO UPDATE SET document = EXCLUDED.document, "
            "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding"
            if upsert
            else "ON CONFLICT (id) DO NOTHING"
        )
        sql = (
            f"INSERT INTO {self._fqtn} (id, document, metadata, embedding) "
            f"VALUES (%s, %s, %s::jsonb, %s::vector) {conflict}"
        )
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)

    def update(self, *, ids, documents=None, metadatas=None, embeddings=None) -> None:
        """Atomic, update-only write (matches the Chroma reference, not the base
        get+merge+upsert default): rows whose ``id`` is absent are left untouched
        — ``update`` never creates rows. Use ``upsert`` to create-or-replace.
        """
        if documents is None and metadatas is None and embeddings is None:
            raise ValueError("update requires at least one of documents, metadatas, embeddings")
        # Pre-flight length validation (mirrors BaseCollection.update) so a
        # mismatch fails cleanly instead of an IndexError mid-transaction.
        n = len(ids)
        for label, value in (
            ("documents", documents),
            ("metadatas", metadatas),
            ("embeddings", embeddings),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{label} length {len(value)} does not match ids length {n}")
        if embeddings is not None:
            self._check_dim(embeddings)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                for i, rid in enumerate(ids):
                    sets = []
                    params: list[Any] = []
                    if documents is not None:
                        sets.append("document = %s")
                        params.append(documents[i])
                    if metadatas is not None:
                        sets.append("metadata = metadata || %s::jsonb")
                        params.append(json.dumps(metadatas[i] or {}))
                    if embeddings is not None and embeddings[i] is not None:
                        sets.append("embedding = %s::vector")
                        params.append(_vec_literal(embeddings[i]))
                    if not sets:
                        continue
                    params.append(rid)
                    cur.execute(f"UPDATE {self._fqtn} SET {', '.join(sets)} WHERE id = %s", params)

    def delete(self, *, ids=None, where=None) -> None:
        _validate_where(where)
        clauses = []
        params: list[Any] = []
        if ids is not None:
            clauses.append("id = ANY(%s)")
            params.append(list(ids))
        if where:
            tr = _WhereTranslator()
            frag = tr.translate(where)
            if frag:
                clauses.append(frag)
                params.extend(tr.params)
        # Refuse an unfiltered delete: an empty WHERE would wipe the whole vault.
        # Callers must scope by ids and/or where (use a collection-drop path for
        # intentional full clears).
        if not clauses:
            raise ValueError("delete requires ids and/or a where filter")
        sql = f"DELETE FROM {self._fqtn} WHERE " + " AND ".join(f"({c})" for c in clauses)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)

    # -- reads ------------------------------------------------------------
    def query(
        self,
        *,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        where=None,
        where_document=None,
        include=None,
        restrict_ids=None,
    ) -> QueryResult:
        _validate_where(where)
        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("query requires exactly one of query_texts or query_embeddings")
        chosen = query_texts if query_texts is not None else query_embeddings
        if not chosen:
            raise ValueError("query input must be a non-empty list")

        spec = _IncludeSpec.resolve(include, default_distances=True)

        if query_embeddings is not None:
            self._check_dim(query_embeddings)
            vectors = [list(v) for v in query_embeddings]
        else:
            assert query_texts is not None  # guaranteed by the exactly-one check above
            vectors = self._embed(list(query_texts))

        select_cols = ["id"]
        if spec.documents:
            select_cols.append("document")
        if spec.metadatas:
            select_cols.append("metadata")
        if spec.embeddings:
            select_cols.append("embedding")

        out_ids: list[list[str]] = []
        out_docs: list[list[str]] = []
        out_metas: list[list[dict]] = []
        out_dists: list[list[float]] = []
        out_embs: list[list[list[float]]] = []

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                for vec in vectors:
                    params: list[Any] = [_vec_literal(vec)]
                    where_parts = ["embedding IS NOT NULL"]
                    if where:
                        tr = _WhereTranslator()
                        frag = tr.translate(where)
                        if frag:
                            where_parts.append(frag)
                            params.extend(tr.params)
                    # Optional id pre-filter (entity-scoped search): restrict the
                    # candidate set to specific drawer ids before the vector rank,
                    # so an entity's drawers are considered even when they are
                    # vector-distant from the query text.
                    if restrict_ids is not None:
                        where_parts.append("id = ANY(%s)")
                        params.append(list(restrict_ids))
                    wd = _where_document_clause(where_document, params)
                    if wd:
                        where_parts.append(wd)
                    sql = (
                        f"SELECT {', '.join(select_cols)}, (embedding <=> %s::vector) AS distance "
                        f"FROM {self._fqtn} "
                        f"WHERE {' AND '.join(f'({c})' for c in where_parts)} "
                        f"ORDER BY distance LIMIT %s"
                    )
                    params.append(int(n_results))
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    colnames = [d[0] for d in cur.description]
                    self._collect_query_rows(
                        rows, colnames, spec, out_ids, out_docs, out_metas, out_dists, out_embs
                    )

        if not out_ids:
            return QueryResult.empty(num_queries=len(vectors), embeddings_requested=spec.embeddings)
        return QueryResult(
            ids=out_ids,
            documents=out_docs,
            metadatas=out_metas,
            distances=out_dists,
            embeddings=out_embs if spec.embeddings else None,
        )

    @staticmethod
    def _collect_query_rows(
        rows, colnames, spec, out_ids, out_docs, out_metas, out_dists, out_embs
    ):
        idx = {name: i for i, name in enumerate(colnames)}
        ids, docs, metas, dists, embs = [], [], [], [], []
        for row in rows:
            ids.append(row[idx["id"]])
            if spec.documents:
                docs.append(row[idx["document"]] or "")
            if spec.metadatas:
                metas.append(_loads_meta(row[idx["metadata"]]))
            if spec.distances:
                dists.append(float(row[idx["distance"]]))
            if spec.embeddings:
                embs.append(_parse_vec(row[idx["embedding"]]))
        out_ids.append(ids)
        out_docs.append(docs)
        out_metas.append(metas)
        out_dists.append(dists)
        out_embs.append(embs)

    def keyword_candidates(
        self,
        *,
        query: str,
        n_results: int,
        where=None,
        restrict_ids=None,
    ) -> list[dict]:
        """Trigram-similarity SCORELESS keyword candidates (G002).

        Tokenises ``query`` (≥3-char tokens — the same trigram floor the chroma
        FTS path uses; see ``searcher._bm25_only_via_sqlite``) and selects
        drawers whose ``document`` matches ANY token via ``ILIKE '%token%'``.
        Each ``ILIKE '%...%'`` on a ≥3-char needle is eligible for the G001
        trigram GIN (``{table}_doc_trgm``, ``gin_trgm_ops``) as a Bitmap Index
        Scan. The planner switches to the GIN once the table is large enough for
        it to beat a sequential scan on cost; small/fresh vaults may still Seq
        Scan the multi-token OR (results are identical either way). Rows are
        ordered by ``word_similarity(query, document)`` purely to decide which
        matches survive the ``LIMIT`` — a truncation heuristic, NOT a relevance
        score. The single Python Okapi-BM25 in ``searcher._hybrid_rank`` does the
        real ranking (the merger over-fetches ``n_results*3``).

        The result is SCORELESS by contract (``distance=None``, no
        ``paradedb.score``): the single Python Okapi-BM25 in
        ``searcher._hybrid_rank`` is the only ranker. See
        :meth:`BaseCollection.keyword_candidates` for the full return-shape
        contract.

        When the query yields no ≥3-char token (e.g. "is a"), this returns an
        empty list — there is no usable trigram needle, matching the chroma
        path's documented <3-char floor. ``$contains``/``where_document``
        retrieval (``_where_document_clause``) is unaffected: it still returns
        correct results for short needles via a (non-index-accelerated) ILIKE
        fallback.
        """
        _validate_where(where)
        tokens = [t for t in _tokenize_query(query) if len(t) >= 3]
        if not tokens:
            return []

        params: list[Any] = []
        where_parts = ["embedding IS NOT NULL"]
        if where:
            tr = _WhereTranslator()
            frag = tr.translate(where)
            if frag:
                where_parts.append(frag)
                params.extend(tr.params)
        if restrict_ids is not None:
            where_parts.append("id = ANY(%s)")
            params.append(list(restrict_ids))

        # OR of per-token ILIKE predicates. Each ``document ILIKE '%token%'`` on
        # a ≥3-char token is GIN-accelerated by the trigram index (G001).
        ilike_clauses = []
        for tok in tokens:
            ilike_clauses.append("document ILIKE %s")
            params.append(f"%{tok}%")
        where_parts.append("(" + " OR ".join(ilike_clauses) + ")")

        # word_similarity(query, document) orders by best lexical overlap. The
        # query text is a bind param, NOT a score returned to the pipeline.
        params.append(query)  # word_similarity first arg
        params.append(int(n_results))
        sql = (
            f"SELECT id, document, metadata FROM {self._fqtn} "
            f"WHERE {' AND '.join(f'({c})' for c in where_parts)} "
            "ORDER BY word_similarity(%s, document) DESC "
            "LIMIT %s"
        )

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        candidates: list[dict] = []
        for row in rows:
            doc = row[1] or ""
            meta = _loads_meta(row[2])
            full_source = meta.get("source_file", "") or ""
            candidates.append(
                {
                    "text": doc,
                    "wing": meta.get("wing", "unknown"),
                    "room": meta.get("room", "unknown"),
                    "source_file": _basename(full_source) if full_source else "?",
                    "created_at": meta.get("filed_at", "unknown"),
                    # SCORELESS: no vector distance, no in-DB/BM25 score. The
                    # single Python Okapi-BM25 in _hybrid_rank ranks these.
                    "similarity": None,
                    "distance": None,
                    "matched_via": "keyword_postgres",
                    # Chunk-precise dedup key for _merge_bm25_union_candidates
                    # (searcher.py:684-697); a None key is silently dropped.
                    "_source_file_full": full_source,
                    "_chunk_index": meta.get("chunk_index"),
                }
            )
        return candidates

    def supports_keyword_candidates(self) -> bool:
        """Postgres collections implement the trigram-GIN keyword path (G002)."""
        return True

    def get(
        self,
        *,
        ids=None,
        where=None,
        where_document=None,
        limit=None,
        offset=None,
        include=None,
    ) -> GetResult:
        _validate_where(where)
        spec = _IncludeSpec.resolve(include, default_distances=False)

        select_cols = ["id"]
        if spec.documents:
            select_cols.append("document")
        if spec.metadatas:
            select_cols.append("metadata")
        if spec.embeddings:
            select_cols.append("embedding")

        params: list[Any] = []
        where_parts = []
        if ids is not None:
            where_parts.append("id = ANY(%s)")
            params.append(list(ids))
        if where:
            tr = _WhereTranslator()
            frag = tr.translate(where)
            if frag:
                where_parts.append(frag)
                params.extend(tr.params)
        wd = _where_document_clause(where_document, params)
        if wd:
            where_parts.append(wd)

        sql = f"SELECT {', '.join(select_cols)} FROM {self._fqtn}"
        if where_parts:
            sql += " WHERE " + " AND ".join(f"({c})" for c in where_parts)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        if offset is not None:
            sql += " OFFSET %s"
            params.append(int(offset))

        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                colnames = [d[0] for d in cur.description]

        idx = {name: i for i, name in enumerate(colnames)}
        out_ids, out_docs, out_metas, out_embs = [], [], [], []
        for row in rows:
            out_ids.append(row[idx["id"]])
            if spec.documents:
                out_docs.append(row[idx["document"]] or "")
            if spec.metadatas:
                out_metas.append(_loads_meta(row[idx["metadata"]]))
            if spec.embeddings:
                out_embs.append(_parse_vec(row[idx["embedding"]]))
        return GetResult(
            ids=out_ids,
            documents=out_docs if spec.documents else [],
            metadatas=out_metas if spec.metadatas else [],
            embeddings=out_embs if spec.embeddings else None,
        )

    def count(self) -> int:
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {self._fqtn}")
                return int(cur.fetchone()[0])

    def health(self) -> HealthStatus:
        return self._backend.health()


def _loads_meta(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _parse_vec(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    s = str(value).strip().lstrip("[").rstrip("]")
    if not s:
        return []
    return [float(x) for x in s.split(",")]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class PostgresBackend(BaseBackend):
    """Central Postgres backend with schema-per-team vaults."""

    name = "postgres"
    spec_version = "1.0"
    capabilities = frozenset(
        {
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            # ``$contains`` rides the per-drawer ``document`` trigram GIN
            # (``gin_trgm_ops``) provisioned in ``_ensure_collection`` — a Bitmap
            # Index Scan, not a sequential scan — for needles of ≥3 characters
            # (the trigram floor, matching the Chroma FTS path). Below 3 chars the
            # query is still correct via a fallback scan; the flag describes the
            # indexed ≥3-char case.
            "supports_contains_fast",
            "supports_keyword_candidates",
            "server_mode",
            "multi_tenant",
        }
    )

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        vector_dim: Optional[int] = None,
        embedder: Optional[Callable] = None,
    ):
        self._dsn = dsn
        self._pool_obj = None
        self._closed = False
        self._lock = threading.Lock()
        # Separate lock for one-time bootstrap so it never re-enters ``_lock``
        # via ``_pool()`` (threading.Lock is non-reentrant).
        self._bootstrap_lock = threading.Lock()
        self._bootstrapped = False
        # (schema, table) pairs whose DDL has been ensured. Read/written without
        # a lock on purpose: the underlying DDL is ``CREATE ... IF NOT EXISTS``,
        # so the worst case under a race is a redundant (harmless) ensure call.
        self._ensured: set[tuple[str, str]] = set()
        self._vector_dim = vector_dim or int(
            os.environ.get("MEMPALACE_PG_VECTOR_DIM", _DEFAULT_VECTOR_DIM)
        )
        # Test/optional injection of an embedding function.
        self._embedder = embedder
        # One-time DDL guard for the central WAL audit table (G004).
        self._wal_ready = False

    # -- pool / connection ------------------------------------------------
    def _pool(self):
        if self._closed:
            raise BackendClosedError("PostgresBackend has been closed")
        if self._pool_obj is not None:
            return self._pool_obj
        with self._lock:
            if self._pool_obj is not None:
                return self._pool_obj
            ConnectionPool = _require_psycopg()
            dsn = self._dsn or dsn_from_env()
            max_size = int(os.environ.get("MEMPALACE_PG_POOL_MAX", "5"))
            self._pool_obj = ConnectionPool(
                conninfo=dsn, min_size=1, max_size=max_size, timeout=30.0, open=True
            )
            return self._pool_obj

    def _conn(self):
        """Context manager yielding a pooled connection (commit on clean exit)."""
        return self._pool().connection()

    def _ensure_bootstrap(self) -> None:
        """Create extensions + AGE graph once per process (idempotent)."""
        if self._bootstrapped:
            return
        with self._bootstrap_lock:
            if self._bootstrapped:
                return
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    # pg_trgm backs the trigram GIN on ``document`` and is a
                    # REQUIRED path (the keyword-candidate retrieval substrate),
                    # unlike the optional pg_search/age below. It is core to
                    # Postgres and always available, so a failure here must
                    # surface (propagate) rather than be swallowed and warned.
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                    for ext in ("pg_search", "age"):
                        try:
                            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                        except Exception as e:  # pragma: no cover - optional extensions
                            conn.rollback()
                            logger.warning("optional extension %s unavailable: %s", ext, e)
                    # Provision the AGE graph for traversal / future KG use.
                    try:
                        cur.execute("LOAD 'age'")
                        cur.execute("SET search_path = ag_catalog, public")
                        cur.execute(
                            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (_AGE_GRAPH,)
                        )
                        if cur.fetchone() is None:
                            cur.execute("SELECT create_graph(%s)", (_AGE_GRAPH,))
                        cur.execute("RESET search_path")
                    except Exception as e:  # pragma: no cover - AGE optional at this stage
                        conn.rollback()
                        logger.warning("AGE graph provisioning skipped: %s", e)
            self._bootstrapped = True

    # -- DDL --------------------------------------------------------------
    def _ensure_collection(self, schema: str, table: str, *, create: bool) -> None:
        key = (schema, table)
        if key in self._ensured:
            return
        preexisting_table = False
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,)
                )
                schema_exists = cur.fetchone() is not None
                cur.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
                table_exists = cur.fetchone()[0] is not None
                preexisting_table = table_exists

                if not create:
                    if not schema_exists:
                        raise PalaceNotFoundError(f"team vault {schema!r} does not exist")
                    if not table_exists:
                        raise CollectionNotInitializedError(
                            f"collection {table!r} not initialized in {schema!r}"
                        )
                    self._ensured.add(key)
                    return

                if not schema_exists:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_qi(schema)}")
                if not table_exists:
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {_qi(schema)}.{_qi(table)} ("
                        "  id text PRIMARY KEY,"
                        "  document text NOT NULL DEFAULT '',"
                        "  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,"
                        f"  embedding vector({self._vector_dim})"
                        ")"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi(table + '_meta_gin')} "
                        f"ON {_qi(schema)}.{_qi(table)} USING gin (metadata jsonb_path_ops)"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi(table + '_hnsw')} "
                        f"ON {_qi(schema)}.{_qi(table)} USING hnsw (embedding vector_cosine_ops)"
                    )
                    # Trigram GIN on ``document`` for index-backed keyword
                    # ($contains) retrieval. Created in the SAME transaction as
                    # the table itself, so a fresh collection is atomic: a
                    # mid-statement failure rolls back the whole table-create
                    # (no half-created table missing its trigram index).
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {_qi(table + '_doc_trgm')} "
                        f"ON {_qi(schema)}.{_qi(table)} USING gin (document gin_trgm_ops)"
                    )
        # Existing-vault migration: a table created before the trigram GIN was
        # introduced has no ``{table}_doc_trgm`` index. Add it WITHOUT holding an
        # ACCESS EXCLUSIVE lock on the (possibly large, live) table — i.e. with
        # CREATE INDEX CONCURRENTLY, which cannot run inside the transaction
        # block above and is therefore run on a dedicated autocommit connection
        # OUTSIDE the ``_conn()`` context. Idempotent and a no-op once the index
        # exists; fresh tables already got the GIN atomically in the txn above.
        if preexisting_table:
            self._migrate_doc_trgm_index(schema, table)
        self._ensured.add(key)

    def _migrate_doc_trgm_index(self, schema: str, table: str) -> None:
        """Add the ``{table}_doc_trgm`` trigram GIN to a pre-existing table.

        Uses ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` so building the index on
        a populated, live table does not take an ACCESS EXCLUSIVE lock that would
        block concurrent ``add_drawer``/search traffic. CONCURRENTLY cannot run
        inside a transaction block, so this acquires a pooled connection and
        forces ``autocommit`` (the ``_conn()`` path is a transaction and must NOT
        be used here).

        CONCURRENTLY can leave an INVALID index behind on failure; this detects
        an invalid leftover, drops it, and retries once before surfacing a clear
        error. Calling this when the index already exists (and is valid) is a
        cheap no-op (``IF NOT EXISTS``).
        """
        index_name = table + "_doc_trgm"
        # Validate identifiers up front (defence-in-depth; ``_qi`` would reject
        # an unsafe identifier anyway, but fail fast before any DDL).
        _qi(schema)
        _qi(table)
        _qi(index_name)
        create_sql = (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_qi(index_name)} "
            f"ON {_qi(schema)}.{_qi(table)} USING gin (document gin_trgm_ops)"
        )
        drop_sql = f"DROP INDEX CONCURRENTLY IF EXISTS {_qi(schema)}.{_qi(index_name)}"

        def _index_is_invalid(conn) -> bool:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT i.indisvalid FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "JOIN pg_index i ON i.indexrelid = c.oid "
                    "WHERE n.nspname = %s AND c.relname = %s",
                    (schema, index_name),
                )
                row = cur.fetchone()
            return row is not None and row[0] is False

        with self._pool().connection() as conn:
            # CONCURRENTLY requires autocommit (no surrounding transaction block).
            # Restore autocommit=False unconditionally on exit so the physical
            # connection returned to the pool has normal transactional semantics
            # for the next consumer. psycopg_pool's reset normalises transaction
            # STATUS but does NOT reset the autocommit attribute, so without the
            # finally the next caller inherits autocommit=True and loses atomicity.
            conn.autocommit = True
            try:
                try:
                    with conn.cursor() as cur:
                        cur.execute(create_sql)
                except Exception as e:
                    # A failed CONCURRENTLY build can leave an INVALID index; drop it
                    # and retry once. If the retry also fails, surface a clear error.
                    logger.warning(
                        "concurrent trigram GIN build failed for %s.%s (%s); "
                        "dropping any invalid leftover and retrying once",
                        schema,
                        index_name,
                        e,
                    )
                    with conn.cursor() as cur:
                        cur.execute(drop_sql)
                    try:
                        with conn.cursor() as cur:
                            cur.execute(create_sql)
                    except Exception as retry_err:
                        raise RuntimeError(
                            f"failed to create trigram GIN {schema}.{index_name} "
                            f"concurrently after retry: {retry_err}"
                        ) from retry_err
                # Guard: even a "successful" CONCURRENTLY can leave an invalid index
                # if interrupted; if so, drop it and rebuild once so we never leave a
                # silently-unused invalid index behind.
                if _index_is_invalid(conn):
                    with conn.cursor() as cur:
                        cur.execute(drop_sql)
                        cur.execute(create_sql)
                    if _index_is_invalid(conn):
                        raise RuntimeError(
                            f"trigram GIN {schema}.{index_name} remains INVALID after rebuild"
                        )
            finally:
                # Restore transactional semantics before the connection goes back
                # to the pool. The connection is IDLE in autocommit here (all DDL
                # committed individually via autocommit), so this is safe.
                conn.autocommit = False

    # -- contract ---------------------------------------------------------
    def get_collection(self, *args, **kwargs) -> PostgresCollection:
        """Obtain a team-vault collection.

        Supports the RFC 001 ``palace=PalaceRef`` form and the legacy positional
        ``(palace_path, collection_name, create)`` form. The team is taken from
        ``PalaceRef.namespace`` and falls back to ``MEMPALACE_TEAM`` / ``default``.
        """
        palace_ref, collection_name, create, options = _normalize_get_collection_args(args, kwargs)
        if options and isinstance(options, dict) and options.get("vector_dim"):
            self._vector_dim = int(options["vector_dim"])
        self._ensure_bootstrap()
        schema = team_schema(palace_ref.namespace)
        table = collection_name
        _qi(table)  # validate
        self._ensure_collection(schema, table, create=bool(create))
        return PostgresCollection(self, schema, table, self._vector_dim)

    def list_vaults(self) -> list[str]:
        """Return the team names (namespaces) that have a vault schema."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name LIKE 'team\\_%' ORDER BY schema_name"
                )
                return [r[0][len("team_") :] for r in cur.fetchall()]

    def close_palace(self, palace) -> None:
        ns = palace.namespace if isinstance(palace, PalaceRef) else None
        schema = team_schema(ns)
        self._ensured = {k for k in self._ensured if k[0] != schema}

    def close(self) -> None:
        with self._lock:
            if self._pool_obj is not None:
                try:
                    self._pool_obj.close()
                except Exception:
                    logger.exception("error closing postgres pool")
                self._pool_obj = None
            self._closed = True

    def reconnect(self) -> None:
        """Drop the current pool so the next operation opens a fresh one.

        Unlike :meth:`close`, the backend stays usable — this is the analog of
        the chroma client-cache reset used by ``mempalace_reconnect`` after an
        external change. The next ``_pool()`` call lazily re-opens against the
        current DSN; ``_ensured`` is cleared so table DDL is re-checked (the
        DDL is ``IF NOT EXISTS``, so re-checking is harmless).
        """
        with self._lock:
            if self._pool_obj is not None:
                try:
                    self._pool_obj.close()
                except Exception:
                    logger.exception("error closing postgres pool during reconnect")
                self._pool_obj = None
            self._ensured = set()
            self._wal_ready = False

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        if self._closed:
            return HealthStatus.unhealthy("backend closed")
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return HealthStatus.healthy()
        except Exception as e:
            return HealthStatus.unhealthy(str(e))

    # -- write-ahead audit log (central sink) -----------------------------
    def _ensure_wal_table(self) -> None:
        if self._wal_ready:
            return
        schema = _qi(_WAL_AUDIT_SCHEMA)
        table = f"{schema}.{_qi(_WAL_AUDIT_TABLE)}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ("
                    "  id bigserial PRIMARY KEY,"
                    "  ts timestamptz NOT NULL DEFAULT now(),"
                    "  operation text NOT NULL,"
                    "  team text,"
                    "  params jsonb,"
                    "  result jsonb"
                    ")"
                )
        self._wal_ready = True

    def wal_append(self, *, timestamp, operation, team, params, result) -> None:
        """Persist one (already-redacted) audit entry to the central table.

        The caller redacts sensitive values before calling this; here we only
        store them. ``team`` is the vault the write targeted (``None`` outside
        team-vault mode). ``params`` / ``result`` land in ``jsonb`` columns.
        """
        self._ensure_wal_table()
        table = f"{_qi(_WAL_AUDIT_SCHEMA)}.{_qi(_WAL_AUDIT_TABLE)}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {table} (ts, operation, team, params, result) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)",
                    (
                        timestamp,
                        operation,
                        team,
                        json.dumps(params, default=str),
                        json.dumps(result, default=str),
                    ),
                )

    @classmethod
    def detect(cls, path: str) -> bool:
        # Postgres is never auto-detected from a filesystem path; it is selected
        # explicitly via config/env (MEMPALACE_BACKEND=postgres).
        return False


# ---------------------------------------------------------------------------
# Arg normalisation (mirrors chroma's legacy/new dual signature)
# ---------------------------------------------------------------------------


def _normalize_get_collection_args(args, kwargs):
    """Return ``(PalaceRef, collection_name, create, options)`` for either the
    new ``palace=PalaceRef`` form or the legacy positional/keyword path form."""
    if "palace" in kwargs:
        palace_ref = kwargs.pop("palace")
        if not isinstance(palace_ref, PalaceRef):
            raise TypeError("palace= must be a PalaceRef instance")
        collection_name = kwargs.pop("collection_name")
        create = kwargs.pop("create", False)
        options = kwargs.pop("options", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        if args:
            raise TypeError("positional args not allowed with palace= kwarg")
        return palace_ref, collection_name, create, options

    if args:
        palace_path = args[0]
        rest = list(args[1:])
        collection_name = kwargs.pop("collection_name", None) or (rest.pop(0) if rest else None)
        if collection_name is None:
            raise TypeError("collection_name is required")
        create = kwargs.pop("create", False)
        if rest:
            create = rest.pop(0)
        namespace = kwargs.pop("namespace", None)
        if rest:
            raise TypeError(f"unexpected positional args: {rest!r}")
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        return (
            PalaceRef(id=palace_path, local_path=palace_path, namespace=namespace),
            collection_name,
            bool(create),
            None,
        )

    if "palace_path" in kwargs:
        palace_path = kwargs.pop("palace_path")
        collection_name = kwargs.pop("collection_name")
        create = kwargs.pop("create", False)
        namespace = kwargs.pop("namespace", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        return (
            PalaceRef(id=palace_path, local_path=palace_path, namespace=namespace),
            collection_name,
            bool(create),
            None,
        )

    raise TypeError("get_collection requires palace=PalaceRef or a palace_path")
