"""Conformance tests for the PostgreSQL storage backend.

These run against a live Postgres with pgvector (the bundled docker-compose db,
or any instance reachable via ``MEMPALACE_TEST_PG_URL`` /
``MEMPALACE_DATABASE_URL``). They are skipped automatically when psycopg is not
installed or the database is unreachable, so they never break a chroma-only CI.

Start the database with::

    cd deploy && docker compose up -d --build

The suite mirrors the contract exercised against ChromaBackend: typed
QueryResult/GetResult, the full where-clause operator set, namespace (team)
isolation, and the RFC 001 error hierarchy.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends import (  # noqa: E402
    CollectionNotInitializedError,
    DimensionMismatchError,
    GetResult,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    available_backends,
)
from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402

DIM = 384
COLLECTION = "mempalace_drawers"


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


pytestmark = pytest.mark.skipif(
    not _reachable(_dsn()),
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)


def _fake_embed(texts):
    """Deterministic embedder: sha256 bytes seeded into the first 32 dims.

    Identical text -> identical vector (cosine distance 0), so exact-text
    queries rank their own document first. Enough to validate the vector path
    without downloading the ONNX model.
    """
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
    name = "t" + uuid.uuid4().hex[:10]
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


# --------------------------------------------------------------------------
# Registration / lifecycle
# --------------------------------------------------------------------------


def test_postgres_registered():
    assert "postgres" in available_backends()


def test_health_ok(backend):
    assert backend.health().ok


def test_create_false_missing_vault_raises(backend, team):
    with pytest.raises(PalaceNotFoundError):
        _col(backend, team, create=False)


def test_create_true_then_collection_not_initialized(backend, team):
    # Create the schema via one collection, then ask for a *different*,
    # never-created collection with create=False -> CollectionNotInitialized.
    _col(backend, team, create=True)
    with pytest.raises(CollectionNotInitializedError):
        backend.get_collection(
            palace=PalaceRef(id=team, namespace=team),
            collection_name="mempalace_closets",
            create=False,
        )


def test_count_empty(backend, team):
    col = _col(backend, team)
    assert col.count() == 0


# --------------------------------------------------------------------------
# Writes + typed reads
# --------------------------------------------------------------------------


def test_add_and_get_typed(backend, team):
    col = _col(backend, team)
    col.add(
        documents=["alpha doc", "beta doc"],
        ids=["a", "b"],
        metadatas=[{"wing": "code", "n": 1}, {"wing": "people", "n": 2}],
    )
    assert col.count() == 2
    res = col.get(ids=["a"])
    assert isinstance(res, GetResult)
    assert res.ids == ["a"]
    assert res.documents == ["alpha doc"]
    assert res.metadatas[0]["wing"] == "code"


def test_add_on_conflict_do_nothing(backend, team):
    col = _col(backend, team)
    col.add(documents=["v1"], ids=["x"], metadatas=[{"k": "1"}])
    col.add(documents=["v2"], ids=["x"], metadatas=[{"k": "2"}])
    res = col.get(ids=["x"])
    assert res.documents == ["v1"]  # add does not overwrite


def test_upsert_overwrites(backend, team):
    col = _col(backend, team)
    col.upsert(documents=["v1"], ids=["x"], metadatas=[{"k": "1"}])
    col.upsert(documents=["v2"], ids=["x"], metadatas=[{"k": "2"}])
    res = col.get(ids=["x"])
    assert res.documents == ["v2"]
    assert res.metadatas[0]["k"] == "2"


def test_update_merges_metadata(backend, team):
    col = _col(backend, team)
    col.upsert(documents=["doc"], ids=["x"], metadatas=[{"a": "1", "b": "2"}])
    col.update(ids=["x"], metadatas=[{"b": "9", "c": "3"}])
    meta = col.get(ids=["x"]).metadatas[0]
    assert meta == {"a": "1", "b": "9", "c": "3"}


def test_query_typed_and_ranked(backend, team):
    col = _col(backend, team)
    col.add(documents=["red apple", "blue ocean", "green field"], ids=["1", "2", "3"])
    res = col.query(query_texts=["blue ocean"], n_results=3)
    assert isinstance(res, QueryResult)
    assert len(res.ids) == 1  # one query
    assert res.ids[0][0] == "2"  # exact-text match ranks first
    assert res.distances[0][0] == pytest.approx(0.0, abs=1e-6)


def test_query_with_explicit_embeddings(backend, team):
    col = _col(backend, team)
    e1 = _fake_embed(["one"])[0]
    e2 = _fake_embed(["two"])[0]
    col.add(documents=["one", "two"], ids=["1", "2"], embeddings=[e1, e2])
    res = col.query(query_embeddings=[e2], n_results=1)
    assert res.ids[0][0] == "2"


def test_query_empty_collection_preserves_shape(backend, team):
    col = _col(backend, team)
    res = col.query(query_texts=["anything"], n_results=5)
    assert res.ids == [[]]
    assert res.distances == [[]]


# --------------------------------------------------------------------------
# Where-clause operators
# --------------------------------------------------------------------------


@pytest.fixture()
def populated(backend, team):
    col = _col(backend, team)
    col.add(
        documents=["doc one contains apple", "doc two", "doc three", "doc four"],
        ids=["1", "2", "3", "4"],
        metadatas=[
            {"wing": "code", "score": 10, "tag": "x"},
            {"wing": "code", "score": 20, "tag": "y"},
            {"wing": "people", "score": 30, "tag": "x"},
            {"wing": "people", "score": 40, "tag": "z"},
        ],
    )
    return col


def _ids(res):
    return sorted(res.ids)


def test_where_eq(populated):
    assert _ids(populated.get(where={"wing": "code"})) == ["1", "2"]


def test_where_explicit_eq(populated):
    assert _ids(populated.get(where={"wing": {"$eq": "people"}})) == ["3", "4"]


def test_where_ne(populated):
    assert _ids(populated.get(where={"wing": {"$ne": "code"}})) == ["3", "4"]


def test_where_in(populated):
    assert _ids(populated.get(where={"tag": {"$in": ["x", "z"]}})) == ["1", "3", "4"]


def test_where_nin(populated):
    assert _ids(populated.get(where={"tag": {"$nin": ["x"]}})) == ["2", "4"]


def test_where_gt_gte_lt_lte(populated):
    assert _ids(populated.get(where={"score": {"$gt": 20}})) == ["3", "4"]
    assert _ids(populated.get(where={"score": {"$gte": 20}})) == ["2", "3", "4"]
    assert _ids(populated.get(where={"score": {"$lt": 20}})) == ["1"]
    assert _ids(populated.get(where={"score": {"$lte": 20}})) == ["1", "2"]


def test_where_and(populated):
    res = populated.get(where={"$and": [{"wing": "code"}, {"tag": "y"}]})
    assert _ids(res) == ["2"]


def test_where_or(populated):
    res = populated.get(where={"$or": [{"score": {"$lt": 15}}, {"score": {"$gt": 35}}]})
    assert _ids(res) == ["1", "4"]


def test_where_document_contains(populated):
    res = populated.get(where_document={"$contains": "apple"})
    assert _ids(res) == ["1"]


def test_unknown_operator_raises(populated):
    with pytest.raises(UnsupportedFilterError):
        populated.get(where={"score": {"$regex": ".*"}})


def test_query_with_where_filter(populated):
    res = populated.query(query_texts=["doc three"], n_results=10, where={"wing": "people"})
    assert set(res.ids[0]) == {"3", "4"}
    assert res.ids[0][0] == "3"  # exact match first


# --------------------------------------------------------------------------
# Delete
# --------------------------------------------------------------------------


def test_delete_by_ids(populated):
    populated.delete(ids=["1", "2"])
    assert _ids(populated.get()) == ["3", "4"]


def test_delete_by_where(populated):
    populated.delete(where={"wing": "people"})
    assert _ids(populated.get()) == ["1", "2"]


def test_delete_without_filter_refuses_mass_wipe(populated):
    # Regression: an unfiltered delete must NOT wipe the whole vault.
    with pytest.raises(ValueError):
        populated.delete()
    assert populated.count() == 4  # untouched


def test_update_length_mismatch_raises(backend, team):
    col = _col(backend, team)
    col.upsert(documents=["a", "b"], ids=["1", "2"])
    with pytest.raises(ValueError):
        col.update(ids=["1", "2"], documents=["only one"])


def test_update_missing_id_is_noop(backend, team):
    # update is update-only (matches the Chroma reference): a missing id is a
    # silent no-op, never an insert and never an error.
    col = _col(backend, team)
    col.update(ids=["ghost"], metadatas=[{"k": "v"}])
    assert col.count() == 0


# --------------------------------------------------------------------------
# Team isolation + vault listing
# --------------------------------------------------------------------------


def test_namespace_isolation(backend):
    a = "t" + uuid.uuid4().hex[:10]
    b = "t" + uuid.uuid4().hex[:10]
    try:
        ca = _col(backend, a)
        cb = _col(backend, b)
        ca.add(documents=["secret-a"], ids=["1"], metadatas=[{"team": "a"}])
        cb.add(documents=["secret-b"], ids=["1"], metadatas=[{"team": "b"}])
        assert ca.get(ids=["1"]).documents == ["secret-a"]
        assert cb.get(ids=["1"]).documents == ["secret-b"]
        assert ca.count() == 1 and cb.count() == 1
        vaults = backend.list_vaults()
        assert a in vaults and b in vaults
    finally:
        for t in (a, b):
            with psycopg.connect(_dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
                conn.commit()


# --------------------------------------------------------------------------
# Dimension validation + legacy call form
# --------------------------------------------------------------------------


def test_dimension_mismatch_raises(backend, team):
    col = _col(backend, team)
    with pytest.raises(DimensionMismatchError):
        col.add(documents=["x"], ids=["1"], embeddings=[[0.1, 0.2, 0.3]])


def test_legacy_positional_call_form(backend, team):
    # palace.py / mcp_server call get_collection(palace_path, collection_name, create)
    # positionally with namespace passed via kwarg.
    col = backend.get_collection(team, COLLECTION, True, namespace=team)
    col.add(documents=["legacy"], ids=["1"])
    assert col.count() == 1


def test_bootstrap_first_call_does_not_deadlock():
    """Regression: get_collection as the very first backend op must not deadlock.

    ``_ensure_bootstrap`` once held the pool lock while creating the pool, so a
    fresh backend whose first operation triggered bootstrap re-entered a
    non-reentrant lock and hung. Run it in a watchdog thread and fail fast.
    """
    import threading

    be = PostgresBackend(dsn=_dsn(), vector_dim=DIM, embedder=_fake_embed)
    name = "t" + uuid.uuid4().hex[:10]
    done = threading.Event()
    box = {}

    def run():
        try:
            col = be.get_collection(
                palace=PalaceRef(id=name, namespace=name),
                collection_name=COLLECTION,
                create=True,
            )
            box["count"] = col.count()
        except Exception as e:  # pragma: no cover - surfaced via assert below
            box["err"] = e
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    finished = done.wait(timeout=30)
    try:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')
            conn.commit()
    finally:
        be.close()
    assert finished, "get_collection as first op deadlocked (bootstrap re-entered pool lock)"
    assert "err" not in box, f"unexpected error: {box.get('err')}"
    assert box["count"] == 0
