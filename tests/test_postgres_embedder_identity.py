"""Embedder-identity persistence + enforcement on the Postgres peer backend (A4).

This is the dedicated model-swap test the v3.5.0 integration plan requires
(Acceptance Gate A4 / Expanded Test Plan). It proves the fork's Postgres path
PERSISTS and ENFORCES embedder identity per RFC 001 through a real
``team_<slug>.embedder_identity`` table — NOT the no-op
``BaseCollection.set_embedder_identity`` default (base.py).

The whole point of the gate: each test below FAILS if the no-op default were
inherited, because nothing would ever be recorded and reloaded — so a model
swap would silently pass (corrupting recall) while the suite still showed green.

These run against a live Postgres with pgvector (the bundled docker-compose db,
or any instance reachable via ``MEMPALACE_TEST_PG_URL`` / ``MEMPALACE_DATABASE_URL``)
and are skipped when psycopg is missing or the database is unreachable. No
embedding model is loaded: writes pass explicit vectors and the enforcement
*check* needs only the configured model name (cheap).
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.base import (  # noqa: E402
    DimensionMismatchError,
    EmbedderIdentity,
    EmbedderIdentityMismatchError,
    PalaceRef,
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


def _vec(seed: float, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[0] = seed
    return v


@pytest.fixture(scope="module")
def backend():
    be = PostgresBackend(dsn=_dsn(), vector_dim=DIM)
    yield be
    be.close()


@pytest.fixture()
def team(backend):
    """A unique, isolated team vault per test; schema dropped on teardown."""
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
# Persistence roundtrip — proves the slot is REAL, not the no-op default.
# --------------------------------------------------------------------------


def test_identity_starts_unknown(backend, team):
    col = _col(backend, team, create=True)
    assert col.get_stored_embedder_identity() is None


def test_identity_roundtrip_persists_in_team_schema(backend, team):
    """set -> get returns the recorded identity. FAILS under the no-op default
    (the no-op records nothing, so get would still return None)."""
    col = _col(backend, team, create=True)
    col.set_embedder_identity(EmbedderIdentity("minilm", DIM))
    got = col.get_stored_embedder_identity()
    assert got is not None, "identity was not persisted — no-op default inherited?"
    assert got.model_name == "minilm"
    assert got.dimension == DIM


def test_identity_row_lands_in_team_embedder_identity_table(backend, team):
    """The identity physically lands in ``team_<slug>.embedder_identity`` keyed
    on the collection table name — the schema-per-team persistence site."""
    col = _col(backend, team, create=True)
    col.set_embedder_identity(EmbedderIdentity("minilm", DIM))
    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT model_name, dimension FROM "{schema}".embedder_identity '
                "WHERE collection = %s",
                (COLLECTION,),
            )
            row = cur.fetchone()
    assert row is not None and row[0] == "minilm" and int(row[1]) == DIM


# --------------------------------------------------------------------------
# Enforcement via palace.get_collection (the real choke point, RFC 001 / A4).
# --------------------------------------------------------------------------


@pytest.fixture
def clear_identity_cache():
    from mempalace import palace

    palace._VALIDATED_IDENTITY.clear()
    yield
    palace._VALIDATED_IDENTITY.clear()


@pytest.fixture
def server_env(monkeypatch):
    """Route ``palace.get_collection`` to the postgres peer for this test DB."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "minilm")
    yield


def _seed_with_identity(backend, team, model, *, dim=DIM):
    col = _col(backend, team, create=True)
    col.add(documents=["x"], ids=["a"], metadatas=[{}], embeddings=[_vec(0.5, dim)])
    if model is not None:
        col.set_embedder_identity(EmbedderIdentity(model, dim))
    return col


def test_enforcement_first_write_records_identity(backend, team, server_env, clear_identity_cache):
    """Opening a brand-new vault with create=True records the current model.

    FAILS under the no-op default: nothing is recorded, so the reload below
    returns None and the assert trips — exactly the regression A4 guards.
    """
    from mempalace import palace as P

    P._VALIDATED_IDENTITY.clear()
    col = P.get_collection(team, collection_name=COLLECTION, create=True, team=team)
    got = col.get_stored_embedder_identity()
    assert got is not None and got.model_name == "minilm"


def test_enforcement_model_swap_raises(backend, team, server_env, clear_identity_cache):
    """A same-dimension model swap on an unforced reopen raises."""
    from mempalace import palace as P

    _seed_with_identity(backend, team, "minilm")
    P._VALIDATED_IDENTITY.clear()
    os.environ["MEMPALACE_EMBEDDING_MODEL"] = "embeddinggemma"
    try:
        with pytest.raises(EmbedderIdentityMismatchError):
            P.get_collection(team, collection_name=COLLECTION, create=False, team=team)
    finally:
        os.environ["MEMPALACE_EMBEDDING_MODEL"] = "minilm"


def test_enforcement_dimension_swap_raises(
    backend, team, server_env, clear_identity_cache, monkeypatch
):
    """A dimension change is physically unusable — raises DimensionMismatchError.

    The width check is gated on BOTH identities carrying a real (non-zero)
    dimension, which only happens once the embedder is loaded. We supply a
    real-dimension *effective* identity (384) for the open so the enforcement
    path compares it against the stored 768 and raises the dimension error
    (checked before the name swap) without a model load.
    """
    from mempalace import palace as P
    from mempalace.backends.postgres import PostgresCollection

    col = _col(backend, team, create=True)
    col.add(documents=["x"], ids=["a"], metadatas=[{}], embeddings=[_vec(0.5)])
    # Record a 768-d identity (a real, different width than the 384-d backend).
    col.set_embedder_identity(EmbedderIdentity("minilm", 768))
    P._VALIDATED_IDENTITY.clear()

    # Make the open report a real-dim effective identity at 384 so the width
    # mismatch (stored 768 vs current 384) is detectable without a model load.
    def _effective(self):
        return EmbedderIdentity("minilm", DIM)

    monkeypatch.setattr(PostgresCollection, "effective_embedder_identity", _effective)
    with pytest.raises(DimensionMismatchError):
        P.get_collection(team, collection_name=COLLECTION, create=False, team=team)


def test_collection_write_rejects_wrong_dimension(backend, team):
    """The Postgres write path's own dimension guard rejects an off-width vector.

    This is the physical dim protection on the peer backend (``_check_dim``):
    a vector whose width != the collection dim raises ``DimensionMismatchError``
    on write, independent of the identity bookkeeping.
    """
    col = _col(backend, team, create=True)
    with pytest.raises(DimensionMismatchError):
        col.add(documents=["x"], ids=["a"], metadatas=[{}], embeddings=[_vec(0.5, dim=128)])


def test_force_re_records_and_proceeds(backend, team, server_env, clear_identity_cache):
    """``set_palace_embedder_identity(force=True)`` overwrites the recorded
    identity; the next open against the new model then passes.
    """
    from mempalace import palace as P

    _seed_with_identity(backend, team, "minilm")
    P._VALIDATED_IDENTITY.clear()

    # Without force, recording a different model is refused.
    with pytest.raises(EmbedderIdentityMismatchError):
        P.set_palace_embedder_identity(team, model="embeddinggemma", force=False, team=team)

    # With force, the identity is re-recorded (name only, no foreign load).
    old, new = P.set_palace_embedder_identity(team, model="embeddinggemma", force=True, team=team)
    assert old.model_name == "minilm" and new.model_name == "embeddinggemma"

    # The new identity is what is now persisted, so a reopen under the new model
    # no longer raises.
    P._VALIDATED_IDENTITY.clear()
    os.environ["MEMPALACE_EMBEDDING_MODEL"] = "embeddinggemma"
    try:
        P.get_collection(team, collection_name=COLLECTION, create=False, team=team)
    finally:
        os.environ["MEMPALACE_EMBEDDING_MODEL"] = "minilm"
