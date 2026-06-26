"""Production-safe embedder-identity backfill on the Postgres peer backend (A4).

A fork upgrade adds the per-team ``team_<slug>.embedder_identity`` slot (RFC 001).
Vaults that already hold drawers but predate the slot must be stamped with the
CURRENT embedder identity by a one-time backfill — additively, and WITHOUT
materialising a ``mempalace_drawers`` table on any schema that never had one.

The naive approach this guards against: "open every team schema with
``create=True`` and record identity" — that runs through ``_ensure_collection``,
which creates ``mempalace_drawers`` when ``create=True``, leaving EMPTY drawers
tables on empty/non-vault team schemas. The proper backfill enumerates EXISTING
vaults only (schemas whose drawers table already exists) and opens them
``create=False``.

These assert the backfill:
  (i)   records identity into EXISTING populated vaults' embedder_identity tables,
  (ii)  creates NO new ``mempalace_drawers`` tables anywhere (empty/non-vault
        team schemas are left untouched — no drawers table appears),
  (iii) is idempotent on a second run,
  (iv)  leaves the A4 model-swap contract intact (a reopen under a different
        model still raises EmbedderIdentityMismatchError after the backfill).

Run against a live Postgres with pgvector (the bundled docker-compose db, or any
instance reachable via ``MEMPALACE_TEST_PG_URL`` / ``MEMPALACE_DATABASE_URL``);
skipped when psycopg is missing or the database is unreachable. No embedding
model is loaded: writes pass explicit vectors and the backfill records an
explicit, supplied identity.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.base import (  # noqa: E402
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
def names():
    """Unique team/schema names per test; every team_<slug> schema dropped on teardown.

    Returns a small namespace object with helpers to register the names this
    test will use so teardown is exhaustive even for schemas created directly
    via SQL (the empty/non-vault ones).
    """
    created: list[str] = []

    class _N:
        def team(self) -> str:
            t = "bf" + uuid.uuid4().hex[:10]
            created.append(t)
            return t

    yield _N()
    for t in created:
        schema = team_schema(t)
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()


def _col(backend, team, create=True):
    return backend.get_collection(
        palace=PalaceRef(id=team, namespace=team), collection_name=COLLECTION, create=create
    )


def _seed_populated_vault(backend, team) -> None:
    """A real vault: drawers table created + a row, NO identity recorded yet
    (the legacy state the backfill repairs)."""
    col = _col(backend, team, create=True)
    col.add(documents=["x"], ids=["a"], metadatas=[{}], embeddings=[_vec(0.5)])


def _make_empty_team_schema(team) -> None:
    """An empty/non-vault team schema: schema present, NO drawers table.

    Mirrors a ``team_<slug>`` schema created by some side path (link-store, WAL,
    entity-index, etc.) that never received a drawers collection. The backfill
    must skip these and must not create a drawers table inside them.
    """
    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.commit()


def _drawers_table_exists(team) -> bool:
    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{schema}.{COLLECTION}",))
            return cur.fetchone()[0] is not None


def _identity_row(team):
    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{schema}.embedder_identity",))
            if cur.fetchone()[0] is None:
                return None
            cur.execute(
                f'SELECT model_name, dimension FROM "{schema}".embedder_identity '
                "WHERE collection = %s",
                (COLLECTION,),
            )
            return cur.fetchone()


# --------------------------------------------------------------------------
# (i) records identity into EXISTING populated vaults
# (ii) creates NO new drawers tables anywhere (empty/non-vault schemas skipped)
# --------------------------------------------------------------------------


def test_backfill_records_into_existing_vaults_only(backend, names):
    """Two populated vaults get identity; two empty/non-vault team schemas are
    skipped — and crucially NO drawers table is materialised in the empties."""
    pop_a = names.team()
    pop_b = names.team()
    empty_a = names.team()
    empty_b = names.team()

    _seed_populated_vault(backend, pop_a)
    _seed_populated_vault(backend, pop_b)
    _make_empty_team_schema(empty_a)
    _make_empty_team_schema(empty_b)

    # Pre-conditions: populated vaults have a drawers table; empties do not.
    assert _drawers_table_exists(pop_a) and _drawers_table_exists(pop_b)
    assert not _drawers_table_exists(empty_a) and not _drawers_table_exists(empty_b)
    # Pre-condition: populated vaults have NO identity recorded yet (legacy state).
    assert _identity_row(pop_a) is None
    assert _identity_row(pop_b) is None

    identity = EmbedderIdentity("embeddinggemma", DIM)
    results = backend.backfill_embedder_identity(identity, collection=COLLECTION)

    backfilled = {r["team"] for r in results}
    # (i) identity recorded into BOTH existing populated vaults.
    assert pop_a in backfilled and pop_b in backfilled
    for team in (pop_a, pop_b):
        row = _identity_row(team)
        assert row is not None, f"{team} identity not recorded"
        assert row[0] == "embeddinggemma" and int(row[1]) == DIM

    # Empty/non-vault schemas are NOT in the backfill result.
    assert empty_a not in backfilled and empty_b not in backfilled

    # (ii) NO new drawers table materialised anywhere — empties stay empty.
    assert not _drawers_table_exists(empty_a), "backfill materialised a drawers table in empty_a!"
    assert not _drawers_table_exists(empty_b), "backfill materialised a drawers table in empty_b!"
    # And the empty schemas got no identity row either (skipped entirely).
    assert _identity_row(empty_a) is None
    assert _identity_row(empty_b) is None


# --------------------------------------------------------------------------
# (iii) idempotent on a second run
# --------------------------------------------------------------------------


def test_backfill_is_idempotent(backend, names):
    """A second backfill changes nothing: same recorded identity, no new tables."""
    pop = names.team()
    empty = names.team()
    _seed_populated_vault(backend, pop)
    _make_empty_team_schema(empty)

    identity = EmbedderIdentity("embeddinggemma", DIM)
    first = backend.backfill_embedder_identity(identity, collection=COLLECTION)
    assert any(r["team"] == pop for r in first)

    second = backend.backfill_embedder_identity(identity, collection=COLLECTION)
    assert any(r["team"] == pop for r in second)

    row = _identity_row(pop)
    assert row is not None and row[0] == "embeddinggemma" and int(row[1]) == DIM
    # Idempotent re-run still must not have materialised a drawers table.
    assert not _drawers_table_exists(empty)


def test_backfill_scoped_to_named_vault(backend, names):
    """``teams=[name]`` restricts the backfill to the one existing vault; another
    existing vault is left untouched (no identity)."""
    target = names.team()
    other = names.team()
    _seed_populated_vault(backend, target)
    _seed_populated_vault(backend, other)

    identity = EmbedderIdentity("embeddinggemma", DIM)
    results = backend.backfill_embedder_identity(identity, teams=[target], collection=COLLECTION)

    assert [r["team"] for r in results] == [target]
    assert _identity_row(target) is not None
    assert _identity_row(other) is None, "scoped backfill leaked into an unrelated vault"


# --------------------------------------------------------------------------
# (iv) A4 model-swap contract still caught after the backfill
# --------------------------------------------------------------------------


def test_backfill_then_model_swap_still_raises(backend, names, monkeypatch):
    """After the backfill stamps a vault, reopening it under a DIFFERENT model
    raises EmbedderIdentityMismatchError — the A4 enforcement contract holds on
    a backfilled vault, not just on first-write-recorded ones."""
    from mempalace import palace as P

    team = names.team()
    _seed_populated_vault(backend, team)

    # Backfill records embeddinggemma (the current model) onto the existing vault.
    backend.backfill_embedder_identity(
        EmbedderIdentity("embeddinggemma", DIM), collection=COLLECTION
    )
    assert _identity_row(team)[0] == "embeddinggemma"

    # Route palace.get_collection to this postgres test DB.
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    # Configure a DIFFERENT model than what the backfill recorded.
    monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "minilm")
    P._VALIDATED_IDENTITY.clear()

    with pytest.raises(EmbedderIdentityMismatchError):
        P.get_collection(team, collection_name=COLLECTION, create=False, team=team)
