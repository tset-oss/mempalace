"""Tests for pg_trgm provisioning + the per-drawer trigram GIN (story G001).

The live-DB tests run against a real Postgres with pgvector (the bundled
docker-compose db, or any instance reachable via ``MEMPALACE_TEST_PG_URL`` /
``MEMPALACE_DATABASE_URL``). They self-skip when psycopg is missing or the
database is unreachable, mirroring ``tests/test_postgres_backend.py``.

Covered acceptance (G001):
  (a) after bootstrap, ``pg_trgm`` is installed (``pg_extension`` has a row);
  (b) a freshly-created collection has the ``{table}_doc_trgm`` trigram GIN;
  (c) the existing-vault migration adds the GIN to a table that lacks it,
      without an ACCESS EXCLUSIVE lock (CONCURRENTLY on an autocommit
      connection), leaving a VALID index, and is idempotent (re-runnable no-op);
  (d) the ``CREATE EXTENSION ... pg_trgm`` is NOT swallowed: a simulated failure
      propagates out of ``_ensure_bootstrap`` (unlike the swallowed pg_search/age
      creates), proving it sits outside the swallow loop and is a required path.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends import PalaceRef  # noqa: E402
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


_LIVE = _reachable(_dsn())
live_only = pytest.mark.skipif(
    not _LIVE,
    reason="no reachable Postgres (start deploy/docker-compose.yml or set MEMPALACE_TEST_PG_URL)",
)


def _fake_embed(texts):
    """Deterministic embedder so collection creation never downloads a model."""
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
    name = "trgm" + uuid.uuid4().hex[:8]
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


def _trgm_index_row(schema: str, index_name: str):
    """Return (indisvalid, indexdef) for the trigram GIN, or None if absent."""
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT i.indisvalid, pg_get_indexdef(c.oid) "
                "FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN pg_index i ON i.indexrelid = c.oid "
                "WHERE n.nspname = %s AND c.relname = %s",
                (schema, index_name),
            )
            return cur.fetchone()


# --------------------------------------------------------------------------
# (a) extension provisioned by bootstrap
# --------------------------------------------------------------------------


@live_only
def test_bootstrap_installs_pg_trgm(backend, team):
    # Creating any collection triggers _ensure_bootstrap.
    _col(backend, team, create=True)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
            assert cur.fetchone() is not None, "pg_trgm not installed after bootstrap"


# --------------------------------------------------------------------------
# (b) fresh collection carries the trigram GIN
# --------------------------------------------------------------------------


@live_only
def test_fresh_collection_has_doc_trgm_gin(backend, team):
    _col(backend, team, create=True)
    schema = team_schema(team)
    index_name = COLLECTION + "_doc_trgm"
    row = _trgm_index_row(schema, index_name)
    assert row is not None, f"{index_name} missing on a freshly-created collection"
    indisvalid, indexdef = row
    assert indisvalid is True, f"{index_name} is INVALID on a fresh collection"
    assert "gin" in indexdef.lower()
    assert "gin_trgm_ops" in indexdef.lower()


# --------------------------------------------------------------------------
# (a-flag) supports_contains_fast is accurate: the backend advertises it, and
# the backing trigram GIN it claims actually exists on a fresh collection.
# Honesty post-condition (story G006): the flag is true IFF $contains is
# GIN-backed, so this ties the capability to the index it depends on.
# --------------------------------------------------------------------------


def test_supports_contains_fast_capability_advertised():
    # No DB needed: the capability is a class attribute.
    assert "supports_contains_fast" in PostgresBackend.capabilities


@live_only
def test_supports_contains_fast_is_backed_by_doc_trgm_gin(backend, team):
    # The flag is only honest if the trigram GIN that makes $contains fast
    # actually exists on a freshly-created collection.
    assert "supports_contains_fast" in PostgresBackend.capabilities
    _col(backend, team, create=True)
    schema = team_schema(team)
    row = _trgm_index_row(schema, COLLECTION + "_doc_trgm")
    assert row is not None, (
        "supports_contains_fast is advertised but the trigram GIN that backs it "
        "is missing — the capability flag would be a false claim"
    )
    indisvalid, indexdef = row
    assert indisvalid is True
    assert "gin_trgm_ops" in indexdef.lower()


# --------------------------------------------------------------------------
# (c) existing-vault migration (the CONCURRENTLY path)
# --------------------------------------------------------------------------


@live_only
def test_existing_vault_migration_adds_valid_gin_and_is_idempotent(backend, team):
    # Create the collection (which builds the GIN), then DROP the trigram GIN to
    # simulate a vault provisioned before this change.
    _col(backend, team, create=True)
    schema = team_schema(team)
    index_name = COLLECTION + "_doc_trgm"

    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP INDEX IF EXISTS "{schema}"."{index_name}"')
        conn.commit()
    assert _trgm_index_row(schema, index_name) is None, "precondition: GIN should be dropped"

    # Run the migration helper directly. It runs CREATE INDEX CONCURRENTLY on an
    # autocommit connection (NOT _conn(), which is a transaction block); a full
    # lock-level assertion is impractical here, so we assert the observable
    # post-conditions: the index exists, is VALID, and the helper is re-runnable.
    backend._migrate_doc_trgm_index(schema, COLLECTION)

    row = _trgm_index_row(schema, index_name)
    assert row is not None, "migration did not create the trigram GIN"
    indisvalid, indexdef = row
    assert indisvalid is True, "CONCURRENTLY left an INVALID index"
    assert "gin_trgm_ops" in indexdef.lower()

    # Idempotent: a second call is a no-op and leaves the index VALID.
    backend._migrate_doc_trgm_index(schema, COLLECTION)
    row2 = _trgm_index_row(schema, index_name)
    assert row2 is not None and row2[0] is True


@live_only
def test_migration_runs_on_autocommit_not_in_transaction(monkeypatch, backend, team):
    """The migration must run CONCURRENTLY, which requires autocommit.

    CONCURRENTLY cannot run inside a transaction block, and ``_conn()`` is one,
    so the helper acquires a pooled connection and sets ``autocommit = True``
    before issuing any DDL. We wrap the pooled connection and record its
    ``autocommit`` flag at the moment a cursor is requested for DDL (the helper
    only calls ``cursor()`` after flipping autocommit). A real index also lands.
    """
    _col(backend, team, create=True)
    schema = team_schema(team)
    index_name = COLLECTION + "_doc_trgm"
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP INDEX IF EXISTS "{schema}"."{index_name}"')
        conn.commit()

    seen = {"autocommit_at_cursor": []}
    real_connection = backend._pool().connection

    class _ConnWrapper:
        """Proxies a real connection; records ``autocommit`` on every cursor()."""

        def __init__(self, cm):
            self._cm = cm
            self._conn = None

        def __enter__(self):
            self._conn = self._cm.__enter__()
            return self

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

        def cursor(self, *a, **k):
            # Capture the connection's autocommit state at DDL time.
            seen["autocommit_at_cursor"].append(self._conn.autocommit)
            return self._conn.cursor(*a, **k)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            if name in ("_cm", "_conn"):
                object.__setattr__(self, name, value)
            else:
                setattr(self._conn, name, value)

    def _wrapped_connection(*a, **k):
        return _ConnWrapper(real_connection(*a, **k))

    monkeypatch.setattr(backend._pool(), "connection", _wrapped_connection)
    backend._migrate_doc_trgm_index(schema, COLLECTION)

    assert seen["autocommit_at_cursor"], "helper never opened a cursor"
    assert all(seen["autocommit_at_cursor"]), (
        "CREATE INDEX CONCURRENTLY ran without autocommit (would fail inside a txn block)"
    )
    # And it produced a VALID index.
    row = _trgm_index_row(schema, index_name)
    assert row is not None and row[0] is True


# --------------------------------------------------------------------------
# (d) the pg_trgm create is NOT swallowed (required path, propagation test)
# --------------------------------------------------------------------------


class _FakeCursor:
    """Cursor that raises on the pg_trgm CREATE EXTENSION, succeeds otherwise."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        text = str(sql)
        if "pg_trgm" in text:
            self._conn.executed.append(("FAIL", text))
            raise RuntimeError("simulated pg_trgm failure")
        self._conn.executed.append(("OK", text))

    def fetchone(self):
        return None


class _FakeConn:
    def __init__(self):
        self.executed: list[tuple[str, str]] = []
        self.rolled_back = 0

    def cursor(self):
        return _FakeCursor(self)

    def rollback(self):
        self.rolled_back += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_pg_trgm_create_failure_propagates_not_swallowed(monkeypatch):
    """A failing pg_trgm CREATE EXTENSION must propagate out of bootstrap.

    Unlike the swallowed pg_search/age creates (which rollback + warn), pg_trgm
    is a required path: a hard failure must surface. We monkeypatch _conn() to
    yield a fake connection whose cursor raises specifically on the pg_trgm
    statement and assert the exception is NOT swallowed.
    """
    be = PostgresBackend(dsn="postgresql://unused", vector_dim=DIM)
    fake = _FakeConn()
    monkeypatch.setattr(be, "_conn", lambda: fake)

    with pytest.raises(RuntimeError, match="simulated pg_trgm failure"):
        be._ensure_bootstrap()

    # The vector create ran (OK) and the pg_trgm create was attempted and FAILED,
    # and no rollback-and-continue happened for pg_trgm (it is outside the loop).
    statements = [s for _, s in fake.executed]
    assert any("vector" in s for s in statements)
    assert any("pg_trgm" in s for s in statements)
    # The pg_search/age loop must NOT have been reached (failure short-circuited).
    assert not any("pg_search" in s for s in statements)
    assert not any("age" in s.lower() for s in statements)
    # bootstrapped flag must remain False so a retry can re-run.
    assert be._bootstrapped is False


# --------------------------------------------------------------------------
# autocommit-leak regression (the HIGH blocking bug from code review)
# --------------------------------------------------------------------------


@live_only
def test_migrate_restores_autocommit_on_pooled_connection(team):
    """_migrate_doc_trgm_index must restore autocommit=False before returning.

    psycopg_pool's reset normalises transaction STATUS but does NOT reset the
    autocommit attribute. Without the try/finally in the migration helper, the
    physical connection returned to the pool retains autocommit=True and the
    next consumer loses all-or-nothing semantics.

    We use a pool of max_size=1 so the pool hands back the SAME physical
    connection after the migration. Two sub-tests:

    (A) direct attribute check — the connection drawn from the pool after
        migration has autocommit=False.
    (B) atomicity check — a deliberately-failing two-statement block on a
        pooled connection rolls back the first statement (would NOT happen
        if autocommit were still True, because each statement would be
        auto-committed individually before the exception).
    """
    import psycopg_pool  # already a dep; importorskip guards psycopg above

    schema = team_schema(team)
    dsn = _dsn()

    # Use a dedicated single-connection pool so we're guaranteed to get back
    # the same physical connection the migration used.
    pool = psycopg_pool.ConnectionPool(
        conninfo=dsn, min_size=1, max_size=1, timeout=10.0, open=True
    )
    try:
        be = PostgresBackend(dsn=dsn, vector_dim=DIM, embedder=_fake_embed)
        be._pool_obj = pool  # inject the single-slot pool

        # Create the collection (triggers bootstrap + _ensure_collection new-table
        # path, which runs INSIDE the normal _conn() transaction — not the
        # CONCURRENTLY helper). Then drop the trigram GIN to simulate a
        # pre-existing vault, forcing the CONCURRENTLY migration path.
        be.get_collection(
            palace=PalaceRef(id=team, namespace=team),
            collection_name=COLLECTION,
            create=True,
        )
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP INDEX IF EXISTS "{schema}"."{COLLECTION}_doc_trgm"')
            conn.commit()

        # Clear the ensured-set so _ensure_collection re-runs for this table,
        # hitting the preexisting_table=True branch and calling the migration.
        be._ensured.discard((schema, COLLECTION))
        be.get_collection(
            palace=PalaceRef(id=team, namespace=team),
            collection_name=COLLECTION,
            create=True,
        )

        # --- (A) autocommit attribute check ---
        with pool.connection() as conn:
            assert conn.autocommit is False, (
                "pool connection has autocommit=True after migration — "
                "_migrate_doc_trgm_index leaked autocommit=True to the pool"
            )

        # --- (B) atomicity check ---
        # Insert a row then force an error in the SAME block. With autocommit=False
        # both statements share one transaction; the error aborts it and the
        # rollback drops the INSERT. If autocommit had leaked True, the first INSERT
        # would have been auto-committed before the error and would survive.
        # Use a REGULAR table in the team schema (not a TEMP table): temp tables are
        # session-local, so the pooled connection's session could not see one created
        # on a separate setup connection. The finally-block schema DROP cleans it up.
        leak_tbl = f'"{schema}"."_trgm_leak_test"'
        with psycopg.connect(dsn) as setup_conn:
            with setup_conn.cursor() as cur:
                cur.execute(f"CREATE TABLE IF NOT EXISTS {leak_tbl} (v int)")
            setup_conn.commit()

        with pool.connection() as conn:
            assert conn.autocommit is False  # pre-condition
            try:
                with conn.cursor() as cur:
                    # Statement 1: insert a row.
                    cur.execute(f"INSERT INTO {leak_tbl} VALUES (42)")
                    # Statement 2: force a hard error (divide by zero).
                    cur.execute("SELECT 1/0")
            except Exception:
                conn.rollback()

            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {leak_tbl}")
                count = cur.fetchone()[0]
            assert count == 0, (
                f"INSERT was not rolled back (count={count}); "
                "pool connection still has autocommit=True — try/finally not working"
            )

        be.close()
    finally:
        pool.close()
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()
