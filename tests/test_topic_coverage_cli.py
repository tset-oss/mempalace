"""Integration tests for the ``topic-coverage`` CLI subcommand (S4 / FU5).

Verifies the read-only per-team topic-coverage report against a live Postgres
instance. Skipped when psycopg is absent or the DB is unreachable.
"""

from __future__ import annotations

import os
import types
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.link_store_postgres import PostgresLinkStore  # noqa: E402


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


@pytest.fixture
def backend():
    b = PostgresBackend(dsn=_dsn())
    yield b
    b.close()


def _fresh_team(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _drop_schema(backend, team_name: str) -> None:
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_name)}" CASCADE')


def _run_topic_coverage(monkeypatch, backend, team_name: str, all_vaults: bool = False):
    """Invoke cmd_topic_coverage with a minimal args namespace and return stdout."""
    import io

    from mempalace.cli import cmd_topic_coverage

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)

    args = types.SimpleNamespace(
        vault=None if all_vaults else team_name,
        all_vaults=all_vaults,
    )

    captured = io.StringIO()
    import builtins

    real_print = builtins.print

    def _capture(*a, **kw):
        kw.setdefault("file", captured)
        real_print(*a, **kw)

    monkeypatch.setattr(builtins, "print", _capture)
    cmd_topic_coverage(args)
    monkeypatch.setattr(builtins, "print", real_print)
    return captured.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Core correctness: seeded counts appear in the report
# ─────────────────────────────────────────────────────────────────────────────


def test_topic_coverage_counts_match_seeded_data(monkeypatch, backend):
    """Reported row/topic/wing counts match what was seeded into wing_topics."""
    team = _fresh_team("tcov")
    try:
        store = PostgresLinkStore(backend, team=team)
        # 3 rows, 2 distinct topics, 2 wings with topics
        store.add_topics("wing_alpha", ["Angular", "OpenAPI"])
        store.add_topics("wing_beta", ["Angular"])

        out = _run_topic_coverage(monkeypatch, backend, team)

        assert "3 wing_topics rows" in out
        assert "2 distinct topics" in out
        assert "2 wings with topics" in out
    finally:
        _drop_schema(backend, team)


def test_topic_coverage_empty_vault_reports_no_table(monkeypatch, backend):
    """A vault with no wing_topics rows reports the no-labels message."""
    team = _fresh_team("tcov_empty")
    try:
        # Force schema creation without any topics so the vault exists but
        # the wing_topics table is absent (never populated).
        out = _run_topic_coverage(monkeypatch, backend, team)

        # Either "no wing_topics table" (table was never created) or 0 rows.
        assert "no wing_topics" in out or "0 wing_topics rows" in out
    finally:
        _drop_schema(backend, team)


# ─────────────────────────────────────────────────────────────────────────────
# Isolation: team X's report does NOT include team Y's topics
# ─────────────────────────────────────────────────────────────────────────────


def test_topic_coverage_isolated_per_team(monkeypatch, backend):
    """Team X's report contains only X's counts, not Y's topics."""
    team_x = _fresh_team("tcov_x")
    team_y = _fresh_team("tcov_y")
    try:
        store_x = PostgresLinkStore(backend, team=team_x)
        store_y = PostgresLinkStore(backend, team=team_y)

        # Team X: 2 wings, 3 labels, 3 rows
        store_x.add_topics("wing_a", ["Alpha", "Beta"])
        store_x.add_topics("wing_b", ["Gamma"])

        # Team Y: 1 wing, 5 labels, 5 rows — deliberately larger to prove isolation
        store_y.add_topics("wing_c", ["X1", "X2", "X3", "X4", "X5"])

        out_x = _run_topic_coverage(monkeypatch, backend, team_x)
        out_y = _run_topic_coverage(monkeypatch, backend, team_y)

        # X's report must show X's 3 rows / 3 topics / 2 wings
        assert "3 wing_topics rows" in out_x
        assert "3 distinct topics" in out_x
        assert "2 wings with topics" in out_x

        # Y's report must show Y's 5 rows / 5 topics / 1 wing
        assert "5 wing_topics rows" in out_y
        assert "5 distinct topics" in out_y
        assert "1 wings with topics" in out_y

        # X's output must not mention Y's team slug or Y's row counts
        assert team_y not in out_x
        assert "5 wing_topics rows" not in out_x

        # Y's output must not mention X's team slug or X's row counts
        assert team_x not in out_y
        assert "3 wing_topics rows" not in out_y
    finally:
        _drop_schema(backend, team_x)
        _drop_schema(backend, team_y)


# ─────────────────────────────────────────────────────────────────────────────
# Backend guard: chroma backend prints clear not-applicable message, no crash
# ─────────────────────────────────────────────────────────────────────────────


def test_topic_coverage_chroma_backend_prints_not_applicable(monkeypatch, capsys):
    """On chroma/local backend the command prints a clear message and exits cleanly."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")

    from mempalace.cli import cmd_topic_coverage

    args = types.SimpleNamespace(vault=None, all_vaults=False)

    # Must not raise; must print the not-applicable notice.
    cmd_topic_coverage(args)

    captured = capsys.readouterr()
    assert "postgres backend only" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# Pure DB read: no writes issued during the report
# ─────────────────────────────────────────────────────────────────────────────


def test_topic_coverage_issues_only_reads(monkeypatch, backend):
    """The report executes only SELECT statements — no INSERT/UPDATE/DELETE."""
    team = _fresh_team("tcov_readonly")
    try:
        store = PostgresLinkStore(backend, team=team)
        store.add_topics("wing_a", ["Topic1", "Topic2"])

        write_calls: list[str] = []
        real_conn = backend._conn

        class _WatchConn:
            def __init__(self, conn):
                self._conn = conn

            def cursor(self):
                return _WatchCursor(self._conn.cursor())

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._conn.__exit__(*a)

        class _WatchCursor:
            def __init__(self, cur):
                self._cur = cur

            def execute(self, sql, *a, **kw):
                verb = sql.strip().split()[0].upper()
                if verb in (
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "TRUNCATE",
                    "DROP",
                    "CREATE",
                    "ALTER",
                    "MERGE",
                ):
                    write_calls.append(sql)
                return self._cur.execute(sql, *a, **kw)

            def fetchone(self):
                return self._cur.fetchone()

            def fetchall(self):
                return self._cur.fetchall()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self._cur.__exit__(*a)

        import contextlib

        @contextlib.contextmanager
        def _patched_conn():
            with real_conn() as conn:
                yield _WatchConn(conn)

        monkeypatch.setattr(backend, "_conn", _patched_conn)

        _run_topic_coverage(monkeypatch, backend, team)

        assert write_calls == [], f"Unexpected write statements during report: {write_calls}"
    finally:
        _drop_schema(backend, team)
