"""Server-mode (postgres) wiring for tool_status / tool_reconnect (G003).

These verify the backend-aware branching: in postgres mode status/reconnect
report PG health + the active vault and reconnect the pool, while the
chroma-only paths (the vector_disabled HNSW probe, client-cache reset) do not
run. They use fakes for the backend/collection, so they need neither a live
database nor the optional psycopg/serve extras and always run.
"""

from __future__ import annotations

import pytest

import mempalace.mcp_server as m


class _FakeHealth:
    def __init__(self, ok: bool, detail: str = ""):
        self.ok = ok
        self.detail = detail


class _FakeBackend:
    def __init__(self, ok=True, vaults=None):
        self._ok = ok
        self._vaults = vaults or []
        self.reconnected = False
        self.closed = False

    def health(self, palace=None):
        return _FakeHealth(self._ok, "" if self._ok else "connection refused")

    def list_vaults(self):
        return list(self._vaults)

    def reconnect(self):
        self.reconnected = True

    def close(self):
        self.closed = True


class _FakeCol:
    def __init__(self, n=3):
        self._n = n

    def count(self):
        return self._n


@pytest.fixture
def pg_env(monkeypatch):
    """Force postgres backend + a known primary team via env (read live)."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", "frontend")
    # Guard: ensure no leftover session team leaks into _resolve_team().
    token = m._active_team_var.set(None)
    yield monkeypatch
    m._active_team_var.reset(token)


def test_status_server_reports_health_and_active_vault(pg_env):
    monkeypatch = pg_env
    fake = _FakeBackend(ok=True, vaults=["frontend", "backend"])
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: fake)
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: _FakeCol(5))
    monkeypatch.setattr(m, "_get_cached_metadata", lambda col: [{"wing": "w", "room": "r"}] * 5)

    # The chroma-only HNSW probe must NOT run in server mode.
    probe_called = {"hit": False}
    monkeypatch.setattr(
        m, "_refresh_vector_disabled_flag", lambda: probe_called.__setitem__("hit", True)
    )

    res = m.tool_status()
    assert res["backend"] == "postgres"
    assert res["vault"] == "frontend"
    assert res["healthy"] is True
    assert res["vaults"] == ["frontend", "backend"]
    assert res["total_drawers"] == 5
    # No chroma-only signals leak into the server-mode payload.
    assert "vector_disabled" not in res
    assert "vector_disabled_reason" not in res
    assert probe_called["hit"] is False


def test_status_server_unhealthy_skips_collection_probe(pg_env):
    monkeypatch = pg_env
    fake = _FakeBackend(ok=False)
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: fake)

    col_probed = {"hit": False}

    def _boom_collection(*a, **k):
        col_probed["hit"] = True
        return _FakeCol()

    monkeypatch.setattr(m, "_get_collection", _boom_collection)

    res = m.tool_status()
    assert res["healthy"] is False
    assert res["health_detail"] == "connection refused"
    # When the server is unreachable we report health without touching a vault.
    assert "total_drawers" not in res
    assert col_probed["hit"] is False


def test_reconnect_server_reconnects_pool_and_drains_kg_cache(pg_env):
    monkeypatch = pg_env
    monkeypatch.setenv("MEMPALACE_TEAM", "backend")
    fake = _FakeBackend(ok=True, vaults=["backend"])
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: fake)
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: _FakeCol(2))

    closed = {"kg": False}

    class _FakeKG:
        def close(self):
            closed["kg"] = True

    with m._kg_cache_lock:
        m._kg_by_path["pgkg::backend"] = _FakeKG()

    res = m.tool_reconnect()
    assert fake.reconnected is True
    assert res["success"] is True
    assert res["backend"] == "postgres"
    assert res["vault"] == "backend"
    assert res["healthy"] is True
    assert res["drawers"] == 2
    # The cached KG handle was closed and the cache drained.
    assert closed["kg"] is True
    assert "pgkg::backend" not in m._kg_by_path


def test_reconnect_server_reports_errors_when_backend_unhealthy(pg_env):
    monkeypatch = pg_env
    fake = _FakeBackend(ok=False)
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: fake)
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: _FakeCol(0))

    res = m.tool_reconnect()
    assert fake.reconnected is True  # still attempted the pool reconnect
    assert res["healthy"] is False
    assert res["success"] is False


def test_metadata_cache_bypassed_in_server_mode(pg_env):
    # The process-global metadata cache is vault-agnostic; in multi-tenant
    # server mode it must be bypassed so one vault's wing/room breakdown is
    # never served for another vault within the TTL.
    monkeypatch = pg_env
    monkeypatch.setattr(m, "_metadata_cache", [{"wing": "OTHER_VAULT"}])
    monkeypatch.setattr(m, "_metadata_cache_time", 10**12)  # "fresh" far-future stamp
    fresh = [{"wing": "frontend_w", "room": "r"}]
    monkeypatch.setattr(m, "_fetch_all_metadata", lambda col, where=None: fresh)

    out = m._get_cached_metadata(object())
    assert out == fresh, "server mode must fetch fresh, not serve the stale global cache"
    # And it must not have written this vault's data into the shared cache.
    assert m._metadata_cache == [{"wing": "OTHER_VAULT"}]


def test_postgres_backend_reconnect_drops_pool_but_stays_usable():
    # No DB / psycopg needed: reconnect() only manipulates the cached pool ref.
    from mempalace.backends.postgres import PostgresBackend

    backend = PostgresBackend(dsn="postgresql://unused/never-opened")

    class _FakePool:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    pool = _FakePool()
    backend._pool_obj = pool
    backend._ensured = {("team_a", "kg_entities")}

    backend.reconnect()

    assert pool.closed is True, "the live pool must be closed"
    assert backend._pool_obj is None, "next op must lazily re-open a fresh pool"
    assert backend._ensured == set(), "DDL-ensured cache must be cleared"
    assert backend._closed is False, "reconnect (unlike close) keeps the backend usable"
