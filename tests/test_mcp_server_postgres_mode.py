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


def test_wal_sink_config_validation(monkeypatch):
    from mempalace.config import MempalaceConfig

    cfg = MempalaceConfig()
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "postgres")
    assert cfg.wal_sink == "postgres"
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "POSTGRES")  # case-insensitive
    assert cfg.wal_sink == "postgres"
    # Unrecognised -> the backend-aware default (chroma here -> jsonl).
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "garbage")
    assert cfg.wal_sink == "jsonl"


def test_wal_sink_defaults_to_postgres_on_postgres_backend(monkeypatch):
    """The central/postgres deploy defaults the audit sink to the team-tagged
    postgres table (not the host-global jsonl file), while chroma stays jsonl
    and an explicit value always wins."""
    from mempalace.config import MempalaceConfig

    monkeypatch.delenv("MEMPALACE_WAL_SINK", raising=False)

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    assert MempalaceConfig().wal_sink == "postgres"

    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    assert MempalaceConfig().wal_sink == "jsonl"

    # An explicit override beats the backend-aware default in both directions.
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "jsonl")
    assert MempalaceConfig().wal_sink == "jsonl"


def test_wal_log_routes_to_postgres_sink_with_redaction_and_team(pg_env):
    monkeypatch = pg_env
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "postgres")

    calls = []

    class _SinkBackend:
        def wal_append(self, **kw):
            calls.append(kw)

    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: _SinkBackend())
    jsonl_writes = []
    monkeypatch.setattr(m, "_wal_log_jsonl", lambda entry: jsonl_writes.append(entry))

    m._wal_log("add_drawer", {"content": "secret note", "wing": "w"})

    assert len(calls) == 1
    kw = calls[0]
    assert kw["operation"] == "add_drawer"
    assert kw["team"] == "frontend"  # the active vault (pg_env default)
    # Redaction is applied before the sink sees it.
    assert kw["params"]["content"].startswith("[REDACTED")
    assert kw["params"]["wing"] == "w"
    # On success it does NOT also write the local jsonl file.
    assert jsonl_writes == []


def test_wal_log_falls_back_to_jsonl_when_pg_sink_fails(pg_env):
    monkeypatch = pg_env
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "postgres")

    class _FailingBackend:
        def wal_append(self, **kw):
            raise RuntimeError("database unreachable")

    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: _FailingBackend())
    jsonl_writes = []
    monkeypatch.setattr(m, "_wal_log_jsonl", lambda entry: jsonl_writes.append(entry))

    m._wal_log("delete_drawer", {"drawer_id": "d1"})

    # The audit entry is never dropped — it lands in the local jsonl fallback.
    assert len(jsonl_writes) == 1
    assert jsonl_writes[0]["operation"] == "delete_drawer"
    assert jsonl_writes[0]["team"] == "frontend"


def test_wal_log_jsonl_default_for_chroma_records_no_team(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.delenv("MEMPALACE_WAL_SINK", raising=False)
    wal_file = tmp_path / "write_log.jsonl"
    monkeypatch.setattr(m, "_WAL_FILE", wal_file)

    m._wal_log("add_drawer", {"content": "hi", "safe": "ok"})

    import json

    entry = json.loads(wal_file.read_text().strip())
    assert entry["operation"] == "add_drawer"
    assert entry["team"] is None  # chroma is single-vault
    assert entry["params"]["content"].startswith("[REDACTED")
    assert entry["params"]["safe"] == "ok"


def test_wal_log_redacts_result_not_just_params(pg_env):
    # result now crosses into central storage, so it gets the same redaction.
    monkeypatch = pg_env
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "postgres")
    calls = []

    class _SinkBackend:
        def wal_append(self, **kw):
            calls.append(kw)

    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: _SinkBackend())
    m._wal_log("op", {"safe": "x"}, result={"content": "should be redacted", "ok": True})

    kw = calls[0]
    assert kw["result"]["content"].startswith("[REDACTED")
    assert kw["result"]["ok"] is True


def test_add_drawer_audits_the_vault_it_actually_targeted(pg_env):
    # The cross-vault case: session/default is 'frontend', but the write targets
    # vault='backend'. The audit row MUST record 'backend' (where the data lands),
    # not the session default.
    monkeypatch = pg_env
    captured = {}

    def _spy_wal(operation, params, result=None, team=None):
        captured["operation"] = operation
        captured["team"] = team

    monkeypatch.setattr(m, "_wal_log", _spy_wal)

    class _ExistingResult:
        ids = ["drawer_x"]  # idempotency hit -> add_drawer returns right after WAL

    class _Col:
        def get(self, ids=None, include=None):
            return _ExistingResult()

        def count(self):
            return 1

    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: _Col())

    m.tool_add_drawer("wing", "room", "some verbatim content", vault="backend")
    assert captured["operation"] == "add_drawer"
    assert captured["team"] == "backend"


def test_postgres_wal_append_ensures_table_once_and_inserts_redacted():
    # No DB needed: stub _conn() so we can inspect the SQL/params wal_append runs.
    from mempalace.backends.postgres import PostgresBackend, _WAL_AUDIT_TABLE

    backend = PostgresBackend(dsn="postgresql://unused/never-opened")
    executed = []

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            executed.append((sql, params))

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _Cur()

    backend._conn = lambda: _Conn()

    backend.wal_append(
        timestamp="2026-06-03T00:00:00",
        operation="add_drawer",
        team="frontend",
        params={"content": "[REDACTED 5 chars]", "wing": "w"},
        result=None,
    )
    sqls = " ".join(s for s, _ in executed)
    assert "CREATE SCHEMA IF NOT EXISTS" in sqls
    assert _WAL_AUDIT_TABLE in sqls
    insert = next(e for e in executed if e[0].lstrip().startswith("INSERT"))
    params = insert[1]
    assert params[1] == "add_drawer"
    assert params[2] == "frontend"
    assert "[REDACTED 5 chars]" in params[3]  # params jsonb payload

    # Second append must NOT re-run the DDL (the _wal_ready guard).
    executed.clear()
    backend.wal_append(timestamp="t", operation="op", team=None, params={}, result=None)
    assert not any("CREATE SCHEMA" in s for s, _ in executed)


def test_sanitize_team_caps_length_to_match_reported_slug():
    # The physical schema slug must not exceed the 40-char slug the override
    # validator / _canonical_default_team report, so a >40 name can never make
    # the data land in a differently-named schema than the server advertises.
    from mempalace.backends.postgres import sanitize_team

    assert sanitize_team("a" * 50) == "a" * 40
    assert len(sanitize_team("frontend_" * 10)) <= 40


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


# ==================== _resolve_team_strict (fail-loud team resolution) ====================
#
# _resolve_team_strict returns None on ambiguity so a server-mode writer can
# RAISE, instead of _resolve_team's silent _canonical_default_team() fallback
# (which would re-create the shared-default cross-tenant leak). No DB needed.


def test_resolve_team_strict_returns_none_when_no_explicit_and_no_active_team(monkeypatch):
    monkeypatch.setenv("MEMPALACE_TEAM", "configured_default")
    token = m._active_team_var.set(None)
    try:
        # Sanity: the non-strict resolver DOES default (proving the strict one
        # is not merely echoing an empty configured team).
        assert m._resolve_team() == m._canonical_default_team()
        # Strict: ambiguous -> None, never the canonical default.
        result = m._resolve_team_strict()
        assert result is None
        assert result != m._canonical_default_team()
    finally:
        m._active_team_var.reset(token)


def test_resolve_team_strict_returns_explicit_slug_when_given(monkeypatch):
    token = m._active_team_var.set(None)
    try:
        assert m._resolve_team_strict("backend") == "backend"
    finally:
        m._active_team_var.reset(token)


def test_resolve_team_strict_returns_active_team_when_set(monkeypatch):
    monkeypatch.setenv("MEMPALACE_TEAM", "configured_default")
    token = m._active_team_var.set("frontend")
    try:
        assert m._resolve_team_strict() == "frontend"
    finally:
        m._active_team_var.reset(token)


def test_resolve_team_strict_explicit_overrides_active_team():
    token = m._active_team_var.set("frontend")
    try:
        assert m._resolve_team_strict("backend") == "backend"
    finally:
        m._active_team_var.reset(token)


# ==================== kg_neighbors MCP tool (H004) ====================
#
# The validation / clamp / bad-entity / direction / chroma-unsupported cases
# stub _call_kg so they need no database and always run. The happy path drives
# the real per-team Postgres handle through _get_kg and uses the live-DB
# self-skip convention.

import os  # noqa: E402
import uuid  # noqa: E402

psycopg = pytest.importorskip("psycopg")  # noqa: E402

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.knowledge_graph_postgres import PostgresKnowledgeGraph  # noqa: E402


def _dsn():
    return (
        os.environ.get("MEMPALACE_TEST_PG_URL")
        or os.environ.get("MEMPALACE_DATABASE_URL")
        or "postgresql://mempalace:mempalace@localhost:5432/mempalace"
    )


def _reachable():
    try:
        with psycopg.connect(_dsn(), connect_timeout=3) as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


_PG_LIVE = pytest.mark.skipif(not _reachable(), reason="no reachable Postgres")


def test_kg_neighbors_clamps_depth_to_one_through_four(monkeypatch):
    """depth 0 clamps up to 1 and depth 5 clamps down to 4 (mirrors max_hops)."""
    seen = {}

    def _spy(op):
        class _KG:
            def neighbors(self, name, **kw):
                seen.update(kw)
                seen["name"] = name
                return {"neighbors": [], "truncated": False, "depth": kw["depth"]}

        return op(_KG())

    monkeypatch.setattr(m, "_call_kg", _spy)

    res = m.tool_kg_neighbors("Max", depth=0)
    assert seen["depth"] == 1
    assert res["depth"] == 1

    res = m.tool_kg_neighbors("Max", depth=5)
    assert seen["depth"] == 4
    assert res["depth"] == 4


def test_kg_neighbors_bad_entity_returns_structured_error(monkeypatch):
    """A blank/invalid entity is rejected as {"error": ...}, not an exception."""
    called = {"hit": False}
    monkeypatch.setattr(m, "_call_kg", lambda op: called.__setitem__("hit", True))

    res = m.tool_kg_neighbors("   ")
    assert "error" in res
    assert "unsupported" not in res
    assert called["hit"] is False  # rejected before any graph call


def test_kg_neighbors_bad_target_returns_structured_error(monkeypatch):
    """An invalid target is rejected as {"error": ...} before the graph call."""
    called = {"hit": False}
    monkeypatch.setattr(m, "_call_kg", lambda op: called.__setitem__("hit", True))

    res = m.tool_kg_neighbors("Max", target="   ")
    assert "error" in res
    assert called["hit"] is False


def test_kg_neighbors_invalid_direction_returns_structured_error(monkeypatch):
    """direction outside the allowed set yields a structured error."""
    called = {"hit": False}
    monkeypatch.setattr(m, "_call_kg", lambda op: called.__setitem__("hit", True))

    res = m.tool_kg_neighbors("Max", direction="sideways")
    assert res == {"error": "direction must be 'outgoing', 'incoming', or 'both'"}
    assert called["hit"] is False


def test_kg_neighbors_unsupported_on_chroma_backend_is_structured(monkeypatch):
    """The local SQLite stub's NotImplementedError surfaces as a structured
    {"error": ..., "unsupported": True}, never a crash and never empty."""

    def _raises(op):
        from mempalace.knowledge_graph import KnowledgeGraph

        # Drive the real SQLite stub path (H003): its neighbors() raises.
        return op(
            KnowledgeGraph.__new__(KnowledgeGraph)  # no DB file needed; stub raises first
        )

    monkeypatch.setattr(m, "_call_kg", _raises)

    res = m.tool_kg_neighbors("Max", depth=2)
    assert res.get("unsupported") is True
    assert "error" in res
    assert "Postgres" in res["error"]


# ==================== explicit-tunnel tools through the link-store seam (H010) ====
#
# The four explicit-tunnel CRUD tools (create / list / follow / delete) now route
# through get_link_store(_config, team): on chroma the seam returns the JSON store
# (byte-identical to the prior direct palace_graph calls); on postgres it returns
# the per-team PostgresLinkStore, with the team resolved in-request via the strict
# resolver (RAISE on None — no default vault). The no-DB tests stub the seam /
# palace_graph; the e2e + isolation tests use the live-DB self-skip convention.


def test_tunnel_tools_use_json_store_on_chroma(monkeypatch):
    """On chroma the tools route through the seam to the JSON store, which
    delegates to palace_graph verbatim — byte-identical to the prior direct
    calls. We assert the JSON delegate is hit and no team is required."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    token = m._active_team_var.set(None)  # no team available at all
    try:
        calls = {}

        def _spy_create(*a, **kw):
            calls["create"] = (a, kw)
            return {"id": "T1", "kind": "explicit"}

        monkeypatch.setattr("mempalace.palace_graph.create_tunnel", _spy_create)
        monkeypatch.setattr("mempalace.palace_graph.list_tunnels", lambda wing=None: [{"id": "T1"}])
        monkeypatch.setattr("mempalace.palace_graph.delete_tunnel", lambda tid: {"deleted": tid})
        monkeypatch.setattr(
            "mempalace.palace_graph.follow_tunnels",
            lambda wing, room, col=None, config=None: [{"direction": "outgoing"}],
        )
        monkeypatch.setattr(m, "_get_collection", lambda *a, **k: None)

        # create — no team needed on chroma (single vault); JSON delegate hit.
        res = m.tool_create_tunnel("wing_a", "room_a", "wing_b", "room_b", label="x")
        assert res == {"id": "T1", "kind": "explicit"}
        assert "create" in calls
        assert calls["create"][1]["label"] == "x"

        # list / delete / follow all route to the JSON delegates too.
        assert m.tool_list_tunnels() == [{"id": "T1"}]
        assert m.tool_delete_tunnel("T1") == {"deleted": "T1"}
        assert m.tool_follow_tunnels("wing_a", "room_a") == [{"direction": "outgoing"}]
    finally:
        m._active_team_var.reset(token)


def test_tunnel_tool_write_raises_on_postgres_without_resolvable_team(monkeypatch):
    """fail-loud: a tunnel-tool write on postgres with no explicit team and no
    active session team RAISES rather than routing into a default vault. No DB
    is touched — the strict resolver returns None before any store is built."""
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", "configured_default")
    token = m._active_team_var.set(None)
    try:
        # Sanity: the strict resolver is ambiguous here (no explicit, no active).
        assert m._resolve_team_strict() is None

        # The fail-loud raise propagates (it is NOT swallowed by the tool's
        # name-validation error handler).
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_create_tunnel("wing_a", "room_a", "wing_b", "room_b")
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_list_tunnels()
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_delete_tunnel("some_id")
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_follow_tunnels("wing_a", "room_a")
    finally:
        m._active_team_var.reset(token)


@_PG_LIVE
def test_tunnel_tools_round_trip_in_team_vault_on_postgres(pg_env):
    """e2e: create -> list -> follow -> delete through the tools operate in the
    team vault (PostgresLinkStore) and round-trip via the H008 store."""
    monkeypatch = pg_env
    team = "t" + uuid.uuid4().hex[:10]

    backend = PostgresBackend(dsn=_dsn())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: None)
    token = m._active_team_var.set(team)
    m._link_store_by_team.pop(f"pglink::{team}", None)
    try:
        created = m.tool_create_tunnel(
            "wing_code",
            "auth",
            "wing_people",
            "users",
            label="same concept",
            target_drawer_id="drawer_users_1",
        )
        assert "error" not in created
        assert created["kind"] == "explicit"
        assert created["label"] == "same concept"

        # list — visible through the tool.
        listed = m.tool_list_tunnels()
        assert [t["id"] for t in listed] == [created["id"]]

        # follow — outgoing from the source endpoint.
        connections = m.tool_follow_tunnels("wing_code", "auth")
        assert len(connections) == 1
        assert connections[0]["direction"] == "outgoing"
        assert connections[0]["connected_wing"] == "wing_people"
        assert connections[0]["tunnel_id"] == created["id"]

        # The row physically lives in the H008 per-team store.
        from mempalace.link_store_postgres import PostgresLinkStore

        store = PostgresLinkStore(backend, team=team)
        assert [t["id"] for t in store.list_tunnels()] == [created["id"]]

        # delete — through the tool; the vault is now empty.
        assert m.tool_delete_tunnel(created["id"]) == {"deleted": created["id"]}
        assert m.tool_list_tunnels() == []
    finally:
        m._active_team_var.reset(token)
        m._link_store_by_team.pop(f"pglink::{team}", None)
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
            conn.commit()
        backend.close()


@_PG_LIVE
def test_tunnel_tools_isolate_teams_on_postgres(pg_env):
    """second-team isolation: a tunnel created via team A's context is invisible
    to the same tools running in team B's context."""
    monkeypatch = pg_env
    team_a = "t" + uuid.uuid4().hex[:10]
    team_b = "t" + uuid.uuid4().hex[:10]

    backend = PostgresBackend(dsn=_dsn())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: None)
    for t in (team_a, team_b):
        m._link_store_by_team.pop(f"pglink::{t}", None)
    try:
        # Team A creates a tunnel.
        tok_a = m._active_team_var.set(team_a)
        try:
            created = m.tool_create_tunnel("wing_a", "r1", "wing_b", "r2", label="A-only")
            assert "error" not in created
        finally:
            m._active_team_var.reset(tok_a)

        # Team B sees nothing — the tools query team B's schema only.
        tok_b = m._active_team_var.set(team_b)
        try:
            assert m.tool_list_tunnels() == []
            assert m.tool_follow_tunnels("wing_a", "r1") == []
        finally:
            m._active_team_var.reset(tok_b)

        # Team A still sees its own tunnel.
        tok_a = m._active_team_var.set(team_a)
        try:
            assert [t["id"] for t in m.tool_list_tunnels()] == [created["id"]]
        finally:
            m._active_team_var.reset(tok_a)
    finally:
        for t in (team_a, team_b):
            m._link_store_by_team.pop(f"pglink::{t}", None)
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                for t in (team_a, team_b):
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
            conn.commit()
        backend.close()


@_PG_LIVE
def test_tunnel_tool_writes_no_host_global_json_on_postgres(pg_env, monkeypatch):
    """no host-global JSON: creating a tunnel via the tool on postgres lands in
    the PG table and never calls the host-global tunnels.json writer."""
    monkeypatch = pg_env
    team = "t" + uuid.uuid4().hex[:10]

    # The host-global JSON writer is palace_graph.create_tunnel (it serializes to
    # ~/.mempalace/tunnels.json via _save_tunnels). On the postgres path it must
    # NEVER be reached — the PostgresLinkStore writes a per-team table instead.
    json_writer = {"hit": False}
    monkeypatch.setattr(
        "mempalace.palace_graph.create_tunnel",
        lambda *a, **k: json_writer.__setitem__("hit", True) or {},
    )

    backend = PostgresBackend(dsn=_dsn())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: None)
    token = m._active_team_var.set(team)
    m._link_store_by_team.pop(f"pglink::{team}", None)
    try:
        created = m.tool_create_tunnel("wing_a", "r1", "wing_b", "r2", label="pg-only")
        assert "error" not in created

        # The host-global JSON writer was never invoked on the postgres path.
        assert json_writer["hit"] is False

        # The tunnel really lives in the team's PG table.
        from mempalace.link_store_postgres import PostgresLinkStore

        store = PostgresLinkStore(backend, team=team)
        assert [t["id"] for t in store.list_tunnels()] == [created["id"]]
    finally:
        m._active_team_var.reset(token)
        m._link_store_by_team.pop(f"pglink::{team}", None)
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
            conn.commit()
        backend.close()


def test_miner_post_processing_skips_host_global_json_on_postgres(monkeypatch, tmp_path):
    """miner gating: on a postgres config the miner's host-global link writers
    (topic tunnels / hallways / entity tunnels) are NOT invoked, and on chroma
    they ARE. Exercises the real miner._mine_impl post-mine block by driving a
    zero-file mine (no ChromaDB needed) and spying on the three compute helpers."""
    import mempalace.miner as miner

    called = {"topic": 0, "hallways": 0, "entity": 0}
    monkeypatch.setattr(
        miner,
        "_compute_topic_tunnels_for_wing",
        lambda wing: called.__setitem__("topic", called["topic"] + 1) or 0,
    )
    monkeypatch.setattr(
        miner,
        "compute_hallways_for_wing",
        lambda wing, col=None: called.__setitem__("hallways", called["hallways"] + 1) or [],
    )
    monkeypatch.setattr(
        miner,
        "_compute_entity_tunnels_for_wing",
        lambda wing: called.__setitem__("entity", called["entity"] + 1) or 0,
    )
    # The post-mine block also runs the FTS5 integrity check; stub it (no DB).
    monkeypatch.setattr(miner, "_validate_palace_fts5_after_mine", lambda path: None)
    # The post-mine block runs once after the (single-wing) file loop completes
    # without exception. Stub the file scan + per-file ingest + collection
    # openers so no ChromaDB is needed; process_file returns one drawer so the
    # loop body runs the normal success path.
    (tmp_path / "f.txt").write_text("hello world content for mining" * 5)
    monkeypatch.setattr(miner, "scan_project", lambda *a, **k: [tmp_path / "f.txt"])
    monkeypatch.setattr(miner, "get_collection", lambda path: object())
    monkeypatch.setattr(miner, "get_closets_collection", lambda path: object())
    monkeypatch.setattr(miner, "process_file", lambda **k: (1, "general", None))

    def _run(backend_name):
        monkeypatch.setenv("MEMPALACE_BACKEND", backend_name)
        # On the central backend a mine requires a resolvable team (fail-loud
        # before ingest); set one so the run reaches the post-mine gating under
        # test. Chroma ignores it.
        monkeypatch.setenv("MEMPALACE_TEAM", "tteam")
        called.update(topic=0, hallways=0, entity=0)
        miner._mine_impl(str(tmp_path), str(tmp_path / "palace"), wing_override="w")

    _run("postgres")
    assert called == {"topic": 0, "hallways": 0, "entity": 0}, (
        "postgres must NOT write host-global tunnels/hallways JSON"
    )

    _run("chroma")
    assert called == {"topic": 1, "hallways": 1, "entity": 1}, (
        "chroma must still write the host-global link layer (byte-identical)"
    )


@_PG_LIVE
def test_kg_neighbors_happy_path_multi_hop_against_live_pg(pg_env):
    """End-to-end multi-hop walk through the real per-team Postgres KG handle."""
    monkeypatch = pg_env
    team = "t" + uuid.uuid4().hex[:10]

    backend = PostgresBackend(dsn=_dsn())
    try:
        # Seed A -> B -> C -> D in this team's vault.
        seed = PostgresKnowledgeGraph(backend, team=team)
        seed.add_triple("A", "rel", "B")
        seed.add_triple("B", "rel", "C")
        seed.add_triple("C", "rel", "D")

        # Route the server's per-team handle at the same live backend + vault.
        monkeypatch.setenv("MEMPALACE_TEAM", team)
        monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
        token = m._active_team_var.set(team)
        m._kg_by_path.pop(f"pgkg::{team}", None)
        try:
            res = m.tool_kg_neighbors("A", depth=3, direction="outgoing")
        finally:
            m._active_team_var.reset(token)
            m._kg_by_path.pop(f"pgkg::{team}", None)

        assert res["entity"] == "A"
        assert res["depth"] == 3
        assert res["direction"] == "outgoing"
        assert res["truncated"] is False
        objects_by_hop = {(n["hop"], n["object"]) for n in res["neighbors"]}
        assert objects_by_hop == {(1, "B"), (2, "C"), (3, "D")}
        assert res["count"] == 3
    finally:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
            conn.commit()
        backend.close()
