"""Tests for the FastMCP server surface + per-session team routing (G001/G002).

Two layers are covered:

* Registration parity — the FastMCP server exposes every legacy tool (plus the
  new ``switch_team``) and the handler signatures survive FastMCP's schema
  derivation through the per-session wrapper.
* Team routing — the resolution chain (vault arg > session/switch_team > header
  > server default) and per-session isolation, exercised against the routing
  primitives directly so no live HTTP server / database is required.

These run in chroma/local mode and self-skip when the optional ``serve`` extra
(the ``mcp`` SDK) is not installed.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp.server.fastmcp")

from starlette.datastructures import Headers  # noqa: E402  (provided by mcp[serve])

from mempalace import mcp_fastmcp  # noqa: E402
from mempalace import mcp_server  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Registration parity
# --------------------------------------------------------------------------- #


def test_registers_every_legacy_tool_plus_switch_team():
    server = mcp_fastmcp.build_server()
    tools = _run(server.list_tools())
    names = {t.name for t in tools}
    # Every legacy tool is present...
    assert set(mcp_server.TOOLS).issubset(names), (
        "FastMCP must register every tool from the legacy TOOLS registry"
    )
    # ...plus the per-session switch_team tool, and nothing else.
    assert names == set(mcp_server.TOOLS) | {"mempalace_switch_team"}
    assert len(tools) == len(mcp_server.TOOLS) + 1


def test_every_tool_has_input_schema():
    server = mcp_fastmcp.build_server()
    for t in _run(server.list_tools()):
        assert isinstance(t.inputSchema, dict), f"{t.name} has no input schema"
        assert t.description, f"{t.name} has no description"


def test_list_vaults_roundtrips_through_sdk():
    # chroma/local mode: list_vaults reports the single 'local' vault.
    server = mcp_fastmcp.build_server()
    result = _run(server.call_tool("mempalace_list_vaults", {}))
    text = _extract_text(result)
    assert "local" in text


def test_search_schema_exposes_vault_param():
    # The vault routing param must survive the FastMCP schema derivation from
    # the handler signature even though the handler is wrapped for routing.
    server = mcp_fastmcp.build_server()
    search = next(t for t in _run(server.list_tools()) if t.name == "mempalace_search")
    props = search.inputSchema.get("properties", {})
    assert "vault" in props
    assert "query" in props


def test_switch_team_schema():
    server = mcp_fastmcp.build_server()
    sw = next(t for t in _run(server.list_tools()) if t.name == "mempalace_switch_team")
    assert "team" in sw.inputSchema.get("properties", {})
    assert sw.description


# --------------------------------------------------------------------------- #
# Per-session team routing — fakes for the request context
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Stand-in for an MCP ServerSession: a weak-referenceable identity key."""


class _FakeRequest:
    def __init__(self, headers):
        # Real Starlette Headers so the case-insensitive lookup is exercised.
        self.headers = Headers(headers) if headers is not None else None


class _FakeRequestContext:
    def __init__(self, session, request):
        self.session = session
        self.request = request


class _FakeCtx:
    """Minimal Context shaped like FastMCP's, for the routing primitives."""

    def __init__(self, session, headers=None, has_request=True):
        self._session = session
        request = _FakeRequest(headers) if has_request else None
        self._rc = _FakeRequestContext(session, request)

    @property
    def session(self):
        return self._session

    @property
    def request_context(self):
        return self._rc


def test_header_seeds_active_team():
    ctx = _FakeCtx(_FakeSession(), headers={"X-Mempalace-Team": "Frontend"})

    def probe():
        return mcp_server._active_team_var.get()

    wrapped = mcp_fastmcp._session_wrap(lambda: ctx, probe)
    # Header (any case) is normalised and seeded for the call duration...
    assert wrapped() == "frontend"
    # ...and the contextvar is restored afterwards (no leak between calls).
    assert mcp_server._active_team_var.get() is None


def test_session_team_overrides_header():
    session = _FakeSession()
    ctx = _FakeCtx(session, headers={"X-Mempalace-Team": "frontend"})
    mcp_fastmcp._get_or_create_session(ctx).active_team = "backend"

    def probe():
        return mcp_server._active_team_var.get()

    wrapped = mcp_fastmcp._session_wrap(lambda: ctx, probe)
    assert wrapped() == "backend"  # switch_team beats the header


def test_invalid_header_is_ignored():
    ctx = _FakeCtx(_FakeSession(), headers={"X-Mempalace-Team": "../etc/passwd"})
    assert mcp_fastmcp._seed_for_request(lambda: ctx) is None


def test_no_request_means_no_header_seed():
    # stdio transport: request is None -> header path yields nothing.
    ctx = _FakeCtx(_FakeSession(), has_request=False)
    assert mcp_fastmcp._header_team(ctx) is None
    assert mcp_fastmcp._seed_for_request(lambda: ctx) is None


def test_sessions_are_isolated():
    s1, s2 = _FakeSession(), _FakeSession()
    ctx1, ctx2 = _FakeCtx(s1), _FakeCtx(s2)

    state1 = mcp_fastmcp._get_or_create_session(ctx1)
    state2 = mcp_fastmcp._get_or_create_session(ctx2)
    assert state1 is not state2, "distinct sessions must get distinct state"
    assert mcp_fastmcp._get_or_create_session(ctx1) is state1, "same session is stable"

    # A switch in session 1 must not bleed into session 2.
    state1.active_team = "backend"
    assert mcp_fastmcp._seed_for_request(lambda: ctx1) == "backend"
    assert mcp_fastmcp._seed_for_request(lambda: ctx2) is None


def test_switch_team_sets_and_resets_session_state():
    session = _FakeSession()
    ctx = _FakeCtx(session)
    switch = mcp_fastmcp._make_switch_team(lambda: ctx)

    assert switch("backend") == {"ok": True, "active_team": "backend"}
    assert mcp_fastmcp._get_or_create_session(ctx).active_team == "backend"

    reset = switch("")
    assert reset["ok"] is True and reset["active_team"] is None
    assert mcp_fastmcp._get_or_create_session(ctx).active_team is None


def test_switch_team_rejects_all_and_invalid():
    ctx = _FakeCtx(_FakeSession())
    switch = mcp_fastmcp._make_switch_team(lambda: ctx)
    assert switch("all")["ok"] is False  # use vault='all' on search instead
    assert switch("../bad")["ok"] is False
    # A rejected switch must not mutate session state.
    assert mcp_fastmcp._get_or_create_session(ctx).active_team is None


def test_wrapper_restores_active_team_on_exception():
    ctx = _FakeCtx(_FakeSession(), headers={"X-Mempalace-Team": "frontend"})

    def boom():
        raise RuntimeError("handler blew up")

    wrapped = mcp_fastmcp._session_wrap(lambda: ctx, boom)
    with pytest.raises(RuntimeError):
        wrapped()
    # contextvar must be reset even when the handler raises.
    assert mcp_server._active_team_var.get() is None


def test_resolve_team_precedence():
    # vault arg > session/header (contextvar) > server default.
    assert mcp_server._resolve_team("frontend") == "frontend"
    token = mcp_server._active_team_var.set("backend")
    try:
        assert mcp_server._resolve_team() == "backend"  # session beats default
        assert mcp_server._resolve_team("ml") == "ml"  # explicit beats session
        assert mcp_server._resolve_team("default") == "backend"  # alias = no override
    finally:
        mcp_server._active_team_var.reset(token)
    # With nothing set, falls back to the process default ("default" here).
    assert mcp_server._resolve_team() == "default"


def test_valid_team_canonicalises_or_rejects():
    # Canonical: lowercased, [a-z0-9_], 1-40 chars.
    assert mcp_server._valid_team("Frontend") == "frontend"
    assert mcp_server._valid_team("team_a") == "team_a"
    # Aliases mean "no override".
    assert mcp_server._valid_team("primary") is None
    assert mcp_server._valid_team("default") is None
    assert mcp_server._valid_team("") is None
    # Non-canonical is rejected (NOT silently rewritten) so it can't route to a
    # surprise vault or collide with another at the SQL-sanitisation boundary.
    assert mcp_server._valid_team("front-end") is None
    assert mcp_server._valid_team("../etc/passwd") is None
    assert mcp_server._valid_team("a" * 41) is None
    # Internal underscores are fine, but leading/trailing are not — they would
    # diverge from sanitize_team's strip("_") schema slug (e.g. "_a_" -> team_a).
    assert mcp_server._valid_team("team_a") == "team_a"
    assert mcp_server._valid_team("_a_") is None
    assert mcp_server._valid_team("a_") is None
    assert mcp_server._valid_team("_a") is None


def test_resolve_team_rejects_malformed_explicit_and_falls_back():
    # A malformed explicit vault falls back to the default rather than routing
    # to a rewritten slug — so the returned value always equals the real schema.
    assert mcp_server._resolve_team("front-end") == "default"
    assert mcp_server._resolve_team("DROP TABLE") == "default"
    token = mcp_server._active_team_var.set("backend")
    try:
        # Malformed explicit -> falls back to the session team, not a rewrite.
        assert mcp_server._resolve_team("bad value!") == "backend"
    finally:
        mcp_server._active_team_var.reset(token)


def _extract_text(result) -> str:
    content = result[0] if isinstance(result, tuple) else result
    parts = []
    for item in content:
        parts.append(getattr(item, "text", "") or "")
    return "\n".join(parts)
