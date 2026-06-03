"""FastMCP server for MemPalace — stdio today, streamable-HTTP for central hosting.

Wraps the existing, battle-tested MCP tool handlers (``mcp_server.TOOLS``) onto a
FastMCP server. Tools are registered programmatically from the ``TOOLS`` registry
via :meth:`FastMCP.add_tool`, so the handler bodies — and the tests that exercise
them directly — are unchanged; only the transport/registration layer is new.

This is the basis for the centrally-hosted, team-vaulted deployment: the same
server runs over stdio for a local install and over streamable-HTTP for the
central host (``mempalace serve --transport streamable-http``).

Per-session team routing
-------------------------
The central server is one shared instance for the whole org, so the active team
vault must be resolved *per session*, never process-globally. Mirroring loop-cli's
``SessionState`` / ``switch_repo`` pattern, each MCP session gets its own
:class:`_TeamSession`. For every tool call this module seeds
``mcp_server._active_team_var`` (a contextvar the handlers read via
``_resolve_team``) from, in order:

    session.active_team (set by switch_team)  >  X-Mempalace-Team request header

The header lets the MCP *client config* drive the default with zero agent effort:
a ``user``-scope server entry (``~/.claude.json``) is the developer's machine-wide
default; a ``project``-scope entry (committed ``.mcp.json``) is the repo's default.
``switch_team`` overrides it at runtime; a ``vault=`` argument on an individual
call overrides everything. When neither a session team nor a header is present the
seed is ``None`` and the handler falls back to the server's process default
(``--default-team`` / ``MEMPALACE_TEAM``) — which is also exactly the stdio /
single-user behavior.

Requires the ``serve`` extra (the ``mcp`` SDK): ``pip install 'mempalace[serve]'``.
"""

from __future__ import annotations

import functools
import weakref
from dataclasses import dataclass

from . import mcp_server as _legacy

# Agent self-use protocol — surfaced to the model via FastMCP's instructions.
# Mirrors loop-cli's proactive-use framing, adapted for a memory system.
INSTRUCTIONS = (
    "MemPalace is your long-term memory — not just a Q&A store. Use it PROACTIVELY:\n"
    "recall before you answer from scratch, and file durable facts as you learn them.\n\n"
    "TEAM VAULT ROUTING (central deployments):\n"
    "Call mempalace_list_vaults() once at session start to see the available team\n"
    "vaults and which one is this session's primary. By default you read from and\n"
    "write to that primary vault (set by this machine's / this repo's MCP config).\n"
    "- To work in a different team's vault for the rest of the session, call\n"
    "  mempalace_switch_team(team='<team>'); call it with no team to reset to the\n"
    "  configured default.\n"
    "- To target one call only, pass vault='<team>' on mempalace_search /\n"
    "  mempalace_add_drawer.\n"
    "- To read across every team vault at once, pass vault='all' on mempalace_search.\n\n"
    "AT SESSION START: mempalace_search for prior context on the task/people involved.\n"
    "AS YOU LEARN: mempalace_add_drawer to file verbatim facts; mempalace_kg_add to\n"
    "record relationships (subject -> predicate -> object) with temporal validity.\n"
    "BEFORE assuming: mempalace_kg_query / mempalace_search rather than guessing."
)

_SWITCH_TEAM_DESC = (
    "Set the active team vault for THIS session (mirrors list_vaults' primary). "
    "All subsequent reads/writes route to it until you switch again. Call with no "
    "team (or 'default') to reset to this machine's/repo's configured default. To "
    "read across all vaults use vault='all' on search instead; to target a single "
    "call use that call's vault= argument."
)

# Team-name validation reuses mcp_server._TEAM_SLUG_RE — the single source of
# truth (the sanitize_team fixed-point set). A malformed header / switch_team
# value is dropped (falls back to the default), never silently rewritten, so
# routing never surprises the caller and never disagrees with the schema slug.
_RESET_ALIASES = {"", "primary", "default"}


@dataclass
class _TeamSession:
    """Per-MCP-session routing state. One instance per connected client."""

    active_team: str | None = None


# Per-session registry, keyed on the MCP ServerSession object so state is
# garbage-collected when the client disconnects. The sentinel covers calls made
# outside a live request context (direct unit-test calls, or a transport that
# exposes no session).
_sessions: "weakref.WeakKeyDictionary[object, _TeamSession]" = weakref.WeakKeyDictionary()
_STDIO_SESSION = _TeamSession()


def _clean_team(value) -> str | None:
    """Normalise an external team value to a safe slug, or ``None`` if unusable.

    ``None`` means "no override" — reset/blank/``primary``/``default`` and any
    value outside the safe identifier class all collapse to ``None`` so the
    caller falls back to the session/server default.
    """
    if not value:
        return None
    t = str(value).strip().lower()
    if t in _RESET_ALIASES or not _legacy._TEAM_SLUG_RE.match(t):
        return None
    return t


def _get_or_create_session(ctx) -> _TeamSession:
    """Return the :class:`_TeamSession` for *ctx*'s MCP session.

    Falls back to the module sentinel when no live session is available (e.g.
    the context was created outside a request, as in direct unit-test calls).
    """
    try:
        session_key = ctx.session
    except (ValueError, AttributeError):
        return _STDIO_SESSION
    if session_key is None:
        return _STDIO_SESSION
    return _sessions.setdefault(session_key, _TeamSession())


def _header_team(ctx) -> str | None:
    """Read + sanitise the ``X-Mempalace-Team`` header from *ctx* (HTTP only)."""
    try:
        request = ctx.request_context.request
    except (ValueError, AttributeError):
        return None
    if request is None:  # stdio transport / no HTTP request
        return None
    try:
        raw = request.headers.get("x-mempalace-team")
    except AttributeError:
        return None
    return _clean_team(raw)


def _seed_for_request(get_context) -> str | None:
    """Compute this request's active-team seed: session override, else header."""
    try:
        ctx = get_context()
    except Exception:
        return None
    session = _get_or_create_session(ctx)
    if session.active_team:
        return session.active_team
    return _header_team(ctx)


def _session_wrap(get_context, fn):
    """Wrap a legacy handler so the per-session team is bound for the call.

    Sets ``mcp_server._active_team_var`` from :func:`_seed_for_request` for the
    duration of the call, then restores it. ``functools.wraps`` keeps the
    original signature/annotations so FastMCP derives the same input schema it
    would from the bare handler — the routing is transparent to the schema.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _legacy._active_team_var.set(_seed_for_request(get_context))
        try:
            return fn(*args, **kwargs)
        finally:
            _legacy._active_team_var.reset(token)

    return wrapper


def _make_switch_team(get_context):
    """Build the ``mempalace_switch_team`` tool bound to this server's context."""

    def mempalace_switch_team(team: str = "") -> dict:
        session = _get_or_create_session(get_context())
        raw = (team or "").strip().lower()
        if raw in _RESET_ALIASES:
            session.active_team = None
            return {
                "ok": True,
                "active_team": None,
                "note": "reset to the configured default (header / server default)",
            }
        if raw == "all":
            return {
                "ok": False,
                "error": (
                    "switch_team sets a single active vault; for cross-team reads "
                    "pass vault='all' on mempalace_search instead."
                ),
            }
        if not _legacy._TEAM_SLUG_RE.match(raw):
            return {
                "ok": False,
                "error": (
                    f"invalid team name {team!r}; allowed: [a-z0-9_], 1-40 chars, "
                    "no leading/trailing underscore"
                ),
            }
        session.active_team = raw
        return {"ok": True, "active_team": raw}

    return mempalace_switch_team


def build_server():
    """Construct a FastMCP server with every MemPalace tool registered.

    Built lazily (not at import) so importing this module does not require the
    ``serve`` extra unless a server is actually constructed.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mempalace", instructions=INSTRUCTIONS)
    get_context = mcp.get_context
    for name, spec in _legacy.TOOLS.items():
        mcp.add_tool(
            _session_wrap(get_context, spec["handler"]),
            name=name,
            description=spec["description"],
        )
    mcp.add_tool(
        _make_switch_team(get_context),
        name="mempalace_switch_team",
        description=_SWITCH_TEAM_DESC,
    )
    return mcp


def main() -> None:
    """Run the FastMCP server over stdio (parity with the legacy entry point)."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
