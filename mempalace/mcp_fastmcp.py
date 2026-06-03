"""FastMCP server for MemPalace — stdio today, streamable-HTTP for central hosting.

Wraps the existing, battle-tested MCP tool handlers (``mcp_server.TOOLS``) onto a
FastMCP server. Tools are registered programmatically from the ``TOOLS`` registry
via :meth:`FastMCP.add_tool`, so the handler bodies — and the tests that exercise
them directly — are unchanged; only the transport/registration layer is new.

This is the basis for the centrally-hosted, team-vaulted deployment: the same
server runs over stdio for a local install and over streamable-HTTP for the
central host. Per-session team routing (an ``active_team`` seeded from a request
header + a ``switch_team`` tool) and the HTTP ``mempalace serve`` command land in
the follow-up serve story; this module establishes the FastMCP surface and keeps
stdio parity with the legacy server.

Requires the ``serve`` extra (the ``mcp`` SDK): ``pip install 'mempalace[serve]'``.
"""

from __future__ import annotations

from . import mcp_server as _legacy

# Agent self-use protocol — surfaced to the model via FastMCP's instructions.
# Mirrors loop-cli's proactive-use framing, adapted for a memory system.
INSTRUCTIONS = (
    "MemPalace is your long-term memory — not just a Q&A store. Use it PROACTIVELY:\n"
    "recall before you answer from scratch, and file durable facts as you learn them.\n\n"
    "TEAM VAULT ROUTING (central deployments):\n"
    "Call mempalace_list_vaults() once at session start to see the available team\n"
    "vaults and which one is this machine's primary. Memories file into and recall\n"
    "from your primary team vault by default. To reach another team, pass\n"
    "vault='<team>' on mempalace_search / mempalace_add_drawer; pass vault='all' on\n"
    "search to read across every team vault.\n\n"
    "AT SESSION START: mempalace_search for prior context on the task/people involved.\n"
    "AS YOU LEARN: mempalace_add_drawer to file verbatim facts; mempalace_kg_add to\n"
    "record relationships (subject -> predicate -> object) with temporal validity.\n"
    "BEFORE assuming: mempalace_kg_query / mempalace_search rather than guessing."
)


def build_server():
    """Construct a FastMCP server with every MemPalace tool registered.

    Built lazily (not at import) so importing this module does not require the
    ``serve`` extra unless a server is actually constructed.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mempalace", instructions=INSTRUCTIONS)
    for name, spec in _legacy.TOOLS.items():
        mcp.add_tool(spec["handler"], name=name, description=spec["description"])
    return mcp


def main() -> None:
    """Run the FastMCP server over stdio (parity with the legacy entry point)."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
