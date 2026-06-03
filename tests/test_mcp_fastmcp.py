"""Tests for the FastMCP server surface (G001 of the central-HTTP-server phase).

Verify the FastMCP server registers every legacy tool with parity and that calls
round-trip through the SDK. These run without a database (chroma/local mode) and
self-skip when the optional `serve` extra (the `mcp` SDK) is not installed.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp.server.fastmcp")

from mempalace import mcp_fastmcp  # noqa: E402
from mempalace import mcp_server  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_registers_every_legacy_tool():
    server = mcp_fastmcp.build_server()
    tools = _run(server.list_tools())
    names = {t.name for t in tools}
    assert names == set(mcp_server.TOOLS), (
        "FastMCP tool set must match the legacy TOOLS registry exactly"
    )
    assert len(tools) == len(mcp_server.TOOLS)


def test_every_tool_has_input_schema():
    server = mcp_fastmcp.build_server()
    for t in _run(server.list_tools()):
        assert isinstance(t.inputSchema, dict), f"{t.name} has no input schema"
        assert t.description, f"{t.name} has no description"


def test_list_vaults_roundtrips_through_sdk():
    # chroma/local mode: list_vaults reports the single 'local' vault.
    server = mcp_fastmcp.build_server()
    result = _run(server.call_tool("mempalace_list_vaults", {}))
    # FastMCP returns a content list (or a (content, structured) tuple depending
    # on SDK version); normalise to the text payload and assert the handler ran.
    text = _extract_text(result)
    assert "local" in text


def test_search_schema_exposes_vault_param():
    # The vault routing param (added in the postgres/team work) must survive the
    # FastMCP schema derivation from the handler signature.
    server = mcp_fastmcp.build_server()
    search = next(t for t in _run(server.list_tools()) if t.name == "mempalace_search")
    props = search.inputSchema.get("properties", {})
    assert "vault" in props
    assert "query" in props


def _extract_text(result) -> str:
    # call_tool may return a list[Content] or a (list[Content], structured) tuple.
    content = result[0] if isinstance(result, tuple) else result
    parts = []
    for item in content:
        parts.append(getattr(item, "text", "") or "")
    return "\n".join(parts)
