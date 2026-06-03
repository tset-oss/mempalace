"""DB-less wiring tests for entity-scoped search + the entities tool (G004).

These mock the entity index / search layer, so they need no Postgres and always
run. They verify that tool_search resolves an entity to a drawer-id pre-filter
and short-circuits on no match, and that the entities tool degrades on chroma.
"""

from __future__ import annotations

import mempalace.mcp_server as m


class _FakeIndex:
    def __init__(self, rows):
        self._rows = rows

    def drawers_for_entity(self, entity):
        return list(self._rows)


def test_tool_search_entity_passes_restrict_ids(monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr(m, "_resolve_team", lambda v=None: "frontend")
    monkeypatch.setattr(
        m,
        "_get_entity_index",
        lambda team=None: _FakeIndex(
            [{"drawer_id": "d1", "wing": "w", "room": "r"},
             {"drawer_id": "d2", "wing": "w", "room": "r"}]
        ),
    )
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return {"results": [], "count": 0}

    monkeypatch.setattr(m, "search_memories", fake_search)

    m.tool_search("retry policy", entity="Dana")
    assert captured.get("restrict_ids") == ["d1", "d2"]


def test_tool_search_entity_no_match_short_circuits(monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr(m, "_resolve_team", lambda v=None: "frontend")
    monkeypatch.setattr(m, "_get_entity_index", lambda team=None: _FakeIndex([]))
    called = {"search": False}
    monkeypatch.setattr(
        m, "search_memories", lambda *a, **k: called.__setitem__("search", True)
    )

    res = m.tool_search("anything", entity="Ghost")
    assert res["results"] == [] and res["count"] == 0
    assert res["entity"] == "Ghost"
    assert called["search"] is False  # no unscoped search when the filter matches nothing


def test_tool_search_without_entity_does_not_restrict(monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr(m, "_resolve_team", lambda v=None: "frontend")
    captured = {}
    monkeypatch.setattr(
        m, "search_memories", lambda *a, **k: (captured.update(k), {"results": [], "count": 0})[1]
    )
    m.tool_search("a query")
    assert captured.get("restrict_ids") is None


def test_tool_entities_unavailable_on_chroma(monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    res = m.tool_entities()
    assert res["available"] is False
    assert res["backend"] == "chroma"


def test_cli_reindex_entities_is_postgres_only(monkeypatch, capsys):
    from types import SimpleNamespace

    from mempalace.cli import cmd_reindex_entities

    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    cmd_reindex_entities(SimpleNamespace(vault=None, all_vaults=False))
    out = capsys.readouterr().out
    assert "postgres backend only" in out
