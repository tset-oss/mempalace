"""Tests for the backend-aware link-store seam (mempalace.link_store).

The seam is the single route explicit-tunnel writers and hallway readers
take so the chroma (JSON) store and the team-scoped store cannot drift.
These tests assert:

  * JsonLinkStore round-trips create -> list -> follow -> delete through the
    seam with the SAME effect as calling palace_graph directly (same
    tunnels.json, same symmetric IDs, same dynamics preservation on
    re-create);
  * the hallway read path delegates to hallways.compute_hallways_for_wing;
  * get_link_store returns JsonLinkStore on chroma and raises
    NotImplementedError on postgres (the placeholder);
  * the fail-loud contract: a team-scoped write that resolves NO team via
    the strict resolver RAISES rather than writing to a default vault.

Conventions mirror tests/test_palace_graph_tunnels.py and
tests/test_hallways.py (chromadb mocked at import; per-test tmp JSON files).
"""

from unittest.mock import MagicMock, patch

import pytest

with patch.dict("sys.modules", {"chromadb": MagicMock()}):
    import mempalace.hallways as hallways_mod
    import mempalace.link_store as link_store
    import mempalace.palace_graph as palace_graph


def _use_tmp_tunnel_file(monkeypatch, tmp_path):
    """Redirect the tunnel-file resolver + legacy check at tmp_path and
    neutralize _get_collection (so endpoint-existence validation falls
    through to the permissive branch). Mirrors the helper in
    tests/test_palace_graph_tunnels.py."""
    tunnel_file = tmp_path / "tunnels.json"
    monkeypatch.setattr(palace_graph, "_get_tunnel_file", lambda *a, **kw: str(tunnel_file))
    monkeypatch.setattr(
        palace_graph,
        "_legacy_tunnel_file",
        lambda: str(tmp_path / "legacy-tunnels.json"),
    )
    monkeypatch.setattr(palace_graph, "_get_collection", lambda *a, **kw: None)
    return tunnel_file


def _use_tmp_hallway_file(monkeypatch, tmp_path):
    """Redirect hallway persistence to a per-test JSON file. Mirrors the
    helper in tests/test_hallways.py."""
    hallway_file = tmp_path / "hallways.json"
    monkeypatch.setattr(hallways_mod, "_get_hallway_file", lambda *a, **kw: str(hallway_file))
    monkeypatch.setattr(
        hallways_mod,
        "_legacy_hallway_file",
        lambda: str(tmp_path / "legacy-hallways.json"),
    )
    return hallway_file


def _fake_collection(drawers):
    """MagicMock collection over ``drawers`` supporting the paginated fetch
    (count() + get(limit=, offset=)) compute_hallways_for_wing uses."""
    col = MagicMock()
    metas = list(drawers)
    col.count.return_value = len(metas)

    def _get(limit=None, offset=0, include=None, where=None, ids=None, **kwargs):
        page = metas[offset : offset + limit] if limit is not None else metas
        return {
            "ids": [f"drawer_{i}" for i in range(offset, offset + len(page))],
            "metadatas": page,
        }

    col.get.side_effect = _get
    return col


class _FakeConfig:
    """Minimal stand-in for MempalaceConfig — only ``.backend`` is read."""

    def __init__(self, backend):
        self.backend = backend


# ─────────────────────────────────────────────────────────────────────────────
# Contract: JsonLinkStore round-trip through the seam matches direct calls
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonLinkStoreContract:
    def test_create_list_follow_delete_round_trip_through_seam(self, tmp_path, monkeypatch):
        tunnel_file = _use_tmp_tunnel_file(monkeypatch, tmp_path)
        store = link_store.JsonLinkStore()

        created = store.create_tunnel(
            "wing_code",
            "auth",
            "wing_people",
            "users",
            label="same concept",
            target_drawer_id="drawer_users_1",
        )

        # The seam wrote the same tunnels.json the direct path would, with the
        # symmetric canonical ID.
        assert tunnel_file.exists()
        assert created["id"] == palace_graph._canonical_tunnel_id(
            "wing_code", "auth", "wing_people", "users"
        )
        assert created["kind"] == "explicit"

        # list through the seam == direct list.
        seam_list = store.list_tunnels()
        assert seam_list == palace_graph.list_tunnels()
        assert len(seam_list) == 1

        # filter by either endpoint.
        assert len(store.list_tunnels("wing_people")) == 1
        assert len(store.list_tunnels("wing_code")) == 1

        # follow through the seam == direct follow.
        col = MagicMock()
        col.get.return_value = {
            "ids": ["drawer_users_1"],
            "documents": ["A" * 400],
            "metadatas": [{}],
        }
        connections = store.follow_tunnels("wing_code", "auth", col=col)
        assert len(connections) == 1
        assert connections[0]["direction"] == "outgoing"
        assert connections[0]["connected_wing"] == "wing_people"
        assert connections[0]["tunnel_id"] == created["id"]

        # delete through the seam.
        assert store.delete_tunnel(created["id"]) == {"deleted": created["id"]}
        assert store.list_tunnels() == []
        assert palace_graph.list_tunnels() == []

    def test_seam_create_equals_direct_create_same_file_effect(self, tmp_path, monkeypatch):
        """Creating via the seam produces the same stored record as creating
        the identical tunnel directly through palace_graph."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)

        direct = palace_graph.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="lbl")
        # Same canonical id => the seam re-creates (updates) the same record,
        # not a duplicate. Use a fresh label so the update is observable.
        seam = link_store.JsonLinkStore().create_tunnel(
            "wing_b", "room_y", "wing_a", "room_x", label="lbl2"
        )

        assert seam["id"] == direct["id"]
        assert len(palace_graph.list_tunnels()) == 1
        assert seam["label"] == "lbl2"

    def test_recreate_through_seam_preserves_dynamics(self, tmp_path, monkeypatch):
        """A second create through the seam must preserve accumulated L7
        dynamics — same guarantee as the direct create_tunnel path."""
        _use_tmp_tunnel_file(monkeypatch, tmp_path)
        store = link_store.JsonLinkStore()

        first = store.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="initial")

        # Simulate accumulated activity.
        stored = palace_graph._load_tunnels()
        for t in stored:
            if t["id"] == first["id"]:
                t["strength"] = 2.7
                t["access_count"] = 12
                t["stability"] = 1.5
        palace_graph._save_tunnels(stored)

        # Re-create through the seam with a new label.
        second = store.create_tunnel("wing_a", "room_x", "wing_b", "room_y", label="updated")
        assert second["id"] == first["id"]
        assert second["label"] == "updated"
        assert second["strength"] == 2.7
        assert second["access_count"] == 12
        assert second["stability"] == 1.5

    def test_hallway_read_delegates_to_hallways_module(self, tmp_path, monkeypatch):
        """compute_hallways_for_wing through the seam == direct hallways call,
        and list_hallways reflects the persisted records."""
        _use_tmp_hallway_file(monkeypatch, tmp_path)
        store = link_store.JsonLinkStore()

        drawers = [
            {"wing": "wing_aya", "room": "diary", "entities": "Aya;Lumi"},
            {"wing": "wing_aya", "room": "letters", "entities": "Aya;Lumi"},
        ]
        col = _fake_collection(drawers)

        hallways = store.compute_hallways_for_wing("wing_aya", col=col, min_count=2)
        assert len(hallways) == 1
        assert {hallways[0]["entity_a"], hallways[0]["entity_b"]} == {"Aya", "Lumi"}
        assert hallways[0]["co_occurrence_count"] == 2

        listed = store.list_hallways("wing_aya")
        assert listed == hallways_mod.list_hallways("wing_aya")
        assert len(listed) == 1

    def test_json_link_store_is_a_link_store(self):
        assert isinstance(link_store.JsonLinkStore(), link_store.LinkStore)


# ─────────────────────────────────────────────────────────────────────────────
# Resolver dispatch: chroma -> JsonLinkStore; postgres -> NotImplementedError
# ─────────────────────────────────────────────────────────────────────────────


class TestGetLinkStore:
    def test_chroma_returns_json_store(self):
        store = link_store.get_link_store(_FakeConfig("chroma"))
        assert isinstance(store, link_store.JsonLinkStore)

    def test_chroma_ignores_team_argument(self):
        """The chroma JSON store is host-local single-vault, so a team arg is
        accepted (uniform signature) but not required."""
        store = link_store.get_link_store(_FakeConfig("chroma"), team="frontend")
        assert isinstance(store, link_store.JsonLinkStore)

    def test_postgres_with_team_returns_postgres_store(self):
        """The postgres path now returns a per-team PostgresLinkStore. Backend
        construction is lazy (no connection until first use), so this builds the
        store without a live DB."""
        from mempalace.link_store_postgres import PostgresLinkStore

        store = link_store.get_link_store(_FakeConfig("postgres"), team="frontend")
        assert isinstance(store, PostgresLinkStore)

    def test_postgres_without_team_raises(self):
        """On postgres a team is MANDATORY (reads AND writes are team-scoped).
        A None team fails loud rather than routing into a default vault."""
        with pytest.raises(ValueError, match="no team resolved"):
            link_store.get_link_store(_FakeConfig("postgres"), team=None)


# ─────────────────────────────────────────────────────────────────────────────
# Fail loud: a team-scoped write with no resolved team RAISES (no default vault)
# ─────────────────────────────────────────────────────────────────────────────


class TestFailLoudNoTeam:
    def test_require_write_team_raises_on_none(self):
        """The seam's fail-loud check raises when no team is resolved, rather
        than routing the write into a default vault."""
        with pytest.raises(ValueError, match="no team resolved"):
            link_store.require_write_team(None)

    def test_require_write_team_returns_slug_when_set(self):
        assert link_store.require_write_team("frontend") == "frontend"

    def test_strict_resolver_with_no_team_makes_seam_write_raise(self, monkeypatch):
        """End-to-end fail-loud contract: with NO explicit team and NO active
        session team, the strict resolver returns None, and feeding that to the
        seam's write-team check RAISES — it does NOT write to a default vault.

        This guards against the strict-resolver RAISE path becoming dead code:
        even though the team-scoped store lands later, the contract is
        established here.
        """
        import mempalace.mcp_server as m

        # No explicit team, and force no active session team.
        token = m._active_team_var.set(None)
        try:
            resolved = m._resolve_team_strict(explicit=None)
            assert resolved is None
            with pytest.raises(ValueError, match="no team resolved"):
                link_store.require_write_team(resolved)
        finally:
            m._active_team_var.reset(token)

    def test_strict_resolver_with_active_team_resolves_for_seam_write(self, monkeypatch):
        """Sanity counter-case: when an active session team IS set, the strict
        resolver returns it and the seam's write-team check passes it through."""
        import mempalace.mcp_server as m

        token = m._active_team_var.set("frontend")
        try:
            resolved = m._resolve_team_strict(explicit=None)
            assert resolved == "frontend"
            assert link_store.require_write_team(resolved) == "frontend"
        finally:
            m._active_team_var.reset(token)
