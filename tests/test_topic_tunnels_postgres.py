"""Integration tests for the server-derived TOPIC tunnels (Postgres parity).

Topic tunnels link two wings that share one or more TOPIC labels. The MATCHING
is pure case-insensitive string overlap of per-wing labels (the unchanged
``palace_graph.compute_topic_tunnels``) — there is NO LLM at tunnel time. The
labels are agent/miner-supplied: an agent passes ``topics`` on
``tool_add_drawer`` (or a ``topic`` on ``tool_diary_write``), and on the central
backend the miner persists confirmed topics too. On chroma the labels live in
the host-global ``known_entities.json["topics_by_wing"]``; on the central
multi-team backend that one host file is a cross-tenant leak, so each team's
labels live in its own ``team_<slug>.wing_topics`` table and topic tunnels are
derived per team via the SAME debounced ``(team, wing)`` rebuild.

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.
Each test uses a per-test team and drops its schema CASCADE on teardown.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import team_schema  # noqa: E402
from mempalace.link_store_postgres import PostgresLinkStore  # noqa: E402
from mempalace.palace_graph import (  # noqa: E402
    _normalize_topic,
    compute_topic_tunnels,
)


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
    from mempalace.backends.postgres import PostgresBackend

    b = PostgresBackend(dsn=_dsn())
    yield b
    b.close()


@pytest.fixture
def team(backend):
    name = "topic_" + uuid.uuid4().hex[:12]
    yield name
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')


def _topic_tunnels(store):
    return [t for t in store.list_tunnels() if t["kind"] == "topic"]


# ─────────────────────────────────────────────────────────────────────────────
# wing_topics population + topic-tunnel derivation (the string-overlap matcher)
# ─────────────────────────────────────────────────────────────────────────────


def test_add_topics_populates_wing_topics_team_scoped(backend, team):
    store = PostgresLinkStore(backend, team=team)
    store.add_topics("wing_aya", ["Angular", "OpenAPI"])
    store.add_topics("wing_work", ["openapi"])  # different casing

    topics_map = store.topics_by_wing()
    assert topics_map["wing_aya"] == ["Angular", "OpenAPI"]
    assert topics_map["wing_work"] == ["openapi"]

    # The labels physically live in this team's wing_topics table only.
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM "{team_schema(team)}"."wing_topics"')
            assert cur.fetchone()[0] == 3


def test_topic_overlap_derives_topic_tunnel(backend, team):
    """Two wings sharing a label (case-insensitively) get a kind='topic' tunnel."""
    store = PostgresLinkStore(backend, team=team)
    store.add_topics("wing_aya", ["Angular", "OpenAPI"])
    store.add_topics("wing_work", ["openapi"])  # overlaps OpenAPI case-insensitively

    result = store.rebuild_derived_links_for_wing("wing_aya")
    assert result["topic_tunnels"] >= 1

    tunnels = _topic_tunnels(store)
    assert len(tunnels) == 1
    t = tunnels[0]
    assert {t["source"]["wing"], t["target"]["wing"]} == {"wing_aya", "wing_work"}
    # The synthetic room is topic:<casing>; the overlap is on the normalized key.
    assert _normalize_topic("OpenAPI") in t["source"]["room"].lower()


def test_no_overlap_no_topic_tunnel(backend, team):
    store = PostgresLinkStore(backend, team=team)
    store.add_topics("wing_aya", ["Angular"])
    store.add_topics("wing_work", ["Django"])
    store.rebuild_derived_links_for_wing("wing_aya")
    assert _topic_tunnels(store) == []


# ─────────────────────────────────────────────────────────────────────────────
# Per-team isolation: team A's topics never link team B's wings
# ─────────────────────────────────────────────────────────────────────────────


def test_topic_tunnels_isolated_per_team(backend):
    team_a = "topic_" + uuid.uuid4().hex[:12]
    team_b = "topic_" + uuid.uuid4().hex[:12]
    try:
        store_a = PostgresLinkStore(backend, team=team_a)
        store_b = PostgresLinkStore(backend, team=team_b)

        # Both teams use the same wing names + same shared topic.
        store_a.add_topics("wing_x", ["Shared"])
        store_a.add_topics("wing_y", ["Shared"])
        # Team B only has ONE wing carrying the topic -> no overlap of its own.
        store_b.add_topics("wing_x", ["Shared"])

        store_a.rebuild_derived_links_for_wing("wing_x")
        store_b.rebuild_derived_links_for_wing("wing_x")

        # Team A derives a topic tunnel from its two wings.
        assert len(_topic_tunnels(store_a)) == 1
        # Team B sees neither team A's labels nor a tunnel (its wing_y is absent).
        assert _topic_tunnels(store_b) == []
        assert "wing_y" not in store_b.topics_by_wing()
    finally:
        with backend._conn() as conn:
            with conn.cursor() as cur:
                for t in (team_a, team_b):
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# The three tunnel kinds coexist + purge INDEPENDENTLY per wing
# ─────────────────────────────────────────────────────────────────────────────


def test_topic_rebuild_preserves_explicit_and_entity_tunnels(backend, team):
    """A topic-tunnel rebuild must not purge explicit or entity tunnels."""
    from mempalace.entity_index_postgres import PostgresEntityIndex

    store = PostgresLinkStore(backend, team=team)
    idx = PostgresEntityIndex(backend, team=team)

    # An explicit (user-authored) tunnel between two wings.
    store.create_tunnel("wing_aya", "diary", "wing_work", "notes", label="see also")
    # An entity tunnel substrate: the same pair co-occurs in BOTH wings.
    idx.add(["a1"], ["Aya", "Lumi"], "wing_aya", "diary")
    idx.add(["a2"], ["Aya", "Lumi"], "wing_aya", "letters")
    idx.add(["b1"], ["Aya", "Lumi"], "wing_work", "notes")
    idx.add(["b2"], ["Aya", "Lumi"], "wing_work", "tasks")
    # Topic labels shared by the two wings.
    store.add_topics("wing_aya", ["Angular"])
    store.add_topics("wing_work", ["angular"])

    # Build both wings so entity + topic tunnels all materialize.
    store.rebuild_derived_links_for_wing("wing_aya", min_count=2)
    store.rebuild_derived_links_for_wing("wing_work", min_count=2)

    kinds_before = sorted(t["kind"] for t in store.list_tunnels())
    assert "explicit" in kinds_before
    assert "entity" in kinds_before
    assert "topic" in kinds_before

    explicit_before = [t for t in store.list_tunnels() if t["kind"] == "explicit"]
    entity_before = [t for t in store.list_tunnels() if t["kind"] == "entity"]
    topic_before = _topic_tunnels(store)
    assert len(explicit_before) == 1
    assert len(entity_before) >= 1
    assert len(topic_before) == 1

    # Re-running the rebuild purges + re-derives ONLY entity + topic tunnels for
    # the wing; the explicit tunnel and the other kinds survive unchanged.
    store.rebuild_derived_links_for_wing("wing_aya", min_count=2)
    store.rebuild_derived_links_for_wing("wing_work", min_count=2)

    explicit_after = [t for t in store.list_tunnels() if t["kind"] == "explicit"]
    entity_after = [t for t in store.list_tunnels() if t["kind"] == "entity"]
    topic_after = _topic_tunnels(store)
    assert {t["id"] for t in explicit_after} == {t["id"] for t in explicit_before}
    assert {t["id"] for t in entity_after} == {t["id"] for t in entity_before}
    assert {t["id"] for t in topic_after} == {t["id"] for t in topic_before}


def test_topic_purge_does_not_touch_explicit_topic_room(backend, team):
    """A user-authored explicit tunnel using a topic-like room is NOT purged.

    The purge keys on kind='topic', NOT on the room string, so an explicit
    tunnel survives a topic rebuild even if its room happens to look like one.
    """
    store = PostgresLinkStore(backend, team=team)
    store.create_tunnel("wing_aya", "topic:Angular", "wing_work", "topic:Angular", label="manual")
    store.add_topics("wing_aya", ["Other"])
    store.add_topics("wing_work", ["Other"])

    store.rebuild_derived_links_for_wing("wing_aya")
    store.rebuild_derived_links_for_wing("wing_work")

    explicit = [t for t in store.list_tunnels() if t["kind"] == "explicit"]
    assert len(explicit) == 1  # the manual one survives
    assert len(_topic_tunnels(store)) == 1  # the derived "Other" overlap


# ─────────────────────────────────────────────────────────────────────────────
# Full parity: identical {wing:[topics]} produce the SAME topic tunnels on chroma
# (host-global, via compute_topic_tunnels) and on PG (wing_topics).
# ─────────────────────────────────────────────────────────────────────────────


def test_full_parity_chroma_and_pg_same_labels(backend, team):
    labels = {
        "wing_aya": ["Angular", "OpenAPI"],
        "wing_work": ["openapi", "Django"],
        "wing_solo": ["RustLang"],
    }

    # Chroma side: drive the host-global matcher directly with an injected
    # collector (no host-global file write), capturing the tunnel ids it emits.
    chroma_tunnels: list[dict] = []

    def _collect(source_wing, source_room, target_wing, target_room, label="", kind="topic"):
        from mempalace.palace_graph import _canonical_tunnel_id

        tid = _canonical_tunnel_id(source_wing, source_room, target_wing, target_room)
        rec = {"id": tid, "source_wing": source_wing, "target_wing": target_wing, "kind": kind}
        chroma_tunnels.append(rec)
        return rec

    compute_topic_tunnels(labels, create_tunnel_fn=_collect)

    # PG side: persist the SAME labels into wing_topics, derive per wing.
    store = PostgresLinkStore(backend, team=team)
    for wing, ts in labels.items():
        store.add_topics(wing, ts)
    for wing in labels:
        store.rebuild_derived_links_for_wing(wing)

    chroma_ids = sorted({t["id"] for t in chroma_tunnels})
    pg_ids = sorted({t["id"] for t in _topic_tunnels(store)})
    assert chroma_ids == pg_ids
    # And it is the expected single OpenAPI overlap (Angular/Django/RustLang
    # are not shared across wings).
    assert len(pg_ids) == 1


# ─────────────────────────────────────────────────────────────────────────────
# MCP wiring: add_drawer topics + diary topic + fail-loud + no host JSON
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pg_mcp(monkeypatch, backend):
    """Server-mode mcp_server pinned to postgres + a per-test team via config."""
    import mempalace.mcp_server as mcp
    from mempalace.backends import get_backend
    from mempalace.config import MempalaceConfig

    t = "topic_" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setenv("MEMPALACE_TEAM", t)
    monkeypatch.setenv("MEMPALACE_DERIVE_DEBOUNCE_SECONDS", "0")
    monkeypatch.setattr(mcp, "_config", MempalaceConfig())
    # Deterministic embedder so add_drawer does not download ONNX weights.
    get_backend("postgres")._embedder = _embed_384
    # Set the per-request contextvar so strict resolvers see the team.
    token = mcp._active_team_var.set(t)
    mcp._link_store_by_team.clear()
    mcp._entity_index_by_team.clear()
    mcp._kg_by_path.clear()
    mcp._derived_link_debouncer = None
    try:
        yield mcp, t
    finally:
        mcp._active_team_var.reset(token)
        get_backend("postgres")._embedder = None
        mcp._link_store_by_team.clear()
        mcp._derived_link_debouncer = None
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')


def _embed_384(texts):
    import hashlib

    out = []
    for txt in texts:
        v = [0.0] * 384
        for i, b in enumerate(hashlib.sha256((txt or "").encode()).digest()):
            v[i] = b / 255.0
        out.append(v)
    return out


def test_add_drawer_topics_populate_wing_topics(pg_mcp):
    mcp, t = pg_mcp
    res = mcp.tool_add_drawer(
        wing="project", room="api", content="some api notes", topics=["Angular", "OpenAPI"]
    )
    assert res.get("success") is True, res

    store = PostgresLinkStore(_pg_backend(), team=t)
    topics_map = store.topics_by_wing()
    assert sorted(topics_map.get("project", [])) == ["Angular", "OpenAPI"]


def test_diary_topic_wired_to_wing_topics(pg_mcp):
    mcp, t = pg_mcp
    token = mcp._active_team_var.set(t)
    try:
        res = mcp.tool_diary_write(
            agent_name="claude", entry="worked on the parser", topic="Parsing"
        )
    finally:
        mcp._active_team_var.reset(token)
    assert res.get("success") is True, res

    store = PostgresLinkStore(_pg_backend(), team=t)
    topics_map = store.topics_by_wing()
    # Diary wing defaults to wing_<agent>.
    assert "Parsing" in topics_map.get("wing_claude", [])


def test_no_host_global_topics_json_on_pg(pg_mcp, tmp_path, monkeypatch):
    """The host-global known_entities.json topics map is never written on PG."""
    import mempalace.miner as miner

    registry = tmp_path / "known_entities.json"
    monkeypatch.setattr(miner, "_ENTITY_REGISTRY_PATH", str(registry))

    mcp, t = pg_mcp
    mcp.tool_add_drawer(wing="project", room="api", content="x", topics=["Angular"])
    token = mcp._active_team_var.set(t)
    try:
        mcp.tool_diary_write(agent_name="claude", entry="y", topic="Parsing")
    finally:
        mcp._active_team_var.reset(token)

    # Nothing in the host-global registry; the labels went to the team vault.
    if registry.exists():
        import json

        data = json.loads(registry.read_text())
        assert "topics_by_wing" not in data
    store = PostgresLinkStore(_pg_backend(), team=t)
    assert store.topics_by_wing()  # labels landed in the vault instead


def test_fail_loud_no_team_raises_on_topic_store():
    """A topic-store write with no resolvable team RAISES (no default vault)."""
    import mempalace.mcp_server as mcp

    token = mcp._active_team_var.set(None)
    try:
        with pytest.raises(ValueError):
            # Strict resolution returns None -> require_write_team RAISES.
            mcp._get_link_store(None)
    finally:
        mcp._active_team_var.reset(token)


def test_miner_persists_topics_to_wing_topics_on_pg(backend, team, monkeypatch):
    """On PG add_to_known_entities routes confirmed topics to wing_topics."""
    import mempalace.miner as miner

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    monkeypatch.setenv("MEMPALACE_TEAM", team)
    # Point the host-global registry at a temp path so a regression would be
    # observable; on PG it must stay untouched for the topics map.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        reg = os.path.join(d, "known_entities.json")
        monkeypatch.setattr(miner, "_ENTITY_REGISTRY_PATH", reg)
        # Confirmed-entities shape with a topics list, as cmd_init produces.
        miner.add_to_known_entities({"topics": ["Angular", "OpenAPI"]}, wing="project")

        # The topics landed in the team vault, NOT the host-global topics map.
        store = PostgresLinkStore(backend, team=team)
        topics_map = store.topics_by_wing()
        assert sorted(topics_map.get("project", [])) == ["Angular", "OpenAPI"]
        if os.path.exists(reg):
            import json

            data = json.loads(open(reg).read())
            assert "topics_by_wing" not in data


def _pg_backend():
    from mempalace.backends.postgres import PostgresBackend

    return PostgresBackend(dsn=_dsn())
