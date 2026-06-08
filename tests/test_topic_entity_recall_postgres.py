"""Caller-supplied topic labels are recall-only entity rows (central Postgres).

A caller that files a drawer with ``topics=['infra.eden']`` should be able to
find that drawer via ``entity='infra.eden'`` EVEN WHEN the label never appears in
the drawer text — the label is the strongest explicit signal of what the drawer
is about. But a topic label must NOT pollute the co-occurrence derive that builds
entity hallways/tunnels (the ``entity_occurrences`` table is dual-purpose: it is
both the ``entity=`` recall index and the co-occurrence substrate). These tests
pin both halves: topic labels are findable, and they are isolated from the derive.

The isolation rests on an ``is_topic`` column excluded from
``_co_occurrence_for_wing``. Because that column is owned by the entity index but
read by the link store, the tests also cover the migration ordering hazard: a
rebuild that runs on a vault whose ``entity_occurrences`` predates the column
(before any new-code write) must not crash, and the link store's column probe must
not pin a stale "absent" result past a later write that adds the column.

Skipped when no Postgres is reachable. The rebuild is driven SYNCHRONOUSLY
(``rebuild_derived_links_for_wing``) so the assertions never race the debouncer.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends import get_backend  # noqa: E402
from mempalace.backends.postgres import team_schema  # noqa: E402
from mempalace.link_store_postgres import PostgresLinkStore  # noqa: E402

DIM = 384


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


pytestmark = pytest.mark.skipif(not _reachable(), reason="no reachable Postgres")


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * DIM
        for i, b in enumerate(hashlib.sha256((t or "").encode()).digest()):
            v[i] = b / 255.0
        out.append(v)
    return out


def _drop(*teams):
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            for t in teams:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
        conn.commit()


def _use_team(mcp, monkeypatch, team):
    from mempalace.config import MempalaceConfig

    monkeypatch.setenv("MEMPALACE_TEAM", team)
    monkeypatch.setattr(mcp, "_config", MempalaceConfig())
    mcp._kg_by_path.clear()


def _hallway_pairs(mcp, team):
    store = mcp._get_link_store(team)
    return {(h["entity_a"], h["entity_b"]) for h in store.list_hallways()}


def _entity_tunnel_endpoints(mcp, team):
    store = mcp._get_link_store(team)
    return {
        (t.get("source", {}).get("wing"), t.get("target", {}).get("wing"))
        for t in store.list_tunnels()
        if t.get("kind") == "entity"
    }


def _setup(monkeypatch, team):
    import mempalace.mcp_server as mcp

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    get_backend("postgres")._embedder = _fake_embed
    _use_team(mcp, monkeypatch, team)
    return mcp


def test_topic_label_findable_via_entity_when_absent_from_text(monkeypatch):
    """1A: a topics= label is findable via entity= even when not in the drawer text."""
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        add = mcp.tool_add_drawer(
            wing="ops",
            room="r1",
            content="Short note about the renovate rollout.",  # label NOT present
            topics=["infra.eden"],
        )
        assert add.get("success") is True, add

        rows = mcp._get_entity_index(team).drawers_for_entity("infra.eden")
        assert len(rows) >= 1, rows  # findable despite absence from text

        found = mcp.tool_search("renovate rollout", entity="infra.eden")
        texts = " ".join(r.get("text", "") for r in found.get("results", []))
        assert "renovate rollout" in texts, found
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_topic_only_add_does_not_pollute_cooccurrence(monkeypatch):
    """Isolation: a topics-only add leaves the entity-tunnel + hallway SETS unchanged.

    Asserts SET equality (not counts) and that the topic label appears in no
    hallway pair. The rebuild is driven synchronously to avoid debouncer races.
    """
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    store = mcp._get_link_store(team)
    try:
        # Build a REAL hallway: Dana+Aya co-occur in two physical chunks of one
        # wing so the pair clears the min_count=2 hallway gate.
        for room in ("r1", "r2"):
            mcp.tool_add_drawer(
                wing="ops",
                room=room,
                content="Dana met Aya. Dana and Aya synced again with Dana and Aya.",
            )
        store.rebuild_derived_links_for_wing("ops")
        pairs_before = _hallway_pairs(mcp, team)
        tunnels_before = _entity_tunnel_endpoints(mcp, team)
        assert any("Dana" in p and "Aya" in p for p in pairs_before), pairs_before

        # A topics-only add (label absent from text) must change neither set.
        mcp.tool_add_drawer(
            wing="ops",
            room="r3",
            content="An unrelated rollout note with no recurring proper nouns here.",
            topics=["infra.eden"],
        )
        store.rebuild_derived_links_for_wing("ops")
        pairs_after = _hallway_pairs(mcp, team)
        tunnels_after = _entity_tunnel_endpoints(mcp, team)

        assert pairs_after == pairs_before, (pairs_before, pairs_after)
        assert tunnels_after == tunnels_before, (tunnels_before, tunnels_after)
        assert not any("infra.eden" in p for p in pairs_after), pairs_after

        # ...yet the topic label IS findable (recall preserved).
        assert mcp._get_entity_index(team).drawers_for_entity("infra.eden"), "recall lost"
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_topic_label_demoted_to_cooccurrence_when_organically_mentioned(monkeypatch):
    """M1: a label written topic-first then organically extracted ends is_topic=False.

    The monotone ON CONFLICT (is_topic = old AND new) means any organic extraction
    permanently demotes the row so it participates in co-occurrence again.
    """
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        # First a topic-only mention (label absent from text) -> is_topic=True.
        mcp.tool_add_drawer(wing="ops", room="r1", content="note one", topics=["infra.eden"])
        # Then a drawer that organically mentions infra.eden -> is_topic=False row.
        mcp.tool_add_drawer(
            wing="ops",
            room="r2",
            content="infra.eden had an incident. infra.eden recovered after Dana and Dana paged.",
            topics=["infra.eden"],
        )
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT is_topic FROM "{team_schema(team)}".entity_occurrences '
                    "WHERE entity = %s",
                    ("infra.eden",),
                )
                flags = [r[0] for r in cur.fetchall()]
        assert flags, "no infra.eden rows"
        assert False in flags, f"expected an organic is_topic=False row, got {flags}"
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_same_key_demotion_in_place(monkeypatch):
    """M1 (same-key): when one drawer's text ALSO contains its topics= label, the
    SINGLE row for that (entity, drawer) is is_topic=False — proving the monotone
    ON CONFLICT demotes in place (extracted is_topic=False then label is_topic=True
    on the same key resolves to False), not that a second distinct row appears."""
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        # One drawer, single chunk: infra.eden is BOTH a topic label AND present
        # in the text, so the extracted (False) and label (True) writes hit the
        # same (entity, drawer_id) key.
        add = mcp.tool_add_drawer(
            wing="ops",
            room="r1",
            content="infra.eden rollout. infra.eden stabilized after the change.",
            topics=["infra.eden"],
        )
        assert add.get("chunks", 1) == 1, add  # single key, no chunk fan-out
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT drawer_id, is_topic FROM "{team_schema(team)}".entity_occurrences '
                    "WHERE entity = %s",
                    ("infra.eden",),
                )
                rows = cur.fetchall()
        assert len(rows) == 1, f"expected exactly one (entity,drawer) row, got {rows}"
        assert rows[0][1] is False, f"same-key row should be demoted to False, got {rows}"
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_chunked_topic_label_stamped_on_all_chunks(monkeypatch):
    """A topics= label on an oversized (chunked) drawer tags every physical chunk."""
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        big = "The renovate operator rollout note. " * 60  # > chunk_size -> chunked
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=big, topics=["infra.eden"])
        assert add.get("chunks", 1) > 1, add
        rows = mcp._get_entity_index(team).drawers_for_entity("infra.eden")
        stamped = {r["drawer_id"] for r in rows}
        assert set(add["chunk_ids"]) <= stamped, (add["chunk_ids"], stamped)
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_cooccurrence_rebuild_first_on_pre_is_topic_vault(monkeypatch):
    """C1: a rebuild that runs BEFORE any new-code write on a column-less vault
    must not crash, and must still derive hallways correctly."""
    _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])  # sets backend/env/embedder
    team = "t" + uuid.uuid4().hex[:10]
    sch = team_schema(team)
    try:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
                # Old-shape table: no is_topic column.
                cur.execute(
                    f'CREATE TABLE "{sch}".entity_occurrences ('
                    "entity text NOT NULL, drawer_id text NOT NULL, wing text, room text,"
                    " PRIMARY KEY (entity, drawer_id))"
                )
                cur.executemany(
                    f'INSERT INTO "{sch}".entity_occurrences VALUES (%s,%s,%s,%s)',
                    [
                        ("Dana", "d1", "ops", "r1"),
                        ("Aya", "d1", "ops", "r1"),
                        ("Dana", "d2", "ops", "r1"),
                        ("Aya", "d2", "ops", "r1"),
                    ],
                )
            conn.commit()
        store = PostgresLinkStore(get_backend("postgres"), team)
        result = store.rebuild_derived_links_for_wing("ops")  # must NOT raise
        assert result["hallways"] >= 1, result
        assert store._entity_has_is_topic is False  # column genuinely absent
        pairs = {(h["entity_a"], h["entity_b"]) for h in store.list_hallways()}
        assert ("Aya", "Dana") in pairs, pairs
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_cooccurrence_column_probe_caches_on_present_only(monkeypatch):
    """C1 stale-False: a store that probed the column absent must re-probe and
    exclude topic rows once a later write (same process) adds the column."""
    _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = "t" + uuid.uuid4().hex[:10]
    sch = team_schema(team)
    try:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
                cur.execute(
                    f'CREATE TABLE "{sch}".entity_occurrences ('
                    "entity text NOT NULL, drawer_id text NOT NULL, wing text, room text,"
                    " PRIMARY KEY (entity, drawer_id))"
                )
                cur.executemany(
                    f'INSERT INTO "{sch}".entity_occurrences VALUES (%s,%s,%s,%s)',
                    [
                        ("Dana", "d1", "ops", "r1"),
                        ("Aya", "d1", "ops", "r1"),
                        ("Dana", "d2", "ops", "r1"),
                        ("Aya", "d2", "ops", "r1"),
                    ],
                )
            conn.commit()
        store = PostgresLinkStore(get_backend("postgres"), team)
        store.rebuild_derived_links_for_wing("ops")  # probes absent, caches nothing
        assert store._entity_has_is_topic is False

        # A new-code write adds the column (via _ensure) + a topic-only row.
        from mempalace.entity_index_postgres import PostgresEntityIndex

        idx = PostgresEntityIndex(get_backend("postgres"), team)
        idx.add(["d1"], ["infra.eden"], "ops", "r1", is_topic=True)
        idx.add(["d2"], ["infra.eden"], "ops", "r1", is_topic=True)

        store.rebuild_derived_links_for_wing("ops")  # SAME instance must re-probe
        assert store._entity_has_is_topic is True
        pairs = {(h["entity_a"], h["entity_b"]) for h in store.list_hallways()}
        assert not any("infra.eden" in p for p in pairs), pairs  # excluded
        assert ("Aya", "Dana") in pairs, pairs  # real hallway preserved
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_is_topic_alter_is_idempotent(monkeypatch):
    """Calling _ensure on a vault whose table predates is_topic adds the column once
    and is harmless on repeat."""
    _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    from mempalace.entity_index_postgres import PostgresEntityIndex

    team = "t" + uuid.uuid4().hex[:10]
    sch = team_schema(team)
    try:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
                cur.execute(
                    f'CREATE TABLE "{sch}".entity_occurrences ('
                    "entity text NOT NULL, drawer_id text NOT NULL, wing text, room text,"
                    " PRIMARY KEY (entity, drawer_id))"
                )
            conn.commit()
        idx = PostgresEntityIndex(get_backend("postgres"), team)
        idx._ensure()  # adds is_topic
        idx._ensured = False
        idx._ensure()  # idempotent re-run
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns WHERE table_schema=%s "
                    "AND table_name='entity_occurrences' AND column_name='is_topic'",
                    (sch,),
                )
                assert cur.fetchone() is not None
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_update_drawer_preserves_topic_recall(monkeypatch):
    """Editing a topic-tagged drawer must NOT drop its entity= recall: update_drawer
    re-derives extracted entities from new content but re-stamps the captured topic
    labels, so a topics= label stays findable across a content/wing/room edit."""
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        add = mcp.tool_add_drawer(
            wing="ops", room="r1", content="Short renovate note.", topics=["infra.eden"]
        )
        assert add.get("success") is True
        assert mcp._get_entity_index(team).drawers_for_entity("infra.eden"), "precondition"

        # Edit the content AND move the drawer to a new wing — the label is NOT
        # present in the new text either. The preserved topic row must follow the
        # drawer to the new wing.
        upd = mcp.tool_update_drawer(
            add["drawer_id"],
            content="Edited note about the rollout, no label in text.",
            wing="archive",
        )
        assert upd.get("success") is True, upd

        rows = mcp._get_entity_index(team).drawers_for_entity("infra.eden")
        assert len(rows) >= 1, f"topic recall lost on update: {rows}"
        new_wing = mcp.tool_get_drawer(add["drawer_id"])["wing"]
        assert all(r["wing"] == new_wing for r in rows), (rows, new_wing)
        found = mcp.tool_search("rollout", entity="infra.eden")
        assert found.get("results"), found
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_topic_labels_for_parent_captures_chunk_rows(monkeypatch):
    """topic_labels_for_parent must capture topic rows keyed on a chunked drawer's
    physical chunk ids ({id}_chunk_*), not just the bare id — this is the row set
    update_drawer preserves. Pinned directly since update_drawer can't target a
    chunked logical id (it reports not-found before re-indexing)."""
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        big = "The renovate operator rollout note. " * 60  # > chunk_size -> chunked
        add = mcp.tool_add_drawer(wing="ops", room="r1", content=big, topics=["infra.eden"])
        assert add.get("chunks", 1) > 1, add
        # Another drawer's topic must NOT bleed into this parent's capture.
        mcp.tool_add_drawer(wing="ops", room="r2", content="unrelated", topics=["other-theme"])

        labels = mcp._get_entity_index(team).topic_labels_for_parent(add["drawer_id"])
        assert "infra.eden" in labels, labels  # captured via the {id}_chunk_* pattern
        assert "other-theme" not in labels, labels  # scoped to this parent only
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)


def test_top_entities_excludes_topic_rows(monkeypatch):
    """The top_entities overview reflects organically-extracted entities only — a
    recall-only topic label (stamped on every chunk) must not rank there — while it
    stays findable via drawers_for_entity (recall is unaffected)."""
    mcp = _setup(monkeypatch, "t" + uuid.uuid4().hex[:10])
    team = mcp._resolve_team(None)
    try:
        # 'Dana' organically mentioned >=2x (extracted, is_topic=false);
        # 'secret-theme' supplied only as a topic label (is_topic=true).
        mcp.tool_add_drawer(
            wing="proj",
            room="r1",
            content="Dana shipped it. Dana confirmed. Dana again.",
            topics=["secret-theme"],
        )
        names = [e["entity"] for e in mcp._get_entity_index(team).top_entities()]
        assert "Dana" in names, names  # organic entity present
        assert "secret-theme" not in names, names  # topic excluded from overview
        # ...but the topic label is still recallable.
        assert mcp._get_entity_index(team).drawers_for_entity("secret-theme"), "recall lost"
    finally:
        get_backend("postgres")._embedder = None
        _drop(team)
