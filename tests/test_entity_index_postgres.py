"""Integration tests for the per-vault PostgresEntityIndex (G003).

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.entity_index_postgres import PostgresEntityIndex  # noqa: E402


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
    b = PostgresBackend(dsn=_dsn())
    yield b
    b.close()


@pytest.fixture
def team(backend):
    name = "eidx_" + uuid.uuid4().hex[:12]
    yield name
    # Tear down the whole vault schema (entity_occurrences + any kg tables).
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')


def test_add_query_top_and_delete(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1", "d2"], ["Dana", "Ingest"], "people", "2026-06-03")
    idx.add(["d3"], ["Dana"], "people", "2026-06-03")

    # Case-insensitive lookup returns every physical drawer mentioning the entity.
    rows = idx.drawers_for_entity("dana")
    assert {r["drawer_id"] for r in rows} == {"d1", "d2", "d3"}

    top = {t["entity"]: t["count"] for t in idx.top_entities()}
    assert top["Dana"] == 3  # d1, d2, d3 (count of DISTINCT drawer_id)
    assert top["Ingest"] == 2  # d1, d2

    # Deleting drawers removes their rows; Dana now only in d3.
    idx.delete_by_drawer(["d1", "d2"])
    rows = idx.drawers_for_entity("Dana")
    assert {r["drawer_id"] for r in rows} == {"d3"}


def test_add_is_idempotent_per_pair(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1"], ["Dana"], "w", "r")
    idx.add(["d1"], ["Dana"], "w", "r")  # ON CONFLICT DO NOTHING
    assert len(idx.drawers_for_entity("Dana")) == 1


def test_delete_by_parent_clears_bare_id_and_all_chunk_rows(backend, team):
    """delete_by_parent removes the bare id AND every {parent}_chunk_* row.

    Guards the update-path orphan fix: re-indexing a drawer must clear ALL its
    prior physical rows regardless of the prior chunk count, while leaving an
    UNRELATED drawer that merely shares a name prefix untouched.
    """
    idx = PostgresEntityIndex(backend, team=team)
    parent = "drawer_w_r_p"
    idx.add([parent], ["Dana"], "w", "r")  # legacy single-row write
    idx.add([f"{parent}_chunk_000000"], ["Dana"], "w", "r")
    idx.add([f"{parent}_chunk_000001"], ["Zelda"], "w", "r")
    # An unrelated drawer whose id starts with the same human prefix but is NOT
    # a chunk of `parent` (no `_chunk_` suffix) must survive.
    idx.add([f"{parent}x"], ["Other"], "w", "r")

    idx.delete_by_parent(parent)

    assert idx.drawers_for_entity("Dana") == []
    assert idx.drawers_for_entity("Zelda") == []
    assert {r["drawer_id"] for r in idx.drawers_for_entity("Other")} == {f"{parent}x"}


def test_update_drawer_re_chunk_leaves_no_orphan_rows(backend, team, monkeypatch):
    """An update that shrinks the entity footprint leaves no stale chunk rows.

    Simulates a drawer originally filed as 3 chunks (entity rows under
    {parent}_chunk_000000..2), then re-indexed via the update path's
    parent-prefix delete + per-chunk re-index. The stale high-index rows from
    the larger prior chunking must be gone — only the freshly re-indexed row
    survives.
    """
    import mempalace.mcp_server as m

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_entity_index_by_team", {})

    idx = m._get_entity_index(team)
    parent = "drawer_w_r_upd"
    # Prior chunked add: 3 physical chunk rows, all mentioning Dana.
    for i in range(3):
        idx.add([f"{parent}_chunk_{i:06d}"], ["Dana"], "w", "r")
    assert len(idx.drawers_for_entity("Dana")) == 3

    # Update path: clear ALL prior physical rows by parent prefix, then
    # re-index the new (smaller) content keyed on the single physical id.
    idx.delete_by_parent(parent)
    m._index_drawer_entities(team, [(parent, "Zelda Zelda only now.")], "w", "r")

    # No orphaned Dana rows from the larger prior chunking remain.
    assert idx.drawers_for_entity("Dana") == []
    assert {r["drawer_id"] for r in idx.drawers_for_entity("Zelda")} == {parent}


def test_top_entities_wing_scope_and_min_count(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1"], ["Alpha", "Beta"], "wing_a", "r")
    idx.add(["d2"], ["Alpha"], "wing_a", "r")
    idx.add(["d3"], ["Gamma"], "wing_b", "r")

    wing_a = {t["entity"] for t in idx.top_entities(wing="wing_a")}
    assert wing_a == {"Alpha", "Beta"}  # Gamma is in wing_b
    frequent = {t["entity"] for t in idx.top_entities(min_count=2)}
    assert frequent == {"Alpha"}  # only Alpha appears in >=2 drawers


def test_known_entities_accumulates_and_seeds_from_kg(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    idx.add(["d1"], ["Dana"], "people", "r")
    assert "Dana" in idx.known_entities()

    # kg_add entities (relational kg_entities, same vault) seed the known set.
    from mempalace.knowledge_graph_postgres import PostgresKnowledgeGraph

    kg = PostgresKnowledgeGraph(backend, team=team)
    kg.add_triple("Zelda", "owns", "ingest-pipeline")

    # Fresh index instance so the TTL cache does not mask the new kg entities.
    known = PostgresEntityIndex(backend, team=team).known_entities()
    assert "Dana" in known
    assert "Zelda" in known
    assert "ingest-pipeline" in known


def test_known_entities_empty_vault_is_empty(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    # _ensure creates the (empty) table; no kg table yet -> empty known set.
    assert idx.known_entities() == frozenset()


def test_write_path_helper_indexes_per_chunk_against_live_db(backend, team, monkeypatch):
    """The mcp_server write-path helper extracts + indexes PER CHUNK against PG.

    Exercises _index_drawer_entities (the function tool_add_drawer calls) end to
    end minus the ONNX embedding — per-chunk extraction -> per-vault index —
    confirming each physical chunk id is tagged with the entities in its OWN
    text slice (matching the chroma miner), not the whole-drawer set fanned out
    onto every id.
    """
    import mempalace.mcp_server as m

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_entity_index_by_team", {})  # isolate the per-vault cache
    assert m._config.backend != "chroma"

    # Two physical chunks, each mentioning a DIFFERENT entity twice (the
    # frequency tier catches a capitalized word appearing >=2x). Per-chunk
    # tagging must land Zelda only on dX and Ingest only on dY.
    m._index_drawer_entities(
        team,
        [
            ("dX", "Zelda met Zelda about the plan."),
            ("dY", "The Ingest Ingest pipeline shipped."),
        ],
        "people",
        "today",
    )

    idx = m._get_entity_index(team)
    assert {r["drawer_id"] for r in idx.drawers_for_entity("Zelda")} == {"dX"}
    assert {r["drawer_id"] for r in idx.drawers_for_entity("Ingest")} == {"dY"}

    # And the delete helper clears them.
    m._unindex_drawer_entities(team, ["dX", "dY"])
    assert m._get_entity_index(team).drawers_for_entity("Zelda") == []
    assert m._get_entity_index(team).drawers_for_entity("Ingest") == []


def test_query_restrict_ids_prefilters_candidates(team):
    """PostgresCollection.query(restrict_ids=...) restricts the candidate set."""
    import hashlib

    from mempalace.backends import PalaceRef

    def _embed(texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [0.0] * 384
            for i in range(32):
                v[i] = h[i] / 255.0
            out.append(v)
        return out

    b = PostgresBackend(dsn=_dsn(), embedder=_embed)
    try:
        col = b.get_collection(
            palace=PalaceRef(id="x", namespace=team),
            collection_name="mempalace_drawers",
            create=True,
        )
        col.upsert(
            ids=["a", "b", "c"],
            documents=["alpha", "beta", "gamma"],
            metadatas=[{}, {}, {}],
        )
        res = col.query(query_texts=["alpha"], n_results=10, restrict_ids=["a", "c"])
        ids = set(res.ids[0])
        assert ids <= {"a", "c"}  # 'b' excluded by the id pre-filter
        assert "a" in ids
    finally:
        b.close()  # the `team` fixture drops the schema on teardown


def test_tool_entities_lists_and_looks_up_per_chunk(backend, team, monkeypatch):
    """The mempalace_entities tool reads the live per-vault index, per chunk.

    Two physical chunks each carry their own entity (per-chunk tagging). Both
    chunks here are single-drawer ids (their own parents), so the navigator's
    parent-resolved ``drawers`` lists each chunk as its own logical memory.
    """
    import mempalace.mcp_server as m

    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
    monkeypatch.setattr(m, "_entity_index_by_team", {})
    monkeypatch.setattr(m, "_resolve_team", lambda v=None: team)
    # No drawers collection rows for d1/d2 -> parent resolution falls back to
    # each id being its own parent (the best-effort metadata read finds nothing).
    monkeypatch.setattr(m, "_get_collection", lambda *a, **k: None)

    m._index_drawer_entities(
        team,
        [
            ("d1", "Dana Dana met about the plan today."),
            ("d2", "The Ingest Ingest pipeline shipped today."),
        ],
        "people",
        "r",
    )

    looked_up = m.tool_entities(entity="Dana")
    assert {r["drawer_id"] for r in looked_up["drawers"]} == {"d1"}
    assert looked_up["drawer_count"] == 1
    # The mentioning chunk id stays available for verbatim fetch.
    assert looked_up["drawers"][0]["chunk_ids"] == ["d1"]

    listing = m.tool_entities(min_count=1)  # each entity appears in one drawer
    names = {e["entity"] for e in listing["entities"]}
    assert "Dana" in names
    assert "Ingest" in names


def test_tool_entities_resolves_chunk_to_parent_whole_memory(backend, team, monkeypatch):
    """HARD recall: a multi-chunk drawer whose entity is in ONLY one chunk is
    reachable as the WHOLE parent memory from the navigator output.

    Seeds a real drawers collection with two chunks of one parent drawer; the
    entity is mentioned only in the second chunk. The navigator must surface the
    PARENT drawer id (not the lone chunk) so the whole memory stays reachable,
    while keeping the mentioning chunk id available for verbatim fetch.
    """
    import hashlib

    import mempalace.mcp_server as m
    from mempalace.backends import PalaceRef

    def _embed(texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [0.0] * 384
            for i in range(32):
                v[i] = h[i] / 255.0
            out.append(v)
        return out

    embedding_backend = PostgresBackend(dsn=_dsn(), embedder=_embed)
    try:
        col = embedding_backend.get_collection(
            palace=PalaceRef(id=team, namespace=team),
            collection_name="mempalace_drawers",
            create=True,
        )
        parent = "drawer_people_r_parentmemory"
        c0 = f"{parent}_chunk_000000"
        c1 = f"{parent}_chunk_000001"
        col.upsert(
            ids=[c0, c1],
            documents=["A short first chunk with no proper nouns.", "Zelda Zelda again."],
            metadatas=[
                {"wing": "people", "room": "r", "chunk_index": 0, "parent_drawer_id": parent},
                {"wing": "people", "room": "r", "chunk_index": 1, "parent_drawer_id": parent},
            ],
        )

        monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
        monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
        monkeypatch.setattr(m, "_entity_index_by_team", {})
        monkeypatch.setattr(m, "_resolve_team", lambda v=None: team)
        monkeypatch.setattr(m, "_get_collection", lambda *a, **k: col)

        # Entity X (Zelda) appears in ONLY the second chunk -> the index holds
        # the lone chunk id c1.
        m._index_drawer_entities(team, [(c0, ""), (c1, "Zelda Zelda again.")], "people", "r")
        assert {r["drawer_id"] for r in m._get_entity_index(team).drawers_for_entity("Zelda")} == {
            c1
        }

        looked_up = m.tool_entities(entity="Zelda")
        # The navigator surfaces the PARENT (whole memory), not the lone chunk.
        assert {r["drawer_id"] for r in looked_up["drawers"]} == {parent}
        assert looked_up["drawer_count"] == 1
        # ...while the mentioning chunk id is retained for verbatim fetch.
        assert looked_up["drawers"][0]["chunk_ids"] == [c1]
    finally:
        embedding_backend.close()


def test_diary_write_indexes_entities_per_vault(backend, team, monkeypatch):
    """tool_diary_write wires the per-chunk entity index (same as add/update).

    A diary entry's entities must become queryable via the per-team entity
    index — the diary collection is already team-scoped; this is the
    completeness fix. Drives the live PG path with a real diary collection.
    """
    import hashlib

    import mempalace.mcp_server as m
    from mempalace.backends import PalaceRef

    def _embed(texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [0.0] * 384
            for i in range(32):
                v[i] = h[i] / 255.0
            out.append(v)
        return out

    embedding_backend = PostgresBackend(dsn=_dsn(), embedder=_embed)
    try:
        col = embedding_backend.get_collection(
            palace=PalaceRef(id=team, namespace=team),
            collection_name="mempalace_drawers",
            create=True,
        )

        monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
        monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)
        monkeypatch.setattr(m, "_entity_index_by_team", {})
        monkeypatch.setattr(m, "_resolve_team", lambda v=None: team)
        monkeypatch.setattr(m, "_get_collection", lambda *a, **k: col)

        res = m.tool_diary_write(
            agent_name="claude",
            entry="Today Zelda Zelda reviewed the ingest plan.",
            topic="work",
        )
        assert res["success"] is True

        # The diary entry's entity is now queryable in the per-team index.
        rows = m._get_entity_index(team).drawers_for_entity("Zelda")
        assert {r["drawer_id"] for r in rows} == {res["entry_id"]}
    finally:
        embedding_backend.close()


def test_backfill_populates_and_is_idempotent(backend, team):
    """backfill() indexes an existing drawers collection (paginated)."""
    from mempalace.miner import _extract_entities_for_metadata

    idx = PostgresEntityIndex(backend, team=team)

    class _FakeResult:
        def __init__(self, ids, documents, metadatas):
            self.ids = ids
            self.documents = documents
            self.metadatas = metadatas

    class _FakeCol:
        def __init__(self, rows):
            self._rows = rows  # list of (id, document, metadata)

        def count(self):
            return len(self._rows)

        def get(self, limit=None, offset=0, include=None):
            page = self._rows[offset : offset + (limit or len(self._rows))]
            return _FakeResult([r[0] for r in page], [r[1] for r in page], [r[2] for r in page])

    rows = [
        ("d1", "Dana owns the ingest work.", {"wing": "people", "room": "r"}),
        ("d2", "Dana and Zelda met.", {"wing": "people", "room": "r"}),
        ("d3", "", {"wing": "people", "room": "r"}),  # empty content -> no entities
        ("d4", "sentinel", {"is_sentinel": True}),  # sentinel -> skipped entirely
    ]
    known = frozenset({"Dana", "Zelda", "ingest"})

    def extract(content):
        return [e for e in _extract_entities_for_metadata(content, known=known).split(";") if e]

    stats = idx.backfill(_FakeCol(rows), extract)
    assert stats["drawers"] == 3  # d1, d2, d3 scanned; d4 (sentinel) skipped

    assert {r["drawer_id"] for r in idx.drawers_for_entity("Dana")} == {"d1", "d2"}
    assert {r["drawer_id"] for r in idx.drawers_for_entity("Zelda")} == {"d2"}

    # Idempotent: a second backfill does not duplicate rows.
    idx.backfill(_FakeCol(rows), extract)
    assert {r["drawer_id"] for r in idx.drawers_for_entity("Dana")} == {"d1", "d2"}


def test_search_memories_scopes_to_restrict_ids(team, monkeypatch):
    """Regression: search_memories must thread restrict_ids into the drawer query.

    Drives searcher.search_memories against a live drawers collection and asserts
    a drawer outside the restrict set is excluded — the seam where the entity=
    filter was silently dropped (accepted to query, never forwarded).
    """
    import hashlib

    import mempalace.searcher as searcher
    from mempalace.backends import PalaceRef

    def _embed(texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [0.0] * 384
            for i in range(32):
                v[i] = h[i] / 255.0
            out.append(v)
        return out

    backend = PostgresBackend(dsn=_dsn(), embedder=_embed)
    try:
        col = backend.get_collection(
            palace=PalaceRef(id=team, namespace=team),
            collection_name="mempalace_drawers",
            create=True,
        )
        col.upsert(
            ids=["a", "b", "c"],
            documents=["alpha alpha", "beta beta", "gamma gamma"],
            metadatas=[{"wing": "w", "room": "r"}] * 3,
        )
        # search_memories resolves these from its own namespace.
        monkeypatch.setattr(searcher, "get_collection", lambda *a, **k: col)
        monkeypatch.setattr(
            searcher,
            "get_closets_collection",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no closets")),
        )

        scoped = searcher.search_memories(
            "beta", palace_path="ignored", n_results=10, restrict_ids=["a", "c"]
        )
        texts = " ".join(r.get("text", "") for r in scoped.get("results", []))
        assert "beta" not in texts  # 'b' is outside the restrict set -> excluded

        unscoped = searcher.search_memories("beta", palace_path="ignored", n_results=10)
        texts2 = " ".join(r.get("text", "") for r in unscoped.get("results", []))
        assert "beta" in texts2  # control: visible without the filter
    finally:
        backend.close()
