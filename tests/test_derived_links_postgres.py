"""Integration tests for the server-derived link layer (hallways + entity tunnels).

The derived link layer is computed per team from the already-vaulted, per-chunk
``entity_occurrences`` rows: a same-wing entity pair becomes a within-wing
HALLWAY; a cross-wing entity pair becomes a derived ENTITY TUNNEL
(``kind='entity'``) in the team's ``tunnels`` table. It is maintained
incrementally on the write path via a ``(team, wing)``-keyed debounced worker
that mirrors the closet rebuild coalescer (team captured at enqueue; the worker
never resolves the team; ``flush()`` is deterministic).

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.
Each test uses a per-test team and drops its schema CASCADE on teardown.
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.entity_index_postgres import PostgresEntityIndex  # noqa: E402
from mempalace.link_store import DerivedLinkDebouncer  # noqa: E402
from mempalace.link_store_postgres import WING_SCAN_CAP, PostgresLinkStore  # noqa: E402
from mempalace.palace_graph import _canonical_tunnel_id  # noqa: E402


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
    name = "dlnk_" + uuid.uuid4().hex[:12]
    yield name
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(name)}" CASCADE')


def _seed_chunk(idx: PostgresEntityIndex, drawer_id, entities, wing, room):
    """Index one physical chunk's entity set (the per-chunk write-path shape)."""
    idx.add([drawer_id], entities, wing, room)


# ─────────────────────────────────────────────────────────────────────────────
# Same-wing pair -> hallway; cross-wing pair -> entity tunnel
# ─────────────────────────────────────────────────────────────────────────────


def test_same_wing_pair_becomes_hallway(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    # Two chunks in one wing, each carrying the same pair -> count 2 >= min_count.
    _seed_chunk(idx, "c1", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, "c2", ["Aya", "Lumi"], "wing_aya", "letters")

    store = PostgresLinkStore(backend, team=team)
    hallways = store.compute_hallways_for_wing("wing_aya", min_count=2)

    assert len(hallways) == 1
    h = hallways[0]
    assert h["wing"] == "wing_aya"
    assert {h["entity_a"], h["entity_b"]} == {"Aya", "Lumi"}
    assert h["co_occurrence_count"] == 2
    assert sorted(h["rooms"]) == ["diary", "letters"]


def test_cross_wing_pair_becomes_entity_tunnel(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    # Same pair co-occurs (>= min_count) inside EACH of two wings, so each wing
    # gets a hallway and the shared entities bridge the two wings as tunnels.
    _seed_chunk(idx, "a1", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, "a2", ["Aya", "Lumi"], "wing_aya", "letters")
    _seed_chunk(idx, "b1", ["Aya", "Lumi"], "wing_work", "notes")
    _seed_chunk(idx, "b2", ["Aya", "Lumi"], "wing_work", "tasks")

    store = PostgresLinkStore(backend, team=team)
    # Rebuild both wings so both wings' hallways exist before tunnel derivation.
    store.rebuild_derived_links_for_wing("wing_aya", min_count=2)
    result = store.rebuild_derived_links_for_wing("wing_work", min_count=2)

    # Derived entity tunnels live in the tunnels table with kind='entity'.
    entity_tunnels = [t for t in store.list_tunnels() if t["kind"] == "entity"]
    # One tunnel per shared entity (Aya, Lumi) bridging the two wings.
    assert len(entity_tunnels) == 2
    bridged_entities = {t["source"]["room"].split("entity:")[1] for t in entity_tunnels}
    assert bridged_entities == {"Aya", "Lumi"}
    # Endpoints are the two wings (synthetic entity:<name> rooms).
    for t in entity_tunnels:
        assert {t["source"]["wing"], t["target"]["wing"]} == {"wing_aya", "wing_work"}
    assert result["entity_tunnels"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk no-over-count: two chunks of one logical drawer counted per chunk
# ─────────────────────────────────────────────────────────────────────────────


def test_per_chunk_no_over_count(backend, team):
    """A logical drawer split into chunks counts co-occurrence PER PHYSICAL chunk.

    Two chunks of one logical drawer, each carrying the same pair, contribute a
    co-occurrence of 2 (one per chunk) — NOT inflated. This is the regression
    vs whole-drawer over-count (where the whole-drawer entity set stamped on
    every chunk id would multiply the count by chunk fan-out).
    """
    idx = PostgresEntityIndex(backend, team=team)
    parent = "drawerX"
    _seed_chunk(idx, f"{parent}_chunk_000000", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, f"{parent}_chunk_000001", ["Aya", "Lumi"], "wing_aya", "diary")

    store = PostgresLinkStore(backend, team=team)
    hallways = store.compute_hallways_for_wing("wing_aya", min_count=2)

    assert len(hallways) == 1
    # Exactly 2 (one per physical chunk), not 4 (whole-drawer fan-out) or more.
    assert hallways[0]["co_occurrence_count"] == 2


def test_single_mention_chunk_does_not_pair(backend, team):
    """A chunk mentioning only one entity contributes no co-occurrence pair."""
    idx = PostgresEntityIndex(backend, team=team)
    _seed_chunk(idx, "c1", ["Aya"], "wing_aya", "diary")
    _seed_chunk(idx, "c2", ["Lumi"], "wing_aya", "diary")

    store = PostgresLinkStore(backend, team=team)
    hallways = store.compute_hallways_for_wing("wing_aya", min_count=1)
    # Aya and Lumi never share a chunk -> no pair.
    assert hallways == []


# ─────────────────────────────────────────────────────────────────────────────
# Incremental-equals-full after add / update / delete (same backend)
# ─────────────────────────────────────────────────────────────────────────────


def _hallway_snapshot(store, wing):
    """Comparable, dynamics-free snapshot of a wing's hallways."""
    return sorted(
        (h["entity_a"], h["entity_b"], h["co_occurrence_count"], tuple(sorted(h["rooms"])))
        for h in store.list_hallways(wing)
    )


def _entity_tunnel_snapshot(store):
    return sorted(t["id"] for t in store.list_tunnels() if t["kind"] == "entity")


def test_incremental_equals_full_add_update_delete(backend, team):
    """The incremental derive after add/update/delete equals a from-scratch rebuild.

    We mutate the entity_occurrences substrate (the add/update/delete effects)
    and rebuild incrementally after each, then drop ALL derived records and do a
    single from-scratch rebuild of the same wing — the two must agree.
    """
    idx = PostgresEntityIndex(backend, team=team)
    store = PostgresLinkStore(backend, team=team)
    wing = "wing_aya"

    # ADD: two chunks with the same pair.
    _seed_chunk(idx, "c1", ["Aya", "Lumi"], wing, "diary")
    _seed_chunk(idx, "c2", ["Aya", "Lumi"], wing, "letters")
    store.rebuild_derived_links_for_wing(wing, min_count=2)

    # UPDATE: a drawer changes its entity set (simulate update = delete+re-add of
    # the physical row). Replace c2's pair with a new pair.
    idx.delete_by_drawer(["c2"])
    _seed_chunk(idx, "c2", ["Aya", "Mira"], wing, "letters")
    _seed_chunk(idx, "c3", ["Aya", "Mira"], wing, "ideas")
    store.rebuild_derived_links_for_wing(wing, min_count=2)

    # DELETE: drop c1 entirely.
    idx.delete_by_drawer(["c1"])
    store.rebuild_derived_links_for_wing(wing, min_count=2)

    incremental_hallways = _hallway_snapshot(store, wing)
    incremental_tunnels = _entity_tunnel_snapshot(store)

    # From-scratch: purge ALL derived records, then one fresh rebuild.
    schema = team_schema(team)
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DELETE FROM "{schema}"."hallways"')
            cur.execute(f'DELETE FROM "{schema}"."tunnels" WHERE kind = \'entity\'')
    store.rebuild_derived_links_for_wing(wing, min_count=2)

    full_hallways = _hallway_snapshot(store, wing)
    full_tunnels = _entity_tunnel_snapshot(store)

    assert incremental_hallways == full_hallways
    assert incremental_tunnels == full_tunnels
    # Sanity: after delete of c1, only the Aya/Mira pair survives at count 2.
    assert full_hallways == [("Aya", "Mira", 2, ("ideas", "letters"))]


# ─────────────────────────────────────────────────────────────────────────────
# Debounced worker: ONE rebuild per (team, wing) burst; team captured at enqueue
# ─────────────────────────────────────────────────────────────────────────────


def test_debouncer_coalesces_one_rebuild_per_team_wing_burst():
    """N rapid enqueues for one (team, wing) -> exactly ONE rebuild; no DB."""
    calls = []
    lock = threading.Lock()

    def _stub(team, wing):
        with lock:
            calls.append((team, wing))
        return {"hallways": 0, "entity_tunnels": 0}

    deb = DerivedLinkDebouncer(rebuild_fn=_stub, debounce_seconds=0.05)
    deb.start()
    for _ in range(50):
        deb.enqueue("teamA", "wing_aya")
    assert deb.flush(timeout=10.0)
    assert len(calls) == 1
    assert calls[0] == ("teamA", "wing_aya")


def test_debouncer_team_captured_at_enqueue_worker_never_resolves():
    """The coalescing key includes team; the worker fires with the captured team.

    Two teams enqueuing the same wing yield TWO independent rebuilds, each with
    its own captured team — the worker never resolves the team itself.
    """
    calls = []
    lock = threading.Lock()

    def _stub(team, wing):
        with lock:
            calls.append((team, wing))
        return {}

    deb = DerivedLinkDebouncer(rebuild_fn=_stub, debounce_seconds=0.05)
    deb.start()
    for _ in range(10):
        deb.enqueue("teamA", "wing_aya")
        deb.enqueue("teamB", "wing_aya")
    assert deb.flush(timeout=10.0)
    assert set(calls) == {("teamA", "wing_aya"), ("teamB", "wing_aya")}


def test_debouncer_zero_delay_flush_deterministic():
    calls = []

    def _stub(team, wing):
        calls.append((team, wing))
        return {}

    deb = DerivedLinkDebouncer(rebuild_fn=_stub, debounce_seconds=0.0)
    deb.start()
    deb.enqueue("teamA", "w")
    assert deb.flush(timeout=10.0)
    assert calls == [("teamA", "w")]


def test_debouncer_rebuild_failure_swallowed():
    """A raising rebuild_fn never escapes the worker (best-effort)."""
    fired = []

    def _boom(team, wing):
        fired.append((team, wing))
        raise RuntimeError("simulated derive failure")

    deb = DerivedLinkDebouncer(rebuild_fn=_boom, debounce_seconds=0.0)
    deb.start()
    deb.enqueue("teamA", "w")
    assert deb.flush(timeout=10.0)
    assert fired == [("teamA", "w")]
    # Worker still alive after a failure.
    deb.enqueue("teamA", "w2")
    assert deb.flush(timeout=10.0)
    assert ("teamA", "w2") in fired


def test_debouncer_drives_real_rebuild_against_db(backend, team):
    """End-to-end: the debouncer fires the real rebuild_fn against a live DB."""
    idx = PostgresEntityIndex(backend, team=team)
    _seed_chunk(idx, "c1", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, "c2", ["Aya", "Lumi"], "wing_aya", "letters")

    store = PostgresLinkStore(backend, team=team)

    deb = DerivedLinkDebouncer(
        rebuild_fn=lambda t, w: store.rebuild_derived_links_for_wing(w, min_count=2),
        debounce_seconds=0.0,
    )
    deb.start()
    for _ in range(5):
        deb.enqueue(team, "wing_aya")
    assert deb.flush(timeout=15.0)

    hallways = store.list_hallways("wing_aya")
    assert len(hallways) == 1
    assert hallways[0]["co_occurrence_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Explicit tunnels (kind='explicit') survive a derived rebuild
# ─────────────────────────────────────────────────────────────────────────────


def test_explicit_tunnels_not_purged_by_derived_rebuild(backend, team):
    store = PostgresLinkStore(backend, team=team)

    # A user-authored explicit tunnel touching wing_aya. Explicit tunnels are
    # user data and are NEVER purged by a derived rebuild (entity OR topic).
    explicit = store.create_tunnel(
        "wing_aya", "diary", "wing_work", "notes", label="user link", kind="explicit"
    )
    # A topic tunnel backed by wing_topics labels: it survives because the
    # rebuild RE-DERIVES it from the labels (purge + recompute), not because it
    # is left untouched. Topic tunnels are now a DERIVED kind (H014t).
    store.add_topics("wing_aya", ["focus"])
    store.add_topics("wing_work", ["focus"])

    # Seed co-occurrence so a derived rebuild creates entity records for the wing.
    idx = PostgresEntityIndex(backend, team=team)
    _seed_chunk(idx, "c1", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, "c2", ["Aya", "Lumi"], "wing_aya", "letters")

    store.rebuild_derived_links_for_wing("wing_aya", min_count=2)
    # wing_work needs a rebuild too so the cross-wing topic pair materializes.
    store.rebuild_derived_links_for_wing("wing_work", min_count=2)

    ids = {t["id"] for t in store.list_tunnels()}
    kinds = {t["id"]: t["kind"] for t in store.list_tunnels()}
    # The explicit tunnel survives the derived rebuild untouched.
    assert explicit["id"] in ids
    assert kinds[explicit["id"]] == "explicit"
    # A topic tunnel for the shared "focus" label is derived (the kind coexists).
    topic_tunnels = [t for t in store.list_tunnels() if t["kind"] == "topic"]
    assert len(topic_tunnels) == 1

    # A second rebuild still leaves the explicit tunnel and re-derives the topic.
    store.rebuild_derived_links_for_wing("wing_aya", min_count=2)
    store.rebuild_derived_links_for_wing("wing_work", min_count=2)
    ids2 = {t["id"] for t in store.list_tunnels()}
    assert explicit["id"] in ids2
    assert len([t for t in store.list_tunnels() if t["kind"] == "topic"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics preserved across re-created derived records
# ─────────────────────────────────────────────────────────────────────────────


def test_dynamics_preserved_across_derived_hallway_rebuild(backend, team):
    idx = PostgresEntityIndex(backend, team=team)
    _seed_chunk(idx, "c1", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, "c2", ["Aya", "Lumi"], "wing_aya", "letters")

    store = PostgresLinkStore(backend, team=team)
    first = store.compute_hallways_for_wing("wing_aya", min_count=2)
    assert len(first) == 1
    hid = first[0]["id"]

    # Mutate the dynamics columns to simulate accumulated activity.
    schema = team_schema(team)
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'UPDATE "{schema}"."hallways" '
                "SET strength = 3.3, stability = 1.7, access_count = 9 WHERE id = %s",
                (hid,),
            )

    # Add another co-occurring chunk and rebuild — the count changes but the
    # accumulated dynamics on the surviving pair must be preserved.
    _seed_chunk(idx, "c3", ["Aya", "Lumi"], "wing_aya", "ideas")
    second = store.compute_hallways_for_wing("wing_aya", min_count=2)

    assert len(second) == 1
    rec = second[0]
    assert rec["id"] == hid
    assert rec["co_occurrence_count"] == 3  # recomputed
    # Dynamics PRESERVED across the recompute (merge_dynamics).
    assert rec["strength"] == 3.3
    assert rec["stability"] == 1.7
    assert rec["access_count"] == 9


# ─────────────────────────────────────────────────────────────────────────────
# Wing-scan-bound: exceeding WING_SCAN_CAP logs a truncation (observability)
# ─────────────────────────────────────────────────────────────────────────────


def test_wing_scan_bound_logs_truncation(backend, team, monkeypatch, caplog):
    """Exceeding the pinned wing-scan cap clips the scan AND logs a truncation."""
    import mempalace.link_store_postgres as lsp

    # Shrink the cap so the test seeds only a handful of rows. The derive reads
    # the module-level constant inside _co_occurrence_for_wing.
    monkeypatch.setattr(lsp, "WING_SCAN_CAP", 4)

    idx = PostgresEntityIndex(backend, team=team)
    # 3 chunks * 2 entities = 6 entity rows > cap (4) -> truncation.
    for i in range(3):
        _seed_chunk(idx, f"c{i}", ["Aya", "Lumi"], "wing_aya", "diary")

    store = PostgresLinkStore(backend, team=team)
    with caplog.at_level("WARNING", logger="mempalace.link_store_postgres"):
        store.compute_hallways_for_wing("wing_aya", min_count=1)

    assert any("truncated" in r.getMessage() for r in caplog.records)


def test_wing_scan_cap_default_is_explicit_value():
    """The cap is an explicit chosen value, not silently inherited from closets.

    REBUILD_FETCH_CAP=500 / RECONCILE_SCAN_CAP=5000 are closet-group caps; the
    wing scan picks its own (larger) value.
    """
    from mempalace import closet_rebuild

    assert WING_SCAN_CAP == 50_000
    assert WING_SCAN_CAP not in (
        closet_rebuild.REBUILD_FETCH_CAP,
        closet_rebuild.RECONCILE_SCAN_CAP,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-team isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_per_team_isolation(backend):
    team_a = "dlnk_" + uuid.uuid4().hex[:12]
    team_b = "dlnk_" + uuid.uuid4().hex[:12]
    try:
        idx_a = PostgresEntityIndex(backend, team=team_a)
        _seed_chunk(idx_a, "a1", ["Aya", "Lumi"], "wing_shared", "diary")
        _seed_chunk(idx_a, "a2", ["Aya", "Lumi"], "wing_shared", "letters")

        store_a = PostgresLinkStore(backend, team=team_a)
        store_b = PostgresLinkStore(backend, team=team_b)

        store_a.rebuild_derived_links_for_wing("wing_shared", min_count=2)

        # Team A has the hallway; team B (no occurrences) has none.
        assert len(store_a.list_hallways("wing_shared")) == 1
        assert store_b.list_hallways("wing_shared") == []
    finally:
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_a)}" CASCADE')
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team_b)}" CASCADE')


# ─────────────────────────────────────────────────────────────────────────────
# entity_tunnels_for_wing consumes PG-sourced hallway records (adapted)
# ─────────────────────────────────────────────────────────────────────────────


def test_entity_tunnels_for_wing_consumes_pg_hallway_records(backend, team):
    """The shared palace_graph.entity_tunnels_for_wing reads the PG list_hallways
    output directly (entity_a/entity_b/wing dicts) and routes the tunnel write to
    the PG store via the injected callback — not re-implemented per backend."""
    from mempalace.palace_graph import entity_tunnels_for_wing

    idx = PostgresEntityIndex(backend, team=team)
    _seed_chunk(idx, "a1", ["Aya", "Lumi"], "wing_aya", "diary")
    _seed_chunk(idx, "a2", ["Aya", "Lumi"], "wing_aya", "letters")
    _seed_chunk(idx, "b1", ["Aya", "Lumi"], "wing_work", "notes")
    _seed_chunk(idx, "b2", ["Aya", "Lumi"], "wing_work", "tasks")

    store = PostgresLinkStore(backend, team=team)
    store.compute_hallways_for_wing("wing_aya", min_count=2)
    store.compute_hallways_for_wing("wing_work", min_count=2)

    # Feed the PG-sourced hallway records straight into the shared function.
    pg_hallways = store.list_hallways()
    assert pg_hallways  # PG-sourced, hallway-shaped

    created = entity_tunnels_for_wing(
        "wing_aya",
        pg_hallways,
        create_tunnel_fn=lambda **kw: store.create_tunnel(**kw),
    )

    # Each shared entity (Aya, Lumi) bridges wing_aya<->wing_work.
    assert len(created) == 2
    for t in created:
        assert t["kind"] == "entity"
        assert {t["source"]["wing"], t["target"]["wing"]} == {"wing_aya", "wing_work"}
        # Symmetric canonical id, same scheme the JSON path uses.
        assert t["id"] == _canonical_tunnel_id(
            t["source"]["wing"], t["source"]["room"], t["target"]["wing"], t["target"]["room"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-backend equivalence (structural): identical chunk entity sets + same
# known set -> identical co-occurrence pairs on chroma (hallways.py) and PG.
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_backend_equivalence_same_chunks_same_known_set(backend, team, monkeypatch):
    """With the SAME per-chunk entity sets fed to both backends, the chroma
    hallway computation and the PG derive produce the SAME co-occurrence pairs.

    Both paths call the same _extract_entities_for_metadata over the same chunk
    text with the SAME injected known set, so the per-chunk entity sets match;
    co-occurrence is then counted over those sets identically. This is the
    structural equivalence claim (identical chunking + identical known set), not
    arbitrary-corpus byte-equality.
    """
    from unittest.mock import MagicMock, patch

    with patch.dict("sys.modules", {"chromadb": MagicMock()}):
        import mempalace.hallways as hallways_mod
        from mempalace.miner import _extract_entities_for_metadata

    known = frozenset({"Aya", "Lumi"})
    chunk_texts = [
        "Aya and Lumi spoke at length about the project.",
        "Later Aya and Lumi reviewed the notes together.",
    ]

    # --- PG side: per-chunk extraction -> entity index -> derive ---
    idx = PostgresEntityIndex(backend, team=team)
    for i, text in enumerate(chunk_texts):
        ents = [e for e in _extract_entities_for_metadata(text, known=known).split(";") if e]
        idx.add([f"c{i}"], ents, "wing_aya", "diary")
    store = PostgresLinkStore(backend, team=team)
    pg_hallways = store.compute_hallways_for_wing("wing_aya", min_count=2)
    pg_pairs = sorted(
        (tuple(sorted([h["entity_a"], h["entity_b"]])), h["co_occurrence_count"])
        for h in pg_hallways
    )

    # --- chroma side: same chunks, same known set, hallways.py compute ---
    hallway_file = str(uuid.uuid4()) + ".json"  # never written (we read return)
    monkeypatch.setattr(hallways_mod, "_HALLWAY_FILE", "/tmp/" + hallway_file)
    drawers = []
    for text in chunk_texts:
        ents_str = _extract_entities_for_metadata(text, known=known)
        drawers.append({"wing": "wing_aya", "room": "diary", "entities": ents_str})
    col = MagicMock()
    col.count.return_value = len(drawers)

    def _get(limit=None, offset=0, include=None, **kw):
        page = drawers[offset : offset + limit] if limit is not None else drawers
        return {"ids": [f"d{i}" for i in range(offset, offset + len(page))], "metadatas": page}

    col.get.side_effect = _get
    chroma_hallways = hallways_mod.compute_hallways_for_wing("wing_aya", col=col, min_count=2)
    chroma_pairs = sorted(
        (tuple(sorted([h["entity_a"], h["entity_b"]])), h["co_occurrence_count"])
        for h in chroma_hallways
    )

    assert pg_pairs == chroma_pairs
    assert pg_pairs == [(("Aya", "Lumi"), 2)]
