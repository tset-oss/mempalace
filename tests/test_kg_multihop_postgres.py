"""CTE-correctness unit suite for the bounded multi-hop KG read.

Covers ``PostgresKnowledgeGraph.neighbors``: depth clamp, the triple-id cycle
guard (including a ``direction="both"`` cycle), the per-level frontier/expansion
cap plus the total-row cap and the ``truncated`` flag, point-in-time per-edge
temporal validity applied to every hop (including a disjoint-validity-window
path), direction at depth > 1, the pairwise ``target`` filter, the typed
``predicates`` filter, and per-team isolation. Skipped when no Postgres is
reachable.
"""

from __future__ import annotations

import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

from mempalace.backends.postgres import PostgresBackend, team_schema  # noqa: E402
from mempalace.knowledge_graph_postgres import PostgresKnowledgeGraph  # noqa: E402


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


@pytest.fixture(scope="module")
def backend():
    be = PostgresBackend(dsn=_dsn())
    yield be
    be.close()


@pytest.fixture()
def kg(backend):
    team = "t" + uuid.uuid4().hex[:10]
    graph = PostgresKnowledgeGraph(backend, team=team)
    yield graph
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
        conn.commit()


def _objects_by_hop(result):
    return {(r["hop"], r["object"]) for r in result["neighbors"]}


# -- depth cap ------------------------------------------------------------


def test_depth_cap_stops_at_requested_depth(kg):
    # A -> B -> C -> D -> E
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")
    kg.add_triple("C", "rel", "D")
    kg.add_triple("D", "rel", "E")

    two = kg.neighbors("A", depth=2, direction="outgoing")
    assert _objects_by_hop(two) == {(1, "B"), (2, "C")}
    assert "E" not in {r["object"] for r in two["neighbors"]}

    four = kg.neighbors("A", depth=4, direction="outgoing")
    assert _objects_by_hop(four) == {(1, "B"), (2, "C"), (3, "D"), (4, "E")}


def test_depth_is_clamped_to_four(kg):
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")
    kg.add_triple("C", "rel", "D")
    kg.add_triple("D", "rel", "E")
    kg.add_triple("E", "rel", "F")
    kg.add_triple("F", "rel", "G")

    out = kg.neighbors("A", depth=99, direction="outgoing")
    assert out["depth"] == 4
    assert max(r["hop"] for r in out["neighbors"]) == 4
    # G is six hops out; the clamp keeps it unreachable.
    assert "G" not in {r["object"] for r in out["neighbors"]}


# -- cycle guard ----------------------------------------------------------


def test_cycle_guard_terminates_outgoing(kg):
    # A -> B -> C -> A is a cycle; the triple-id path guard prevents revisiting
    # any edge, so the walk terminates with a finite result.
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")
    kg.add_triple("C", "rel", "A")

    out = kg.neighbors("A", depth=4, direction="outgoing")
    # Each distinct triple is traversed at most once along a path.
    assert {(r["hop"], r["subject"], r["object"]) for r in out["neighbors"]} == {
        (1, "A", "B"),
        (2, "B", "C"),
        (3, "C", "A"),
    }


def test_cycle_guard_terminates_both_direction(kg):
    # X <-> Y two-edge cycle traversed with direction="both".
    kg.add_triple("X", "rel", "Y")
    kg.add_triple("Y", "rel", "X")

    out = kg.neighbors("X", depth=4, direction="both")
    hops = {(r["hop"], r["subject"], r["object"]) for r in out["neighbors"]}
    # No hop beyond 2 is possible: every path of length 2 has already visited
    # both triple-ids, so the guard blocks any further step.
    assert max(h for h, _, _ in hops) == 2
    # The walk is finite (no runaway) and contains both edges.
    assert (1, "X", "Y") in hops
    assert (1, "Y", "X") in hops


# -- frontier / expansion cap + total-row cap ----------------------------


def test_expansion_cap_marks_truncated_and_bounds_frontier(kg):
    # A hub with more out-edges than the per-node expansion cap.
    for i in range(10):
        kg.add_triple("Hub", "rel", f"N{i:02d}")

    out = kg.neighbors("Hub", depth=1, direction="outgoing", expand_cap=3)
    assert out["truncated"] is True
    # The frontier is bounded: at most expand_cap + 1 edges kept per node
    # (the probe row that reveals the overflow).
    assert len(out["neighbors"]) <= 4
    assert len(out["neighbors"]) >= 3


def test_no_truncation_when_within_caps(kg):
    kg.add_triple("Hub", "rel", "One")
    kg.add_triple("Hub", "rel", "Two")
    out = kg.neighbors("Hub", depth=1, direction="outgoing", expand_cap=10)
    assert out["truncated"] is False
    assert len(out["neighbors"]) == 2


def test_total_row_limit_marks_truncated(kg):
    for i in range(6):
        kg.add_triple("Hub", "rel", f"N{i}")
    # expand_cap is generous; the total-row LIMIT alone clips here.
    out = kg.neighbors("Hub", depth=1, direction="outgoing", limit=3, expand_cap=100)
    assert out["truncated"] is True
    assert len(out["neighbors"]) == 3


# -- temporal as-of on every hop -----------------------------------------


def test_temporal_as_of_applies_to_every_hop(kg):
    # All three edges current; an as-of in their validity window returns all.
    kg.add_triple("A", "rel", "B", valid_from="2024-01-01")
    kg.add_triple("B", "rel", "C", valid_from="2024-01-01")
    kg.add_triple("C", "rel", "D", valid_from="2024-01-01")
    out = kg.neighbors("A", depth=3, direction="outgoing", as_of="2025-01-01")
    assert _objects_by_hop(out) == {(1, "B"), (2, "C"), (3, "D")}
    # Before any edge is valid, nothing is reachable.
    early = kg.neighbors("A", depth=3, direction="outgoing", as_of="2020-01-01")
    assert early["neighbors"] == []


def test_point_in_time_per_edge_expired_middle_edge(kg):
    # A->B current; B->C valid only in H1 2025; C->D current.
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C", valid_from="2025-01-01", valid_to="2025-06-01")
    kg.add_triple("C", "rel", "D")

    # As of today the middle edge is expired: the walk cannot pass B, so D is
    # unreachable even though A->B and C->D are individually current.
    today = kg.neighbors("A", depth=3, direction="outgoing", as_of="2026-06-06")
    assert _objects_by_hop(today) == {(1, "B")}

    # Before the middle edge expired, the whole chain is traversable.
    before = kg.neighbors("A", depth=3, direction="outgoing", as_of="2025-03-15")
    assert _objects_by_hop(before) == {(1, "B"), (2, "C"), (3, "D")}


def test_disjoint_validity_windows_each_valid_at_as_of(kg):
    # Two edges with DISJOINT-but-overlapping-at-as_of validity windows: each
    # is independently valid at 2025-03-15, so the whole path returns. The
    # per-edge predicate is what makes this work — a path is traversable iff
    # every edge is valid at the single as-of instant.
    kg.add_triple("P", "rel", "Q", valid_from="2025-01-01", valid_to="2025-12-31")
    kg.add_triple("Q", "rel", "R", valid_from="2025-02-01", valid_to="2025-04-30")

    both_valid = kg.neighbors("P", depth=2, direction="outgoing", as_of="2025-03-15")
    assert _objects_by_hop(both_valid) == {(1, "Q"), (2, "R")}

    # At a later as-of the second edge has expired while the first is still
    # valid: R drops out, Q remains.
    later = kg.neighbors("P", depth=2, direction="outgoing", as_of="2025-08-15")
    assert _objects_by_hop(later) == {(1, "Q")}


# -- direction at depth > 1 ----------------------------------------------


def test_direction_incoming_multi_hop(kg):
    # A -> B -> C; querying C incoming reaches B (hop 1) then A (hop 2).
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")
    out = kg.neighbors("C", depth=2, direction="incoming")
    assert {(r["hop"], r["subject"]) for r in out["neighbors"]} == {
        (1, "B"),
        (2, "A"),
    }
    assert all(r["direction"] == "incoming" for r in out["neighbors"])


def test_direction_both_multi_hop(kg):
    # A -> B -> C plus D -> B. From B "both" at depth 2 reaches A and C and D.
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")
    kg.add_triple("D", "rel", "B")
    out = kg.neighbors("B", depth=2, direction="both")
    reached = {
        r["object"] if r["direction"] == "outgoing" else r["subject"] for r in out["neighbors"]
    }
    assert {"A", "C", "D"} <= reached


# -- pairwise target filter ----------------------------------------------


def test_target_filter_returns_only_paths_reaching_target(kg):
    kg.add_triple("A", "rel", "B")
    kg.add_triple("B", "rel", "C")
    kg.add_triple("B", "rel", "X")  # a branch that does NOT reach the target
    kg.add_triple("C", "rel", "D")

    out = kg.neighbors("A", depth=4, direction="outgoing", target="D")
    objs = {r["object"] for r in out["neighbors"]}
    # Only the edge that lands on D is returned.
    assert objs == {"D"}
    assert all(r["object"] == "D" for r in out["neighbors"])


# -- typed predicates filter ---------------------------------------------


def test_predicates_filter_applied_every_hop(kg):
    kg.add_triple("A", "knows", "B")
    kg.add_triple("B", "knows", "C")
    kg.add_triple("B", "dislikes", "Z")  # off-predicate branch
    kg.add_triple("C", "knows", "D")

    out = kg.neighbors("A", depth=3, direction="outgoing", predicates=["knows"])
    objs = {r["object"] for r in out["neighbors"]}
    assert objs == {"B", "C", "D"}
    assert "Z" not in objs
    assert all(r["predicate"] == "knows" for r in out["neighbors"])


# -- edge names + triple-id path -----------------------------------------


def test_rows_carry_names_hop_and_triple_id_path(kg):
    t1 = kg.add_triple("Anna", "rel", "Bob")
    t2 = kg.add_triple("Bob", "rel", "Carol")
    out = kg.neighbors("Anna", depth=2, direction="outgoing")
    by_hop = {r["hop"]: r for r in out["neighbors"]}
    # Edge endpoints are NAMES (joined from kg_entities), not entity ids.
    assert by_hop[1]["subject"] == "Anna"
    assert by_hop[1]["object"] == "Bob"
    assert by_hop[2]["subject"] == "Bob"
    assert by_hop[2]["object"] == "Carol"
    # The path carries the triple-ids taken to reach each edge.
    assert by_hop[1]["path"] == [t1]
    assert by_hop[2]["path"] == [t1, t2]


# -- per-team isolation ---------------------------------------------------


def test_per_team_isolation(backend):
    a = "t" + uuid.uuid4().hex[:10]
    b = "t" + uuid.uuid4().hex[:10]
    kga = PostgresKnowledgeGraph(backend, team=a)
    kgb = PostgresKnowledgeGraph(backend, team=b)
    try:
        kga.add_triple("A", "rel", "B")
        kga.add_triple("B", "rel", "C")
        # Team B has its own (empty) chain anchor; the team-A chain is invisible.
        team_b_view = kgb.neighbors("A", depth=4, direction="outgoing")
        assert team_b_view["neighbors"] == []
        # Team A sees its own chain.
        team_a_view = kga.neighbors("A", depth=4, direction="outgoing")
        assert {r["object"] for r in team_a_view["neighbors"]} == {"B", "C"}
    finally:
        for t in (a, b):
            with psycopg.connect(_dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
                conn.commit()


# -- observability: bounded recursive frontier (EXPLAIN evidence) ---------


def test_explain_recursive_term_is_bounded(kg, backend):
    # A dense hub: EXPLAIN the recursive walk and assert the planner caps the
    # per-node expansion (the LIMIT on the lateral expansion bounds the
    # frontier; a final LIMIT alone would not). This is frontier-cap evidence,
    # not a final-output row count.
    for i in range(20):
        kg.add_triple("Hub", "rel", f"N{i:02d}")
        kg.add_triple(f"N{i:02d}", "rel", f"M{i:02d}")

    # Build the same SQL the method runs and EXPLAIN it.
    eid = kg._entity_id("Hub")
    params = {
        "start_id": eid,
        "max_depth": 4,
        "expand_probe": 6,
        "row_limit": 500,
    }
    triples = kg._triples()
    entities = kg._entities()

    def _expansion(frontier_expr, path_expr):
        leg_out = (
            "SELECT e.id AS triple_id, e.object AS next_id, e.predicate, "
            "e.valid_from, e.valid_to, e.confidence, e.source_closet, "
            "e.subject AS edge_subject, e.object AS edge_object, "
            "'outgoing' AS edge_direction "
            f"FROM {triples} e "
            f"WHERE e.subject = {frontier_expr} "
            f"AND NOT (e.id = ANY({path_expr}))"
        )
        return (
            "SELECT * FROM ( " + leg_out + " ) step ORDER BY step.triple_id LIMIT %(expand_probe)s"
        )

    anchor = _expansion("%(start_id)s", "ARRAY[]::text[]")
    recursive = _expansion("w.endpoint", "w.path")
    sql = (
        "EXPLAIN WITH RECURSIVE walk AS ( "
        "  SELECT s.triple_id, s.next_id AS endpoint, s.predicate, "
        "    s.valid_from, s.valid_to, s.confidence, s.source_closet, "
        "    s.edge_subject, s.edge_object, s.edge_direction, "
        "    1 AS hop, ARRAY[s.triple_id] AS path "
        "  FROM ( " + anchor + " ) s "
        "  UNION ALL "
        "  SELECT n.triple_id, n.next_id AS endpoint, n.predicate, "
        "    n.valid_from, n.valid_to, n.confidence, n.source_closet, "
        "    n.edge_subject, n.edge_object, n.edge_direction, "
        "    w.hop + 1 AS hop, w.path || n.triple_id AS path "
        "  FROM walk w "
        "  CROSS JOIN LATERAL ( " + recursive + " ) n "
        "  WHERE w.hop < %(max_depth)s "
        ") "
        "SELECT w.triple_id, w.hop, w.predicate, w.valid_from, w.valid_to, "
        "  w.confidence, w.source_closet, w.edge_direction, w.path, "
        "  subj.name, obj.name "
        "FROM walk w "
        f"JOIN {entities} subj ON w.edge_subject = subj.id "
        f"JOIN {entities} obj ON w.edge_object = obj.id "
        "ORDER BY w.hop, w.triple_id LIMIT %(row_limit)s + 1"
    )
    with backend._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            plan = "\n".join(row[0] for row in cur.fetchall())
    # The lateral expansion's per-node LIMIT shows up in the plan, proving the
    # recursive frontier is bounded rather than fully materialized.
    assert "Recursive Union" in plan
    assert "Limit" in plan


# -- 1-hop parity gate vs query_entity -----------------------------------
#
# ``neighbors(depth=1, direction=...)`` must agree with ``query_entity`` on the
# edge identity + validity tuple ``(subject, predicate, object, valid_from,
# valid_to)`` ONLY. The fields ``direction``, ``current``, ``confidence`` and
# ``source_closet`` are EXCLUDED from the comparison: ``neighbors`` is allowed to
# shape them differently; only the edge + its validity window must match.
#
# ``query_entity`` (knowledge_graph_postgres.py:281-336) emits the OUTGOING block
# THEN the INCOMING block, concatenated and NOT deduped, so a symmetric pair
# X<->Y surfaces both edges from each side. The parity assertion preserves that
# block ordering: every outgoing tuple precedes every incoming tuple in both
# methods. Within a single block the underlying SQL is unordered (no ``ORDER
# BY``), so each block is compared as a sorted multiset to stay deterministic
# while still asserting the non-deduped block-concatenation order.

_PARITY_TUPLE_KEYS = ("subject", "predicate", "object", "valid_from", "valid_to")


def _project(row):
    return tuple(row[k] for k in _PARITY_TUPLE_KEYS)


def _split_blocks(rows):
    """Split a list of edge dicts into (outgoing-tuples, incoming-tuples).

    Preserves the original sequence so the caller can assert that the outgoing
    block precedes the incoming block (the concatenated, non-deduped ordering
    ``query_entity`` produces).
    """
    outgoing = [_project(r) for r in rows if r["direction"] == "outgoing"]
    incoming = [_project(r) for r in rows if r["direction"] == "incoming"]
    return outgoing, incoming


def _assert_outgoing_precedes_incoming(rows):
    """The outgoing block is contiguous and comes before the incoming block."""
    dirs = [r["direction"] for r in rows]
    if "incoming" in dirs and "outgoing" in dirs:
        # No outgoing edge may appear after the first incoming edge.
        first_incoming = dirs.index("incoming")
        assert "outgoing" not in dirs[first_incoming:], (
            "outgoing block must precede incoming block (concatenated, non-deduped)"
        )


@pytest.mark.parametrize("direction", ["outgoing", "incoming", "both"])
@pytest.mark.parametrize("as_of", [None, "2025-01-01"])
def test_neighbors_depth1_parity_with_query_entity(kg, direction, as_of):
    # Give every edge a validity window that the fixed as-of sits inside, so
    # both the as_of=None and as_of=fixed runs return the full edge set and the
    # parity comparison is NON-VACUOUS for every parametrization.
    kg.add_triple("Hub", "rel", "Out1", valid_from="2024-01-01")
    kg.add_triple("Hub", "rel", "Out2", valid_from="2024-01-01")
    kg.add_triple("In1", "rel", "Hub", valid_from="2024-01-01")
    kg.add_triple("In2", "rel", "Hub", valid_from="2024-01-01")
    kg.add_triple("Hub", "knows", "Peer", valid_from="2024-01-01")
    kg.add_triple("Peer", "knows", "Hub", valid_from="2024-01-01")

    expected = kg.query_entity("Hub", as_of=as_of, direction=direction)
    got = kg.neighbors("Hub", depth=1, direction=direction, as_of=as_of)["neighbors"]

    exp_out, exp_in = _split_blocks(expected)
    got_out, got_in = _split_blocks(got)

    # Edge identity + validity match per block (multiset; within-block SQL order
    # is undefined so we sort each block before comparing).
    assert sorted(got_out) == sorted(exp_out)
    assert sorted(got_in) == sorted(exp_in)

    # The concatenated NON-deduped block ordering is preserved: outgoing block
    # first, then incoming block, in BOTH methods.
    _assert_outgoing_precedes_incoming(expected)
    _assert_outgoing_precedes_incoming(got)

    # Non-vacuous: the relevant block(s) actually carry edges.
    if direction in ("outgoing", "both"):
        assert len(got_out) >= 2
    if direction in ("incoming", "both"):
        assert len(got_in) >= 2
    if direction == "both":
        # The symmetric Hub<->Peer pair survives the non-deduped concatenation:
        # "Peer" appears as an outgoing object AND as an incoming subject.
        assert ("Hub", "knows", "Peer", "2024-01-01", None) in got_out
        assert ("Peer", "knows", "Hub", "2024-01-01", None) in got_in


def test_neighbors_depth1_parity_excludes_nonidentity_fields(kg):
    # A confidence/source_closet difference must NOT break parity: the compared
    # tuple is edge identity + validity only.
    kg.add_triple("Hub", "rel", "Out1", confidence=0.5, source_closet="c1")
    kg.add_triple("In1", "rel", "Hub", confidence=0.9, source_closet="c2")

    expected = kg.query_entity("Hub", direction="both")
    got = kg.neighbors("Hub", depth=1, direction="both")["neighbors"]

    exp_out, exp_in = _split_blocks(expected)
    got_out, got_in = _split_blocks(got)
    assert sorted(got_out) == sorted(exp_out)
    assert sorted(got_in) == sorted(exp_in)
    # Both methods agree on the identity tuple even though confidence differs
    # per edge (3rd assertion is the proof the excluded fields are irrelevant).
    assert sorted(_project(r) for r in got) == sorted(_project(r) for r in expected)


# -- per-team isolation of the 1-hop read --------------------------------


def test_neighbors_depth1_per_team_isolation(backend):
    # A chain written in team A is invisible to a neighbors() call routed to a
    # distinct team B; team A still sees its own edges.
    a = "t" + uuid.uuid4().hex[:10]
    b = "t" + uuid.uuid4().hex[:10]
    kga = PostgresKnowledgeGraph(backend, team=a)
    kgb = PostgresKnowledgeGraph(backend, team=b)
    try:
        kga.add_triple("Hub", "rel", "Out1")
        kga.add_triple("In1", "rel", "Hub")

        # Team B has never written "Hub": the 1-hop read is empty in every
        # direction (the team-A chain does not leak across the schema boundary).
        for direction in ("outgoing", "incoming", "both"):
            view_b = kgb.neighbors("Hub", depth=1, direction=direction)
            assert view_b["neighbors"] == []

        # Team A sees its own edges, and they match query_entity within team A.
        got_a = kga.neighbors("Hub", depth=1, direction="both")["neighbors"]
        exp_a = kga.query_entity("Hub", direction="both")
        assert sorted(_project(r) for r in got_a) == sorted(_project(r) for r in exp_a)
        assert {r["object"] for r in got_a if r["direction"] == "outgoing"} == {"Out1"}
        assert {r["subject"] for r in got_a if r["direction"] == "incoming"} == {"In1"}
    finally:
        for t in (a, b):
            with psycopg.connect(_dsn()) as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
                conn.commit()


# -- truncation at the cap for a 1-hop fan-out ---------------------------


def test_neighbors_depth1_truncates_at_expand_cap(kg):
    # A 1-hop fan-out exceeding expand_cap returns truncated=True (the per-node
    # expansion cap clips the frontier even at a single hop).
    for i in range(10):
        kg.add_triple("Hub", "rel", f"N{i:02d}")

    out = kg.neighbors("Hub", depth=1, direction="outgoing", expand_cap=3)
    assert out["truncated"] is True
    # The kept frontier is bounded at expand_cap + 1 (the overflow probe row).
    assert len(out["neighbors"]) <= 4

    # Within the cap there is no truncation.
    ok = kg.neighbors("Hub", depth=1, direction="outgoing", expand_cap=50)
    assert ok["truncated"] is False
    assert len(ok["neighbors"]) == 10
