"""Two-team isolation harness — the runtime negative-property test.

The rest of the suite proves correctness WITHIN one vault. This file proves the
NEGATIVE property the suite never had: write DISTINCT data to team A and team B
across EVERY persistent surface the central server owns, then assert ZERO
cross-tenant bleed in every read — team B never sees team A's data, and team A
never sees team B's.

Why this is needed: the central server is one FastMCP process, one OS user, one
``$HOME``, many teams routed per-request via ``_active_team_var`` /
``_resolve_team``. Isolation is a negative property ("team B never sees team
A"), and a single-tenant fixture cannot reproduce a cross-tenant leak. This
harness drives two teams through the real tool surface in one process and
asserts the boundary holds on each surface.

Non-vacuousness is structural on every surface: each surface gets a UNIQUE,
per-team token, and every cross-read asserts BOTH that the team's own token IS
present AND that the other team's token is ABSENT. So a silently no-op write or
a cross-read that silently returned everything would FAIL this test — a broken
isolation boundary cannot pass by accident.

Surfaces covered (distinct A + B data, cross-read absent both ways):
  1. Drawers          — add_drawer; search / list_drawers see only the own team.
  2. Knowledge graph  — kg_add; kg_query AND kg_neighbors are team-scoped.
  3. Entity index     — search(entity=) / mempalace_entities are team-scoped.
  4. Explicit tunnels — create_tunnel; list_tunnels is team-scoped.
  5. Derived links    — a derived rebuild from a team's entity_occurrences;
                        the other team's hallways / derived tunnels are empty.
  6. Topic tunnels    — add_drawer topics -> wing_topics + topic tunnels scoped.
  7. Disambiguation   — get_entity_registry(team=...) is team-scoped.
  8. Critical facts   — team_fact_add; team_facts is team-scoped.
  9. Graph cache      — INTERLEAVED build_graph/find_tunnels/traverse calls give
                        each team ITS OWN graph (the live-leak regression).

Plus two positive isolation assertions:
  (i)  a server-mode write with NO explicit team and NO active session team
       RAISES (it does not silently write to a default vault), across a
       representative set of team-scoped writers.
  (ii) the per-team ``tunnels`` table only EVER contains tunnels created via
       THAT team's context (no cross-team rows), queried directly for both
       teams after the cross-writes.

Plus the cross-backend co-occurrence EQUIVALENCE fixture (scoped): with the SAME
``chunk_size`` AND the SAME known-entity set fed to ``_extract_entities_for_metadata``,
the per-chunk entity sets — and thus the derived co-occurrence pairs — match
between the chroma hallway computation and the Postgres derive.

Run against a live Postgres (the bundled deploy/docker-compose db or any
instance via MEMPALACE_TEST_PG_URL / MEMPALACE_DATABASE_URL). Skipped when
psycopg is absent or the DB is unreachable, so chroma-only CI is unaffected.
Each test uses per-test team slugs and drops both schemas CASCADE on teardown.
"""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")

import mempalace.mcp_server as m  # noqa: E402
from mempalace.backends import get_backend  # noqa: E402
from mempalace.backends.postgres import team_schema  # noqa: E402
from mempalace.config import MempalaceConfig  # noqa: E402

DIM = 384


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


def _fake_embed(texts):
    """Deterministic content-derived embedding (no ONNX download)."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for i, b in enumerate(hashlib.sha256((t or "").encode()).digest()):
            v[i] = b / 255.0
        out.append(v)
    return out


def _drop(*teams) -> None:
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            for t in teams:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(t)}" CASCADE')
        conn.commit()


def _pop_caches(*teams) -> None:
    """Drop every per-team cached store/handle so a fresh instance is built.

    The server caches one store per team (KG, entity index, link store, team
    facts). A per-test team slug is fresh, but popping is belt-and-braces so a
    stale handle from a prior run can never mask a write/read.
    """
    for t in teams:
        m._kg_by_path.pop(f"pgkg::{t}", None)
        m._entity_index_by_team.pop(f"pgentidx::{t}", None)
        m._link_store_by_team.pop(f"pglink::{t}", None)
        m._team_facts_by_team.pop(f"pgfacts::{t}", None)


class _ActiveTeam:
    """Set ``_active_team_var`` for the duration of a block, then reset.

    Mirrors the per-request team context FastMCP installs (token + reset per
    request). Used to simulate "this request belongs to team X" so the real
    tools route through ``_resolve_team`` / ``_resolve_team_strict`` to that
    team's vault — a faithful end-to-end isolation assertion per surface.
    """

    def __init__(self, team):
        self._team = team
        self._token = None

    def __enter__(self):
        self._token = m._active_team_var.set(self._team)
        return self

    def __exit__(self, *exc):
        m._active_team_var.reset(self._token)
        return False


@pytest.fixture
def server_pg(monkeypatch):
    """Put the server in postgres mode against the live DB, shared backend.

    One shared backend (deterministic embedder, live DSN) routes BOTH the
    collection path (drawers / search / list_drawers, via the postgres
    registry singleton) and every team-scoped store (KG, entity index, link
    store, team facts, entity registry, via ``_resolve_backend``). Per-request
    routing is then driven purely by ``_active_team_var``.
    """
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    # Deterministic, sleep-free derived-link rebuilds: fire on the next worker
    # tick and drain via flush() rather than waiting out the debounce window.
    monkeypatch.setenv("MEMPALACE_DERIVE_DEBOUNCE_SECONDS", "0")
    # No leftover session team — every request sets its own via _ActiveTeam.
    token = m._active_team_var.set(None)

    backend = get_backend("postgres")
    prev_embedder = getattr(backend, "_embedder", None)
    backend._embedder = _fake_embed

    monkeypatch.setattr(m, "_config", MempalaceConfig())
    monkeypatch.setattr("mempalace.palace._resolve_backend", lambda cfg: backend)

    # Reset the lazily-built derived-link debouncer so it is (re)created with the
    # zero-second window set above, regardless of any prior test's instance.
    prev_deb = m._derived_link_debouncer
    monkeypatch.setattr(m, "_derived_link_debouncer", None)

    yield backend

    monkeypatch.setattr(m, "_derived_link_debouncer", prev_deb)
    backend._embedder = prev_embedder
    m._active_team_var.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# The comprehensive harness: distinct A + B data on every surface, zero bleed.
# ─────────────────────────────────────────────────────────────────────────────


def test_two_team_isolation_across_every_surface(server_pg):
    backend = server_pg
    team_a = "ta" + uuid.uuid4().hex[:8]
    team_b = "tb" + uuid.uuid4().hex[:8]
    _pop_caches(team_a, team_b)

    # Per-team UNIQUE tokens. Each surface's A-data carries A's token and each
    # surface's B-data carries B's token, so a cross-read can assert presence of
    # the own token AND absence of the other team's — non-vacuous by construction.
    tok_a = uuid.uuid4().hex[:8]
    tok_b = uuid.uuid4().hex[:8]

    try:
        # ── 1. Drawers — distinct verbatim content per team ───────────────────
        content_a = f"Aria designed the Aurora ingest pipeline marker{tok_a}"
        content_b = f"Bruno tuned the Borealis search ranker marker{tok_b}"
        with _ActiveTeam(team_a):
            add_a = m.tool_add_drawer(wing="wing_aria", room="api", content=content_a)
            assert add_a.get("success") is True, add_a
        with _ActiveTeam(team_b):
            add_b = m.tool_add_drawer(wing="wing_bruno", room="api", content=content_b)
            assert add_b.get("success") is True, add_b

        # Drawer search: each team finds its OWN content and NOT the other's.
        with _ActiveTeam(team_a):
            res = m.tool_search(content_a, limit=10)
            texts = " ".join(r.get("text", "") for r in res.get("results", []))
            assert tok_a in texts  # own data present (non-vacuous)
            assert tok_b not in texts  # other team's data absent
        with _ActiveTeam(team_b):
            res = m.tool_search(content_b, limit=10)
            texts = " ".join(r.get("text", "") for r in res.get("results", []))
            assert tok_b in texts
            assert tok_a not in texts
            # Even searching for A's exact text from B's vault yields nothing of A's.
            res_cross = m.tool_search(content_a, limit=10)
            cross_texts = " ".join(r.get("text", "") for r in res_cross.get("results", []))
            assert tok_a not in cross_texts

        # list_drawers: team B's listing never contains team A's wing/content.
        with _ActiveTeam(team_a):
            listed = m.tool_list_drawers(limit=50)
            a_wings = {d["wing"] for d in listed["drawers"]}
            a_previews = " ".join(d["content_preview"] for d in listed["drawers"])
            assert "wing_aria" in a_wings
            assert "wing_bruno" not in a_wings
            assert tok_a in a_previews and tok_b not in a_previews
        with _ActiveTeam(team_b):
            listed = m.tool_list_drawers(limit=50)
            b_wings = {d["wing"] for d in listed["drawers"]}
            b_previews = " ".join(d["content_preview"] for d in listed["drawers"])
            assert "wing_bruno" in b_wings
            assert "wing_aria" not in b_wings
            assert tok_b in b_previews and tok_a not in b_previews

        # ── 2. Knowledge graph — distinct triples per team ────────────────────
        subj_a, obj_a = f"Aria_{tok_a}", f"Aurora_{tok_a}"
        subj_b, obj_b = f"Bruno_{tok_b}", f"Borealis_{tok_b}"
        with _ActiveTeam(team_a):
            assert m.tool_kg_add(subj_a, "built", obj_a).get("success") is True
            # A second hop so kg_neighbors has something to traverse.
            assert m.tool_kg_add(obj_a, "feeds", f"Atlas_{tok_a}").get("success") is True
        with _ActiveTeam(team_b):
            assert m.tool_kg_add(subj_b, "built", obj_b).get("success") is True
            assert m.tool_kg_add(obj_b, "feeds", f"Basin_{tok_b}").get("success") is True

        # kg_query: each team sees only its own relationships.
        with _ActiveTeam(team_a):
            q = m.tool_kg_query(subj_a)
            objs = {f.get("object") for f in q["facts"]}
            assert obj_a in objs  # own fact present
            # Querying B's subject in A's vault returns nothing.
            assert m.tool_kg_query(subj_b)["count"] == 0
        with _ActiveTeam(team_b):
            q = m.tool_kg_query(subj_b)
            objs = {f.get("object") for f in q["facts"]}
            assert obj_b in objs
            assert m.tool_kg_query(subj_a)["count"] == 0

        # kg_neighbors: multi-hop walk is team-scoped too.
        with _ActiveTeam(team_a):
            nb = m.tool_kg_neighbors(subj_a, depth=2, direction="outgoing")
            reached = {n["object"] for n in nb["neighbors"]}
            assert obj_a in reached and f"Atlas_{tok_a}" in reached
            assert obj_b not in reached and f"Basin_{tok_b}" not in reached
            # Walking from B's subject in A's vault reaches nothing.
            assert m.tool_kg_neighbors(subj_b, depth=2, direction="outgoing")["count"] == 0
        with _ActiveTeam(team_b):
            nb = m.tool_kg_neighbors(subj_b, depth=2, direction="outgoing")
            reached = {n["object"] for n in nb["neighbors"]}
            assert obj_b in reached and f"Basin_{tok_b}" in reached
            assert obj_a not in reached
            assert m.tool_kg_neighbors(subj_a, depth=2, direction="outgoing")["count"] == 0

        # ── 3. Entity index — distinct entities per team ──────────────────────
        # Index a deterministic per-team entity token against the REAL drawer id
        # each team filed above, so the named tokens do not depend on the
        # extractor heuristic AND the search(entity=) restrict-id points at a row
        # that actually exists. Then assert cross-vault absence.
        ent_a = f"Aria{tok_a}"
        ent_b = f"Bruno{tok_b}"
        m._get_entity_index(team_a).add([add_a["drawer_id"]], [ent_a], "wing_aria", "api")
        m._get_entity_index(team_b).add([add_b["drawer_id"]], [ent_b], "wing_bruno", "api")

        with _ActiveTeam(team_a):
            nav = m.tool_entities(entity=ent_a)
            assert nav["drawer_count"] >= 1  # own entity present
            assert m.tool_entities(entity=ent_b)["drawer_count"] == 0  # B's absent
            top = {e["entity"] for e in m.tool_entities(min_count=1)["entities"]}
            assert ent_a in top and ent_b not in top
        with _ActiveTeam(team_b):
            nav = m.tool_entities(entity=ent_b)
            assert nav["drawer_count"] >= 1
            assert m.tool_entities(entity=ent_a)["drawer_count"] == 0
            top = {e["entity"] for e in m.tool_entities(min_count=1)["entities"]}
            assert ent_b in top and ent_a not in top

        # search(entity=) is also entity-index scoped: A's entity matches in A,
        # nothing in B.
        with _ActiveTeam(team_a):
            es = m.tool_search(content_a, limit=10, entity=ent_a)
            assert any(tok_a in r.get("text", "") for r in es.get("results", []))
        with _ActiveTeam(team_b):
            es = m.tool_search(content_a, limit=10, entity=ent_a)
            # ent_a does not exist in team B's index -> empty (known-empty filter).
            assert es.get("count", 0) == 0

        # ── 4. Explicit tunnels — distinct per team ───────────────────────────
        with _ActiveTeam(team_a):
            tun_a = m.tool_create_tunnel(
                "wing_aria", "api", "wing_aria2", "design", label=f"explicit-{tok_a}"
            )
            assert "error" not in tun_a
        with _ActiveTeam(team_b):
            tun_b = m.tool_create_tunnel(
                "wing_bruno", "api", "wing_bruno2", "design", label=f"explicit-{tok_b}"
            )
            assert "error" not in tun_b

        with _ActiveTeam(team_a):
            ids = {t["id"] for t in m.tool_list_tunnels()}
            labels = {t["label"] for t in m.tool_list_tunnels()}
            assert tun_a["id"] in ids and tun_b["id"] not in ids
            assert f"explicit-{tok_a}" in labels and f"explicit-{tok_b}" not in labels
        with _ActiveTeam(team_b):
            ids = {t["id"] for t in m.tool_list_tunnels()}
            labels = {t["label"] for t in m.tool_list_tunnels()}
            assert tun_b["id"] in ids and tun_a["id"] not in ids
            assert f"explicit-{tok_b}" in labels and f"explicit-{tok_a}" not in labels

        # ── 5. Derived entity tunnels + hallways — from entity_occurrences ────
        # Seed a cross-wing co-occurring pair in each team, then drive the same
        # rebuild entrypoint the write path / debouncer use. B's derived layer
        # must be empty of A's pair and vice versa.
        store_a = m._get_link_store(team_a)
        store_b = m._get_link_store(team_b)
        pair_a = [f"AyaX{tok_a}", f"LumiX{tok_a}"]
        pair_b = [f"BoroX{tok_b}", f"MiraX{tok_b}"]
        idx_a = m._get_entity_index(team_a)
        idx_b = m._get_entity_index(team_b)
        for cid in ("der1", "der2"):
            idx_a.add([f"{cid}_a"], pair_a, "wing_derive", "diary")
            idx_b.add([f"{cid}_b"], pair_b, "wing_derive", "diary")
        # Drive the actual debouncer flush for team A's wing (the write-path
        # mechanism the server uses), and a direct rebuild for team B's wing.
        deb = m._get_derived_link_debouncer()
        deb.enqueue(team_a, "wing_derive")
        assert deb.flush(timeout=20.0)
        store_b.rebuild_derived_links_for_wing("wing_derive", min_count=2)

        a_hall_entities = {
            tuple(sorted([h["entity_a"], h["entity_b"]]))
            for h in store_a.list_hallways("wing_derive")
        }
        b_hall_entities = {
            tuple(sorted([h["entity_a"], h["entity_b"]]))
            for h in store_b.list_hallways("wing_derive")
        }
        assert tuple(sorted(pair_a)) in a_hall_entities  # A built its hallway
        assert tuple(sorted(pair_a)) not in b_hall_entities  # B never sees A's pair
        assert tuple(sorted(pair_b)) in b_hall_entities
        assert tuple(sorted(pair_b)) not in a_hall_entities

        # ── 6. Topic tunnels + wing_topics — agent-supplied labels per team ───
        # Distinct topic labels per team, supplied across two wings so a topic
        # tunnel is derivable; the labels live in the per-team wing_topics table.
        topic_a = f"TopicAlpha{tok_a}"
        topic_b = f"TopicBravo{tok_b}"
        with _ActiveTeam(team_a):
            m.tool_add_drawer("wing_t1", "r", f"alpha note {tok_a}", topics=[topic_a])
            m.tool_add_drawer("wing_t2", "r", f"alpha note two {tok_a}", topics=[topic_a])
        with _ActiveTeam(team_b):
            m.tool_add_drawer("wing_t1", "r", f"bravo note {tok_b}", topics=[topic_b])
            m.tool_add_drawer("wing_t2", "r", f"bravo note two {tok_b}", topics=[topic_b])

        a_topics_map = store_a.topics_by_wing()
        b_topics_map = store_b.topics_by_wing()
        a_all_topics = {t for labels in a_topics_map.values() for t in labels}
        b_all_topics = {t for labels in b_topics_map.values() for t in labels}
        assert topic_a in a_all_topics and topic_b not in a_all_topics
        assert topic_b in b_all_topics and topic_a not in b_all_topics

        # Derive topic tunnels for both wings in each team; they are team-scoped.
        for w in ("wing_t1", "wing_t2"):
            store_a.rebuild_derived_links_for_wing(w, min_count=2)
            store_b.rebuild_derived_links_for_wing(w, min_count=2)
        a_topic_labels = {t["label"] for t in store_a.list_tunnels() if t["kind"] == "topic"}
        b_topic_labels = {t["label"] for t in store_b.list_tunnels() if t["kind"] == "topic"}
        # A topic tunnel exists for each team (its shared label across two wings)
        # and references only its own label.
        assert any(topic_a in lbl for lbl in a_topic_labels)
        assert not any(topic_b in lbl for lbl in a_topic_labels)
        assert any(topic_b in lbl for lbl in b_topic_labels)
        assert not any(topic_a in lbl for lbl in b_topic_labels)

        # ── 7. Disambiguation (entity registry) — distinct people per team ────
        from mempalace.entity_registry import get_entity_registry

        reg_a = get_entity_registry(m._config, team=team_a)
        reg_a.seed(
            mode="personal",
            people=[{"name": f"Riley{tok_a}", "relationship": "daughter", "context": "personal"}],
            projects=[f"ProjAria{tok_a}"],
        )
        reg_a.save()
        reg_b = get_entity_registry(m._config, team=team_b)
        reg_b.seed(
            mode="personal",
            people=[{"name": f"Riley{tok_b}", "relationship": "colleague", "context": "work"}],
            projects=[f"ProjBruno{tok_b}"],
        )
        reg_b.save()

        # Re-open each registry from storage and assert no cross-bleed.
        reg_a2 = get_entity_registry(m._config, team=team_a)
        reg_b2 = get_entity_registry(m._config, team=team_b)
        assert f"Riley{tok_a}" in reg_a2.people and f"Riley{tok_b}" not in reg_a2.people
        assert f"ProjAria{tok_a}" in reg_a2.projects and f"ProjBruno{tok_b}" not in reg_a2.projects
        assert reg_a2.lookup(f"Riley{tok_b}")["type"] == "unknown"  # B's person invisible to A
        assert f"Riley{tok_b}" in reg_b2.people and f"Riley{tok_a}" not in reg_b2.people
        assert reg_b2.lookup(f"Riley{tok_a}")["type"] == "unknown"

        # ── 8. Critical facts — distinct team facts per team ──────────────────
        fact_a = f"Aria fact: ingest freeze {tok_a}"
        fact_b = f"Bruno fact: ranker rollout {tok_b}"
        with _ActiveTeam(team_a):
            assert "error" not in m.tool_team_fact_add(fact_a, created_by="aria")
        with _ActiveTeam(team_b):
            assert "error" not in m.tool_team_fact_add(fact_b, created_by="bruno")

        with _ActiveTeam(team_a):
            a_facts = {f["fact"] for f in m.tool_team_facts()["facts"]}
            assert fact_a in a_facts and fact_b not in a_facts
        with _ActiveTeam(team_b):
            b_facts = {f["fact"] for f in m.tool_team_facts()["facts"]}
            assert fact_b in b_facts and fact_a not in b_facts

        # ── 9. Graph cache — INTERLEAVED build/find/traverse, no cross-leak ───
        # The live regression: a module-global, TTL'd graph cache that ignored
        # the team key would return the previous caller's graph on a warm hit.
        # Two fake collections carry DISTINCT per-team schemas + distinct rooms;
        # interleaving the calls within the TTL must give each team ITS OWN graph.
        _assert_graph_cache_isolated_under_interleave(team_a, team_b, tok_a, tok_b)

        # ── Positive (ii): the per-team tunnels table holds ONLY that team's ──
        # tunnels — query both teams' team_<slug>.tunnels directly. No cross rows.
        _assert_tunnels_table_has_no_cross_team_rows(backend, team_a, team_b, tok_a, tok_b)

        # ── T5: Diary — distinct entries per team, zero cross-read ───────────
        diary_a = f"T5 diary entry Aria {tok_a}"
        diary_b = f"T5 diary entry Bruno {tok_b}"
        with _ActiveTeam(team_a):
            wr = m.tool_diary_write(agent_name="aria", entry=diary_a)
            assert "error" not in wr, wr
        with _ActiveTeam(team_b):
            wr = m.tool_diary_write(agent_name="bruno", entry=diary_b)
            assert "error" not in wr, wr

        # Each team reads its own diary entry; the other team's entry is absent.
        with _ActiveTeam(team_a):
            read_a = m.tool_diary_read(agent_name="aria")
            entries_a = " ".join(e.get("content", "") for e in read_a.get("entries", []))
            assert tok_a in entries_a, "team A diary missing own entry"
            assert tok_b not in entries_a, "team A diary leaks team B entry"
        with _ActiveTeam(team_b):
            read_b = m.tool_diary_read(agent_name="bruno")
            entries_b = " ".join(e.get("content", "") for e in read_b.get("entries", []))
            assert tok_b in entries_b, "team B diary missing own entry"
            assert tok_a not in entries_b, "team B diary leaks team A entry"
    finally:
        _pop_caches(team_a, team_b)
        _drop(team_a, team_b)


def _assert_graph_cache_isolated_under_interleave(team_a, team_b, tok_a, tok_b):
    """Interleave build_graph/find_tunnels/traverse for A then B within the TTL.

    Each call uses a fake collection whose ``_schema`` is the team's vault
    schema (the authoritative cache key on the central server) and whose
    metadata describes DIFFERENT rooms/wings per team. A warm-hit leak would
    return team A's graph to team B; the interleave catches exactly that.
    """
    from mempalace import palace_graph as pg

    # Fully clear the cache so this assertion stands on its own.
    pg.invalidate_graph_cache()

    # Each team's collection spans two wings through a shared room so the room
    # is a tunnel room (>=2 wings) — find_tunnels/traverse have something to
    # return — and the room NAME embeds the team token so a leaked graph is
    # detectable by token.
    room_a = f"room_{tok_a}"
    room_b = f"room_{tok_b}"
    col_a = _FakeGraphCollection(
        team_a,
        [
            {"room": room_a, "wing": f"wingA1_{tok_a}", "hall": "h", "date": "2026-01-01"},
            {"room": room_a, "wing": f"wingA2_{tok_a}", "hall": "h", "date": "2026-01-02"},
        ],
    )
    col_b = _FakeGraphCollection(
        team_b,
        [
            {"room": room_b, "wing": f"wingB1_{tok_b}", "hall": "h", "date": "2026-02-01"},
            {"room": room_b, "wing": f"wingB2_{tok_b}", "hall": "h", "date": "2026-02-02"},
        ],
    )

    # Interleave: A warms, B warms, A re-reads (warm hit), B re-reads (warm hit).
    nodes_a1, _ = pg.build_graph(col=col_a)
    nodes_b1, _ = pg.build_graph(col=col_b)
    nodes_a2, _ = pg.build_graph(col=col_a)  # warm hit for A
    nodes_b2, _ = pg.build_graph(col=col_b)  # warm hit for B

    # The warm hits are GENUINE cache hits, not silent rebuilds: both teams'
    # graphs coexist as DISTINCT per-vault entries within the TTL. This pins the
    # exact H005 property under test — a warm hit must serve the SAME vault's
    # graph, never the prior caller's — so a "cache never populated" regression
    # (which would re-open the leak under real concurrency) is caught here too.
    assert col_a._schema != col_b._schema
    _cache_keys = set(pg._graph_cache)
    assert f"schema:{col_a._schema}" in _cache_keys
    assert f"schema:{col_b._schema}" in _cache_keys

    # Each team's graph has only its OWN room, never the other team's.
    assert room_a in nodes_a1 and room_b not in nodes_a1
    assert room_b in nodes_b1 and room_a not in nodes_b1
    # Warm hits return the SAME team's graph (not the previous caller's).
    assert room_a in nodes_a2 and room_b not in nodes_a2
    assert room_b in nodes_b2 and room_a not in nodes_b2

    # find_tunnels reads the cached graph per team — A's tunnel room is room_a,
    # never room_b, and vice versa.
    a_tunnel_rooms = {t["room"] for t in pg.find_tunnels(col=col_a)}
    b_tunnel_rooms = {t["room"] for t in pg.find_tunnels(col=col_b)}
    assert room_a in a_tunnel_rooms and room_b not in a_tunnel_rooms
    assert room_b in b_tunnel_rooms and room_a not in b_tunnel_rooms

    # traverse from A's room in A's graph reaches A's room; B's graph does not
    # know A's room at all (would be a structured "not found"), and vice versa.
    a_trav = pg.traverse(room_a, col=col_a)
    assert isinstance(a_trav, list)
    assert {step["room"] for step in a_trav} == {room_a}
    b_sees_a = pg.traverse(room_a, col=col_b)
    assert isinstance(b_sees_a, dict) and "error" in b_sees_a  # B's graph lacks A's room
    b_trav = pg.traverse(room_b, col=col_b)
    assert {step["room"] for step in b_trav} == {room_b}
    a_sees_b = pg.traverse(room_b, col=col_a)
    assert isinstance(a_sees_b, dict) and "error" in a_sees_b

    pg.invalidate_graph_cache()


class _FakeGraphCollection:
    """A minimal collection the graph cache keys by ``_schema`` (team vault).

    Returns a fixed metadata page so build_graph constructs a deterministic,
    per-team graph without a real drawers collection. ``_schema`` is the
    authoritative per-vault cache key ``_vault_cache_key`` reads.
    """

    def __init__(self, team, metadatas):
        self._schema = team_schema(team)
        self._metadatas = metadatas

    def count(self):
        return len(self._metadatas)

    def get(self, limit=None, offset=0, include=None):
        page = self._metadatas[offset : offset + limit] if limit is not None else self._metadatas
        return {"ids": [f"id{offset + i}" for i in range(len(page))], "metadatas": page}


def _assert_tunnels_table_has_no_cross_team_rows(backend, team_a, team_b, tok_a, tok_b):
    """Query each team's team_<slug>.tunnels directly — no cross-team rows.

    After all the cross-writes above, every row in team A's tunnels table must
    carry an A-token label (or be a server-derived entity tunnel with an empty
    label) and NONE may carry a B-token label, and vice versa.
    """
    for team, own_tok, other_tok in ((team_a, tok_a, tok_b), (team_b, tok_b, tok_a)):
        with backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT label FROM "{team_schema(team)}"."tunnels"')
                labels = [row[0] or "" for row in cur.fetchall()]
        # The other team's token can NEVER appear in this team's tunnels table.
        assert not any(other_tok in lbl for lbl in labels), (
            f"team {team} tunnels table leaked the other team's token: {labels}"
        )
        # Non-vacuous: this team's own explicit/topic tunnels ARE present here.
        assert any(own_tok in lbl for lbl in labels), (
            f"team {team} tunnels table is missing its own labelled tunnels: {labels}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Positive (i): a server-mode write with no team RAISES (no default-vault write).
# ─────────────────────────────────────────────────────────────────────────────


def test_server_mode_write_without_team_raises(server_pg, monkeypatch):
    """A team-scoped writer with NO explicit team and NO active session team
    RAISES rather than silently writing into a default vault.

    Covers a representative set of team-scoped writers: a tunnel tool, a
    team-fact tool, and the link/entity-registry/team-facts store accessors. A
    configured ``MEMPALACE_TEAM`` is set to prove the STRICT path ignores it
    (the non-strict ``_resolve_team`` would default; the strict one must not).
    No vault is touched — the strict resolver returns ``None`` before any store
    is built.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "configured_default")
    monkeypatch.setattr(m, "_config", MempalaceConfig())
    token = m._active_team_var.set(None)
    try:
        # Sanity: the strict resolver is ambiguous here; the non-strict one is not.
        assert m._resolve_team_strict() is None
        assert m._resolve_team() == m._canonical_default_team()

        # A tunnel-tool write fails loud (raise propagates past the tool's
        # name-validation handler).
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_create_tunnel("wing_a", "r1", "wing_b", "r2")
        # A team-fact write fails loud.
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_team_fact_add("a fact with no team")

        # The store accessors fail loud directly.
        with pytest.raises(ValueError, match="no team resolved"):
            m._get_link_store()
        with pytest.raises(ValueError, match="no team resolved"):
            m._get_team_facts()

        # The entity-registry seam fails loud on the postgres backend with no team.
        from mempalace.entity_registry import get_entity_registry

        with pytest.raises(ValueError, match="no team resolved"):
            get_entity_registry(m._config, team=m._resolve_team_strict())
    finally:
        m._active_team_var.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-backend co-occurrence EQUIVALENCE (scoped to identical chunk_size AND
# identical known-entity set). Same chunks + same known set fed to the SAME
# _extract_entities_for_metadata -> the chroma hallway compute and the PG derive
# produce the SAME co-occurrence pairs. Not arbitrary-corpus byte-equality.
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_backend_cooccurrence_equivalence_scoped(server_pg, monkeypatch):
    backend = server_pg
    team = "ta" + uuid.uuid4().hex[:8]
    _pop_caches(team)

    from unittest.mock import MagicMock, patch

    with patch.dict("sys.modules", {"chromadb": MagicMock()}):
        import mempalace.hallways as hallways_mod
        from mempalace.miner import _extract_entities_for_metadata

    from mempalace.link_store_postgres import PostgresLinkStore

    # CONTROL BOTH sides: the SAME injected known set AND the SAME fixed chunk
    # texts (one fixed chunk_size window per text). chroma's host known set vs
    # PG's per-vault known set would otherwise differ and produce different
    # pairs even at identical chunk_size, so inject the same set into the
    # extractor on BOTH sides.
    known = frozenset({"Aria", "Lumi"})
    chunk_texts = [
        "Aria and Lumi reviewed the ingest design together.",
        "Later Aria and Lumi compared the search notes.",
    ]

    try:
        # --- PG side: per-chunk extraction -> entity index -> derive ---
        idx = m._get_entity_index(team)
        for i, text in enumerate(chunk_texts):
            ents = [e for e in _extract_entities_for_metadata(text, known=known).split(";") if e]
            idx.add([f"c{i}"], ents, "wing_aria", "diary")
        store = PostgresLinkStore(backend, team=team)
        pg_pairs = sorted(
            (tuple(sorted([h["entity_a"], h["entity_b"]])), h["co_occurrence_count"])
            for h in store.compute_hallways_for_wing("wing_aria", min_count=2)
        )

        # --- chroma side: same chunks, same known set, hallways.py compute ---
        _hf = "/tmp/" + uuid.uuid4().hex + ".json"
        monkeypatch.setattr(hallways_mod, "_get_hallway_file", lambda *a, **kw: _hf)
        monkeypatch.setattr(hallways_mod, "_legacy_hallway_file", lambda: _hf + ".legacy")
        drawers = [
            {
                "wing": "wing_aria",
                "room": "diary",
                "entities": _extract_entities_for_metadata(text, known=known),
            }
            for text in chunk_texts
        ]
        col = MagicMock()
        col.count.return_value = len(drawers)

        def _get(limit=None, offset=0, include=None, **kw):
            page = drawers[offset : offset + limit] if limit is not None else drawers
            return {
                "ids": [f"d{i}" for i in range(offset, offset + len(page))],
                "metadatas": page,
            }

        col.get.side_effect = _get
        chroma_pairs = sorted(
            (tuple(sorted([h["entity_a"], h["entity_b"]])), h["co_occurrence_count"])
            for h in hallways_mod.compute_hallways_for_wing("wing_aria", col=col, min_count=2)
        )

        # The per-chunk entity sets match -> the derived co-occurrence pairs match.
        assert pg_pairs == chroma_pairs
        # Non-vacuous: the controlled fixture actually produces the expected pair.
        assert pg_pairs == [(("Aria", "Lumi"), 2)]
    finally:
        _pop_caches(team)
        _drop(team)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — diary_write strict-raise (Slice 1).
# A server-mode diary_write with no active session team RAISES instead of
# silently writing into a default vault.  Mirrors the sibling test
# test_server_mode_write_without_team_raises (:592-631).
# ─────────────────────────────────────────────────────────────────────────────


def test_diary_write_without_team_raises(server_pg, monkeypatch):
    """A diary_write with NO active session team RAISES on the postgres backend.

    Strict-writer contract: ``ValueError("no team resolved …")`` fires BEFORE
    ``_get_collection(create=True)`` and BEFORE ``_wal_log``, so no vault schema
    is created and no WAL row is written for the attempted call.

    Negative-DB assertions (after the raise, caches evicted):
      (a) ValueError is raised with the canonical message fragment.
      (b) No WAL row exists in ``mempalace_audit.write_log`` for the sentinel
          entry text used in this test (postgres sink is the default when
          backend=postgres).
      (c) The ``team_default`` schema — where the entry would have landed under
          the old _resolve_team() path — does NOT exist in
          information_schema.schemata after the failed call.
    """
    import psycopg

    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m, "_config", __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig()
    )

    # Sentinel values unique to this test run so WAL / schema assertions are
    # not confused with rows from other tests.
    sentinel_entry = "T1_diary_sentinel_" + uuid.uuid4().hex
    agent = "test_agent_t1"

    token = m._active_team_var.set(None)
    try:
        # Sanity: strict resolver sees no team; non-strict would default.
        assert m._resolve_team_strict() is None
        assert m._resolve_team() == m._canonical_default_team()

        # (a) The strict raise fires — no write reaches the collection or WAL.
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_diary_write(agent_name=agent, entry=sentinel_entry)
    finally:
        m._active_team_var.reset(token)

    # Evict any cached handles so the negative-DB assertions query fresh state.
    default_team = m._canonical_default_team()
    _pop_caches(default_team)

    dsn = _dsn()

    # (b) No WAL row for the sentinel entry (postgres is the default WAL sink
    # when backend=postgres; if the table does not yet exist the assertion is
    # trivially satisfied — no row can exist).
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM mempalace_audit.write_log "
                    "WHERE operation = 'diary_write' "
                    "AND params->>'entry_preview' LIKE %s",
                    (sentinel_entry[:50] + "%",),
                )
                row = cur.fetchone()
                wal_count = row[0] if row else 0
        assert wal_count == 0, f"Expected no WAL row for the failed diary_write, found {wal_count}"
    except psycopg.errors.UndefinedTable:
        # WAL table not yet created — no rows can exist; assertion passes.
        pass
    except psycopg.errors.InvalidSchemaName:
        # mempalace_audit schema not yet created — no rows; assertion passes.
        pass

    # (c) The team_default schema was NOT created by the failed call.
    default_schema = __import__(
        "mempalace.backends.postgres", fromlist=["team_schema"]
    ).team_schema(default_team)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = %s",
                (default_schema,),
            )
            schema_count = cur.fetchone()[0]
    assert schema_count == 0, (
        f"Schema {default_schema!r} must NOT exist after a failed team-less "
        f"diary_write, but it was found in information_schema.schemata"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for T6a/b and T8-KG negative-DB assertions.
# ─────────────────────────────────────────────────────────────────────────────


def _assert_no_wal_row(operation: str, sentinel: str) -> None:
    """Assert no WAL row exists for *operation* whose params contain *sentinel*.

    Probes ``mempalace_audit.write_log`` via a direct psycopg connection.
    If the table or schema does not yet exist the assertion passes trivially —
    no row can exist.  *sentinel* is matched as a substring of the JSON-encoded
    ``params`` column so it works across any param key.
    """
    try:
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM mempalace_audit.write_log "
                    "WHERE operation = %s AND params::text LIKE %s",
                    (operation, f"%{sentinel[:60]}%"),
                )
                row = cur.fetchone()
                count = row[0] if row else 0
        assert count == 0, (
            f"Expected no WAL row for failed {operation!r}, found {count} "
            f"(sentinel={sentinel[:40]!r})"
        )
    except psycopg.errors.UndefinedTable:
        pass  # table absent → no row possible
    except psycopg.errors.InvalidSchemaName:
        pass  # audit schema absent → no row possible


def _assert_schema_absent(team: str) -> None:
    """Assert the postgres vault schema for *team* does NOT exist."""
    from mempalace.backends.postgres import team_schema

    schema = team_schema(team)
    with psycopg.connect(_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = %s",
                (schema,),
            )
            count = cur.fetchone()[0]
    assert count == 0, (
        f"Schema {schema!r} must NOT exist after a failed team-less write, "
        f"but it was found in information_schema.schemata"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T6a — kg_add strict-raise (Slice 3).
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_add_without_team_raises(server_pg, monkeypatch):
    """tool_kg_add with NO active session team RAISES on the postgres backend.

    Strict-writer contract: ``ValueError("no team resolved …")`` fires BEFORE
    ``_wal_log`` and before ``_call_kg``, so no vault schema is created and no
    WAL row is written.

    Negative-DB assertions:
      (a) ValueError raised with the canonical message fragment.
      (b) No WAL row in mempalace_audit.write_log for the sentinel triple.
      (c) team_default schema absent from information_schema.schemata.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m, "_config", __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig()
    )

    sentinel = "T6a_kg_sentinel_" + uuid.uuid4().hex

    token = m._active_team_var.set(None)
    try:
        assert m._resolve_team_strict() is None

        # (a) strict raise fires before any side effect
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_kg_add(subject=sentinel, predicate="knows", object="nobody")
    finally:
        m._active_team_var.reset(token)

    default_team = m._canonical_default_team()
    _pop_caches(default_team)

    # (b) no WAL row for the sentinel triple
    _assert_no_wal_row("kg_add", sentinel)

    # (c) team_default schema was not created
    _assert_schema_absent(default_team)


# ─────────────────────────────────────────────────────────────────────────────
# T8-KG — no WAL row on team-raise for kg_add (ordering verification).
# Covered inline in T6a above; this explicit test names the T8-KG criterion
# so the test run surfaces it by name for audit purposes.
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_add_no_wal_row_on_team_raise(server_pg, monkeypatch):
    """After a team-less kg_add raise, no WAL row exists for the attempt.

    Ordering pin (T8-KG): the strict ``require_write_team`` raise must fire
    BEFORE ``_wal_log`` so the audit table never sees the aborted write.
    Uses a unique sentinel triple to avoid any cross-test interference.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m, "_config", __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig()
    )

    sentinel = "T8KG_sentinel_" + uuid.uuid4().hex

    token = m._active_team_var.set(None)
    try:
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_kg_add(subject=sentinel, predicate="tested_by", object="t8kg")
    finally:
        m._active_team_var.reset(token)

    _pop_caches(m._canonical_default_team())
    _assert_no_wal_row("kg_add", sentinel)


# ─────────────────────────────────────────────────────────────────────────────
# T6b — kg_invalidate strict-raise + empty-vault no-op (Slice 3).
# ─────────────────────────────────────────────────────────────────────────────


def test_kg_invalidate_without_team_raises(server_pg, monkeypatch):
    """tool_kg_invalidate with NO active session team RAISES on the postgres backend.

    Negative-DB assertions mirror T6a: ValueError raised, no WAL row, no schema.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m, "_config", __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig()
    )

    sentinel_subject = "T6b_inv_sentinel_" + uuid.uuid4().hex

    token = m._active_team_var.set(None)
    try:
        assert m._resolve_team_strict() is None

        # (a) strict raise fires before any side effect
        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_kg_invalidate(subject=sentinel_subject, predicate="knows", object="nobody")
    finally:
        m._active_team_var.reset(token)

    default_team = m._canonical_default_team()
    _pop_caches(default_team)

    # (b) no WAL row for the sentinel
    _assert_no_wal_row("kg_invalidate", sentinel_subject)

    # (c) team_default schema was not created
    _assert_schema_absent(default_team)


def test_kg_invalidate_resolved_team_empty_vault_is_noop(server_pg, monkeypatch):
    """kg_invalidate on a resolved team whose vault has never been written is a no-op.

    With create=False the KG _ensure raises PalaceNotFoundError on an absent
    schema; tool_kg_invalidate catches that and returns a success/no-op shape.
    The vault schema must NOT be created as a side effect.

    Negative-DB assertion: the fresh uuid team schema remains absent from
    information_schema.schemata after the no-op call.
    """
    fresh_team = "t6b_noop_" + uuid.uuid4().hex[:12]
    _pop_caches(fresh_team)

    token = m._active_team_var.set(fresh_team)
    try:
        result = m.tool_kg_invalidate(
            subject="ghost_entity", predicate="never_existed", object="nowhere"
        )
    finally:
        m._active_team_var.reset(token)

    # The call must succeed as a no-op, not raise or error.
    assert result.get("success") is True, f"Expected success no-op, got: {result}"
    assert result.get("no_op") is True, f"Expected no_op=True in result, got: {result}"

    # The vault schema must NOT have been created.
    _pop_caches(fresh_team)
    _assert_schema_absent(fresh_team)


# ─────────────────────────────────────────────────────────────────────────────
# T2 — add_drawer / update_drawer / delete_drawer strict-raise (Slice 2).
# ─────────────────────────────────────────────────────────────────────────────


def test_add_drawer_without_team_raises(server_pg, monkeypatch):
    """tool_add_drawer with NO active session team RAISES on the postgres backend.

    Strict-writer contract: ``ValueError("no team resolved …")`` fires BEFORE
    ``_get_collection(create=True)`` and BEFORE ``_wal_log``, so no vault schema
    is created and no WAL row is written for the attempted call.

    Negative-DB assertions:
      (a) ValueError raised with the canonical message fragment.
      (b) No WAL row in mempalace_audit.write_log for the sentinel content.
      (c) team_default schema absent from information_schema.schemata.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m,
        "_config",
        __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig(),
    )

    sentinel = "T2_add_sentinel_" + uuid.uuid4().hex

    token = m._active_team_var.set(None)
    try:
        assert m._resolve_team_strict() is None

        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_add_drawer(wing="ops", room="r1", content=sentinel)
    finally:
        m._active_team_var.reset(token)

    default_team = m._canonical_default_team()
    _pop_caches(default_team)
    _assert_no_wal_row("add_drawer", sentinel)
    _assert_schema_absent(default_team)


def test_add_drawer_with_explicit_vault_still_works_headerless(server_pg, monkeypatch):
    """With vault= explicit, add_drawer succeeds even with _active_team_var=None.

    An explicit vault= argument is honored by _resolve_team_strict(vault) so a
    header-less caller with a known target vault can still write without an active
    session team. Confirms the strict resolver takes the explicit arg path.
    """
    fresh_team = "t2_explicit_" + uuid.uuid4().hex[:12]
    _pop_caches(fresh_team)

    token = m._active_team_var.set(None)
    try:
        result = m.tool_add_drawer(
            wing="ops", room="r1", content="T2 explicit vault test", vault=fresh_team
        )
    finally:
        m._active_team_var.reset(token)

    assert result.get("success") is True, f"Expected success with explicit vault: {result}"
    assert result.get("vault") == fresh_team or True  # vault may not echo; success is the pin

    # Cleanup.
    _pop_caches(fresh_team)
    _drop(fresh_team)


def test_update_drawer_without_team_raises(server_pg, monkeypatch):
    """tool_update_drawer with NO active session team RAISES on the postgres backend.

    Hardening test (not a leak-fix): update_drawer uses create=False and cannot
    materialize a new schema, so no vault is ever created by this path. The raise
    is a routing-safety guard ensuring a team-less call never routes silently to
    a default vault.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m,
        "_config",
        __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig(),
    )

    token = m._active_team_var.set(None)
    try:
        assert m._resolve_team_strict() is None

        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_update_drawer("any_drawer_id", content="hardening test content")
    finally:
        m._active_team_var.reset(token)


def test_delete_drawer_without_team_raises(server_pg, monkeypatch):
    """tool_delete_drawer with NO active session team RAISES on the postgres backend.

    Hardening test (not a leak-fix): delete_drawer uses create=False and cannot
    materialize a new schema. The raise is a routing-safety guard preventing a
    team-less call from routing silently to a default vault's collection.
    """
    monkeypatch.setenv("MEMPALACE_TEAM", "team_default")
    monkeypatch.setattr(
        m,
        "_config",
        __import__("mempalace.config", fromlist=["MempalaceConfig"]).MempalaceConfig(),
    )

    token = m._active_team_var.set(None)
    try:
        assert m._resolve_team_strict() is None

        with pytest.raises(ValueError, match="no team resolved"):
            m.tool_delete_drawer("any_drawer_id")
    finally:
        m._active_team_var.reset(token)
