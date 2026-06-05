"""No-DB unit tests for the server-side closet rebuild coalescer (G005).

These exercise the debounce/tenant-isolation/best-effort/fallback-key behaviour
WITHOUT a database by injecting a counting stub rebuild function into the
:class:`ClosetDebouncer`. They always run (no live-skip).
"""

from __future__ import annotations

import threading
import time

from mempalace.closet_rebuild import (
    ClosetDebouncer,
    _group_where_filter,
    closet_grouping_key,
)


def test_coalesces_rapid_adds_to_one_rebuild():
    """N rapid enqueues for the SAME (team, grouping_key) -> exactly ONE rebuild."""
    calls = []
    lock = threading.Lock()

    def _stub(team, grouping_key, wing, room):
        with lock:
            calls.append((team, grouping_key, wing, room))
        return 1

    # Small but non-zero window so all 50 enqueues land before the worker fires.
    deb = ClosetDebouncer(rebuild_fn=_stub, debounce_seconds=0.05)
    deb.start()
    for _ in range(50):
        deb.enqueue("teamA", "notes.md", "people", "alice")
    assert deb.flush(timeout=10.0)
    assert len(calls) == 1, f"expected 1 coalesced rebuild, got {len(calls)}: {calls}"
    assert calls[0] == ("teamA", "notes.md", "people", "alice")


def test_tenant_isolation_two_teams_two_rebuilds():
    """Same grouping_key under two DIFFERENT teams -> TWO independent rebuilds.

    The coalescing key INCLUDES team, and each rebuild fires with its own
    captured team. Two teams writing into the same (wing, room) fallback group
    never collide.
    """
    calls = []
    lock = threading.Lock()

    def _stub(team, grouping_key, wing, room):
        with lock:
            calls.append((team, grouping_key))
        return 1

    deb = ClosetDebouncer(rebuild_fn=_stub, debounce_seconds=0.05)
    deb.start()
    key = closet_grouping_key("", "people", "alice")  # same fallback key for both
    for _ in range(10):
        deb.enqueue("teamA", key, "people", "alice")
        deb.enqueue("teamB", key, "people", "alice")
    assert deb.flush(timeout=10.0)
    assert len(calls) == 2, f"expected 2 rebuilds (one per team), got {len(calls)}: {calls}"
    assert set(calls) == {("teamA", key), ("teamB", key)}


def test_distinct_groups_fire_independently():
    """Different grouping keys under one team each get their own rebuild."""
    calls = []
    lock = threading.Lock()

    def _stub(team, grouping_key, wing, room):
        with lock:
            calls.append(grouping_key)
        return 1

    deb = ClosetDebouncer(rebuild_fn=_stub, debounce_seconds=0.05)
    deb.start()
    deb.enqueue("teamA", "a.md", "w", "r")
    deb.enqueue("teamA", "b.md", "w", "r")
    deb.enqueue("teamA", "a.md", "w", "r")  # coalesces with the first a.md
    assert deb.flush(timeout=10.0)
    assert sorted(calls) == ["a.md", "b.md"]


def test_rebuild_failure_is_swallowed_never_propagates():
    """A raising rebuild_fn never escapes the worker (best-effort)."""
    fired = []

    def _boom(team, grouping_key, wing, room):
        fired.append((team, grouping_key))
        raise RuntimeError("simulated closet rebuild failure")

    deb = ClosetDebouncer(rebuild_fn=_boom, debounce_seconds=0.0)
    deb.start()
    deb.enqueue("teamA", "notes.md", "w", "r")
    # flush returns True (queue drained) even though the rebuild raised — the
    # failure was contained, the worker stayed alive.
    assert deb.flush(timeout=10.0)
    assert fired == [("teamA", "notes.md")]
    # The worker thread is still alive and usable after a failure.
    deb.enqueue("teamA", "second.md", "w", "r")
    assert deb.flush(timeout=10.0)
    assert ("teamA", "second.md") in fired


def test_zero_delay_debounce_flushes_promptly():
    """debounce_seconds=0 makes flush() resolve without waiting out a window."""
    calls = []

    def _stub(team, grouping_key, wing, room):
        calls.append(grouping_key)
        return 1

    deb = ClosetDebouncer(rebuild_fn=_stub, debounce_seconds=0.0)
    deb.start()
    started = time.monotonic()
    deb.enqueue("teamA", "k", "w", "r")
    assert deb.flush(timeout=10.0)
    assert calls == ["k"]
    assert time.monotonic() - started < 5.0


# ---------------------------------------------------------------------------
# Fallback-key threading: the same key must flow through all four sites.
# ---------------------------------------------------------------------------


def test_fallback_key_is_deterministic_non_empty_and_room_granular():
    k1 = closet_grouping_key("", "people", "alice")
    k2 = closet_grouping_key("", "people", "alice")
    assert k1 == k2 == "wingroom:people/alice"
    assert k1  # non-empty
    # A different room is a DIFFERENT group (no cross-room collapse).
    assert closet_grouping_key("", "people", "bob") != k1
    # A different wing is a different group too.
    assert closet_grouping_key("", "projects", "alice") != k1


def test_non_empty_source_returns_source_unchanged():
    assert closet_grouping_key("docs/readme.md", "w", "r") == "docs/readme.md"


def test_group_where_filter_matches_the_grouping_key_semantics():
    # Non-empty source: group is "every drawer from that file".
    assert _group_where_filter("docs/readme.md", "w", "r") == {"source_file": "docs/readme.md"}
    # Fallback key: group is "every empty-source drawer in that wing/room".
    fb = closet_grouping_key("", "people", "alice")
    assert _group_where_filter(fb, "people", "alice") == {
        "$and": [{"wing": "people"}, {"room": "alice"}, {"source_file": ""}]
    }
