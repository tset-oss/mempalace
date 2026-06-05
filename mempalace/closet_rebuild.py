"""Server-side closet rebuild on the MCP write path (eventually-consistent).

The AAAK closet index — the searchable pointer layer that ``search_memories``
uses as a ranking boost — is normally built by the CLI miner. A miner-less
central deployment, where agents file memories through the MCP ``add_drawer``
tool, never runs the miner, so its closets collection stays EMPTY and the
closet ranking boost in :mod:`mempalace.searcher` silently degrades to pure
drawer search.

This module closes that gap by rebuilding a group's closets server-side after
each ``add_drawer`` write. Closets are a NON-GATING ranking signal (the direct
drawer query is always the recall floor), so the rebuild is intentionally
best-effort and eventually-consistent:

* It reuses the deterministic shared builders in :mod:`mempalace.palace`
  (``purge_file_closets`` -> ``build_closet_lines`` -> ``upsert_closet_lines``)
  verbatim — there is NO second closet algorithm and NO LLM here.
* Unlike the miner, it does NOT reconstruct drawer ids from a formula. The
  miner derives ids as ``sha256(source_file + chunk_index)`` while ``add_drawer``
  derives them as ``sha256(wing + room + content)`` (plus a ``_chunk_NNNNNN``
  suffix on the chunked path). Reconstructing the miner-style id would point the
  closet at rows that do not exist. Instead the rebuild QUERIES the collection
  by metadata and passes the ACTUAL stored ids into ``build_closet_lines``.
* A debounce coalescer batches a burst of writes to the same group into a single
  rebuild, and a bounded startup reconcile re-enqueues groups whose closets are
  missing — both off the request-critical path.

A rebuild failure never fails or undoes the verbatim drawer write; callers wrap
``enqueue`` in their own best-effort try/except, mirroring the existing
``_index_drawer_entities`` pattern in the MCP server.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("mempalace.closet_rebuild")


# Default debounce window. A burst of rapid ``add_drawer`` calls to the same
# group collapses into ONE rebuild fired this many seconds after the last
# enqueue. Tests set ``debounce_seconds=0`` for deterministic, sleep-free flushing.
DEFAULT_DEBOUNCE_SECONDS = 2.0

# Bounded startup reconcile cap: the maximum number of distinct groups
# re-enqueued per vault when the server reconciles missing closets. Keeps the
# reconcile cheap and off the boot-critical path even on a large vault.
DEFAULT_RECONCILE_CAP = 200

# Bound the per-rebuild and per-reconcile drawer scans so a pathologically large
# group / vault cannot load unbounded rows into memory on a central deployment.
# The closet pointer is intentionally lossy (build_closet_lines uses only the
# first 3 ids + a ~5000-char window), so capping the fetch is behaviour-preserving
# for the closet content while bounding memory/IO. Tunable defaults.
REBUILD_FETCH_CAP = 500
RECONCILE_SCAN_CAP = 5000


def closet_grouping_key(source_file: str, wing: str, room: str) -> str:
    """Return the deterministic, non-empty grouping key for a drawer.

    ``add_drawer`` stores ``source_file or ""`` (an agent-curated write often
    has no file of origin). An empty ``source_file`` can never anchor a closet:
    the miner keys closets by ``source_file``, ``purge_file_closets`` deletes
    ``where={"source_file": ...}``, and the boost lookup in
    :mod:`mempalace.searcher` skips empty closet/drawer sources (``if source``).
    So when ``source_file`` is empty we synthesise a stable, non-empty key.

    GRANULARITY TRADEOFF (a deliberate, documented choice):
      * ``(wing, room)``-granular (the default below, ``wingroom:{wing}/{room}``)
        AGGREGATES every empty-source drawer in a room into ONE closet group.
        This matches the AAAK room model and keeps the index compact, but it is
        coarse: distinct memories filed into the same room share one closet.
      * ``drawer_id``-granular (one closet group per drawer) would never
        aggregate and never collapse unrelated drawers, at the cost of index
        size. It is the safer choice if per-drawer closet precision ever matters.

    The default is ``(wing, room)``-granular. Critically it is NOT the miner's
    ``sha256(source_file)`` collapse, which would map ALL empty-source files in a
    wing/room to the SAME closet id base across unrelated groups — this key keeps
    one bucket PER room intentionally rather than colliding cross-group.

    A non-empty ``source_file`` is returned unchanged, so existing miner-built
    closets and the non-empty boost path are completely unaffected.
    """
    if source_file:
        return source_file
    return f"wingroom:{wing}/{room}"


def _group_where_filter(grouping_key: str, wing: str, room: str) -> dict:
    """Metadata filter that selects exactly the drawers belonging to a group.

    For a real ``source_file`` the group is "every drawer from that file"
    (matching the miner's per-file closet model). For the synthetic
    ``wingroom:`` fallback key the group is "every empty-source drawer in that
    wing/room", expressed as the literal stored ``source_file == ""`` plus the
    wing/room — so two teams' rooms, or two different rooms, never bleed together.
    """
    if grouping_key.startswith("wingroom:"):
        return {"$and": [{"wing": wing}, {"room": room}, {"source_file": ""}]}
    return {"source_file": grouping_key}


def rebuild_group_closets(
    palace_path: str,
    team: Optional[str],
    grouping_key: str,
    wing: str,
    room: str,
) -> int:
    """Rebuild the closets for one ``(team, grouping_key)`` group.

    Queries the live collection for the group's drawers (harvesting their REAL
    stored ids), then reuses the shared deterministic palace builders to purge
    and re-upsert the closet pointers. Returns the number of drawers the closet
    was built from (0 when the group is empty — e.g. the drawers were deleted
    between enqueue and rebuild).

    Raises on hard failure; the debounce worker and the enqueue caller both wrap
    this in best-effort try/except so a closet rebuild never fails the drawer
    write.
    """
    # Imported lazily so this module stays import-cheap and free of the heavy
    # backend/embedding import graph until a rebuild actually runs.
    from .palace import (
        build_closet_lines,
        get_closets_collection,
        get_collection,
        purge_file_closets,
        upsert_closet_lines,
    )

    drawers_col = get_collection(palace_path, create=False, team=team)
    where = _group_where_filter(grouping_key, wing, room)

    # Harvest the ACTUAL stored ids + content for the group. We intentionally do
    # NOT derive ids from a formula (see module docstring): the add_drawer id
    # scheme differs from the miner's, so reconstructed ids would be dead pointers.
    fetched = drawers_col.get(
        where=where, include=["documents", "metadatas"], limit=REBUILD_FETCH_CAP
    )
    ids = list(fetched.ids or [])
    if not ids:
        return 0

    docs = list(fetched.documents or [])
    metas = list(fetched.metadatas or [])

    # Build closet pointer content from the concatenated group text. The miner
    # builds per file from the full file content; here the group's drawers ARE
    # the content, so join them. Order by chunk_index when present so a chunked
    # drawer reads in order; otherwise preserve the collection's id order.
    def _chunk_index(meta):
        try:
            return int((meta or {}).get("chunk_index", 0))
        except (TypeError, ValueError):
            return 0

    order = sorted(range(len(ids)), key=lambda i: _chunk_index(metas[i] if i < len(metas) else {}))
    ordered_ids = [ids[i] for i in order]
    ordered_docs = [docs[i] if i < len(docs) else "" for i in order]
    ordered_metas = [metas[i] if i < len(metas) else {} for i in order]
    content = "\n\n".join(d for d in ordered_docs if d)

    closet_lines = build_closet_lines(
        grouping_key,
        ordered_ids,
        content,
        wing,
        room,
        drawer_metas=ordered_metas,
    )

    # closet_id_base, closet_meta["source_file"], and the drawer-fetch filter ALL
    # key off the SAME grouping_key so the boost lookup in searcher matches: the
    # closet's stored source_file must equal the drawers' effective grouping key.
    import hashlib

    closet_id_base = (
        f"closet_{wing}_{room}_{hashlib.sha256(grouping_key.encode()).hexdigest()[:24]}"
    )
    closet_meta = {
        "wing": wing,
        "room": room,
        "source_file": grouping_key,
        "drawer_count": len(ordered_ids),
        "filed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    closets_col = get_closets_collection(palace_path, create=True, team=team)
    purge_file_closets(closets_col, grouping_key)
    upsert_closet_lines(closets_col, closet_id_base, closet_lines, closet_meta)
    return len(ordered_ids)


class ClosetDebouncer:
    """In-process, single-thread debounce coalescer keyed by ``(team, grouping_key)``.

    A burst of rapid ``add_drawer`` writes to the same group enqueues many times
    but only the LAST pending entry survives in the coalescing map, so exactly
    ONE rebuild fires per group per debounce window. ``team`` is part of the key,
    so two teams writing into the same ``(wing, room)`` fallback group never
    collide — each rebuilds against its OWN captured team.

    The worker thread does NOT inherit the request's team contextvar (a
    background thread has no request context), so the team is CAPTURED AT
    ENQUEUE TIME and passed explicitly to the rebuild — the worker never resolves
    the team itself.

    Tests get determinism via ``debounce_seconds=0`` (fire on the next worker
    tick) plus :meth:`flush` (block until the queue drains).
    """

    def __init__(
        self,
        rebuild_fn: Callable[[Optional[str], str, str, str], int] | None = None,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ):
        # rebuild_fn(team, grouping_key, wing, room) -> drawer_count. Injectable
        # so the no-DB unit tests can substitute a counting stub.
        self._rebuild_fn = rebuild_fn or self._default_rebuild
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        # key (team, grouping_key) -> (due_monotonic, wing, room)
        self._pending: dict[tuple, tuple] = {}
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._idle = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._in_flight = 0
        self._palace_path: Optional[str] = None

    def _default_rebuild(self, team, grouping_key, wing, room) -> int:
        # The production rebuild binds the palace_path captured at start().
        return rebuild_group_closets(self._palace_path, team, grouping_key, wing, room)

    def start(self, palace_path: Optional[str] = None) -> None:
        """Lazily start the worker thread (idempotent)."""
        with self._lock:
            if palace_path is not None:
                self._palace_path = palace_path
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run, name="mempalace-closet-debounce", daemon=True
            )
            self._thread.start()
        atexit.register(self._atexit_flush)

    def enqueue(
        self,
        team: Optional[str],
        grouping_key: str,
        wing: str,
        room: str,
        palace_path: Optional[str] = None,
    ) -> None:
        """Schedule a rebuild for ``(team, grouping_key)``, coalescing duplicates.

        Lazy-starts the worker on first call. Capturing ``team`` here — at
        enqueue time, inside the request context — is what makes the worker
        tenant-correct without ever calling the team resolver in-thread.
        """
        if not self._running:
            self.start(palace_path)
        elif palace_path is not None and self._palace_path is None:
            self._palace_path = palace_path
        due = time.monotonic() + self._debounce_seconds
        with self._wake:
            # Coalesce: a fresh enqueue for the same key overwrites the prior
            # pending entry, so N rapid adds yield ONE rebuild.
            self._pending[(team, grouping_key)] = (due, wing, room)
            self._wake.notify_all()

    def _run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._pending:
                    self._wake.wait()
                if not self._running and not self._pending:
                    return
                now = time.monotonic()
                ready = [k for k, (due, _, _) in self._pending.items() if due <= now]
                if not ready:
                    # Sleep until the soonest due time (or until a new enqueue
                    # wakes us). Bounded so a 0-delay queue never busy-spins.
                    soonest = min(due for due, _, _ in self._pending.values())
                    self._wake.wait(timeout=max(0.0, soonest - now) or 0.01)
                    continue
                batch = []
                for key in ready:
                    due, wing, room = self._pending.pop(key)
                    team, grouping_key = key
                    batch.append((team, grouping_key, wing, room))
                self._in_flight += len(batch)
            for team, grouping_key, wing, room in batch:
                self._fire(team, grouping_key, wing, room)
            with self._idle:
                self._in_flight -= len(batch)
                if not self._pending and self._in_flight == 0:
                    self._idle.notify_all()

    def _fire(self, team, grouping_key, wing, room) -> None:
        started = time.monotonic()
        try:
            count = self._rebuild_fn(team, grouping_key, wing, room)
            duration_ms = (time.monotonic() - started) * 1000.0
            logger.info(
                "closet rebuild done team=%s grouping_key=%s drawer_count=%s duration_ms=%.1f",
                team,
                grouping_key,
                count,
                duration_ms,
            )
        except Exception:
            # Best-effort: a rebuild failure never propagates — the verbatim
            # drawer write already committed and closets are a non-gating signal.
            logger.debug(
                "closet rebuild failed team=%s grouping_key=%s (best-effort)",
                team,
                grouping_key,
                exc_info=True,
            )

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until all pending + in-flight rebuilds finish.

        Forces every pending entry due immediately so tests do not have to wait
        out the debounce window. Returns ``True`` when the queue drained within
        ``timeout``.
        """
        with self._wake:
            now = time.monotonic()
            for key, (_, wing, room) in list(self._pending.items()):
                self._pending[key] = (now, wing, room)
            self._wake.notify_all()
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending or self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(timeout=remaining)
        return True

    def _atexit_flush(self) -> None:
        try:
            self.flush(timeout=5.0)
        except Exception:
            pass


def reconcile_closets(
    palace_path: str,
    team: Optional[str],
    debouncer: "ClosetDebouncer",
    cap: int = DEFAULT_RECONCILE_CAP,
) -> int:
    """Re-enqueue groups whose closets are missing — bounded, off the boot path.

    Pending debounce timers live only in this process, so a restart can lose a
    rebuild that never fired. This reconcile scans the vault's drawers, derives
    each drawer's grouping key, and re-enqueues any group that has no closet yet,
    up to ``cap`` distinct groups (so a large vault never turns this into an
    unbounded sweep). It logs the per-vault re-enqueue count.

    Returns the number of groups re-enqueued. The caller MUST run this off the
    import/startup-critical path (a daemon thread) so it never blocks boot or the
    first ``add_drawer``. The eventual-consistency window is: a group missing its
    closet at boot becomes searchable-with-boost only after this reconcile (or
    the next write to that group) fires.
    """
    from .palace import get_closets_collection, get_collection

    try:
        drawers_col = get_collection(palace_path, create=False, team=team)
        drawer_rows = drawers_col.get(include=["metadatas"], limit=RECONCILE_SCAN_CAP)
    except Exception:
        logger.debug("closet reconcile: drawers unavailable for team=%s", team, exc_info=True)
        return 0

    try:
        closets_col = get_closets_collection(palace_path, create=False, team=team)
        closet_rows = closets_col.get(include=["metadatas"])
        existing_keys = {(m or {}).get("source_file", "") for m in (closet_rows.metadatas or [])}
    except Exception:
        existing_keys = set()

    seen_groups: set = set()
    group_meta: dict = {}
    for meta in drawer_rows.metadatas or []:
        meta = meta or {}
        wing = meta.get("wing", "unknown")
        room = meta.get("room", "unknown")
        key = closet_grouping_key(meta.get("source_file", "") or "", wing, room)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        group_meta[key] = (wing, room)

    re_enqueued = 0
    for key, (wing, room) in group_meta.items():
        if key in existing_keys:
            continue
        if re_enqueued >= cap:
            logger.info(
                "closet reconcile: hit cap=%s for team=%s; remaining groups deferred", cap, team
            )
            break
        debouncer.enqueue(team, key, wing, room, palace_path=palace_path)
        re_enqueued += 1

    logger.info(
        "closet reconcile team=%s re_enqueued=%s scanned_groups=%s",
        team,
        re_enqueued,
        len(seen_groups),
    )
    return re_enqueued
