#!/usr/bin/env python3
"""
MemPalace MCP Server — read/write palace access for Claude Code
================================================================
Install: claude mcp add mempalace -- mempalace-mcp [--palace /path/to/palace]

Tools (read):
  mempalace_status          — total drawers, wing/room breakdown
  mempalace_list_wings      — all wings with drawer counts
  mempalace_list_rooms      — rooms within a wing
  mempalace_get_taxonomy    — full wing → room → count tree
  mempalace_search          — semantic search, optional wing/room filter
  mempalace_check_duplicate — check if content already exists before filing

Tools (write):
  mempalace_add_drawer      — file verbatim content into a wing/room
  mempalace_delete_drawer   — remove a drawer by ID

Tools (maintenance):
  mempalace_reconnect       — force cache invalidation and reconnect after external writes
"""

import contextvars
import os
import sys

# --- MCP stdio protection (issue #225) -----------------------------------
# The MCP protocol multiplexes JSON-RPC over stdio: stdout MUST carry only
# valid JSON-RPC messages, stderr is for human-readable logs. Some
# transitive dependencies (chromadb → onnxruntime, posthog telemetry) print
# banners and error messages directly to stdout — sometimes at C level —
# which breaks Claude Desktop's JSON parser. Redirect stdout → stderr at
# both the Python and file-descriptor level before heavy imports, then
# restore the real stdout in main() before entering the protocol loop.
_REAL_STDOUT = sys.stdout
_REAL_STDOUT_FD = None
try:
    _REAL_STDOUT_FD = os.dup(1)
    os.dup2(2, 1)
except (OSError, AttributeError):
    # Environments without fd-level stdio (embedded interpreters, some test
    # harnesses). The Python-level redirect below still applies.
    pass
sys.stdout = sys.stderr

import argparse  # noqa: E402  (deferred until after stdio protection above)
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import hashlib  # noqa: E402
import sqlite3  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import date, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from .config import (  # noqa: E402
    MempalaceConfig,
    sanitize_kg_value,
    sanitize_name,
    sanitize_content,
    sanitize_iso_temporal,
    strip_lone_surrogates,
)
from .version import __version__  # noqa: E402
from chromadb.errors import NotFoundError as _ChromaNotFoundError  # noqa: E402

from .backends.chroma import (  # noqa: E402
    ChromaBackend,
    ChromaCollection,
    _HNSW_BLOAT_GUARD,
    _pin_hnsw_threads,
    hnsw_capacity_status,
)
from .query_sanitizer import sanitize_query  # noqa: E402
from .searcher import search_memories  # noqa: E402
from .palace_graph import (  # noqa: E402
    traverse,
    find_tunnels,
    graph_stats,
    create_tunnel,
    list_tunnels,
    delete_tunnel,
    follow_tunnels,
)

from .knowledge_graph import KnowledgeGraph, DEFAULT_KG_PATH  # noqa: E402


def _init_logging() -> None:
    """Root-logger init: always stderr, optionally append to ``MEMPALACE_LOG_FILE``.

    Stderr-only is the default. When ``MEMPALACE_LOG_FILE`` is set, a
    ``FileHandler`` is attached so MCP-client failures that the client
    does not surface (e.g. the ``-32000`` cold-load timeout in #1495)
    remain diagnosable from the file.

    Failure modes:

    * Invalid path (missing directory, no perms, Windows NUL byte) →
      stderr-only with a warning. The env var must not become a new
      server-start failure surface — that would defeat the diagnostic
      goal. ``ValueError`` is included in the catch because Windows
      raises it for paths with embedded NUL bytes, not ``OSError``.
    * Root logger already configured (host app embedding the server,
      transitive imports touching ``logging``) → ``force=True`` resets
      the handlers so MEMPALACE_LOG_FILE's contract holds regardless
      of what touched root logging first. Without ``force=True``,
      ``basicConfig`` is a no-op when handlers exist and the env var
      silently does nothing — exactly the diagnostic black hole #1495
      exists to close.
    * Concurrent writers (multiple ``mempalace-mcp`` processes pointing
      at the same path) interleave at the line level. The handler uses
      append mode so nothing is overwritten, but operators running
      Claude Code + Claude Desktop simultaneously should give each
      process its own log path.

    ``delay=True`` is intentionally NOT set: deferring the open means an
    invalid path raises at ``emit()`` time (unhandled), defeating the
    fail-soft contract. With eager open the same error surfaces inside
    ``FileHandler.__init__`` and lands in our ``except`` below.

    Module-level invocation: this function runs at import time, preserving
    the side effect of the previous module-level ``logging.basicConfig``
    call. Callers that import ``mempalace.mcp_server`` for introspection
    (``TOOLS`` dict, handler functions) inherit the reset; this matches
    pre-PR behaviour and is intentional for an MCP entry-point module.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    # MEMPALACE_LOG_FILE is operator-supplied and opt-in; this is a
    # local-first server (CLAUDE.md design principle), so no path
    # sanitization — the operator's process UID is the trust boundary.
    log_file = os.environ.get("MEMPALACE_LOG_FILE", "").strip()
    file_handler_error: Exception | None = None
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Fail-soft: see "Invalid path" failure mode above. Broad on
            # (OSError, ValueError) because Windows raises ValueError for
            # NUL-byte paths while POSIX uses OSError for missing-dir / EPERM.
            file_handler_error = exc
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)
    if file_handler_error is not None:
        logging.getLogger("mempalace_mcp").warning(
            "MEMPALACE_LOG_FILE=%r could not be opened (%s); using stderr only",
            log_file,
            file_handler_error,
        )


_init_logging()
logger = logging.getLogger("mempalace_mcp")


def _parse_args():
    parser = argparse.ArgumentParser(description="MemPalace MCP Server")
    parser.add_argument(
        "--palace",
        metavar="PATH",
        help="Path to the palace directory (overrides config file and env var)",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("Ignoring unknown args: %s", unknown)
    return args


_args = _parse_args()

if _args.palace:
    os.environ["MEMPALACE_PALACE_PATH"] = os.path.abspath(_args.palace)

_config = MempalaceConfig()

# --- Per-session team-vault routing (central HTTP server) ----------------
# The active team for the *current request*, set per-call by the FastMCP
# server wrapper (``mcp_fastmcp``) from ``session.active_team`` (switch_team)
# or the ``X-Mempalace-Team`` request header. Default ``None``: the stdio /
# legacy JSON-RPC loop (:func:`main`) never sets it, so :func:`_resolve_team`
# falls back to the process-global ``_config.team`` and every existing code
# path behaves exactly as before. This is the only piece of per-session state
# the tool handlers read; everything else stays process-global.
_active_team_var: contextvars.ContextVar = contextvars.ContextVar(
    "mempalace_active_team", default=None
)


# Canonical team-name rule — the single source of truth, reused by
# mcp_fastmcp's header / switch_team validation. A team maps to the Postgres
# schema ``team_<slug>``. This pattern is exactly the fixed-point set of
# backends.postgres.sanitize_team (``re.sub([^a-z0-9_], "_").strip("_")``): no
# leading/trailing underscore, ``[a-z0-9_]`` body, 1-40 chars. Matching the
# fixed-point set (not just ``[a-z0-9_]+``) means a validated slug passes
# sanitize_team UNCHANGED, so the vault reported to the caller, the KG cache
# key, and "is this vault known" membership can never disagree with the schema
# the data physically lives in. Keeping the resolver STRICT (reject, never
# silently rewrite) is what makes that guarantee hold.
_TEAM_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_]{0,38}[a-z0-9])?$")


# Tokens that mean "no specific team / use the default", not a literal vault.
# ``all`` is here too: it is the cross-team *search* selector, never a writable
# vault — so as an explicit/header/session team it means "fall back", which
# keeps it from ever routing a write into a literal ``team_all`` schema.
_TEAM_RESET_ALIASES = frozenset({"", "primary", "default", "all"})


def _valid_team(value):
    """Return *value* as a canonical team slug, or ``None`` if it is not one.

    The reset aliases (blank / ``primary`` / ``default`` / ``all``) mean "no
    override" (``None``). A value outside the ``[a-z0-9_]`` slug class (e.g. a
    typo, a hyphen, an over-long name, or a stray header) is rejected (``None``)
    rather than silently rewritten, so it falls back to the configured default
    instead of routing to a surprise vault. This is the single team-name
    validator — the FastMCP header / switch_team paths reuse it.
    """
    if not value:
        return None
    t = str(value).strip().lower()
    if t in _TEAM_RESET_ALIASES or not _TEAM_SLUG_RE.match(t):
        return None
    return t


def _canonical_default_team():
    """Canonicalise the trusted process default the way the backend would.

    ``_config.team`` comes from ``--default-team`` / ``MEMPALACE_TEAM`` / config
    (trusted), so unlike the strict-validated override paths it is normalised
    (collapse to ``[a-z0-9_]``, cap length) to match the schema slug rather than
    rejected.

    Intentionally re-derives the normalisation rather than importing
    ``backends.postgres.sanitize_team`` so this module stays backend-agnostic
    (sanitize_team also reads ``MEMPALACE_TEAM`` with its own fallback). Both
    apply the same ``[^a-z0-9_]→_``, strip-underscore, ``[:40]`` rule, so the
    slug they produce for a given input is identical.
    """
    raw = re.sub(r"[^a-z0-9_]+", "_", (_config.team or "default").strip().lower()).strip("_")
    return (raw or "default")[:40]


def _resolve_team(explicit=None):
    """Resolve which team vault this call routes to (server-mode only).

    Order, highest priority first: explicit ``vault=`` / ``team=`` argument >
    per-session active team (``switch_team`` / ``X-Mempalace-Team`` header, via
    :data:`_active_team_var`) > the process default (:func:`_canonical_default_team`).
    The returned value is always a canonical ``[a-z0-9_]`` slug — identical to
    the Postgres schema slug the backend derives from it.
    """
    return _valid_team(explicit) or _valid_team(_active_team_var.get()) or _canonical_default_team()


_kg_by_path: dict[str, KnowledgeGraph] = {}
_kg_cache_lock = threading.Lock()
_palace_flag_given: bool = bool(_args.palace)

# MCP server idle auto-exit (#1552).  Stale MCP servers from ended Claude
# Code sessions do not self-terminate, accumulating ChromaDB/HNSW file
# handles on Windows.  When MEMPALACE_MCP_IDLE_HOURS is set (or defaults
# to 8 h), a background daemon thread exits the process once no request
# has been handled for that long.  Set to 0 to disable.
_MCP_IDLE_HOURS_ENV = "MEMPALACE_MCP_IDLE_HOURS"
_MCP_IDLE_HOURS_DEFAULT = 8.0
_last_request_time: float = time.monotonic()


def _mcp_idle_timeout_secs() -> float:
    """Return the configured MCP idle timeout in seconds (0 = disabled)."""
    raw = os.environ.get(_MCP_IDLE_HOURS_ENV, "")
    if raw:
        try:
            hours = float(raw)
            return max(0.0, hours) * 3600
        except ValueError:
            return 0.0
    return _MCP_IDLE_HOURS_DEFAULT * 3600


def _resolve_kg_path() -> str:
    if _palace_flag_given:
        return os.path.join(_config.palace_path, "knowledge_graph.sqlite3")
    return DEFAULT_KG_PATH


def _canonicalize_kg_path(path: str) -> str:
    """Canonicalize a KG cache key so aliases collapse onto one entry.

    ``realpath`` resolves symlinks: two tenants pointing at the same
    SQLite file via different layouts (``/srv/A`` and
    ``/srv/link-to-A``) hit a single cached ``KnowledgeGraph`` rather
    than opening duplicate connections. ``normcase`` normalizes Windows
    drive-letter casing (``C:\\palace`` vs ``c:\\palace``) and
    path-separator style; on POSIX it returns the input unchanged.
    """
    return os.path.normcase(os.path.realpath(path))


def _get_kg(canonical_path=None) -> KnowledgeGraph:
    """Return the cached ``KnowledgeGraph`` for the resolved palace.

    When ``canonical_path`` is ``None`` (default), the path is resolved
    from module state and canonicalized. Callers like :func:`_call_kg`
    that have already captured a canonical key before entering a retry
    loop should pass it through here so the dict insertion uses the same
    key the caller will later use for eviction. Recomputing the key
    inside this function would let ``MEMPALACE_PALACE_PATH`` rotation,
    a symlink remap, or a mount remap between the captured value and
    this call drift the insert and evict keys apart, stranding a closed
    handle under one key while the lookup probes another.
    """
    # Server-mode backends store the KG centrally in the team's vault schema
    # (PostgresKnowledgeGraph), sharing the storage backend's connection pool.
    # Cached per team rather than per filesystem path.
    if _config.backend != "chroma":
        from .knowledge_graph_postgres import PostgresKnowledgeGraph
        from .palace import _resolve_backend

        team = _resolve_team()
        key = f"pgkg::{team}"
        kg = _kg_by_path.get(key)
        if kg is not None:
            return kg
        with _kg_cache_lock:
            kg = _kg_by_path.get(key)
            if kg is None:
                kg = PostgresKnowledgeGraph(_resolve_backend(_config), team=team)
                _kg_by_path[key] = kg
        return kg

    path = (
        canonical_path if canonical_path is not None else _canonicalize_kg_path(_resolve_kg_path())
    )
    kg = _kg_by_path.get(path)
    if kg is not None:
        return kg
    with _kg_cache_lock:
        kg = _kg_by_path.get(path)
        if kg is None:
            kg = KnowledgeGraph(db_path=path)
            _kg_by_path[path] = kg
    return kg


def _call_kg(op):
    """Run ``op(kg)`` against the cached KG with one-shot retry on close.

    Race we're guarding against: a handler grabs ``kg = _get_kg()`` and is
    about to call ``kg.add_triple(...)`` when ``tool_reconnect`` fires on
    another thread, drains ``_kg_by_path``, and closes the underlying
    sqlite3.Connection. The handler's call then raises
    ``sqlite3.ProgrammingError: Cannot operate on a closed database`` and
    bubbles up as a -32000 to the MCP client even though the user just
    asked for a reconnect.

    Catch that single class of error, evict the stale entry from the
    cache (only if it still points at the closed instance — another
    thread may have already replaced it), and try once more with a fresh
    KG. Beyond one retry give up: a second close means we're losing a
    sustained race we won't win in this loop, and a hung loop is worse
    than a clear failure surface.

    The canonical path is captured once at the top and threaded through
    every ``_get_kg`` call plus the eviction lookup. Doing canonicalize
    only here means an ``OSError`` from ``realpath`` (transient Windows
    junction loss, broken mount) surfaces cleanly before any handler
    runs instead of masking a ``sqlite3.ProgrammingError`` mid-retry.
    Passing the captured key through to ``_get_kg`` also locks the
    insert key to the evict key even if FS or env state mutates between
    attempts, preventing a closed handle from leaking under a stale
    key the lookup no longer matches.
    """
    path = _canonicalize_kg_path(_resolve_kg_path())
    for attempt in range(2):
        kg = _get_kg(path)
        try:
            return op(kg)
        except sqlite3.ProgrammingError:
            if attempt == 0:
                with _kg_cache_lock:
                    if _kg_by_path.get(path) is kg:
                        _kg_by_path.pop(path, None)
                continue
            raise


# ── Per-vault entity index (server mode only) ───────────────────────────────
# Mirrors the KG cache: one PostgresEntityIndex per team, sharing the storage
# backend's connection pool. Drained alongside _kg_by_path on reconnect.
_entity_index_by_team: dict = {}
_entity_index_lock = threading.Lock()


def _get_entity_index(team=None):
    """Return the cached ``PostgresEntityIndex`` for the resolved team."""
    from .entity_index_postgres import PostgresEntityIndex
    from .palace import _resolve_backend

    t = team if team is not None else _resolve_team()
    key = f"pgentidx::{t}"
    idx = _entity_index_by_team.get(key)
    if idx is not None:
        return idx
    with _entity_index_lock:
        idx = _entity_index_by_team.get(key)
        if idx is None:
            idx = PostgresEntityIndex(_resolve_backend(_config), team=t)
            _entity_index_by_team[key] = idx
    return idx


def _index_drawer_entities(team, drawer_ids, content, wing, room):
    """Best-effort: extract entities from ``content`` and index them per vault.

    No-op on chroma (the miner stamps entities there). NEVER raises — an
    entity-index failure must not fail or undo a verbatim drawer write
    (verbatim-always). Entities are extracted once over the full content and the
    same set is written for every physical drawer id (each chunk of a chunked
    drawer), so an ``entity=`` lookup hits whatever physical row search returns.
    """
    if _config.backend == "chroma":
        return
    try:
        from .miner import _extract_entities_for_metadata

        idx = _get_entity_index(team)
        entities_str = _extract_entities_for_metadata(content, known=idx.known_entities())
        entities = [e for e in entities_str.split(";") if e]
        if entities:
            idx.add(drawer_ids, entities, wing, room)
    except Exception:
        logger.debug("entity index update failed (best-effort)", exc_info=True)


def _unindex_drawer_entities(team, drawer_ids):
    """Best-effort: drop a drawer's entity rows. No-op on chroma; never raises."""
    if _config.backend == "chroma":
        return
    try:
        _get_entity_index(team).delete_by_drawer(drawer_ids)
    except Exception:
        logger.debug("entity index delete failed (best-effort)", exc_info=True)


_client_cache = None
_collection_cache = None
_palace_db_inode = 0  # inode of chroma.sqlite3 at cache time
_palace_db_mtime = 0.0  # mtime of chroma.sqlite3 at cache time


def _is_transient_index_error(result) -> bool:
    # Chroma can return "Internal error: Error finding id" during the
    # HNSW flush window after a bulk CLI mine — SQLite rows are
    # committed but the binary segment metadata isn't flushed yet.
    # Self-heals once the flush completes (~30-60s). See issue #1315.
    if not isinstance(result, dict):
        return False
    err = result.get("error", "")
    return isinstance(err, str) and ("Error finding id" in err or "Internal error" in err)


def _force_chroma_cache_reset() -> None:
    # Drop both the MCP-local client cache and the shared backend's
    # per-palace cache so the next call rebuilds against the post-flush
    # state. Without clearing _DEFAULT_BACKEND._clients the retry
    # would just hit the same stale handle, since tool_search routes
    # via search_memories -> palace.get_collection -> backend cache.
    global \
        _client_cache, \
        _collection_cache, \
        _palace_db_inode, \
        _palace_db_mtime, \
        _metadata_cache, \
        _metadata_cache_time
    _client_cache = None
    _collection_cache = None
    _palace_db_inode = 0
    _palace_db_mtime = 0.0
    _metadata_cache = None
    _metadata_cache_time = 0
    try:
        from .palace import _DEFAULT_BACKEND

        _DEFAULT_BACKEND._clients.pop(_config.palace_path, None)
        _DEFAULT_BACKEND._freshness.pop(_config.palace_path, None)
    except Exception:
        pass


# ── Vector-search disabled flag (#1222) ──────────────────────────────────
# Set when ``hnsw_capacity_status`` reports a divergence between sqlite
# and the HNSW segment large enough that chromadb would segfault on
# segment load. While this is set, vector-shaped tools (``search``,
# ``check_duplicate``) route to the sqlite-only BM25 fallback in
# :func:`mempalace.searcher._bm25_only_via_sqlite`. Cleared after a
# successful repair via :func:`tool_reconnect` (which re-runs the probe).
_vector_disabled = False
_vector_disabled_reason = ""
# Optional[dict] (not ``dict | None``) keeps Python 3.9 import-time
# parsing happy — PEP 604 unions in annotations only became unconditional
# at module-eval time in 3.10.
_vector_capacity_status: Optional[dict] = None


def _refresh_vector_disabled_flag() -> None:
    """Re-run the HNSW capacity probe and update the module-level flag.

    Called from :func:`_get_client` whenever the client cache is rebuilt
    (first open or palace replacement). Cheap — pure sqlite + pickle
    read, no chromadb interaction. Never raises: a probe that crashes
    would defeat the point.
    """
    global _vector_disabled, _vector_disabled_reason, _vector_capacity_status
    try:
        info = hnsw_capacity_status(_config.palace_path, _config.collection_name)
    except Exception:
        logger.debug("HNSW capacity probe raised", exc_info=True)
        return
    _vector_capacity_status = info
    if info.get("diverged"):
        if not _vector_disabled:
            logger.warning(
                "HNSW capacity divergence detected (%s) — routing search to "
                "BM25-only sqlite fallback. Run `mempalace repair` to restore "
                "vector search.",
                info.get("message", "unknown"),
            )
        _vector_disabled = True
        _vector_disabled_reason = info.get("message", "")
    else:
        if _vector_disabled:
            logger.info(
                "HNSW capacity within tolerance (%s) — vector search re-enabled",
                info.get("message", ""),
            )
        _vector_disabled = False
        _vector_disabled_reason = ""


# ==================== WRITE-AHEAD LOG ====================
# Every write operation is logged to a JSONL file before execution.
# This provides an audit trail for detecting memory poisoning and
# enables review/rollback of writes from external or untrusted sources.

_WAL_DIR = Path(os.path.expanduser("~/.mempalace/wal"))
_WAL_DIR.mkdir(parents=True, exist_ok=True)
try:
    _WAL_DIR.chmod(0o700)
except (OSError, NotImplementedError):
    pass
_WAL_FILE = _WAL_DIR / "write_log.jsonl"
# Atomically create WAL file with restricted permissions (no TOCTOU race).
# os.open with O_CREAT|O_WRONLY and mode 0o600 creates the file if absent
# or opens it if present, both in a single syscall.
try:
    _fd = os.open(str(_WAL_FILE), os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(_fd)
except (OSError, NotImplementedError):
    pass

# Keys whose values should be redacted in WAL entries to avoid logging sensitive content
_WAL_REDACT_KEYS = frozenset(
    {"content", "content_preview", "document", "entry", "entry_preview", "query", "text"}
)


def _redact_wal_dict(d):
    """Key-based redaction for a WAL payload dict (params or result).

    Applied to both ``params`` and ``result`` because the postgres sink ships
    them to central, org-wide storage — the same redaction discipline the local
    jsonl file gets. Non-dict values (e.g. ``result=None``) pass through.
    """
    if not isinstance(d, dict):
        return d
    safe = {}
    for k, v in d.items():
        if k in _WAL_REDACT_KEYS:
            safe[k] = f"[REDACTED {len(v)} chars]" if isinstance(v, str) else "[REDACTED]"
        else:
            safe[k] = v
    return safe


_wal_sink_unavailable_warned = False


def _wal_log(operation: str, params: dict, result: dict = None, team=None):
    """Append a write operation to the write-ahead audit log.

    Redaction happens here, once, so it is preserved regardless of sink. With
    ``MEMPALACE_WAL_SINK=postgres`` the redacted entry goes to the central audit
    table (tagged with the team vault the write targeted); the local jsonl file
    is the default and the fallback if the database write fails — the audit
    trail is never silently dropped.

    ``team`` is the vault the write actually targeted. Callers that route to a
    non-default vault (``tool_add_drawer(vault=...)``) MUST pass the same value
    they routed the write with, so the audit row matches where the data landed.
    When omitted it resolves to the session/default vault — correct for the
    write tools that take no explicit vault.
    """
    global _wal_sink_unavailable_warned
    safe_params = _redact_wal_dict(params)
    safe_result = _redact_wal_dict(result)
    ts = datetime.now()
    # The vault the write targeted (None on the local single-vault chroma backend).
    if team is None and _config.backend != "chroma":
        team = _resolve_team()
    entry = {
        "timestamp": ts.isoformat(),
        "operation": operation,
        "team": team,
        "params": safe_params,
        "result": safe_result,
    }

    if _config.wal_sink == "postgres":
        try:
            from .palace import _resolve_backend

            backend = _resolve_backend(_config)
            if hasattr(backend, "wal_append"):
                backend.wal_append(
                    timestamp=ts,
                    operation=operation,
                    team=team,
                    params=safe_params,
                    result=safe_result,
                )
                return
            # wal_sink=postgres but the active backend has no central sink (e.g.
            # chroma): fall through to jsonl, but warn once so the operator knows
            # the audit trail is local-only, not centralized as configured.
            if not _wal_sink_unavailable_warned:
                logger.warning(
                    "MEMPALACE_WAL_SINK=postgres but backend %r has no central audit "
                    "sink — writing the audit log to the local jsonl file instead.",
                    _config.backend,
                )
                _wal_sink_unavailable_warned = True
        except Exception as e:
            logger.error("WAL postgres sink failed, falling back to jsonl: %s", e)
    _wal_log_jsonl(entry)


def _wal_log_jsonl(entry: dict) -> None:
    """Append one audit entry to the local append-only jsonl file (0600)."""
    try:
        fd = os.open(str(_WAL_FILE), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.error(f"WAL write failed: {e}")


def _get_client():
    """Return a ChromaDB PersistentClient, reconnecting if the database changed on disk.

    Detects palace rebuilds (repair/nuke/purge) by checking the inode of
    chroma.sqlite3.  A full rebuild replaces the file, changing the inode.
    Also detects external writes (scripts, CLI) via mtime changes — the
    inode check alone misses in-place modifications that invalidate the
    in-memory HNSW index.

    Note: FAT/exFAT may return 0 for st_ino — the ``current_inode != 0``
    guard skips reconnect detection on those filesystems (safe fallback).
    """
    global \
        _client_cache, \
        _collection_cache, \
        _palace_db_inode, \
        _palace_db_mtime, \
        _metadata_cache, \
        _metadata_cache_time
    db_path = os.path.join(_config.palace_path, "chroma.sqlite3")
    try:
        st = os.stat(db_path)
        current_inode = st.st_ino
        current_mtime = st.st_mtime
    except OSError:
        current_inode = 0
        current_mtime = 0.0

    # If the DB file disappeared (e.g. during rebuild) but we have a cached
    # collection, invalidate so we don't serve stale data.  Without this,
    # both stored and current values are 0 on the first call after deletion,
    # making inode_changed and mtime_changed both False.
    if not os.path.isfile(db_path) and _collection_cache is not None:
        _client_cache = None
        _collection_cache = None
        _palace_db_inode = 0
        _palace_db_mtime = 0.0
        # Fall through to normal reconnect which will handle missing DB

    inode_changed = current_inode != 0 and current_inode != _palace_db_inode
    mtime_changed = current_mtime != 0.0 and abs(current_mtime - _palace_db_mtime) > 0.01

    if _client_cache is None or inode_changed or mtime_changed:
        # Run the HNSW capacity probe BEFORE chromadb opens the segment —
        # if the index is severely undersized, segment load can segfault
        # the whole MCP server (#1222). The probe is pure sqlite +
        # metadata-pickle read; never touches the HNSW binary files.
        _refresh_vector_disabled_flag()
        _client_cache = ChromaBackend.make_client(_config.palace_path)
        _collection_cache = None
        _metadata_cache = None
        _metadata_cache_time = 0
        _palace_db_inode = current_inode
        _palace_db_mtime = current_mtime
    return _client_cache


def _get_collection(create=False, team=None):
    """Return the active collection, caching the client between calls.

    For server-mode backends (``MEMPALACE_BACKEND=postgres``) this delegates to
    the backend-aware :func:`mempalace.palace.get_collection`, routing to the
    machine's primary team vault (``config.team``) or an explicit ``team``
    override. The chroma fast-path below — with its client/collection caching
    and stale-HNSW healing — is preserved unchanged for the local default.

    On failure (chroma path), log the exception and retry once after clearing
    the client and collection caches. The retry forces ``_get_client()`` to
    rebuild from scratch (which re-runs ``quarantine_stale_hnsw`` per #1322),
    so the second attempt heals the common stale-handle / stale-HNSW case.
    """
    if _config.backend != "chroma":
        from .backends import CollectionNotInitializedError, PalaceNotFoundError
        from .palace import get_collection as _palace_get_collection

        # Route to the call's explicit ``team`` if given, else the per-session
        # active team (header / switch_team), else the process default. Every
        # read/modify tool reaches its vault through this single resolution.
        resolved_team = _resolve_team(team)
        try:
            return _palace_get_collection(
                _config.palace_path, _config.collection_name, create=create, team=resolved_team
            )
        except (PalaceNotFoundError, CollectionNotInitializedError):
            return None

    global _client_cache, _collection_cache, _metadata_cache, _metadata_cache_time
    for attempt in range(2):
        try:
            client = _get_client()
            # ChromaDB 1.x persists the EF *identity* (its ``name()``) with the
            # collection but not the EF *instance/configuration*. So a reader or
            # writer that omits ``embedding_function=`` silently gets chromadb's
            # built-in ``DefaultEmbeddingFunction`` — its ``name()`` matches the
            # one we spoof in ``mempalace.embedding`` (both report ``"default"``,
            # the identity check passes), but the *provider list* is chromadb's
            # default rather than the user's resolved device. On bleeding-edge
            # interpreters (#1299: python 3.14 + chromadb 1.5.x on Apple Silicon)
            # that default provider selection can SIGSEGV the host process on
            # first ``col.add()``. The miner / Stop hook ingest path avoids this
            # because it routes through ``ChromaBackend.get_collection``, which
            # resolves the EF via ``ChromaBackend._resolve_embedding_function``;
            # the MCP server bypassed that abstraction. Resolve the EF inside the
            # branches that actually open a collection so warm-cache reads stay
            # zero-cost. Reuse the backend helper so the two call sites can't
            # drift on logging or fallback semantics.
            if create:
                ef = ChromaBackend._resolve_embedding_function()
                ef_kwargs = {"embedding_function": ef} if ef is not None else {}
                # hnsw:num_threads=1 disables ChromaDB's multi-threaded ParallelFor
                # HNSW insert path, which has a race in repairConnectionsForUpdate /
                # addPoint (see issues #974, #965). Set via metadata on fresh
                # collections and re-applied via _pin_hnsw_threads() for legacy
                # palaces whose collections were created before this fix (the
                # runtime config does not persist cross-process in chromadb 1.5.x,
                # so the retrofit runs every time _get_collection opens a cache).
                #
                # ChromaDB 1.5.x's Rust binding SIGSEGVs when get_or_create_collection
                # is called with metadata that differs from what's stored. The split
                # below skips the metadata-comparison codepath for existing
                # collections, mirroring the backend-layer fix from #1262.
                try:
                    raw = client.get_collection(_config.collection_name, **ef_kwargs)
                except _ChromaNotFoundError:
                    raw = client.create_collection(
                        _config.collection_name,
                        metadata={
                            "hnsw:space": "cosine",
                            "hnsw:num_threads": 1,
                            **_HNSW_BLOAT_GUARD,
                        },
                        **ef_kwargs,
                    )
                _pin_hnsw_threads(raw)
                _collection_cache = ChromaCollection(raw, palace_path=_config.palace_path)
                _metadata_cache = None
                _metadata_cache_time = 0
            elif _collection_cache is None:
                ef = ChromaBackend._resolve_embedding_function()
                ef_kwargs = {"embedding_function": ef} if ef is not None else {}
                raw = client.get_collection(_config.collection_name, **ef_kwargs)
                _pin_hnsw_threads(raw)
                _collection_cache = ChromaCollection(raw, palace_path=_config.palace_path)
                _metadata_cache = None
                _metadata_cache_time = 0
            return _collection_cache
        except Exception:
            logger.exception(
                "_get_collection attempt %d/2 failed (palace=%s, create=%s)",
                attempt + 1,
                _config.palace_path,
                create,
            )
            if attempt == 0:
                # Reset all caches so the next attempt forces _get_client()
                # to rebuild the chromadb client from scratch — that path
                # re-runs quarantine_stale_hnsw (#1322) and reopens the
                # collection cleanly, healing the common stale-handle case.
                _client_cache = None
                _collection_cache = None
                _metadata_cache = None
                _metadata_cache_time = 0
    return None


def _no_palace():
    return {
        "error": "No palace found",
        "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
    }


# ==================== HELPERS ====================


def _safe_meta(meta):
    """Coerce a Chroma metadata value to a dict.

    ChromaDB's ``col.get()`` / ``col.query()`` can return ``None`` for the
    metadata cell of a partially-flushed row (or any row written without
    metadata in older formats). Indexing the result then yields ``None``,
    and downstream ``.get(...)`` calls raise::

        AttributeError: 'NoneType' object has no attribute 'get'

    This bug bricked the embeddings_queue cleanup path in issue #1426 —
    the handler crashed before reaching the ``DELETE FROM embeddings_queue``
    step, so the queue grew without bound while writes kept appearing
    successful.

    Centralizing the coercion through this helper makes the contract
    explicit and keeps the fix self-documenting at every call site:
    *metadata is always a dict by the time it leaves the boundary*.
    """
    return meta if isinstance(meta, dict) else {}


def _fetch_all_metadata(col, where=None):
    """Paginate col.get() to avoid the 10K silent truncation limit."""
    total = col.count()
    all_meta = []
    offset = 0
    while offset < total:
        kwargs = {"include": ["metadatas"], "limit": 1000, "offset": offset}
        if where:
            kwargs["where"] = where
        batch = col.get(**kwargs)
        if not batch["metadatas"]:
            break
        all_meta.extend(batch["metadatas"])
        offset += len(batch["metadatas"])
    return all_meta


_metadata_cache = None
_metadata_cache_time = 0
_METADATA_CACHE_TTL = 5.0  # seconds
_MAX_RESULTS = 100  # upper bound for search/list limit params


def _get_cached_metadata(col, where=None):
    """Return cached metadata if fresh, else fetch and cache.

    The process-global cache is chroma-only on purpose. It exists to avoid
    re-running chroma's metadata read (and, on the #1222 path, to keep the
    segfault-prone client untouched). In server mode it would be WRONG: the
    cache is vault-agnostic, so on the shared central server one session's
    vault metadata could be served to another session within the TTL. Postgres
    ``_fetch_all_metadata`` is a direct, per-vault indexed query, so bypass the
    cache and always fetch for the collection actually passed in.
    """
    global _metadata_cache, _metadata_cache_time
    if _config.backend != "chroma":
        return _fetch_all_metadata(col, where=where)
    now = time.time()
    if (
        where is None
        and _metadata_cache is not None
        and (now - _metadata_cache_time) < _METADATA_CACHE_TTL
    ):
        return _metadata_cache
    result = _fetch_all_metadata(col, where=where)
    if where is None:
        _metadata_cache = result
        _metadata_cache_time = now
    return result


def _sanitize_optional_name(value: str = None, field_name: str = "name") -> str:
    """Validate optional wing/room-style filters."""
    if value is None or not value.strip():
        return None
    return sanitize_name(value, field_name)


# ==================== READ TOOLS ====================


def _tool_status_via_sqlite() -> dict:
    """Pure-sqlite status reader for the #1222 fallback path.

    When the HNSW capacity probe detects divergence, opening the chromadb
    persistent client can segfault. This reader pulls the same wing/room
    breakdown directly from ``embedding_metadata`` so the operator still
    gets a working status response — and crucially the
    ``vector_disabled`` flag — without us touching the vector segment.
    """
    import sqlite3 as _sqlite3

    db_path = os.path.join(_config.palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return _no_palace()
    collection_name = _config.collection_name

    wings: dict = {}
    rooms: dict = {}
    total = 0
    try:
        conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ?
                """,
                (collection_name,),
            ).fetchone()
            total = int(row[0]) if row and row[0] is not None else 0
            for key, target in (("wing", wings), ("room", rooms)):
                for value, count in conn.execute(
                    """
                    SELECT em.string_value, COUNT(*)
                    FROM embedding_metadata em
                    JOIN embeddings e ON em.id = e.id
                    JOIN segments s ON e.segment_id = s.id
                    JOIN collections c ON s.collection = c.id
                    WHERE c.name = ?
                      AND em.key = ?
                      AND em.string_value IS NOT NULL
                    GROUP BY em.string_value
                    """,
                    (collection_name, key),
                ):
                    target[value] = count
        finally:
            conn.close()
    except _sqlite3.Error:
        logger.exception("tool_status sqlite fallback read failed")

    result = {
        "total_drawers": total,
        "wings": wings,
        "rooms": rooms,
        "protocol": PALACE_PROTOCOL,
        "aaak_dialect": AAAK_SPEC,
        "vector_disabled": True,
        "vector_disabled_reason": _vector_disabled_reason,
    }
    if _vector_capacity_status:
        result["hnsw_capacity"] = {
            "sqlite_count": _vector_capacity_status.get("sqlite_count"),
            "hnsw_count": _vector_capacity_status.get("hnsw_count"),
            "divergence": _vector_capacity_status.get("divergence"),
        }
    return result


def tool_status():
    # Server-mode backends (postgres) report connection health + the active
    # team vault, not the chroma-only on-disk / HNSW / vector_disabled signals.
    if _config.backend != "chroma":
        return _tool_status_server()

    # Run the safe sqlite/pickle probe before we touch chromadb. In the
    # #1222 failure mode, opening the persistent client to call .count()
    # can segfault — short-circuit to a pure-sqlite path when divergence
    # is detected so status stays reachable.
    db_exists = os.path.isfile(os.path.join(_config.palace_path, "chroma.sqlite3"))
    _refresh_vector_disabled_flag()

    if _vector_disabled:
        return _tool_status_via_sqlite()

    # Use create=True only when a palace DB already exists on disk -- this
    # bootstraps the ChromaDB collection on a valid-but-empty palace without
    # accidentally creating a palace in a non-existent directory (#830).
    col = _get_collection(create=db_exists)
    if not col:
        return _no_palace()
    return _status_from_collection(col)


def _status_from_collection(col, extra=None):
    """Build the drawers + wing/room breakdown payload from a collection.

    Shared by the chroma and server-mode status paths so the count/metadata
    aggregation can never drift between them. ``extra`` is merged in first
    (server mode passes backend/vault/health fields).
    """
    wings = {}
    rooms = {}
    result = {
        "total_drawers": col.count(),
        "wings": wings,
        "rooms": rooms,
        "protocol": PALACE_PROTOCOL,
        "aaak_dialect": AAAK_SPEC,
    }
    if extra:
        result.update(extra)
    try:
        for m in _get_cached_metadata(col):
            m = m or {}
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            wings[w] = wings.get(w, 0) + 1
            rooms[r] = rooms.get(r, 0) + 1
    except Exception as e:
        logger.exception("tool_status metadata fetch failed")
        result["error"] = str(e)
        result["partial"] = True
    return result


def _tool_status_server():
    """Status for server-mode backends (postgres): PG health + active vault.

    Reports the backend connection health and the resolved team vault (and the
    available vaults), then the usual drawer/wing/room breakdown for the active
    vault. Deliberately skips the chroma-only signals — on-disk ``chroma.sqlite3``,
    the ``vector_disabled`` HNSW probe, and HNSW capacity — none of which apply
    to the central server.
    """
    from .palace import _resolve_backend

    backend = _resolve_backend(_config)
    result = {"backend": _config.backend, "vault": _resolve_team()}

    try:
        health = backend.health()
        result["healthy"] = health.ok
        if health.detail:
            result["health_detail"] = health.detail
    except Exception as e:
        result["healthy"] = False
        result["health_detail"] = str(e)

    if not result.get("healthy"):
        # Server unreachable — return health without probing collections.
        result["protocol"] = PALACE_PROTOCOL
        result["aaak_dialect"] = AAAK_SPEC
        return result

    try:
        result["vaults"] = backend.list_vaults() if hasattr(backend, "list_vaults") else []
    except Exception as e:
        result["vaults_error"] = str(e)

    col = _get_collection()
    if col is None:
        result["total_drawers"] = 0
        result["protocol"] = PALACE_PROTOCOL
        result["aaak_dialect"] = AAAK_SPEC
        return result
    return _status_from_collection(col, extra=result)


# ── AAAK Dialect Spec ─────────────────────────────────────────────────────────
# Included in status response so the AI learns it on first wake-up call.
# Also available via mempalace_get_aaak_spec tool.

PALACE_PROTOCOL = """IMPORTANT — MemPalace Memory Protocol:
1. ON WAKE-UP: Call mempalace_status to load palace overview + AAAK spec.
2. BEFORE RESPONDING about any person, project, or past event: call mempalace_kg_query or mempalace_search FIRST. Never guess — verify.
3. IF UNSURE about a fact (name, gender, age, relationship): say "let me check" and query the palace. Wrong is worse than slow.
4. AFTER EACH SESSION: call mempalace_diary_write to record what happened, what you learned, what matters.
5. WHEN FACTS CHANGE: call mempalace_kg_invalidate on the old fact, mempalace_kg_add for the new one.

This protocol ensures the AI KNOWS before it speaks. Storage is not memory — but storage + this protocol = memory."""

AAAK_SPEC = """AAAK is a compressed memory dialect that MemPalace uses for efficient storage.
It is designed to be readable by both humans and LLMs without decoding.

FORMAT:
  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.
  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.
  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.
  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).
  IMPORTANCE: ★ to ★★★★★ (1-5 scale).
  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.
  WINGS: wing_user, wing_agent, wing_team, wing_code, wing_myproject, wing_hardware, wing_ue5, wing_ai_research.
  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).

EXAMPLE:
  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)

Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.
When WRITING AAAK: use entity codes, mark emotions, keep structure tight."""


def tool_list_wings():
    col = _get_collection()
    if not col:
        return _no_palace()
    wings = {}
    result = {"wings": wings}
    try:
        all_meta = _get_cached_metadata(col)
        for m in all_meta:
            m = m or {}
            w = m.get("wing", "unknown")
            wings[w] = wings.get(w, 0) + 1
    except Exception as e:
        logger.exception("tool_list_wings metadata fetch failed")
        result["error"] = str(e)
        result["partial"] = True
    return result


def tool_list_rooms(wing: str = None):
    try:
        wing = _sanitize_optional_name(wing, "wing")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    if not col:
        return _no_palace()
    rooms = {}
    result = {"wing": wing or "all", "rooms": rooms}
    try:
        where = {"wing": wing} if wing else None
        all_meta = _fetch_all_metadata(col, where=where)
        for m in all_meta:
            m = m or {}
            r = m.get("room", "unknown")
            rooms[r] = rooms.get(r, 0) + 1
    except Exception as e:
        logger.exception("tool_list_rooms metadata fetch failed")
        result["error"] = str(e)
        result["partial"] = True
    return result


def tool_get_taxonomy():
    col = _get_collection()
    if not col:
        return _no_palace()
    taxonomy = {}
    result = {"taxonomy": taxonomy}
    try:
        all_meta = _get_cached_metadata(col)
        for m in all_meta:
            m = m or {}
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            if w not in taxonomy:
                taxonomy[w] = {}
            taxonomy[w][r] = taxonomy[w].get(r, 0) + 1
    except Exception as e:
        logger.exception("tool_get_taxonomy metadata fetch failed")
        result["error"] = str(e)
        result["partial"] = True
    return result


def tool_search(
    query: str,
    limit: int = 5,
    wing: str = None,
    room: str = None,
    max_distance: float = 1.5,
    min_similarity: float = None,
    context: str = None,
    vault: str = None,
    entity: str = None,
):
    limit = max(1, min(limit, _MAX_RESULTS))
    try:
        wing = _sanitize_optional_name(wing, "wing")
        room = _sanitize_optional_name(room, "room")
    except ValueError as e:
        return {"error": str(e)}
    # Backwards compat: accept old name
    # Backwards compat: convert old similarity scale (higher=stricter) to
    # distance scale (lower=stricter). Similarity 0.8 → distance 0.2.
    dist = (1.0 - min_similarity) if min_similarity is not None else max_distance
    # Mitigate system prompt contamination (Issue #333)
    sanitized = sanitize_query(query)

    # Team-vault routing (server-mode backends only; chroma ignores ``vault``).
    # vault=None / "primary" -> the machine's configured primary team vault.
    # vault="all"            -> search every team vault, returned per-vault.
    # vault="<team>"         -> that team's vault.
    if _config.backend != "chroma" and vault and vault.lower() == "all":
        return _search_all_vaults(sanitized["clean_query"], wing, room, limit, dist)
    # Resolve the vault to search: explicit ``vault=`` > per-session active team
    # (header / switch_team) > process default. Chroma is single-vault, so team
    # stays ``None`` there (the namespace is ignored downstream anyway).
    search_team = _resolve_team(vault) if _config.backend != "chroma" else None

    # Entity-scoped recall (central/postgres only): narrow the candidate set to
    # the drawers that mention `entity` before the vector rank, so an entity's
    # drawers are considered even when vector-distant from the query text. On
    # chroma `entity` is ignored (no per-vault index). A known filter that
    # matches nothing returns an empty result rather than the unscoped search.
    restrict_ids = None
    if entity and _config.backend != "chroma":
        try:
            rows = _get_entity_index(search_team).drawers_for_entity(entity)
            restrict_ids = [r["drawer_id"] for r in rows]
        except Exception:
            logger.debug("entity filter resolution failed (best-effort)", exc_info=True)
            restrict_ids = None
        if restrict_ids == []:
            return {"query": query, "entity": entity, "results": [], "count": 0}

    # Ensure the vector-disabled probe has been run via the safe
    # sqlite/pickle path before we touch chromadb. Calling _get_client()
    # here would defeat the fallback — it constructs a PersistentClient
    # which can segfault on segment load in the #1222 failure mode.
    # Server-mode backends (postgres) have no on-disk HNSW index, so the probe
    # is chroma-only — skipped here (``_vector_disabled`` stays False, i.e.
    # pgvector search stays enabled).
    if _config.backend == "chroma":
        _refresh_vector_disabled_flag()
    result = search_memories(
        sanitized["clean_query"],
        palace_path=_config.palace_path,
        wing=wing,
        room=room,
        n_results=limit,
        max_distance=dist,
        vector_disabled=_vector_disabled,
        collection_name=_config.collection_name,
        team=search_team,
        restrict_ids=restrict_ids,
    )
    if _config.backend == "chroma" and _is_transient_index_error(result):
        # Post-bulk-write HNSW flush window (#1315): drop caches, give
        # the segment a moment to settle, retry once. Caller never sees
        # the transient unless the second attempt also fails.
        _force_chroma_cache_reset()
        time.sleep(2)
        _refresh_vector_disabled_flag()
        result = search_memories(
            sanitized["clean_query"],
            palace_path=_config.palace_path,
            wing=wing,
            room=room,
            n_results=limit,
            max_distance=dist,
            vector_disabled=_vector_disabled,
            team=search_team,
        )
        if not _is_transient_index_error(result):
            result["index_recovered"] = True
    if search_team is not None:
        result["vault"] = search_team
    if _vector_disabled:
        result["vector_disabled"] = True
        result["vector_disabled_reason"] = _vector_disabled_reason
    # Attach sanitizer metadata for transparency
    if sanitized["was_sanitized"]:
        result["query_sanitized"] = True
        result["sanitizer"] = {
            "method": sanitized["method"],
            "original_length": sanitized["original_length"],
            "clean_length": sanitized["clean_length"],
            "clean_query": sanitized["clean_query"],
        }
    if context:
        result["context_received"] = True
    return result


def _search_all_vaults(clean_query, wing, room, limit, dist):
    """Fan a search across every team vault (server-mode ``vault="all"``).

    Returns results grouped per vault rather than a single merged list, so the
    model can see which team each hit came from without the server having to
    re-rank across heterogeneous vaults.
    """
    from .palace import _resolve_backend

    backend = _resolve_backend(_config)
    try:
        teams = backend.list_vaults() if hasattr(backend, "list_vaults") else []
    except Exception as e:
        return {"error": f"could not list vaults: {e}"}
    primary = _resolve_team()
    if primary not in teams:
        teams = [primary, *teams]
    results_by_vault = {}
    for t in teams:
        results_by_vault[t] = search_memories(
            clean_query,
            palace_path=_config.palace_path,
            wing=wing,
            room=room,
            n_results=limit,
            max_distance=dist,
            collection_name=_config.collection_name,
            team=t,
        )
    return {
        "query": clean_query,
        "vault": "all",
        "vaults_searched": teams,
        "results_by_vault": results_by_vault,
    }


def tool_list_vaults():
    """List the team vaults available on the central server.

    Call once per session to discover which vaults exist and which is this
    machine's primary (from local config / ``MEMPALACE_TEAM``). On the local
    chroma backend there is a single implicit ``local`` vault.
    """
    if _config.backend == "chroma":
        return {
            "backend": "chroma",
            "mode": "local",
            "primary": "local",
            "vaults": ["local"],
        }
    from .palace import _resolve_backend

    backend = _resolve_backend(_config)
    try:
        teams = backend.list_vaults() if hasattr(backend, "list_vaults") else []
    except Exception as e:
        return {"error": str(e)}
    primary = _resolve_team()
    return {
        "backend": _config.backend,
        "mode": "central",
        "primary": primary,
        "vaults": teams,
        "hint": "Pass vault='<team>' to read/write another team's vault, or vault='all' to search across all.",
    }


def tool_entities(
    entity: str = None, wing: str = None, min_count: int = 2, limit: int = 50, vault: str = None
):
    """Navigate the per-vault entity index (central server only).

    With ``entity``: the drawers that mention it (ids + wing/room) — the entry
    point for entity-scoped recall; pair with ``mempalace_search(entity=...)``
    for the verbatim content. Without ``entity``: the vault's most-mentioned
    entities, scoped to ``wing`` if given. ``min_count`` (overview only, default
    2) filters one-off extraction noise; pass 1 to see everything. On the local
    chroma backend the entity index does not exist (use ``mempalace mine`` +
    search there).
    """
    if _config.backend == "chroma":
        return {
            "available": False,
            "backend": "chroma",
            "reason": "The entity index is a central-server feature; "
            "on local installs use mempalace mine + search.",
        }
    try:
        wing = _sanitize_optional_name(wing, "wing")
    except ValueError as e:
        return {"error": str(e)}
    team = _resolve_team(vault)
    try:
        idx = _get_entity_index(team)
        if entity:
            rows = idx.drawers_for_entity(entity)
            return {
                "backend": _config.backend,
                "vault": team,
                "entity": entity,
                "drawer_count": len({r["drawer_id"] for r in rows}),
                "drawers": rows,
                "hint": f"Pass entity='{entity}' to mempalace_search for the verbatim content.",
            }
        top = idx.top_entities(
            wing=wing, min_count=max(1, int(min_count)), limit=max(1, min(int(limit), 500))
        )
        return {
            "backend": _config.backend,
            "vault": team,
            "wing": wing or "all",
            "entities": top,
            "count": len(top),
        }
    except Exception as e:
        return {"error": str(e)}


def tool_check_duplicate(content: str, threshold: float = 0.9):
    _refresh_vector_disabled_flag()
    if _vector_disabled:
        # Without a usable HNSW we can't compute cosine similarity for
        # near-duplicate detection. Report the limitation rather than
        # silently returning "not a duplicate" — false negatives here
        # would let the AI re-file content the palace already holds.
        return {
            "is_duplicate": False,
            "matches": [],
            "vector_disabled": True,
            "vector_disabled_reason": _vector_disabled_reason,
            "hint": (
                "duplicate detection requires vector search; run `mempalace repair` to restore"
            ),
        }
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        content = strip_lone_surrogates(content)
        results = col.query(
            query_texts=[content],
            n_results=5,
            include=["metadatas", "documents", "distances"],
        )
        duplicates = []
        if results["ids"] and results["ids"][0]:
            for i, drawer_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                similarity = round(max(0.0, 1 - dist), 3)
                if similarity >= threshold:
                    # Chroma 1.5.x can return None for partially-flushed rows;
                    # coerce to empty sentinels so downstream .get() is safe.
                    meta = _safe_meta(results["metadatas"][0][i])
                    doc = results["documents"][0][i] or ""
                    duplicates.append(
                        {
                            "id": drawer_id,
                            "wing": meta.get("wing", "?"),
                            "room": meta.get("room", "?"),
                            "similarity": similarity,
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        }
                    )
        return {
            "is_duplicate": len(duplicates) > 0,
            "matches": duplicates,
        }
    except Exception:
        logger.exception("check_duplicate failed")
        return {"error": "Duplicate check failed"}


def tool_get_aaak_spec():
    """Return the AAAK dialect specification."""
    return {"aaak_spec": AAAK_SPEC}


def tool_traverse_graph(start_room: str, max_hops: int = 2):
    """Walk the palace graph from a room. Find connected ideas across wings."""
    max_hops = max(1, min(max_hops, 10))
    col = _get_collection()
    if not col:
        return _no_palace()
    return traverse(start_room, col=col, max_hops=max_hops)


def tool_find_tunnels(wing_a: str = None, wing_b: str = None):
    """Find rooms that bridge two wings — the hallways connecting domains."""
    try:
        wing_a = _sanitize_optional_name(wing_a, "wing_a")
        wing_b = _sanitize_optional_name(wing_b, "wing_b")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    if not col:
        return _no_palace()
    return find_tunnels(wing_a, wing_b, col=col)


def tool_graph_stats():
    """Palace graph overview: nodes, tunnels, edges, connectivity."""
    col = _get_collection()
    if not col:
        return _no_palace()
    return graph_stats(col=col)


def tool_create_tunnel(
    source_wing: str,
    source_room: str,
    target_wing: str,
    target_room: str,
    label: str = "",
    source_drawer_id: str = None,
    target_drawer_id: str = None,
):
    """Create an explicit cross-wing tunnel between two palace locations.

    Use when you notice content in one project relates to another project.
    Example: an API design discussion in project_api connects to the
    database schema in project_database.
    """
    # sanitize_name and create_tunnel both raise ValueError for invalid or
    # missing endpoints (empty/non-string names, and create_tunnel's
    # room-existence checks). Catch both so the real reason is surfaced
    # instead of escaping and being wrapped as the opaque "Internal tool
    # error" (#1473), mirroring sibling tools.
    try:
        source_wing = sanitize_name(source_wing, "source_wing")
        source_room = sanitize_name(source_room, "source_room")
        target_wing = sanitize_name(target_wing, "target_wing")
        target_room = sanitize_name(target_room, "target_room")
        return create_tunnel(
            source_wing,
            source_room,
            target_wing,
            target_room,
            label=label,
            source_drawer_id=source_drawer_id,
            target_drawer_id=target_drawer_id,
        )
    except ValueError as e:
        return {"error": str(e)}


def tool_list_tunnels(wing: str = None):
    """List all explicit cross-wing tunnels, optionally filtered by wing."""
    try:
        wing = _sanitize_optional_name(wing, "wing")
    except ValueError as e:
        return {"error": str(e)}
    return list_tunnels(wing)


def tool_delete_tunnel(tunnel_id: str):
    """Delete an explicit tunnel by its ID."""
    if not tunnel_id or not isinstance(tunnel_id, str):
        return {"error": "tunnel_id is required"}
    return delete_tunnel(tunnel_id)


def tool_follow_tunnels(wing: str, room: str):
    """Follow explicit tunnels from a room to see connected drawers in other wings."""
    try:
        wing = sanitize_name(wing, "wing")
        room = sanitize_name(room, "room")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    return follow_tunnels(wing, room, col=col)


# ==================== WRITE TOOLS ====================


def tool_add_drawer(
    wing: str, room: str, content: str, source_file: str = None, added_by: str = "mcp",
    vault: str = None,
):
    """File verbatim content into a wing/room. Checks for duplicates first.

    Content above ``chunk_size`` is split into bounded per-chunk drawers
    via a single batched upsert. Each chunk carries ``parent_drawer_id``
    linkage and ``chunk_index`` metadata so search can rejoin them. The
    returned ``drawer_id`` is the LOGICAL group handle on the chunked
    path; physical drawer ids are in ``chunk_ids`` (#1539). To delete
    or fetch the underlying drawers, iterate ``chunk_ids`` or query by
    ``parent_drawer_id`` — ``tool_get_drawer(drawer_id)`` and
    ``tool_delete_drawer(drawer_id)`` report "not found" on the chunked
    path because no row is stored under the logical group id.
    """
    global _metadata_cache
    try:
        wing = sanitize_name(wing, "wing")
        room = sanitize_name(room, "room")
        content = sanitize_content(content)
        if source_file:
            source_file = strip_lone_surrogates(source_file)
        added_by = strip_lone_surrogates(added_by)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Server-mode write routing: an explicit ``vault=`` targets that team, else
    # the per-session active team (header / switch_team). Resolve ONCE so the
    # collection write and the WAL audit row agree on the target vault. Chroma
    # is single-vault, so team stays None.
    add_team = _resolve_team(vault) if _config.backend != "chroma" else None
    col = _get_collection(create=True, team=add_team)
    if not col:
        return _no_palace()

    drawer_id = (
        f"drawer_{wing}_{room}_{hashlib.sha256((wing + room + content).encode()).hexdigest()[:24]}"
    )

    _wal_log(
        "add_drawer",
        {
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "added_by": added_by,
            "content_length": len(content),
            "content_preview": content[:200],
        },
        team=add_team,
    )

    chunk_size = _config.chunk_size
    base_meta = {
        "wing": wing,
        "room": room,
        "source_file": source_file or "",
        "added_by": added_by,
        "filed_at": datetime.now().isoformat(),
    }

    # Idempotency. Three cases to detect a prior committed write:
    # (a) Single-doc path: drawer_id row exists (the only id used).
    # (b) Chunked path: probe the LAST chunk id — its presence implies
    #     every earlier chunk also landed, since the batched upsert
    #     is all-or-nothing.
    # (c) Legacy pre-#1539 single-row write of oversized content under
    #     drawer_id: probe drawer_id alongside the last chunk id so a
    #     re-call with identical oversized content does not duplicate
    #     the legacy row by adding fresh chunks under different ids.
    if len(content) <= chunk_size:
        idempotency_probe_ids = [drawer_id]
    else:
        last_chunk_idx = (len(content) - 1) // chunk_size
        idempotency_probe_ids = [drawer_id, f"{drawer_id}_chunk_{last_chunk_idx:06d}"]
    try:
        existing = col.get(ids=idempotency_probe_ids, include=[])
        if existing.ids:
            return {"success": True, "reason": "already_exists", "drawer_id": drawer_id}
    except Exception:
        logger.debug("Idempotency pre-check failed for %s", idempotency_probe_ids, exc_info=True)

    try:
        if len(content) <= chunk_size:
            col.upsert(
                ids=[drawer_id],
                documents=[content],
                metadatas=[{**base_meta, "chunk_index": 0}],
            )
            inserted = col.get(ids=[drawer_id], include=[])
            if not inserted.ids:
                raise RuntimeError(
                    "Drawer write was acknowledged but the new ID is not readable. "
                    "The palace index may be stale; run reconnect or repair."
                )
            _metadata_cache = None
            # Best-effort per-vault entity index (server mode); never fails the write.
            _index_drawer_entities(add_team, [drawer_id], content, wing, room)
            logger.info(f"Filed drawer: {drawer_id} → {wing}/{room}")
            return {
                "success": True,
                "drawer_id": drawer_id,
                "wing": wing,
                "room": room,
                "chunks": 1,
            }

        # Oversized content: split into bounded per-chunk drawers so the
        # embedding model never sees a document above ``chunk_size``.
        # Single batched ``upsert`` so the embedding pass either commits
        # every chunk or none — no half-written palace if the embedding
        # model fails mid-loop (#1539).
        chunk_ids: list[str] = []
        chunk_docs: list[str] = []
        chunk_metas: list[dict] = []
        for i in range(0, len(content), chunk_size):
            chunk_idx = i // chunk_size
            chunk_ids.append(f"{drawer_id}_chunk_{chunk_idx:06d}")
            chunk_docs.append(content[i : i + chunk_size])
            chunk_metas.append(
                {**base_meta, "chunk_index": chunk_idx, "parent_drawer_id": drawer_id}
            )
        col.upsert(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas)
        # Probe the LAST chunk id, not the first — its presence confirms
        # the whole batch landed, not just the leading row.
        inserted = col.get(ids=[chunk_ids[-1]], include=[])
        if not inserted.ids:
            raise RuntimeError(
                "Drawer write was acknowledged but the new ID is not readable. "
                "The palace index may be stale; run reconnect or repair."
            )
        _metadata_cache = None
        # Best-effort per-vault entity index (server mode). Keyed on the physical
        # chunk ids actually written so delete/update stay consistent.
        _index_drawer_entities(add_team, chunk_ids, content, wing, room)
        logger.info(f"Filed drawer: {drawer_id} → {wing}/{room} ({len(chunk_ids)} chunks)")
        return {
            "success": True,
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_delete_drawer(drawer_id: str):
    """Delete a single drawer by ID."""
    global _metadata_cache
    col = _get_collection()
    if not col:
        return _no_palace()
    existing = col.get(ids=[drawer_id])
    if not existing["ids"]:
        return {"success": False, "error": f"Drawer not found: {drawer_id}"}

    # Log the deletion with the content being removed for audit trail
    deleted_content = existing.get("documents", [""])[0] if existing.get("documents") else ""
    deleted_meta = _safe_meta(
        existing.get("metadatas", [{}])[0] if existing.get("metadatas") else {}
    )
    _wal_log(
        "delete_drawer",
        {
            "drawer_id": drawer_id,
            "deleted_meta": deleted_meta,
            "content_preview": deleted_content[:200],
        },
    )

    try:
        col.delete(ids=[drawer_id])
        _metadata_cache = None
        # Best-effort: drop the drawer's entity rows (server mode). Same team
        # _get_collection() resolved (implicit session/default).
        _unindex_drawer_entities(
            _resolve_team() if _config.backend != "chroma" else None, [drawer_id]
        )
        logger.info(f"Deleted drawer: {drawer_id}")
        return {"success": True, "drawer_id": drawer_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_sync(project_dir: str = None, wing: str = None, apply: bool = False):
    """Prune drawers whose source files are gitignored, missing, or moved (#1252)."""
    global _metadata_cache
    from .palace import MineAlreadyRunning
    from .sync import sync_palace

    if not _config.palace_path:
        np = _no_palace()
        return {"success": False, "error": np.get("error", "no palace"), "hint": np.get("hint")}
    project_dirs = [project_dir] if project_dir else None
    try:
        try:
            report = sync_palace(
                palace_path=_config.palace_path,
                project_dirs=project_dirs,
                wing=wing,
                dry_run=not apply,
                wal_log=_wal_log,
            )
            return {"success": True, **report}
        # Order matters: typed handlers must precede the bare Exception
        # below, otherwise MineAlreadyRunning and ValueError fall into the
        # generic "sync failed" branch and break the structured-error tests.
        except MineAlreadyRunning as exc:
            return {
                "success": False,
                "error": f"another mine is in progress: {exc}",
                "error_class": "LockHeldByOtherProcess",
            }
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            return {"success": False, "error": f"sync failed: {exc}"}
    finally:
        if apply:
            _metadata_cache = None


def tool_get_drawer(drawer_id: str):
    """Fetch a single drawer by ID. Returns full content and metadata."""
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        result = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return {"error": f"Drawer not found: {drawer_id}"}
        meta = _safe_meta(result["metadatas"][0])
        doc = result["documents"][0]
        # source_file is the absolute filesystem path written by the
        # miners. Reduce to its basename before handing it to the MCP
        # client — same threat model as the palace_path leak fix:
        # nested-agent / multi-server topologies treat the client as a
        # separate trust domain. Basename preserves citation utility.
        # Mirrors the searcher.search_memories() return shape.
        safe_meta = dict(meta) if meta else {}
        if safe_meta.get("source_file"):
            safe_meta["source_file"] = Path(safe_meta["source_file"]).name
        return {
            "drawer_id": drawer_id,
            "content": doc,
            "wing": safe_meta.get("wing", ""),
            "room": safe_meta.get("room", ""),
            "metadata": safe_meta,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_list_drawers(wing: str = None, room: str = None, limit: int = 20, offset: int = 0):
    """List drawers with pagination. Optional wing/room filter."""
    limit = max(1, min(limit, _MAX_RESULTS))
    offset = max(0, offset)
    try:
        wing = _sanitize_optional_name(wing, "wing")
        room = _sanitize_optional_name(room, "room")
    except ValueError as e:
        return {"error": str(e)}
    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        where = None
        conditions = []
        if wing:
            conditions.append({"wing": wing})
        if room:
            conditions.append({"room": room})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        kwargs = {"include": ["documents", "metadatas"], "limit": limit, "offset": offset}
        if where:
            kwargs["where"] = where
        result = col.get(**kwargs)

        # Compute total matching drawers for pagination.
        if where:
            total_result = col.get(where=where, include=[])
            total = len(total_result["ids"])
        else:
            total = col.count()

        drawers = []
        for i, did in enumerate(result["ids"]):
            meta = _safe_meta(result["metadatas"][i])
            doc = result["documents"][i]
            drawers.append(
                {
                    "drawer_id": did,
                    "wing": meta.get("wing", ""),
                    "room": meta.get("room", ""),
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                }
            )
        return {
            "drawers": drawers,
            "total": total,
            "count": len(drawers),
            "offset": offset,
            "limit": limit,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_update_drawer(drawer_id: str, content: str = None, wing: str = None, room: str = None):
    """Update an existing drawer's content and/or metadata."""
    global _metadata_cache

    if content is None and wing is None and room is None:
        return {"success": True, "drawer_id": drawer_id, "noop": True}

    col = _get_collection()
    if not col:
        return _no_palace()
    try:
        existing = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        if not existing["ids"]:
            return {"success": False, "error": f"Drawer not found: {drawer_id}"}

        old_meta = _safe_meta(existing["metadatas"][0])
        old_doc = existing["documents"][0]

        new_doc = old_doc
        if content is not None:
            try:
                new_doc = sanitize_content(content)
            except ValueError as e:
                return {"success": False, "error": str(e)}

        new_meta = dict(old_meta)
        if wing is not None:
            try:
                new_meta["wing"] = sanitize_name(wing, "wing")
            except ValueError as e:
                return {"success": False, "error": str(e)}
        if room is not None:
            try:
                new_meta["room"] = sanitize_name(room, "room")
            except ValueError as e:
                return {"success": False, "error": str(e)}

        _wal_log(
            "update_drawer",
            {
                "drawer_id": drawer_id,
                "old_wing": old_meta.get("wing", ""),
                "old_room": old_meta.get("room", ""),
                "new_wing": new_meta.get("wing", ""),
                "new_room": new_meta.get("room", ""),
                "content_changed": content is not None,
                "content_preview": new_doc[:200] if content is not None else None,
            },
        )

        update_kwargs = {"ids": [drawer_id]}
        if content is not None:
            update_kwargs["documents"] = [new_doc]
        update_kwargs["metadatas"] = [new_meta]
        col.update(**update_kwargs)

        _metadata_cache = None

        # Best-effort: re-index entities (server mode). An update can change the
        # content, wing, or room, any of which alters the entity rows, so drop
        # and re-extract from the new state rather than leaving stale rows.
        if _config.backend != "chroma":
            upd_team = _resolve_team()
            _unindex_drawer_entities(upd_team, [drawer_id])
            _index_drawer_entities(
                upd_team, [drawer_id], new_doc, new_meta.get("wing"), new_meta.get("room")
            )

        logger.info(f"Updated drawer: {drawer_id}")
        return {
            "success": True,
            "drawer_id": drawer_id,
            "wing": new_meta.get("wing", ""),
            "room": new_meta.get("room", ""),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== KNOWLEDGE GRAPH ====================


def tool_kg_query(entity: str, as_of: str = None, direction: str = "both"):
    """Query the knowledge graph for an entity's relationships."""
    try:
        entity = sanitize_kg_value(entity, "entity")
        as_of = sanitize_iso_temporal(as_of, "as_of")
    except ValueError as e:
        return {"error": str(e)}

    if direction not in ("outgoing", "incoming", "both"):
        return {"error": "direction must be 'outgoing', 'incoming', or 'both'"}

    results = _call_kg(lambda kg: kg.query_entity(entity, as_of=as_of, direction=direction))
    return {"entity": entity, "as_of": as_of, "facts": results, "count": len(results)}


def tool_kg_add(
    subject: str,
    predicate: str,
    object: str,
    valid_from: str = None,
    valid_to: str = None,
    source_closet: str = None,
    source_file: str = None,
    source_drawer_id: str = None,
):
    """Add a relationship to the knowledge graph.

    All temporal and provenance fields are optional. ``valid_to`` lets callers
    backfill historical facts with a known end date/time in a single call
    instead of a separate ``kg_invalidate`` call.

    Temporal values accept either ``YYYY-MM-DD`` or canonical UTC datetimes in
    the form ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    try:
        subject = sanitize_kg_value(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_kg_value(object, "object")
        valid_from = sanitize_iso_temporal(valid_from, "valid_from")
        valid_to = sanitize_iso_temporal(valid_to, "valid_to")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    _wal_log(
        "kg_add",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "source_closet": source_closet,
            "source_file": source_file,
            "source_drawer_id": source_drawer_id,
        },
    )

    triple_id = _call_kg(
        lambda kg: kg.add_triple(
            subject,
            predicate,
            object,
            valid_from=valid_from,
            valid_to=valid_to,
            source_closet=source_closet,
            source_file=source_file,
            source_drawer_id=source_drawer_id,
        )
    )
    return {"success": True, "triple_id": triple_id, "fact": f"{subject} → {predicate} → {object}"}


def tool_kg_invalidate(subject: str, predicate: str, object: str, ended: str = None):
    """Mark a fact as no longer true.

    Returns the actual ``ended`` date/time that was stored. When the caller
    omits ``ended``, the underlying graph stamps ``date.today()`` and the
    response reflects that resolved value.

    Temporal values accept either ``YYYY-MM-DD`` or canonical UTC datetimes in
    the form ``YYYY-MM-DDTHH:MM:SSZ``.
    """
    try:
        subject = sanitize_kg_value(subject, "subject")
        predicate = sanitize_name(predicate, "predicate")
        object = sanitize_kg_value(object, "object")
        ended = sanitize_iso_temporal(ended, "ended")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    resolved_ended = ended or date.today().isoformat()

    _wal_log(
        "kg_invalidate",
        {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "ended": resolved_ended,
        },
    )

    _call_kg(lambda kg: kg.invalidate(subject, predicate, object, ended=resolved_ended))
    return {
        "success": True,
        "fact": f"{subject} → {predicate} → {object}",
        "ended": resolved_ended,
    }


def tool_kg_timeline(entity: str = None):
    """Get chronological timeline of facts, optionally for one entity."""
    if entity is not None:
        try:
            entity = sanitize_kg_value(entity, "entity")
        except ValueError as e:
            return {"error": str(e)}
    results = _call_kg(lambda kg: kg.timeline(entity))
    return {"entity": entity or "all", "timeline": results, "count": len(results)}


def tool_kg_stats():
    """Knowledge graph overview: entities, triples, relationship types."""
    return _call_kg(lambda kg: kg.stats())


# ==================== AGENT DIARY ====================


def tool_diary_write(agent_name: str, entry: str, topic: str = "general", wing: str = ""):
    """
    Write a diary entry for this agent. Entries are timestamped and
    accumulate over time in a diary room.

    This is the agent's personal journal — observations, thoughts,
    what it worked on, what it noticed, what it thinks matters.

    Note: ``agent_name`` is normalized to lowercase before storage so
    that diary reads are case-insensitive (see #1243). "Claude",
    "claude", and "CLAUDE" all resolve to the same agent.
    """
    try:
        agent_name = sanitize_name(agent_name, "agent_name").lower()
        entry = sanitize_content(entry)
        topic = sanitize_name(topic, "topic")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if wing:
        wing = sanitize_name(wing)
    else:
        wing = f"wing_{agent_name.replace(' ', '_')}"
    room = "diary"
    col = _get_collection(create=True)
    if not col:
        return _no_palace()

    now = datetime.now()
    entry_id = (
        f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S%f')}_"
        f"{hashlib.sha256(entry.encode()).hexdigest()[:12]}"
    )

    _wal_log(
        "diary_write",
        {
            "agent_name": agent_name,
            "topic": topic,
            "entry_id": entry_id,
            "entry_preview": entry[:200],
        },
    )

    try:
        # TODO: Future versions should expand AAAK before embedding to improve
        # semantic search quality. For now, store raw AAAK in metadata so it's
        # preserved, and keep the document as-is for embedding (even though
        # compressed AAAK degrades embedding quality).
        base_metadata = {
            "wing": wing,
            "room": room,
            "hall": "hall_diary",
            "topic": topic,
            "type": "diary_entry",
            "agent": agent_name,
            "filed_at": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
        }
        chunk_size = _config.chunk_size
        if len(entry) <= chunk_size:
            col.add(
                ids=[entry_id],
                documents=[entry],
                metadatas=[{**base_metadata, "chunk_index": 0}],
            )
            logger.info(f"Diary entry: {entry_id} → {wing}/diary/{topic}")
            return {
                "success": True,
                "entry_id": entry_id,
                "agent": agent_name,
                "topic": topic,
                "timestamp": now.isoformat(),
                "chunks": 1,
            }

        # Oversized entry: split into bounded per-chunk drawers so the
        # embedding model never sees a document above ``chunk_size``.
        # Every chunk carries ``parent_entry_id`` so search can rejoin
        # them and ``chunk_index`` for ordered reconstruction (#1539).
        # Note on ``entry_id`` in the return value: for the chunked
        # path the returned ``entry_id`` is the LOGICAL group handle
        # (no drawer is stored under that exact id). The physical
        # drawer ids are in ``chunk_ids``. Callers wanting to fetch
        # by id should iterate ``chunk_ids``; callers wanting to
        # query by metadata can filter on ``parent_entry_id``.
        # Use a single batched ``add`` so the embedding pass either
        # commits all chunks or none — avoids a half-written palace
        # if the embedding model fails mid-loop. ``col.add`` (not
        # ``upsert``) is intentional here: ``entry_id`` is timestamp-
        # based with microsecond precision, so every call generates a
        # fresh id and a duplicate is by definition a same-microsecond
        # clash that should surface as an error rather than silently
        # overwrite the prior entry (cf. ``tool_add_drawer`` whose
        # content-hash ids are deliberately idempotent and use upsert).
        chunk_ids: list[str] = []
        chunk_docs: list[str] = []
        chunk_metas: list[dict] = []
        for i in range(0, len(entry), chunk_size):
            chunk_idx = i // chunk_size
            chunk_ids.append(f"{entry_id}_chunk_{chunk_idx:06d}")
            chunk_docs.append(entry[i : i + chunk_size])
            chunk_metas.append(
                {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "parent_entry_id": entry_id,
                }
            )
        col.add(ids=chunk_ids, documents=chunk_docs, metadatas=chunk_metas)
        logger.info(f"Diary entry: {entry_id} → {wing}/diary/{topic} ({len(chunk_ids)} chunks)")
        return {
            "success": True,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "timestamp": now.isoformat(),
            "chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_diary_read(agent_name: str, last_n: int = 10, wing: str = ""):
    """
    Read an agent's recent diary entries. Returns the last N entries
    in chronological order — the agent's personal journal.

    When ``wing`` is provided, reads only from that wing. When ``wing``
    is empty or omitted, returns entries from every wing this agent has
    written to. Diary writes from hooks land in project-derived wings
    (``wing_<project>``), so requiring a specific wing on read would
    silo those entries from agent-initiated reads.

    Note: ``agent_name`` is normalized to lowercase before filtering so
    that reads are case-insensitive (see #1243). Entries written under
    pre-fix mixed-case agent names will not match the lowercase filter;
    use ``mempalace repair`` to migrate legacy data if needed.
    """
    try:
        agent_name = sanitize_name(agent_name, "agent_name").lower()
        if wing:
            wing = sanitize_name(wing)
    except ValueError as e:
        return {"error": str(e)}
    last_n = max(1, min(last_n, 100))
    col = _get_collection()
    if not col:
        return _no_palace()

    # Build filter: always scope by agent + room=diary. Wing is optional —
    # when empty, return entries across all wings for this agent (matches
    # the #1097 empty-string-as-no-filter convention for LLM ergonomics).
    conditions = [{"room": "diary"}, {"agent": agent_name}]
    if wing:
        conditions.insert(0, {"wing": wing})

    try:
        results = col.get(
            where={"$and": conditions},
            include=["documents", "metadatas"],
            limit=10000,
        )

        if not results["ids"]:
            return {"agent": agent_name, "entries": [], "message": "No diary entries yet."}

        # Combine and sort by timestamp
        entries = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            meta = _safe_meta(meta)
            entries.append(
                {
                    "date": meta.get("date", ""),
                    "timestamp": meta.get("filed_at", ""),
                    "topic": meta.get("topic", ""),
                    "content": doc,
                }
            )

        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        entries = entries[:last_n]

        return {
            "agent": agent_name,
            "entries": entries,
            "total": len(results["ids"]),
            "showing": len(entries),
        }
    except Exception:
        logger.exception("diary_read failed")
        return {"error": "Failed to read diary entries"}


def tool_hook_settings(silent_save: bool = None, desktop_toast: bool = None):
    """
    Get or set hook behavior settings.

    - silent_save: True = stop hook saves directly (no MCP clutter),
      False = legacy blocking MCP calls. Default: True.
    - desktop_toast: True = show notify-send desktop toast on save,
      False = terminal-only notification. Default: False.

    Call with no arguments to see current settings.
    """
    from .config import MempalaceConfig

    try:
        config = MempalaceConfig()
    except Exception as e:
        return {"success": False, "error": str(e)}

    changed = []
    if silent_save is not None:
        config.set_hook_setting("silent_save", silent_save)
        changed.append(f"silent_save → {silent_save}")
    if desktop_toast is not None:
        config.set_hook_setting("desktop_toast", desktop_toast)
        changed.append(f"desktop_toast → {desktop_toast}")

    # Re-read to return current state
    try:
        config = MempalaceConfig()
    except Exception:
        logger.debug("Could not re-read config after update", exc_info=True)

    result = {
        "success": True,
        "settings": {
            "silent_save": config.hook_silent_save,
            "desktop_toast": config.hook_desktop_toast,
        },
    }
    if changed:
        result["updated"] = changed
    return result


def tool_memories_filed_away():
    """Acknowledge the latest silent checkpoint. Returns a short summary."""
    state_dir = Path.home() / ".mempalace" / "hook_state"
    ack_file = state_dir / "last_checkpoint"
    if not ack_file.is_file():
        return {
            "status": "quiet",
            "message": "No recent journal entry",
            "count": 0,
            "timestamp": None,
        }
    try:
        data = json.loads(ack_file.read_text(encoding="utf-8"))
        ack_file.unlink(missing_ok=True)
        msgs = data.get("msgs", 0)
        return {
            "status": "ok",
            "message": f"\u2726 {msgs} messages tucked into drawers",
            "count": msgs,
            "timestamp": data.get("ts", None),
        }
    except (json.JSONDecodeError, OSError):
        ack_file.unlink(missing_ok=True)
        return {
            "status": "error",
            "message": "\u2726 Journal entry filed in the palace",
            "count": 0,
            "timestamp": None,
        }


# ==================== SETTINGS TOOLS ====================


def tool_reconnect():
    """Force the MCP server to drop cached ChromaDB + KnowledgeGraph state.

    Use after external scripts or CLI commands modify the palace database
    or replace ``knowledge_graph.sqlite3`` directly, which can leave the
    in-memory HNSW index stale or pin a closed-on-disk SQLite connection.

    Server-mode backends (postgres) reconnect the connection pool and drain the
    cached KnowledgeGraph handles instead of resetting chroma client caches.
    """
    if _config.backend != "chroma":
        return _tool_reconnect_server()
    global \
        _client_cache, \
        _collection_cache, \
        _palace_db_inode, \
        _palace_db_mtime, \
        _vector_disabled, \
        _vector_disabled_reason
    from . import palace as palace_module

    close_errors = []
    try:
        palace_module._DEFAULT_BACKEND.close_palace(_config.palace_path)
    except Exception as exc:
        logger.debug("Failed to close shared palace backend during reconnect", exc_info=True)
        close_errors.append(f"backend close_palace failed: {exc}")
    try:
        from chromadb.api.client import SharedSystemClient

        clear_system_cache = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear_system_cache):
            clear_system_cache()
        else:
            logger.debug(
                "SharedSystemClient.clear_system_cache is unavailable; skipping shared Chroma cache clear during reconnect"
            )
    except Exception as exc:
        logger.debug(
            "Failed to clear Chroma shared system cache during reconnect",
            exc_info=True,
        )
        close_errors.append(f"shared Chroma cache clear failed: {exc}")
    _client_cache = None
    _collection_cache = None
    _palace_db_inode = 0
    _palace_db_mtime = 0.0
    # Force probe re-run on next _get_client by clearing the flag now;
    # _refresh_vector_disabled_flag will re-set it if the divergence
    # still applies after the reconnect.
    _vector_disabled = False
    _vector_disabled_reason = ""
    # Drain the per-path KnowledgeGraph cache so a replaced sqlite file is
    # reopened on the next tool call rather than served from a stale handle.
    with _kg_cache_lock:
        for kg in _kg_by_path.values():
            try:
                kg.close()
            except Exception:
                pass
        _kg_by_path.clear()
    # Drain the per-vault entity-index cache too so a reconnected pool is used.
    with _entity_index_lock:
        _entity_index_by_team.clear()
    try:
        col = _get_collection()
        if col is None:
            result = {
                "success": False,
                "message": "No palace found after reconnect",
                "drawers": 0,
                "vector_disabled": _vector_disabled,
            }
            if close_errors:
                result["error"] = "; ".join(close_errors)
            return result
        if close_errors:
            return {
                "success": False,
                "message": "Reconnect reopened the palace but failed to fully reset cached handles",
                "drawers": col.count(),
                "vector_disabled": _vector_disabled,
                "vector_disabled_reason": _vector_disabled_reason,
                "error": "; ".join(close_errors),
            }
        return {
            "success": True,
            "message": "Reconnected to palace",
            "drawers": col.count(),
            "vector_disabled": _vector_disabled,
            "vector_disabled_reason": _vector_disabled_reason,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tool_reconnect_server():
    """Reconnect a server-mode backend (postgres): drop the pool + KG cache.

    The chroma path resets on-disk client/HNSW caches; centrally there is no
    such state. Instead we close the connection pool (the next call lazily
    re-opens it) and drain the cached KnowledgeGraph handles (they share the
    pool, so their close() is a no-op — clearing the cache forces a rebuild
    against the fresh pool), then report PG health + the active vault.
    """
    from .palace import _resolve_backend

    backend = _resolve_backend(_config)
    errors = []
    try:
        if hasattr(backend, "reconnect"):
            backend.reconnect()
        elif hasattr(backend, "close"):
            backend.close()
    except Exception as exc:
        logger.debug("postgres pool reconnect failed", exc_info=True)
        errors.append(f"pool reconnect failed: {exc}")

    with _kg_cache_lock:
        for kg in _kg_by_path.values():
            try:
                kg.close()
            except Exception:
                pass
        _kg_by_path.clear()
    # Drain the per-vault entity-index cache too so a reconnected pool is used.
    with _entity_index_lock:
        _entity_index_by_team.clear()

    result = {"success": not errors, "backend": _config.backend, "vault": _resolve_team()}
    try:
        health = backend.health()
        result["healthy"] = health.ok
        if health.detail:
            result["health_detail"] = health.detail
        if not health.ok:
            result["success"] = False
        col = _get_collection()
        result["drawers"] = col.count() if col is not None else 0
        result["message"] = (
            "Reconnected to central server"
            if result["success"]
            else "Reconnect completed with errors"
        )
    except Exception as exc:
        result["success"] = False
        errors.append(str(exc))
    if errors:
        result["error"] = "; ".join(errors)
    return result


# ==================== MCP PROTOCOL ====================

TOOLS = {
    "mempalace_status": {
        "description": "Palace overview — total drawers, wing and room counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_status,
    },
    "mempalace_list_wings": {
        "description": "List all wings with drawer counts",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_wings,
    },
    "mempalace_list_rooms": {
        "description": "List rooms within a wing (or all rooms if no wing given)",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing to list rooms for (optional)"},
            },
        },
        "handler": tool_list_rooms,
    },
    "mempalace_get_taxonomy": {
        "description": "Full taxonomy: wing → room → drawer count",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_taxonomy,
    },
    "mempalace_get_aaak_spec": {
        "description": "Get the AAAK dialect specification — the compressed memory format MemPalace uses. Call this if you need to read or write AAAK-compressed memories.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_get_aaak_spec,
    },
    "mempalace_kg_query": {
        "description": "Query the knowledge graph for an entity's relationships. Returns typed facts with temporal validity. E.g. 'Max' → child_of Alice, loves chess, does swimming. Filter by date with as_of to see what was true at a point in time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to query (e.g. 'Max', 'MyProject', 'Alice')",
                },
                "as_of": {
                    "type": "string",
                    "description": "Date/datetime filter — only facts valid at this time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ, optional)",
                },
                "direction": {
                    "type": "string",
                    "description": "outgoing (entity→?), incoming (?→entity), or both (default: both)",
                },
            },
            "required": ["entity"],
        },
        "handler": tool_kg_query,
    },
    "mempalace_kg_add": {
        "description": "Add a fact to the knowledge graph. Subject → predicate → object with optional time window. E.g. ('Max', 'started_school', 'Year 7', valid_from='2026-09-01'). Pass valid_to to backfill an already-ended historical fact in a single call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "The entity doing/being something"},
                "predicate": {
                    "type": "string",
                    "description": "The relationship type (e.g. 'loves', 'works_on', 'daughter_of')",
                },
                "object": {"type": "string", "description": "The entity being connected to"},
                "valid_from": {
                    "type": "string",
                    "description": "When this became true (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ, optional)",
                },
                "valid_to": {
                    "type": "string",
                    "description": "When this stopped being true (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ, optional). Use for backfilling already-ended historical facts.",
                },
                "source_closet": {
                    "type": "string",
                    "description": "Closet ID where this fact appears (optional)",
                },
                "source_file": {
                    "type": "string",
                    "description": "Source file path the fact was extracted from (optional)",
                },
                "source_drawer_id": {
                    "type": "string",
                    "description": "Drawer ID the fact was extracted from (optional, RFC 002 provenance)",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": tool_kg_add,
    },
    "mempalace_kg_invalidate": {
        "description": "Mark a fact as no longer true. E.g. ankle injury resolved, job ended, moved house.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Entity"},
                "predicate": {"type": "string", "description": "Relationship"},
                "object": {"type": "string", "description": "Connected entity"},
                "ended": {
                    "type": "string",
                    "description": "When it stopped being true (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ, default: today)",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
        "handler": tool_kg_invalidate,
    },
    "mempalace_kg_timeline": {
        "description": "Chronological timeline of facts. Shows the story of an entity (or everything) in order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to get timeline for (optional — omit for full timeline)",
                },
            },
        },
        "handler": tool_kg_timeline,
    },
    "mempalace_kg_stats": {
        "description": "Knowledge graph overview: entities, triples, current vs expired facts, relationship types.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_kg_stats,
    },
    "mempalace_traverse": {
        "description": "Walk the palace graph from a room. Shows connected ideas across wings — the tunnels. Like following a thread through the palace: start at 'chromadb-setup' in wing_code, discover it connects to wing_myproject (planning) and wing_user (feelings about it).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_room": {
                    "type": "string",
                    "description": "Room to start from (e.g. 'chromadb-setup', 'riley-school')",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "How many connections to follow (default: 2)",
                },
            },
            "required": ["start_room"],
        },
        "handler": tool_traverse_graph,
    },
    "mempalace_find_tunnels": {
        "description": "Find rooms that bridge two wings — the hallways connecting different domains. E.g. what topics connect wing_code to wing_team?",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing_a": {"type": "string", "description": "First wing (optional)"},
                "wing_b": {"type": "string", "description": "Second wing (optional)"},
            },
        },
        "handler": tool_find_tunnels,
    },
    "mempalace_graph_stats": {
        "description": "Palace graph overview: total rooms, tunnel connections, edges between wings.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_graph_stats,
    },
    "mempalace_create_tunnel": {
        "description": "Create a cross-wing tunnel linking two palace locations. Use when content in one project relates to another — e.g., an API design in project_api connects to a database schema in project_database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_wing": {"type": "string", "description": "Wing of the source"},
                "source_room": {"type": "string", "description": "Room in the source wing"},
                "target_wing": {"type": "string", "description": "Wing of the target"},
                "target_room": {"type": "string", "description": "Room in the target wing"},
                "label": {"type": "string", "description": "Description of the connection"},
                "source_drawer_id": {
                    "type": "string",
                    "description": "Optional specific drawer ID",
                },
                "target_drawer_id": {
                    "type": "string",
                    "description": "Optional specific drawer ID",
                },
            },
            "required": ["source_wing", "source_room", "target_wing", "target_room"],
        },
        "handler": tool_create_tunnel,
    },
    "mempalace_list_tunnels": {
        "description": "List all explicit cross-wing tunnels. Optionally filter by wing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {
                    "type": "string",
                    "description": "Filter tunnels by wing (shows tunnels where wing is source or target)",
                },
            },
        },
        "handler": tool_list_tunnels,
    },
    "mempalace_delete_tunnel": {
        "description": "Delete an explicit tunnel by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tunnel_id": {"type": "string", "description": "Tunnel ID to delete"},
            },
            "required": ["tunnel_id"],
        },
        "handler": tool_delete_tunnel,
    },
    "mempalace_follow_tunnels": {
        "description": "Follow tunnels from a room to see what it connects to in other wings. Returns connected rooms with drawer previews.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing to start from"},
                "room": {"type": "string", "description": "Room to follow tunnels from"},
            },
            "required": ["wing", "room"],
        },
        "handler": tool_follow_tunnels,
    },
    "mempalace_search": {
        "description": "Semantic search. Returns verbatim drawer content with similarity scores. IMPORTANT: 'query' must contain ONLY search keywords. Use 'context' for background. Results with cosine distance > max_distance are filtered out.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search query ONLY — keywords or a question. Max 250 chars.",
                    "maxLength": 250,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                    "minimum": 1,
                    "maximum": 100,
                },
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "max_distance": {
                    "type": "number",
                    "description": "Max cosine distance threshold (0=identical, 2=opposite). Results further than this are dropped. Lower = stricter. Default 1.5. Set to 0 to disable.",
                },
                "context": {
                    "type": "string",
                    "description": "Background context for the search (optional). NOT used for embedding — only for future re-ranking.",
                },
                "vault": {
                    "type": "string",
                    "description": "Team vault to search (central/postgres deployments only). Omit for this machine's primary team; '<team>' for a specific team; 'all' to search every team vault. Ignored on local installs.",
                },
                "entity": {
                    "type": "string",
                    "description": "Scope recall to drawers that mention this entity (a person, project, or service) before ranking. Use when the user names a specific entity. Central/postgres deployments only; ignored on local installs.",
                },
            },
            "required": ["query"],
        },
        "handler": tool_search,
    },
    "mempalace_entities": {
        "description": "Navigate the team's entity index (central/postgres deployments only). With 'entity': the drawers mentioning it (pair with mempalace_search entity= for the verbatim text). Without 'entity': the vault's most-mentioned entities. Use to answer 'what/who do we know about X' and to find the right entity name to scope a search by.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity to look up (person/project/service). Omit to list the vault's top entities.",
                },
                "wing": {"type": "string", "description": "Filter the top-entities listing to a wing (optional)"},
                "min_count": {
                    "type": "integer",
                    "description": "Top-entities listing only: minimum drawers an entity must appear in (default 2, filters one-off noise). Pass 1 to see everything.",
                },
                "limit": {"type": "integer", "description": "Max entities in the listing (default 50)"},
                "vault": {
                    "type": "string",
                    "description": "Team vault to inspect (omit for your primary; '<team>' for another team).",
                },
            },
            "required": [],
        },
        "handler": tool_entities,
    },
    "mempalace_check_duplicate": {
        "description": "Check if content already exists in the palace before filing",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to check"},
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold 0-1 (default 0.9)",
                },
            },
            "required": ["content"],
        },
        "handler": tool_check_duplicate,
    },
    "mempalace_add_drawer": {
        "description": "File verbatim content into the palace. Checks for duplicates first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Wing (project name)"},
                "room": {
                    "type": "string",
                    "description": "Room (aspect: backend, decisions, meetings...)",
                },
                "content": {
                    "type": "string",
                    "description": "Verbatim content to store — exact words, never summarized",
                },
                "source_file": {"type": "string", "description": "Where this came from (optional)"},
                "added_by": {"type": "string", "description": "Who is filing this (default: mcp)"},
                "vault": {
                    "type": "string",
                    "description": "Team vault to file into (central/postgres deployments only). Omit for this machine's primary team; '<team>' to file into another team's vault. Ignored on local installs.",
                },
            },
            "required": ["wing", "room", "content"],
        },
        "handler": tool_add_drawer,
    },
    "mempalace_list_vaults": {
        "description": "List the team vaults available on the central server and which is this machine's primary. Call once per session before routing memories to a specific team. On local installs returns a single 'local' vault.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_list_vaults,
    },
    "mempalace_delete_drawer": {
        "description": "Delete a drawer by ID. Irreversible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to delete"},
            },
            "required": ["drawer_id"],
        },
        "handler": tool_delete_drawer,
    },
    "mempalace_sync": {
        "description": "Prune drawers whose source files are gitignored, deleted, or moved. Returns dry-run report by default; pass apply=true to commit deletions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_dir": {
                    "type": "string",
                    "description": "Project root to scope the sync (optional; auto-detected from drawer metadata if omitted)",
                },
                "wing": {"type": "string", "description": "Limit to one wing (optional)"},
                "apply": {
                    "type": "boolean",
                    "description": "Actually delete drawers; default is dry-run preview",
                },
            },
        },
        "handler": tool_sync,
    },
    "mempalace_get_drawer": {
        "description": "Fetch a single drawer by ID — returns full content and metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to fetch"},
            },
            "required": ["drawer_id"],
        },
        "handler": tool_get_drawer,
    },
    "mempalace_list_drawers": {
        "description": "List drawers with pagination. Optional wing/room filter. Returns IDs, wings, rooms, content previews, and total matching count for pagination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Filter by wing (optional)"},
                "room": {"type": "string", "description": "Filter by room (optional)"},
                "limit": {
                    "type": "integer",
                    "description": "Max results per page (default 20, max 100)",
                    "minimum": 1,
                    "maximum": 100,
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset for pagination (default 0)",
                    "minimum": 0,
                },
            },
        },
        "handler": tool_list_drawers,
    },
    "mempalace_update_drawer": {
        "description": "Update an existing drawer's content and/or metadata (wing, room). Fetches existing drawer first; returns error if not found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drawer_id": {"type": "string", "description": "ID of the drawer to update"},
                "content": {
                    "type": "string",
                    "description": "New content (optional — omit to keep existing)",
                },
                "wing": {
                    "type": "string",
                    "description": "New wing (optional — omit to keep existing)",
                },
                "room": {
                    "type": "string",
                    "description": "New room (optional — omit to keep existing)",
                },
            },
            "required": ["drawer_id"],
        },
        "handler": tool_update_drawer,
    },
    "mempalace_diary_write": {
        "description": "Write to your personal agent diary in AAAK format. Your observations, thoughts, what you worked on, what matters. Each agent has their own diary with full history. Write in AAAK for compression — e.g. 'SESSION:2026-04-04|built.palace.graph+diary.tools|ALC.req:agent.diaries.in.aaak|★★★'. Use entity codes from the AAAK spec.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary wing",
                },
                "entry": {
                    "type": "string",
                    "description": "Your diary entry in AAAK format — compressed, entity-coded, emotion-marked",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic tag (optional, default: general)",
                },
                "wing": {
                    "type": "string",
                    "description": "Target wing for this diary entry (optional). If omitted, uses wing_{agent_name}. Use this to write diary entries to a project wing instead of an agent-specific wing.",
                },
            },
            "required": ["agent_name", "entry"],
        },
        "handler": tool_diary_write,
    },
    "mempalace_diary_read": {
        "description": "Read your recent diary entries (in AAAK). See what past versions of yourself recorded — your journal across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Your name — each agent gets their own diary wing",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent entries to read (default: 10)",
                },
                "wing": {
                    "type": "string",
                    "description": "Wing to read diary entries from (optional). If omitted, reads from wing_{agent_name}.",
                },
            },
            "required": ["agent_name"],
        },
        "handler": tool_diary_read,
    },
    "mempalace_hook_settings": {
        "description": (
            "Get or set hook behavior. silent_save: True = save directly "
            "(no MCP clutter), False = legacy blocking. desktop_toast: "
            "True = show desktop notification. Call with no args to view."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "silent_save": {
                    "type": "boolean",
                    "description": "True = silent direct save, False = blocking MCP calls",
                },
                "desktop_toast": {
                    "type": "boolean",
                    "description": "True = show desktop toast via notify-send",
                },
            },
        },
        "handler": tool_hook_settings,
    },
    "mempalace_memories_filed_away": {
        "description": "Check if a recent palace checkpoint was saved. Returns message count and timestamp.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_memories_filed_away,
    },
    "mempalace_reconnect": {
        "description": (
            "Force reconnect to the palace database. Use after external scripts or CLI commands"
            " modified the palace directly, which can leave the in-memory HNSW index stale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
        "handler": tool_reconnect,
    },
}


SUPPORTED_PROTOCOL_VERSIONS = [
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
]


def _internal_tool_error(req_id, tool_name: str, exc: BaseException = None) -> dict:
    logger.exception(f"Tool error in {tool_name}")
    error: dict = {"code": -32000, "message": "Internal tool error"}
    if exc is not None:
        error["data"] = {
            "error_class": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": error,
    }


def handle_request(request):
    global _last_request_time
    if not isinstance(request, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
    _last_request_time = time.monotonic()
    method = request.get("method") or ""
    params = request.get("params") or {}
    req_id = request.get("id")

    if method == "initialize":
        client_version = params.get("protocolVersion", SUPPORTED_PROTOCOL_VERSIONS[-1])
        negotiated = (
            client_version
            if client_version in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace", "version": __version__},
            },
        }
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method.startswith("notifications/"):
        # Notifications (no id) never get a response per JSON-RPC spec
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": n, "description": t["description"], "inputSchema": t["input_schema"]}
                    for n, t in TOOLS.items()
                ]
            },
        }
    elif method == "tools/call":
        if not isinstance(params, dict) or "name" not in params:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: 'name' is required for tools/call",
                },
            }
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        # Whitelist arguments to declared schema properties only.
        # Prevents callers from spoofing internal params like added_by/source_file.
        # Skip filtering if handler explicitly accepts **kwargs (pass-through).
        # Default to filtering on inspect failure (safe fallback).
        import inspect

        schema_props = TOOLS[tool_name]["input_schema"].get("properties", {})
        try:
            handler = TOOLS[tool_name]["handler"]
            sig = inspect.signature(handler)
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            accepts_var_keyword = False
        if not accepts_var_keyword:
            # An unknown kwarg here is almost always a wrong parameter *name*
            # (e.g. text= instead of content=). Silently dropping it makes the
            # cause surface only indirectly as a later "Missing required 'X'",
            # so name it explicitly — symmetric with the missing-required path
            # below. wait_for_previous is an internal transport kwarg in no
            # tool schema; it is popped before dispatch further down, so it
            # must not be reported as unknown here.
            unknown = [k for k in tool_args if k not in schema_props and k != "wait_for_previous"]
            if unknown:
                quoted = ", ".join(f"'{k}'" for k in unknown)
                word = "parameter" if len(unknown) == 1 else "parameters"
                logger.debug("Tool %s: unknown %s %s", tool_name, word, quoted)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": f"Unknown {word} {quoted} for tool {tool_name}",
                    },
                }
            tool_args = {k: v for k, v in tool_args.items() if k in schema_props}
        # Coerce argument types based on input_schema.
        # MCP JSON transport may deliver integers as floats or strings;
        # ChromaDB and Python slicing require native int.
        for key, value in list(tool_args.items()):
            prop_schema = schema_props.get(key, {})
            declared_type = prop_schema.get("type")
            try:
                if declared_type == "integer" and not isinstance(value, int):
                    tool_args[key] = int(value)
                elif declared_type == "number" and not isinstance(value, (int, float)):
                    tool_args[key] = float(value)
            except (ValueError, TypeError):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Invalid value for parameter '{key}'"},
                }
        tool_args.pop("wait_for_previous", None)
        try:
            result = TOOLS[tool_name]["handler"](**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}
                    ]
                },
            }
        except TypeError as e:
            # Qualname match prevents leaking internal helper/param names raised
            # inside the handler body — see test_handler_internal_signature_shape_stays_generic.
            msg = str(e)
            handler = TOOLS[tool_name]["handler"]
            handler_qn = getattr(handler, "__qualname__", None) or getattr(handler, "__name__", "")
            # Qualname can include "<locals>" for nested defs and "<lambda>"
            # for lambdas — accept Python's TypeError emit verbatim.
            m_missing = re.match(
                r"^([\w\.<>]+)\(\) missing \d+ required "
                r"(?:positional |keyword-only )?arguments?: (.+)$",
                msg,
            )
            if m_missing and m_missing.group(1) == handler_qn:
                names = re.findall(r"'(\w+)'", m_missing.group(2))
                if names:
                    quoted = ", ".join(f"'{n}'" for n in names)
                    word = "parameter" if len(names) == 1 else "parameters"
                    logger.debug("Tool %s: missing required %s %s", tool_name, word, quoted)
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": f"Missing required {word} {quoted} for tool {tool_name}",
                        },
                    }
            return _internal_tool_error(req_id, tool_name, e)
        except Exception as exc:
            return _internal_tool_error(req_id, tool_name, exc)

    # Notifications (missing id) must never get a response
    if req_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def _restore_stdout():
    """Restore real stdout for MCP JSON-RPC output (see issue #225)."""
    global _REAL_STDOUT, _REAL_STDOUT_FD
    if _REAL_STDOUT_FD is not None:
        try:
            os.dup2(_REAL_STDOUT_FD, 1)
            os.close(_REAL_STDOUT_FD)
        except OSError:
            pass
        _REAL_STDOUT_FD = None
    sys.stdout = _REAL_STDOUT


_WARMUP_TRUTHY = {"1", "true", "yes", "on"}
_WARMUP_FALSY = {"", "0", "false", "no", "off"}
# Sentinel text for the warmup query. Distinctive so it cannot semantically
# match real drawer content (e.g. a palace containing notes about "warmup"
# routines) and is greppable in chromadb debug logs if the team ever adds
# request instrumentation. Single non-empty string is enough to trigger
# ChromaDB's ONNXMiniLM_L6_V2.__call__ → _download_model_if_not_exists +
# InferenceSession.
_WARMUP_PROBE_TEXT = "__mempalace_warmup_probe__"


def _describe_device_safe() -> str:
    """Return ``embedding.describe_device()`` value or ``"unknown"`` on failure.

    Used only inside warmup-failure log lines; the import is deferred so
    that an embedding-stack import error cannot itself crash the warmup
    diagnostic path.
    """
    try:
        from .embedding import describe_device

        return describe_device()
    except Exception:  # fail-soft: see docstring — log-message helper must not crash
        return "unknown"


def _maybe_eager_warmup_embedder() -> None:
    """Pre-load embedder + HNSW segment at startup when ``MEMPALACE_EAGER_WARMUP`` is truthy.

    The first MCP tool call that touches chromadb (``diary_write``,
    ``add_drawer``, ``search``) otherwise pays two compounding cold-load
    costs that together can exceed the MCP client timeout and surface as
    ``-32000`` "Internal tool error" with no recoverable trace on the
    agent side (#1495):

    1. ONNX/CoreML embedder init in :func:`mempalace.embedding.get_embedding_function`
       (5–30s on first inference; ChromaDB's ``ONNXMiniLM_L6_V2.__call__``
       triggers ``_download_model_if_not_exists`` + ``InferenceSession``).
    2. HNSW segment cold-load (reading ``data_level0.bin`` into RAM on
       first collection operation; seconds on palaces of 50k+ drawers).

    Warming via :func:`_get_collection`'s collection-then-query path
    covers BOTH in a single startup-phase call — mirroring the reporter's
    proposal in #1495 — so users with large existing palaces see the
    same benefit as users on the embedder-only cost path.

    Truthy parsing accepts ``1/true/yes/on`` (case-insensitive); falsy
    set ``0/false/no/off`` and empty/whitespace are silently off; any
    other value logs a warning and stays off so typos like ``tru`` do
    not silently disable the feature.

    Fresh-install guard (pre-check, NOT a catch): ``_get_collection``'s
    retry layer absorbs ``_ChromaNotFoundError`` and returns ``None`` while
    also materialising ``chroma.sqlite3`` on disk via the chromadb client
    constructor. To preserve the documented "no palace yet → nothing to
    warm" contract WITHOUT writing palace scaffolding before
    ``mempalace init`` (which would violate CLAUDE.md "Incremental only"),
    we test for ``chroma.sqlite3`` ourselves before touching the chromadb
    client. Operators who set ``MEMPALACE_EAGER_WARMUP=1`` in their MCP
    config and launch the server before running ``mempalace init`` get a
    single INFO line and no on-disk side effect.

    Fail-soft beyond the fresh-install pre-check:

    * **Backend open failure** (palace path misconfigured, file locked,
      corrupted HNSW that ``quarantine_stale_hnsw`` cannot recover) →
      log exception with device + palace context and return. The next
      embedding-requiring call sees the same fail mode it would have
      without warmup.
    * **`_get_collection` retried and returned None** → palace exists
      but chromadb cannot open the collection (rare; usually a stale
      sqlite + segment-files mismatch surfaced by `_get_client` rebuild).
      A warning suffices because the retry layer already wrote two
      tracebacks with the underlying chromadb error class.
    * **Query failure** (network failure during ONNX model download,
      provider init crash, runtime decoder error) → log exception with
      device + palace context and return. Same fail-mode preservation.

    Note: on an existing palace with an empty collection (created via
    ``mempalace init`` but never written to), ``col.query`` succeeds but
    returns ``{'ids': [[]]}`` without reading any HNSW segment — the
    embedder warms but there is no HNSW segment to load. The success log
    still says ``embedder + HNSW ready`` because the no-HNSW-segment case
    has zero cold-load cost; nothing was skipped that the first real tool
    call would have paid.
    """
    raw = os.environ.get("MEMPALACE_EAGER_WARMUP", "").strip().lower()
    if raw in _WARMUP_FALSY:
        return
    if raw not in _WARMUP_TRUTHY:
        logger.warning(
            "MEMPALACE_EAGER_WARMUP=%r is not recognized (use one of %s); warmup disabled",
            raw,
            sorted(_WARMUP_TRUTHY | (_WARMUP_FALSY - {""})),
        )
        return
    palace_path = _config.palace_path
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        # Pre-check (NOT a try/except on _ChromaNotFoundError, which never
        # propagates out of _get_collection — see docstring). No palace
        # file means nothing to warm AND avoids the chromadb-client
        # side effect of materialising the palace dir.
        logger.info(
            "MEMPALACE_EAGER_WARMUP=%s: no palace at %s — nothing to warm",
            raw,
            palace_path,
        )
        return
    # Cache device once: _describe_device_safe re-imports embedding stack
    # each call, which is wasteful inside a function that already paid
    # that cost via the warmup query below.
    device = _describe_device_safe()
    try:
        col = _get_collection(create=False)
    except Exception as exc:  # fail-soft per docstring — broad on purpose
        logger.exception(
            "MEMPALACE_EAGER_WARMUP=%s: collection open failed (palace=%s, device=%s, error=%s)",
            raw,
            palace_path,
            device,
            type(exc).__name__,
        )
        return
    if col is None:
        logger.warning(
            "MEMPALACE_EAGER_WARMUP=%s: _get_collection returned None for palace=%s — see prior log lines",
            raw,
            palace_path,
        )
        return
    try:
        col.query(query_texts=[_WARMUP_PROBE_TEXT], n_results=1)
    except Exception as exc:  # fail-soft per docstring — broad on purpose
        logger.exception(
            "MEMPALACE_EAGER_WARMUP=%s: warmup query failed (palace=%s, device=%s, error=%s)",
            raw,
            palace_path,
            device,
            type(exc).__name__,
        )
    else:
        logger.info(
            "MEMPALACE_EAGER_WARMUP=%s: embedder + HNSW ready (palace=%s, device=%s)",
            raw,
            palace_path,
            device,
        )


def _start_idle_exit_watchdog() -> None:
    """Start a daemon thread that exits the process after an idle period.

    When no request has been handled for ``MEMPALACE_MCP_IDLE_HOURS``
    (default 8 h), the thread terminates the process so that stale MCP
    servers from ended Claude Code sessions do not accumulate ChromaDB /
    HNSW file handles on Windows (#1552).

    Set ``MEMPALACE_MCP_IDLE_HOURS=0`` to disable the watchdog.
    """
    timeout = _mcp_idle_timeout_secs()
    if timeout <= 0:
        return
    check_interval = min(60.0, timeout / 4)

    def _watchdog() -> None:
        while True:
            time.sleep(check_interval)
            idle = time.monotonic() - _last_request_time
            if idle >= timeout:
                logger.info(
                    "MCP server idle for %.1f h (limit %.1f h); exiting to release file handles.",
                    idle / 3600,
                    timeout / 3600,
                )
                os._exit(0)

    t = threading.Thread(target=_watchdog, name="mcp-idle-watchdog", daemon=True)
    t.start()


def main():
    """MCP server entry point for the ``mempalace-mcp`` console script.

    Side effect: pops ``PYTHONPATH`` from ``os.environ`` (see #1423) so
    any subprocess this server spawns inherits a clean env. Host
    applications that call ``main()`` programmatically should be aware
    that the parent process loses ``PYTHONPATH`` as well. Library imports
    (``import mempalace.searcher`` from a host app) do NOT trigger this
    side effect; only the CLI/MCP entry points pop the env var.
    """
    # Drop leaked PYTHONPATH so any subprocess this server spawns starts
    # with a clean env. The sys.path filter in mempalace/__init__.py
    # already protects this process from the same ABI mismatch; here we
    # extend the protection to children.
    os.environ.pop("PYTHONPATH", None)
    _restore_stdout()
    # Force UTF-8 on stdio. MCP JSON-RPC is UTF-8, but Python on Windows
    # defaults stdin/stdout to the system codepage (e.g. cp1251), which
    # corrupts non-ASCII payloads and surfaces as generic -32000 errors on
    # Cyrillic/CJK content. See PEP 540.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    logger.info("MemPalace MCP Server starting...")
    # Pre-flight: probe HNSW capacity before any tool call so the warning
    # is visible at startup rather than on first use (#1222). Pure
    # filesystem read; never opens a chromadb client.
    _refresh_vector_disabled_flag()
    # Opt-in: pre-load the embedder so the first chromadb-write tool call
    # does not pay the ONNX/CoreML cold-load tax under the MCP client
    # timeout (#1495). Default off — preserves current startup latency.
    _maybe_eager_warmup_embedder()
    # Idle auto-exit: release ChromaDB file handles from stale servers
    # that outlived their Claude Code session (#1552).
    _start_idle_exit_watchdog()
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
