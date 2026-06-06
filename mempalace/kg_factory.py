"""Backend-aware knowledge-graph factory.

Returns the SQLite :class:`KnowledgeGraph` for the local (chroma) backend and the
:class:`PostgresKnowledgeGraph` (team vault schema) for server-mode backends, so
every KG consumer reads/writes the same graph the MCP server does. Server-mode
KGs are cached per team to avoid repeating the lazy schema bootstrap on each
call (they share the storage backend's connection pool, so caching is cheap).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_pg_cache: dict = {}
_pg_lock = threading.Lock()


def get_knowledge_graph(palace_path: Optional[str] = None, team: Optional[str] = None, config=None):
    """Return the KG implementation for the active backend.

    * chroma  -> SQLite KnowledgeGraph at ``<palace_path>/knowledge_graph.sqlite3``
      (or the default path when ``palace_path`` is None).
    * postgres -> PostgresKnowledgeGraph for ``team`` / ``config.team`` / ``default``.
    """
    from .config import MempalaceConfig

    config = config or MempalaceConfig()
    if config.backend == "chroma":
        from .knowledge_graph import DEFAULT_KG_PATH, KnowledgeGraph

        db_path = (
            os.path.join(palace_path, "knowledge_graph.sqlite3") if palace_path else DEFAULT_KG_PATH
        )
        return KnowledgeGraph(db_path=db_path)

    resolved_team = team or config.team or "default"
    kg = _pg_cache.get(resolved_team)
    if kg is not None:
        return kg
    with _pg_lock:
        kg = _pg_cache.get(resolved_team)
        if kg is None:
            from .knowledge_graph_postgres import PostgresKnowledgeGraph
            from .palace import _resolve_backend

            kg = PostgresKnowledgeGraph(_resolve_backend(config), team=resolved_team)
            _pg_cache[resolved_team] = kg
    return kg
