"""Per-team critical-facts store (central, team-vaulted Postgres).

A small team-shared surface for the handful of facts EVERY agent on a team
should see before doing anything — e.g. "the prod DB is read-replica only" or
"release freeze until Q3". It lives in a team's vault schema
(``team_<slug>.critical_facts``) so the facts are shared WITHIN a team and
invisible across teams: a store for team A physically cannot read or write team
B's rows because it queries a different schema.

Relationship to the personal L0 identity layer (``layers.py``): the personal
``~/.mempalace/identity.txt`` (``Layer0``) is HOST-LOCAL and PER-DEVELOPER — "who
am I, who do I serve" for one machine's agent. It is deliberately NOT vaulted and
NOT touched by this store. Team critical-facts are the DISTINCT, complementary
layer: the shared, vaulted facts the whole team must know. One is personal and
local; the other is team-shared and central. They never overlap.

It mirrors :class:`mempalace.entity_index_postgres.PostgresEntityIndex`: the
schema comes from ``team_schema(team)``, identifiers are quoted with ``_qi``, the
DDL runs once behind ``self._ensured`` + a lock, and connections come from
``PostgresBackend._conn()``.

Team is mandatory here, for reads AND writes. Unlike the host-local personal L0
layer (single machine, no team), every operation on this store is scoped to one
team's schema. The accessor in :mod:`mempalace.mcp_server` resolves the team
strictly and fails loud rather than silently routing into a default vault (which
would re-create the shared-default cross-tenant leak).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from .backends.postgres import _qi, team_schema

# Conservative defaults (open tunables — see .omc/plans/open-questions.md). A
# fact is a short, human-readable line, not a document; capping the length keeps
# the surface a scannable "must-know" list rather than a second drawer store. The
# per-team cap keeps the whole surface cheap to read in one wake-up call.
MAX_FACT_CHARS = 2000
MAX_FACTS_PER_TEAM = 200


class PostgresTeamFacts:
    """Team critical-facts stored in a team vault's ``critical_facts`` table."""

    def __init__(self, backend, team: str):
        self._backend = backend
        self._team = team
        self._schema = team_schema(team)
        self._lock = threading.Lock()
        self._ensured = False

    # -- schema -----------------------------------------------------------
    def _table(self) -> str:
        return f"{_qi(self._schema)}.{_qi('critical_facts')}"

    def _ensure(self) -> None:
        """Create schema + ``critical_facts`` table once.

        Guarded behind ``self._ensured`` + a lock (the PostgresEntityIndex
        pattern). The table is intentionally small: an id, the verbatim fact
        text, when it was added, and an optional author label.
        """
        if self._ensured:
            return
        with self._lock:
            if self._ensured:
                return
            with self._backend._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_qi(self._schema)}")
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._table()} ("
                        "  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
                        "  fact text NOT NULL,"
                        "  created_at timestamptz NOT NULL,"
                        "  created_by text"
                        ")"
                    )
            self._ensured = True

    # -- writes -----------------------------------------------------------
    def add_fact(self, fact: str, created_by: Optional[str] = None) -> dict:
        """Add one team critical fact. Returns the stored row.

        Enforces the two conservative caps: a fact longer than
        :data:`MAX_FACT_CHARS` is rejected, and once a team already holds
        :data:`MAX_FACTS_PER_TEAM` facts no more are accepted (so the surface
        stays a short must-know list, not an unbounded log). Both raise
        ``ValueError`` so the caller surfaces a structured error.
        """
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("fact is required")
        fact = fact.strip()
        if len(fact) > MAX_FACT_CHARS:
            raise ValueError(f"fact is too long ({len(fact)} chars); the cap is {MAX_FACT_CHARS}")
        self._ensure()
        now = datetime.now(timezone.utc)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {self._table()}")
                count = cur.fetchone()[0]
                if count >= MAX_FACTS_PER_TEAM:
                    raise ValueError(
                        f"team already has {count} critical facts; the cap is "
                        f"{MAX_FACTS_PER_TEAM}. Remove a fact before adding another."
                    )
                cur.execute(
                    f"INSERT INTO {self._table()} (fact, created_at, created_by) "
                    "VALUES (%s, %s, %s) RETURNING id, fact, created_at, created_by",
                    (fact, now, created_by or None),
                )
                row = cur.fetchone()
        return _row_to_fact(row)

    # -- reads ------------------------------------------------------------
    def list_facts(self) -> list[dict]:
        """Return this team's critical facts, oldest first."""
        self._ensure()
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, fact, created_at, created_by FROM {self._table()} "
                    "ORDER BY created_at, id"
                )
                return [_row_to_fact(r) for r in cur.fetchall()]


def _row_to_fact(row: tuple) -> dict:
    """Map a ``critical_facts`` row to a stable JSON-serialisable dict.

    Normalises the ``timestamptz`` column to an ISO string so the return shape
    is JSON-clean (``json.dumps`` has no ``default=str`` in mcp_server).
    """
    id_, fact, created_at, created_by = row
    created = created_at.isoformat() if hasattr(created_at, "isoformat") else created_at
    return {"id": id_, "fact": fact, "created_at": created, "created_by": created_by}
