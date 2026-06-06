"""Per-team Postgres entity registry (central, team-vaulted).

The team-scoped counterpart to :class:`mempalace.entity_registry.EntityRegistry`.
It holds the same disambiguation data model — the registry of known people
(with their DOB / ID / relationship / context fields), projects, ambiguous-name
flags, and the Wikipedia cache — inside a team's vault schema
(``team_<slug>.entity_registry``) instead of the host-global
``~/.mempalace/entity_registry.json`` that the chroma path uses.

On the central multi-team deployment that single host-global file is a
cross-tenant leak: one file, one host, many teams sharing one "who is Riley"
table. A per-team table makes the disambiguation knowledge isolated and shared
WITHIN a team and invisible across teams — a registry for team A physically
cannot read or write team B's rows because it queries a different schema.

Disambiguation parity is achieved by REUSE, not reimplementation. This class
SUBCLASSES ``EntityRegistry`` and inherits every read/disambiguation method
(``lookup``, ``_disambiguate``, ``seed``, ``research``, ``confirm_research``,
``learn_from_text``, ``extract_people_from_query``, ``summary``, the
``people`` / ``projects`` / ``ambiguous_flags`` / ``mode`` properties) VERBATIM.
Only the persistence is overridden: the in-memory ``self._data`` document is the
exact same shape the JSON store serialises, so resolving an ambiguous name by
context / relationship returns the SAME result on both backends. The single
``self._data`` document is round-tripped to one JSON row per team.

It mirrors :class:`mempalace.entity_index_postgres.PostgresEntityIndex`: the
schema comes from ``team_schema(team)``, identifiers are quoted with ``_qi``, the
DDL runs once behind ``self._ensured`` + a lock, and connections come from
``PostgresBackend._conn()``.

Team is mandatory here. Unlike the host-local chroma JSON store (single-vault,
no team), every operation is scoped to one team's schema. The factory in
:mod:`mempalace.entity_registry` calls ``require_write_team`` before constructing
this store so a missing team fails loud rather than silently routing into a
default vault (which would re-create the shared-default cross-tenant leak).
"""

from __future__ import annotations

import json
import threading

from .backends.postgres import _qi, team_schema
from .entity_registry import EntityRegistry

# The whole registry is one JSON document per team. A fixed single-row key keeps
# the table trivial (no per-name schema migration) while preserving the exact
# data model the JSON store uses, so the inherited disambiguation logic operates
# on identical bytes on both backends.
_DOC_ID = "registry"


class PostgresEntityRegistry(EntityRegistry):
    """Entity registry stored in a team vault's ``entity_registry`` table.

    Inherits the full public method surface and disambiguation semantics of
    :class:`EntityRegistry`; only ``load`` / ``save`` are backend-specific.
    """

    def __init__(self, data: dict, backend, team: str):
        # _path is unused on Postgres (kept for the inherited surface; onboarding
        # prints registry._path). It records the logical location for diagnostics.
        super().__init__(data, f"{team_schema(team)}.entity_registry")
        self._backend = backend
        self._team = team
        self._schema = team_schema(team)
        self._ddl_lock = threading.Lock()
        self._ensured = False

    # -- schema -----------------------------------------------------------
    def _table(self) -> str:
        return f"{_qi(self._schema)}.{_qi('entity_registry')}"

    def _ensure(self) -> None:
        """Create schema + ``entity_registry`` table once.

        Guarded behind ``self._ensured`` + a lock (the PostgresEntityIndex
        pattern). The table is a single-document store: one ``doc`` JSON column
        keyed by a fixed id, holding the same payload the JSON store serialises.
        """
        if self._ensured:
            return
        with self._ddl_lock:
            if self._ensured:
                return
            with self._backend._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_qi(self._schema)}")
                    cur.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._table()} ("
                        "  id text PRIMARY KEY,"
                        "  doc jsonb NOT NULL"
                        ")"
                    )
            self._ensured = True

    # -- load / save ------------------------------------------------------
    @classmethod
    def open(cls, backend, team: str) -> "PostgresEntityRegistry":
        """Construct the store and load this team's registry document.

        Returns a registry seeded with the team's persisted document, or an
        empty registry (``EntityRegistry._empty()``) when the team has none yet —
        matching ``EntityRegistry.load`` returning an empty registry for a
        missing file.
        """
        store = cls(cls._empty(), backend, team)
        store._ensure()
        with store._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT doc FROM {store._table()} WHERE id = %s", (_DOC_ID,))
                row = cur.fetchone()
        if row and row[0]:
            doc = row[0]
            # psycopg returns jsonb as a parsed object; tolerate a text driver.
            store._data = doc if isinstance(doc, dict) else json.loads(doc)
        return store

    def save(self):
        """Persist the in-memory document to the team's single registry row.

        Overrides the JSON store's atomic-file write. The whole ``self._data``
        document is upserted as one JSON row, so the persisted shape is identical
        to the JSON path and the inherited disambiguation logic reads back the
        same bytes.
        """
        self._ensure()
        payload = json.dumps(self._data)
        with self._backend._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._table()} (id, doc) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (id) DO UPDATE SET doc = EXCLUDED.doc",
                    (_DOC_ID, payload),
                )
