"""Backend-aware link store — the single route for explicit tunnels + hallways.

A **link store** owns the agent-authored cross-wing *tunnels* and the
within-wing entity *hallways*. Historically these lived in host-global JSON
files (``tunnels.json`` / ``hallways.json``) maintained directly by
``palace_graph`` and ``hallways``. On the central, multi-team deployment that
host-global file is a cross-tenant leak: one file, one host, many teams.

This module introduces ONE seam — :class:`LinkStore` plus
:func:`get_link_store` — that every explicit-tunnel writer and every
hallway reader routes through, so the local (chroma) JSON store and the
per-team Postgres store cannot drift apart. The Postgres implementation
lands in a later story; today the resolver returns the JSON store for the
chroma backend and raises a clear, deliberate placeholder for postgres.

Contract guarantees the JSON store preserves verbatim (so consumers can
route through the seam with zero behavior change):

  * tunnels persist to the exact same ``MempalaceConfig``-derived
    ``tunnels.json`` path, with the same atomic ``os.replace`` write and
    0600 file / 0700 parent-dir permissions;
  * tunnel IDs are the same symmetric ``_canonical_tunnel_id`` (sorting the
    two endpoints before hashing, so ``create_tunnel(A, B)`` and
    ``create_tunnel(B, A)`` dedup to one record);
  * re-creating an existing tunnel preserves its accumulated dynamics
    (``strength`` / ``stability`` / ``last_activated`` / ``access_count``)
    via the shared ``merge_dynamics`` helper, updating only the label and
    ``updated_at``;
  * hallways persist to the same ``hallways.json`` path and preserve
    dynamics across recomputes the same way.

The return shapes are the existing dicts: tunnels carry ``id`` / ``source``
/ ``target`` / ``label`` / ``kind`` / ``created_at`` (+ ``updated_at`` and
the four dynamics fields); ``follow_tunnels`` returns connection dicts
(``direction`` / ``connected_wing`` / ``connected_room`` / ``label`` /
``drawer_id`` / ``tunnel_id`` [+ ``drawer_preview``]); hallways carry
``id`` / ``wing`` / ``entity_a`` / ``entity_b`` / ``co_occurrence_count`` /
``rooms`` / ``label`` / the dynamics fields.

Team handling — fail loud, never silent default:
  * The chroma JSON store is host-local single-vault, so it needs NO team:
    ``get_link_store`` accepts a team but the JSON store ignores it.
  * A team-scoped (server-mode) backend MUST resolve a team for every write.
    The seam resolves it with ``mcp_server._resolve_team_strict`` and RAISES
    when it returns ``None`` — it never falls back to
    ``_canonical_default_team``. Silently defaulting would re-create the
    shared-``team_default`` cross-tenant leak inside the team-scoped store.
    :func:`require_write_team` is that load-bearing check; it is established
    here in the contract so the RAISE path is reachable (not dead code) once
    the team-scoped store and its wiring land.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from . import hallways as _hallways
from . import palace_graph as _palace_graph


class LinkStore(ABC):
    """Abstract contract for explicit-tunnel CRUD + hallway reads.

    The method signatures match the existing ``palace_graph`` /
    ``hallways`` functions exactly so consumers can route through the seam
    unchanged. Return shapes are documented at module level.
    """

    @abstractmethod
    def create_tunnel(
        self,
        source_wing: str,
        source_room: str,
        target_wing: str,
        target_room: str,
        label: str = "",
        source_drawer_id: str = None,
        target_drawer_id: str = None,
        kind: str = "explicit",
    ) -> dict:
        """Create (or re-create-and-update) a symmetric explicit tunnel.

        Mirrors :func:`palace_graph.create_tunnel`. Returns the stored
        tunnel dict. A second call with the same (symmetric) endpoints
        updates the label and preserves the accumulated dynamics.
        """

    @abstractmethod
    def list_tunnels(self, wing: str = None) -> list[dict]:
        """List explicit tunnels, optionally filtered by *wing*.

        Mirrors :func:`palace_graph.list_tunnels`. Matches *wing* against
        either endpoint (tunnels are symmetric).
        """

    @abstractmethod
    def delete_tunnel(self, tunnel_id: str) -> dict:
        """Delete a tunnel by ID. Mirrors :func:`palace_graph.delete_tunnel`.

        Returns ``{"deleted": <tunnel_id>}``.
        """

    @abstractmethod
    def follow_tunnels(self, wing: str, room: str, col=None, config=None) -> list[dict]:
        """Follow explicit tunnels from a room. Mirrors
        :func:`palace_graph.follow_tunnels`. Returns connection dicts; when
        *col* is supplied, hydrates a ``drawer_preview`` for connected
        drawers.
        """

    @abstractmethod
    def compute_hallways_for_wing(self, wing: str, col=None, min_count: int = 2) -> list[dict]:
        """Compute + persist within-wing entity hallways for *wing*.

        This is the hallway read/derive path consumers need. Mirrors
        :func:`hallways.compute_hallways_for_wing`. Returns this wing's
        hallway dicts; records for other wings are preserved on disk.
        """

    @abstractmethod
    def list_hallways(self, wing: Optional[str] = None) -> list[dict]:
        """List hallway records, optionally filtered by *wing*.

        Mirrors :func:`hallways.list_hallways`.
        """


class JsonLinkStore(LinkStore):
    """Host-local JSON-backed link store (the chroma single-vault path).

    Every method DELEGATES to the existing ``palace_graph`` / ``hallways``
    module functions verbatim — it wraps, it does not reimplement. The exact
    file path, 0600/0700 permissions, atomic ``os.replace``, symmetric
    ``_canonical_tunnel_id``, and ``merge_dynamics``-based dynamics
    preservation are therefore preserved bit-for-bit by reuse.

    This store is single-vault and host-local, so it takes no team.
    """

    def create_tunnel(
        self,
        source_wing: str,
        source_room: str,
        target_wing: str,
        target_room: str,
        label: str = "",
        source_drawer_id: str = None,
        target_drawer_id: str = None,
        kind: str = "explicit",
    ) -> dict:
        return _palace_graph.create_tunnel(
            source_wing,
            source_room,
            target_wing,
            target_room,
            label=label,
            source_drawer_id=source_drawer_id,
            target_drawer_id=target_drawer_id,
            kind=kind,
        )

    def list_tunnels(self, wing: str = None) -> list[dict]:
        return _palace_graph.list_tunnels(wing)

    def delete_tunnel(self, tunnel_id: str) -> dict:
        return _palace_graph.delete_tunnel(tunnel_id)

    def follow_tunnels(self, wing: str, room: str, col=None, config=None) -> list[dict]:
        return _palace_graph.follow_tunnels(wing, room, col=col, config=config)

    def compute_hallways_for_wing(self, wing: str, col=None, min_count: int = 2) -> list[dict]:
        return _hallways.compute_hallways_for_wing(wing, col=col, min_count=min_count)

    def list_hallways(self, wing: Optional[str] = None) -> list[dict]:
        return _hallways.list_hallways(wing)


def require_write_team(team: Optional[str]) -> str:
    """Return *team* for a team-scoped write, or RAISE when it is ``None``.

    This is the fail-loud point of the seam. Callers on a team-scoped
    backend resolve the team with ``mcp_server._resolve_team_strict`` (which
    returns ``None`` on ambiguity, never a silent default) and pass the
    result here. A ``None`` team means a writer could not establish which
    vault to write to; routing it into a default vault would re-create the
    shared-default cross-tenant leak, so we RAISE instead.

    The chroma JSON store does NOT call this — it is host-local single-vault
    and needs no team.
    """
    if team is None:
        raise ValueError(
            "no team resolved for a team-scoped link-store write; refusing to "
            "fall back to a default vault (would leak across teams). Set an "
            "explicit team or an active session team."
        )
    return team


def get_link_store(config, team: Optional[str] = None) -> LinkStore:
    """Return the link store for the configured backend.

    * ``chroma`` (default, host-local single-vault) → :class:`JsonLinkStore`.
      The store is single-vault, so *team* is accepted for a uniform call
      signature but ignored — chroma does not require a team.
    * any team-scoped (server-mode) backend → a per-team Postgres
      :class:`~mempalace.link_store_postgres.PostgresLinkStore`. On Postgres a
      team is MANDATORY for every operation — reads AND writes are scoped to
      one team's schema, so you cannot list/follow another team's tunnels — so
      this path runs *team* through :func:`require_write_team` and RAISES when
      it is ``None`` rather than silently routing into a default vault.

    Args:
        config: a ``MempalaceConfig`` (read for ``.backend``).
        team: the resolved team. Ignored by the chroma JSON store; MANDATORY
            for team-scoped backends (resolve it with the strict resolver and
            pass the result; ``None`` fails loud via :func:`require_write_team`).
    """
    backend = getattr(config, "backend", "chroma")
    if backend == "chroma":
        return JsonLinkStore()
    # Team-scoped (server-mode) backend: team is mandatory for reads AND writes
    # (isolation is structural — one team's schema). Fail loud on None.
    from .link_store_postgres import PostgresLinkStore
    from .palace import _resolve_backend

    return PostgresLinkStore(_resolve_backend(config), team=require_write_team(team))
