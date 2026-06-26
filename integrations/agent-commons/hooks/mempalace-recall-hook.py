#!/usr/bin/env python3
"""MemPalace recall-prime hook (SessionStart) for the central memory server.

Read-side counterpart to the Stop-event save nudge. It cannot query the server
itself (no MCP from hooks, no mempalace dependency), so its whole job is to
inject a 'recall first' reminder into the session context — the agent does the
actual mempalace_search/recall. Python 3 stdlib only.
"""

import json
import os
import sys

# Recall only matters at the start of real work. Skip "compact" (the agent just
# received a summary) so we don't stack a recall prompt on top of it.
PRIME_SOURCES = {"startup", "resume", "clear"}

RECALL_NUDGE = (
    "MemPalace session start — before re-deriving anything, recall what the "
    "team already knows. First confirm the vault: run mempalace_list_vaults "
    "and, if you are not pointed at the team that owns this work, "
    "mempalace_switch_team. Then, for the "
    "task at hand, pull the durable context that already exists — "
    "mempalace_search / mempalace_entities for the people, services, and "
    "topics in play, and mempalace_get_drawer for any decision or runbook a "
    "search surfaces — so you build on prior decisions, owners, and "
    "constraints instead of guessing at them. Recall is read-only and cheap: "
    "pull only what is relevant to this task, not the whole vault. If a search "
    "returns nothing, just proceed — but don't skip the look."
)


def _emit(data: dict) -> None:
    sys.stdout.write(json.dumps(data))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    # Shared kill switch with the save nudge; any non-empty value disables both.
    if os.environ.get("MEMPALACE_NUDGE_DISABLE", "").strip():
        _emit({})
        return

    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError, ValueError):
        _emit({})
        return
    if not isinstance(data, dict):
        _emit({})
        return

    if str(data.get("source", "startup")) not in PRIME_SOURCES:
        _emit({})
        return

    _emit(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": RECALL_NUDGE,
            }
        }
    )


if __name__ == "__main__":
    main()
