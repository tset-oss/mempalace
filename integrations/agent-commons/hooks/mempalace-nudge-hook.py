#!/usr/bin/env python3
"""MemPalace auto-save nudge hook (Stop event) for the central memory server.

This is the *remote* counterpart to MemPalace's upstream local hooks. It is
deliberately minimal and has NO dependency on the ``mempalace`` package: it
does not read/embed transcripts into a local store, spawn miners, or write
anything itself. The only actor that can persist to the central, team-vaulted
MemPalace is the agent, via its MCP tools (``mempalace_add_drawer`` /
``mempalace_diary_write``). So this hook's entire job is to *nudge* the agent
to do that on a cadence — by blocking the Stop event and feeding a reason back
to the model, which is the one Claude Code mechanism that hands control back to
the model so it can act.

Why Stop-only (no PreCompact): a PreCompact hook cannot block for a model turn
or force a tool call before compaction proceeds (its output is advisory only).
The periodic Stop nudge is what keeps team memory current, so by the time
compaction runs the durable facts are already filed; the raw transcript also
survives compaction on disk. Adding a PreCompact hook that cannot save would be
theatre, so we don't.

Protocol: reads the Claude Code hook JSON from stdin, writes a hook-response
JSON to stdout. Requires only Python 3 stdlib (python3 is a system tool, not a
mempalace install).
"""

import json
import os
import re
import sys
from pathlib import Path

# Default user turns between nudges; override per-environment via
# MEMPALACE_NUDGE_INTERVAL (read in _interval()).
DEFAULT_INTERVAL = 15

# The instruction the model reads when we block the Stop. It is intentionally
# conservative: file only durable knowledge, never scratch state or secrets,
# and otherwise just stop. Over-eager filing makes recall worse, not better.
NUDGE_REASON = (
    "MemPalace checkpoint — a moment to capture what this session learned so "
    "the team inherits it. RECALL FIRST: search for an existing drawer on the "
    "topic and UPDATE it (mempalace_update_drawer) instead of filing a "
    "near-duplicate — let one evolving fact converge to a single drawer, and "
    "delete a superseded one. Only add a new drawer when nothing covers it. "
    "File ONLY facts the whole team needs — decisions, owners, constraints, "
    "runbooks. Your own preferences, workflow corrections, and one-off "
    "debugging notes are personal, not team facts; leave those to your "
    "personal/native memory. Confirm you are pointed at the right team vault "
    "before filing (mempalace_list_vaults, then mempalace_switch_team if "
    "needed): every team-scoped write (add_drawer, diary_write, kg_add, "
    "team_fact_add, entity_seed) errors without a resolved team, and "
    "switch_team reports exists:false when the name matches no vault. Match each "
    "durable fact to the right tool and shape per the mempalace skill (the "
    "source of truth for shape→tool routing and the drawer lifecycle). File as "
    "the human operator, not the model: pass added_by as '<operator-email> "
    "(<model-id>)' on drawers (e.g. 'you@example.com "
    "(claude-sonnet-4-6)'), and checkpoint with mempalace_diary_write "
    "(agent_name=<operator handle>, e.g. 'your-handle', topic=<tag>) — an "
    "operator handle keeps "
    "diaries in one stable wing instead of a throwaway wing_<model> per model "
    "version. File "
    "self-contained, durable facts — not raw transcript, and not rot-prone "
    "detail the repo already records (file:line and code mechanics live in "
    "git). Never save throwaway/scratch context, secrets, credentials, tokens, "
    "or PII. If nothing durable happened since the last checkpoint, just stop "
    "without saving."
)


def _interval() -> int:
    raw = os.environ.get("MEMPALACE_NUDGE_INTERVAL", "")
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
    return DEFAULT_INTERVAL


def _state_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "") or str(Path.home() / ".cache")
    return Path(base) / "mempalace-nudge"


def _sanitize_session_id(session_id: str) -> str:
    """Restrict to a filename-safe slug so the marker path can't be steered."""
    return re.sub(r"[^A-Za-z0-9_-]", "", session_id) or "unknown"


def _safe_transcript(transcript_path: str):
    """Resolve a transcript path, rejecting traversal and non-JSONL inputs."""
    if not transcript_path:
        return None
    if ".." in Path(transcript_path).parts:
        return None
    path = Path(transcript_path).expanduser().resolve()
    if path.suffix not in (".jsonl", ".json"):
        return None
    return path


def _count_human_messages(transcript_path: str) -> int:
    """Count user-role turns in the transcript, skipping slash-command turns.

    This mirrors the upstream cadence counter: it counts every user-role entry
    (tool-result turns included) as a unit of progress. That is approximate by
    design — it only drives *when* to nudge, not what gets saved — so we keep
    it simple rather than trying to distinguish tool results from typed input.
    """
    path = _safe_transcript(transcript_path)
    if path is None or not path.is_file():
        return 0
    count = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                msg = entry.get("message")
                if not (isinstance(msg, dict) and msg.get("role") == "user"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                if not isinstance(content, str):
                    content = ""
                # Slash-command envelopes are not conversational progress.
                if "<command-message>" in content or "<command-name>" in content:
                    continue
                count += 1
    except OSError:
        return 0
    return count


def _emit(data: dict) -> None:
    sys.stdout.write(json.dumps(data))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    # Kill switch: any non-empty value disables the nudge entirely.
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

    # Secondary loop guard: if Claude is already continuing because of a prior
    # stop-hook block, never block again. The save marker below is the primary
    # guard; this just honours the field when the runtime supplies it.
    if str(data.get("stop_hook_active", "")).lower() in ("true", "1", "yes"):
        _emit({})
        return

    session_id = _sanitize_session_id(str(data.get("session_id", "unknown")))
    transcript_path = str(data.get("transcript_path", ""))

    count = _count_human_messages(transcript_path)
    if count <= 0:
        _emit({})
        return

    state_dir = _state_dir()
    marker = state_dir / f"{session_id}.last_nudge"
    last = 0
    try:
        last = int(marker.read_text().strip())
    except (OSError, ValueError):
        last = 0

    if count - last < _interval():
        _emit({})
        return

    # Advance the marker BEFORE blocking. The save turn the agent runs in
    # response appends only tool-result entries (which count as user-role
    # turns) — at most a few, well under the interval — so the next Stop sees
    # count - last < interval and does not re-fire. stop_hook_active above is
    # the backstop if a save turn ever appends >= interval entries.
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(count), encoding="utf-8")
    except OSError:
        # If we cannot persist the marker we would re-nudge next Stop; better
        # to skip this one than risk a loop with no way to record progress.
        _emit({})
        return

    _emit({"decision": "block", "reason": NUDGE_REASON})


if __name__ == "__main__":
    main()
