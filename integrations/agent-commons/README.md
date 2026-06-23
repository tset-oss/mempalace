# agent-commons integration (reference)

Reference copies of the **consumer-side** MemPalace integration as it is wired
into an internal Claude Code plugin (`agent-commons`). They are kept here so a
reader of this repo can see how a real client drives the central, team-vaulted
MemPalace server over MCP — the recall/save habits, the team-routing protocol,
and the two hooks that nudge an agent to recall and to file.

These files are **documentation, not runtime code for this package**. Nothing
here is imported by `mempalace/`, run by its CLI, or exercised by its tests. The
server in this repo works without them; they only illustrate the client end.

## Contents

| Path | Event | What it does |
|---|---|---|
| `hooks/mempalace-recall-hook.py` | `SessionStart` | Injects a "recall first" reminder so the agent searches memory before re-deriving context. Pure stdlib; no `mempalace` dependency (hooks cannot call MCP). |
| `hooks/mempalace-nudge-hook.py` | `Stop` | Blocks the Stop event on a cadence (default every 15 user turns, `MEMPALACE_NUDGE_INTERVAL`) and feeds the model a "checkpoint — file durable facts" reason. Pure stdlib. Kill switch: `MEMPALACE_NUDGE_DISABLE`. |
| `skills/mempalace/SKILL.md` | — | The in-session usage protocol: recall-before-reasoning, the SHAPE->TOOL filing lanes (kg_add / entity_seed / team_fact_add / add_drawer / diary_write), the update-don't-duplicate drawer lifecycle, and the team-vault routing precedence. |
| `skills/mempalace/README.md` | — | Overview of the skill above. |

## How the hooks are wired

The host plugin registers both hooks in its `hooks.json` (paths shown relative to
the plugin root):

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [ { "type": "command", "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/mempalace-recall-hook.py\"", "timeout": 10 } ] }
    ],
    "Stop": [
      { "hooks": [ { "type": "command", "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/mempalace-nudge-hook.py\"", "timeout": 10 } ] }
    ]
  }
}
```

The server's *own* hooks (the ones this package ships for local/Chroma palaces)
are separate and live in the top-level [`hooks/`](../../hooks/) directory.

## Provenance & caveats (read before publishing)

- **Source:** copied verbatim from the internal `agent-commons` repo on
  2026-06-23. They are not the canonical copy — the canonical copy lives there.
- **Internal-flavored content.** The skill is namespaced `/tset:mempalace` and
  refers to "tset's shared memory", the `tset` plugin, and internal tooling
  (Jira, Slite, Serena). Generalize this naming if these files are meant to be
  followed by external users rather than just read as an example.
- **Example email.** `markus.burger@tset.com` appears in `SKILL.md` and
  `mempalace-nudge-hook.py` as the illustrative `added_by` / `asserted_by`
  format (`<operator-email> (<model-id>)`). It is an example, not a credential —
  swap it for a placeholder (e.g. `you@example.com`) before a public release if
  you prefer not to ship a real address.
- **Dangling links.** `SKILL.md`/`README.md` link to `docs/mcp/mempalace.md` and
  `docs/mcp/README.md`, which live in `agent-commons`, not here. Those links do
  not resolve in this repo by design — these are reference excerpts, not the full
  client doc set.
- **No secrets.** These four files contain no tokens, passwords, connection
  strings, or internal hostnames. (The bearer token and `*.internal` host
  references that exist elsewhere in `agent-commons` deploy docs were
  deliberately **not** copied.)
