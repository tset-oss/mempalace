# agent-commons integration (reference)

Reference copies of the **consumer-side** MemPalace integration as it can be
wired into a Claude Code plugin. They are kept here so a reader of this repo can
see how a real client drives the central, team-vaulted MemPalace server over
MCP — the recall/save habits, the team-routing protocol, and the two hooks that
nudge an agent to recall and to file.

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

## Adapting these for your environment

These files are a **generalized reference, not a turnkey drop-in.** Adapt them to
your own setup before wiring them into a plugin:

- **Skill namespace.** The skill is namespaced `/mempalace`; rename it to your
  plugin's namespace if needed.
- **Authorship convention.** The skill and the Stop hook use placeholder author
  values — `you@example.com` for the operator email and `your-handle` for the
  operator handle. Substitute your team's convention for crediting who filed a
  memory (the harness-surfaced operator email plus the model id).
- **Tool references.** Where the skill says "your LSP / code-navigation tool" or
  "your issue tracker / docs system", plug in whatever your team actually uses.
- **No secrets.** These files contain no tokens, passwords, connection strings,
  or hostnames. Wire those through your plugin config / environment per
  [`../../deploy/README.md`](../../deploy/README.md).
