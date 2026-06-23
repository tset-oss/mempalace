# /tset:mempalace

The agent-facing usage protocol for the central, team-vaulted **MemPalace**
memory server. It teaches Claude Code when to recall durable team knowledge,
when to file it, and how to route reads and writes to the right team vault.

## What is this skill?

MemPalace is tset's shared long-term agent memory: one central Postgres-backed
server with a separate **vault** per team, reached over MCP. The `tset` plugin
ships the connection (plus a session-start recall nudge and a Stop-hook
auto-save nudge), so every engineer who
installs the plugin is connected with **zero local install** — there is no
`mempalace` package, ChromaDB, or local miner on your machine.

This `SKILL.md` is the short protocol the agent follows in-session. The
exhaustive human reference — deployment, the full tool list, troubleshooting —
lives in [`docs/mcp/mempalace.md`](../../../docs/mcp/mempalace.md).

## Why this exists

Without guidance the agent has the `mempalace_*` MCP tools available but no
sense of *when* to use them. The result is either silence (memory never
consulted) or noise (every scrap filed, which degrades recall). This skill
encodes the two habits that make team memory pay off:

- **Recall before reasoning** — search memory before re-deriving context.
- **File durable facts only** — decisions, owners, constraints, runbook steps;
  never scratch state or secrets.
- **File with transparent authorship** — credit the human operator (plus the
  model that wrote it) on every entry, and key session diaries to the operator
  so memory stays attributable and wings stay stable across model versions.

## How to use it

It triggers on memory-related intents ("what do we know about…", "remember
this", "switch team") or explicitly via `/tset:mempalace`. There is nothing to
run — it points the agent at the right `mempalace_*` tool for the moment:

- `mempalace_search`, `mempalace_kg_query` for recall.
- `mempalace_add_drawer`, `mempalace_diary_write`, `mempalace_kg_add` for
  filing.
- `mempalace_list_vaults`, `mempalace_switch_team`, and the `vault=` argument
  for team routing.

In an autonomous (auto-permission) run, Claude Code's auto-mode classifier may
auto-deny *unrequested* writes to the shared vault, so a Stop-nudge-driven save
can be refused. If proactive filing is blocked, allow the `mempalace_*` write
tools in `~/.claude/settings.json` (`permissions.allow`) or approve the call
interactively — see the troubleshooting table in
[`docs/mcp/mempalace.md`](../../../docs/mcp/mempalace.md).

## When to use / NOT to use

Use it when you want the agent to remember or recall durable context across
sessions and teammates. Do **not** use it for throwaway state, secrets, live
code-structure questions (that is Serena/LSP), or as a system of record (link
to Jira/Slite instead).

The in-session protocol — recall-first, file-durable-only, the routing
precedence, and the common mistakes — lives in
[`SKILL.md`](SKILL.md); this README is the overview, not a second copy.

## See also

- [`docs/mcp/mempalace.md`](../../../docs/mcp/mempalace.md) — full client
  reference and the per-repo / per-machine default-team override recipes.
- [`docs/mcp/README.md`](../../../docs/mcp/README.md) — MCP in Claude Code and
  the other servers tset standardises on.
