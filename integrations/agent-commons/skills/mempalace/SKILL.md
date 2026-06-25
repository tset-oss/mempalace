---
name: mempalace
description: Use the central team-vaulted MemPalace memory server to recall durable team knowledge before re-deriving it, file durable facts so future sessions and teammates inherit them, and route reads and writes to the right team vault.
triggers:
  - mempalace
  - memory
  - remember this
  - recall
  - what do we know about
  - team memory
  - switch team
  - vault
---

# /mempalace — team-vaulted agent memory

<Purpose>

MemPalace is the team's shared long-term memory: a central server (one Postgres
instance, team-isolated vaults) that every engineer's Claude Code reaches over
MCP. The MemPalace plugin ships the connection, so if the plugin is installed you
are already connected — there is nothing to install locally.

This skill is the usage protocol. For the full reference (deployment, every
tool, troubleshooting) see [`deploy/README.md`](../../../../deploy/README.md).

</Purpose>

<Use_When>

- Before re-deriving context you might already know — search memory first.
- When something durable was established that should outlive the session.
- When you need another team's context (a cross-team read).
- At session start, to confirm which team vault you are writing to.

</Use_When>

<Do_Not_Use_When>

- Throwaway/scratch state that only matters this task — do not file it.
- Secrets, credentials, tokens, or PII — never file them; vaults are team-shared.
- Live code-structure questions ("who calls X") — that is your LSP / code-navigation tool, not remembered knowledge.
- As a system of record — link to your issue tracker / docs system for authoritative docs instead of copying them wholesale.
- Rot-prone implementation detail (file:line refs, code mechanics) the repo already records — git/commit/PR/plan is the source of truth. Distill the durable decision or runbook and reference the code; do not re-narrate it (line numbers go stale fast).

</Do_Not_Use_When>

<Scripts_And_References>

- [`deploy/README.md`](../../../../deploy/README.md) — full deployment + client reference: connection, team routing, troubleshooting, and the per-repo / per-machine default-team override recipes.
- [`README.md`](../../../../README.md#central-team-vaults-postgresql) — how the central team-vaulted server is set up and consumed over MCP.

</Scripts_And_References>

<Workflow>

## Recall before reasoning

When a task leans on prior decisions, people, or history, search first:

- `mempalace_search` ("payment retry policy", "who owns ingest") — verbatim
  entries. Add `entity=<name>` to scope to that entity's drawers.
- `mempalace_entities(entity=<name>)` — drawers mentioning it.
  `mempalace_entities()` lists all vault entities when unsure of the exact name.
- `mempalace_kg_query` / `mempalace_kg_timeline` — who/what/when relationships,
  optionally as-of a date.
- `mempalace_kg_neighbors(entity=<name>)` — multi-hop graph walk from an entity;
  surfaces indirect connections (teammates, their projects, owning systems) that
  a single-hop query misses.
- `mempalace_disambiguate(name, context="")` (postgres/central only;
  `available:false` on local) — resolve an ambiguous surface-form name against
  the team's seeded registry before re-deriving who or what it refers to.

## Which memory system? (CLAUDE.md vs native memory vs MemPalace)

Three durable stores coexist; route each fact to exactly one home. **MemPalace
is for TEAM-shared knowledge only** — do not file personal notes or static rules
here.

- **CLAUDE.md / AGENTS.md** — stable *rules* and architectural invariants that
  must be in context on turn 1 (conventions, model routing, "verbatim always").
  No recall step, no expiry, no per-team scope.
- **Claude Code native auto-memory** (`~/.claude/projects/<slug>/memory/`) — your
  PERSONAL, per-machine notebook: your preferences, your workflow corrections,
  your reference links, your in-progress notes. Never team-shared.
- **MemPalace** (this server) — durable, TEAM-relevant facts your teammates and
  future sessions must inherit.

**Authority rule for facts phrased as rules.** The test is not
imperative-vs-declarative wording — it is *"does this become false on a date or a
state change?"* A fact that expires (e.g. "release freeze until Q3") is a MEMORY
even though it reads like a rule: it belongs in MemPalace `team_fact_add`
(Lane 3), which can be updated or removed when the condition lifts — never as a
verbatim line in CLAUDE.md. If turn-1 visibility is required, put a POINTER in
CLAUDE.md ("current operational facts: query `mempalace_team_facts()`"), not the
fact itself.

Decision tree:

```text
Stable invariant / rule / bootstrapping directive, true regardless of
date or state, needed every turn?
  YES -> CLAUDE.md / AGENTS.md. Stop.
Personal to you (prefs, corrections, reference links, in-progress notes,
debugging scratch)?
  YES -> native auto-memory. Stop. Do NOT file in MemPalace.
Durable, team-relevant, and not derivable from repo/git?
  NO  -> do not file (throwaway).
  YES -> MemPalace -- pick a lane below (SHAPE->TOOL).
```

## File durable knowledge — SHAPE→TOOL

Match the shape of the fact to the tool. **Three identity-ish lanes are
distinct** — do not conflate them:

**Lane 1 — Relationships / ownership / who-depends-on-what / people↔systems →
`mempalace_kg_add` triples** (`subject → predicate → object`, e.g. "Dana
owns ingest-pipeline"). Queryable via kg_query / kg_timeline / kg_neighbors.
Always pass `asserted_by` as `<operator-email> (<model-id>)` (e.g.
`you@example.com (claude-opus-4-8)`) so the triple records who asserted
it. Call `mempalace_kg_invalidate` when a fact stops being true. Both
**raise** without a resolved team. File a drawer about the same
person/system, then add the matching triple so it surfaces in graph queries.

**Lane 2 — Name resolution (aliases, ambiguous names, who-is-X) →
`mempalace_entity_seed` + `mempalace_disambiguate`** (postgres/central only;
`available:false` on local). Seed the team's entity registry with known people,
projects, and aliases. **The registry starts EMPTY in production — agents are
the sole labelers; seed early and re-seed as you learn new names.** Aliases go in
ONE direction: `aliases={"DR": "Dana Rivera"}` (key = alias, value = canonical),
or embedded per person as `people=[{"name": "Dana Rivera", "aliases": ["DR"]}]`
— there is no reverse form, and every name must be a non-empty string or the call
raises. `entity_seed` is read-merge-write (nothing overwritten); it **raises**
without a resolved team. Use `mempalace_disambiguate` to resolve a surface form
before acting on it. `entity_seed` also takes `mode=` (default `'personal'`,
applied only on the first seed; later calls silently ignore it) — set the team
mode when seeding the shared team registry; reserve personal mode for local
name-resolution scratch. **Entity type is set by WHICH list a name goes in**
(`people=[...]` → person, `projects=[...]` → project), not by any field: do
**not** pass a `kind` key on a person entry — it is silently dropped, and a
service placed in `people=[...]` is then mis-typed as a person. Systems and services (e.g. `ingest-pipeline`,
`auth-gateway`) have no dedicated bucket — seed them in `projects=[...]` (do not
omit them) so they stay disambiguatable, accepting that they type as `project`.

**Lane 3 — Must-know shared team identity / critical facts →
`mempalace_team_fact_add(fact, created_by=None)` / `mempalace_team_facts()`**
(postgres/central only; `available:false` on local). Short must-know lines every
agent on the team sees ("prod DB is read-replica only", "release freeze until
Q3"). Not per-conversation memories (those go to `add_drawer`); not kg triples;
not name-resolution. `team_fact_add` **raises** without a resolved team. Always
pass `created_by` as `<operator-email> (<model-id>)` — a shared vault otherwise
loses who asserted the fact. **`team_fact` vs `drawer`:** use `team_fact_add` when the fact is SHORT
and must be visible to every agent on every `team_facts()` call without a search;
use a drawer (Lane 4) when it carries rationale, detail, or runbook steps only
some tasks need. If a fact needs both, file the detail as a drawer and a one-line
team_fact pointing to it — never duplicate the full text in both.

**Lane 4 — Verbatim decisions, rationale, runbooks, gotchas →
`mempalace_add_drawer(wing, room, content, topics=[...])`**. Supply
`topics=[...]` labels (e.g. `["Angular", "OpenAPI"]`) — these populate the
wing-topics substrate that forms cross-wing topic tunnels. **Without labels,
topic tunnels stay empty in production.** Agents are the labelers. On the central
backend a `topics=` label also makes the drawer findable via
`mempalace_search(entity=<label>)` / `mempalace_entities(entity=<label>)` even when
the label never appears in the content — so tagging is the deterministic way to
make a drawer recallable by a name the prose doesn't spell out. Always pass
`added_by` as the **human operator plus the model that wrote it** — e.g.
`added_by="you@example.com (claude-opus-4-8)"`: the operator's corporate
email (the address the harness surfaces to you, NOT `git config user.email`,
which may be a personal one) followed by your model id in parentheses. It
defaults to the constant `'mcp'`, and passing your model id alone (e.g.
`"claude"`) erases who is accountable for the memory in a shared vault — the
human operator is the author, the model tag is provenance. If the harness does
not surface your email (e.g. a headless pipeline), use your operator handle alone
(e.g. `your-handle`), never the default `mcp`. Like every team-scoped writer,
`add_drawer` **raises** without a resolved team (`switch_team`, header, or an
explicit `vault=`) — a team-less call fails loud instead of silently landing
in a shared default vault.

**Lane 5 — Session checkpoint →
`mempalace_diary_write(agent_name, entry, topic=<tag>)`**. Set `agent_name` to
the **human operator's handle** (e.g. `agent_name="your-handle"`), NOT your
model id. The server auto-derives the diary's wing as `wing_<agent_name>`, so a
model id spawns a throwaway `wing_claude-sonnet-4-6` that fragments anew on every
model version, whereas an operator handle keeps each person's diaries converging
in one stable, targeted `wing_<operator>`. Set `topic=` too — same substrate as
drawer topics. **Raises** without a resolved team, like the other writers.

File proactively while the fact is fresh. Quality over volume: noisy logging
degrades recall, and a `kg_add` triple recalls far better than the same
relationship buried in paragraph prose.

## Update, don't duplicate

A drawer is mutable: its id is a **logical** id, and `get` / `update` / `delete`
all operate on the **whole drawer** (`mempalace_update_drawer` replaces content
via a verbatim-safe write-first rechunk; `mempalace_delete_drawer` removes the
whole drawer; both accept the drawer's logical id). So when a fact you already
filed evolves, **update the existing drawer instead of filing a near-duplicate**:

1. **Recall first.** Before `add_drawer`, `mempalace_search` /
   `mempalace_entities` for an existing drawer on this topic — the result row
   carries its `drawer_id` (the logical id you pass to update/delete).
2. **If one exists, update it** with `mempalace_update_drawer` so one evolving
   fact converges to a SINGLE drawer across the session — not one drawer per
   checkpoint. Add a new drawer only when nothing covers the topic.
3. **Supersede cleanly.** If a fact merely *evolved*, edit the drawer in place;
   if a drawer is *wrong or replaced* by a different one, `mempalace_delete_drawer`
   it. Never leave a stale drawer beside its replacement.

**Concurrency caution (multi-writer).** `mempalace_update_drawer` REPLACES the
whole drawer — it does not merge, and there is no version check or optimistic
lock. Before you update a drawer you did not just create, re-read it
(`mempalace_get_drawer`) immediately beforehand and preserve everything you are
not deliberately changing; otherwise two teammates editing the same drawer
concurrently will silently lose one set of edits (last write wins). For a
high-churn shared topic, prefer a new dated `add_drawer` entry over rewriting one
shared drawer.

## Team-vault routing

Active vault resolves highest-first: `vault=` arg → `mempalace_switch_team` →
`X-Mempalace-Team` header → server default. The plugin ships no team header, so
set yours explicitly:

- **`mempalace_list_vaults()`** — see all vaults + current; run once at session start.
- **`mempalace_switch_team(team="frontend")`** — set for the session. The
  response reports `exists: false` plus the known vaults when the name matches
  no existing vault — a typo caught here costs nothing; unnoticed, the first
  write creates the misspelled vault.
- **`vault="frontend"`** on a single call to `mempalace_search` / `mempalace_add_drawer`.
- **`vault="all"`** on `mempalace_search` — read-only sweep across every vault.

For a persistent default, use the per-repo `.mcp.json` or per-machine
`claude mcp add --scope user` recipes in
[`deploy/README.md`](../../../../deploy/README.md).

</Workflow>

<Stop_Conditions>

## Common mistakes

- **Writing to the wrong vault.** If `mempalace_list_vaults` shows the wrong
  primary, set it (`switch_team`, or a per-repo/per-machine override) before
  bulk writes — scattering a team's memory is hard to undo.
- **Filing transcript noise.** File durable, self-contained facts, not raw chat scrollback.
- **Expecting cross-team writes.** `vault="all"` is read-only; to write into
  another team's vault, target it explicitly.
- **Treating `default` as a team.** `default`, `primary` and `all` are
  reset/sweep ALIASES, never writable vaults — `switch_team(team="default")`
  clears the session team and `vault="default"` resolves to no team, so both
  land on "no team resolved". Always use a real team name from
  `mempalace_list_vaults`.
- **Leaving the entity registry empty.** `mempalace_disambiguate` always returns
  `found:false` until you seed with `mempalace_entity_seed` — seed early and
  re-seed as you learn new names.
- **Omitting topics.** A drawer or diary entry filed without `topics=` / `topic=`
  contributes nothing to cross-wing topic tunnels.
- **Filing as the model instead of the operator.** Passing `added_by="claude"`
  (your model id) on a drawer, or a model id as `diary_write`'s `agent_name`,
  erases human authorship and spawns a fresh `wing_<model>` per model version.
  Author drawers as `<operator-email> (<model-id>)` and key diary wings to the
  operator handle so authorship is transparent and wings stay stable.
- **Duplicating instead of updating.** Don't file a new drawer that supersedes an
  earlier one and leave both. Update the existing drawer in place
  (`mempalace_update_drawer`) or delete the superseded one
  (`mempalace_delete_drawer`) — two drawers telling different versions of one
  fact poison recall.

</Stop_Conditions>
