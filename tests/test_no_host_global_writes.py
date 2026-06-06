"""Subtractive proof: no NEW server-reachable host-global write path may appear.

The central deployment runs one OS user, one ``$HOME``, and many team vaults in
a single process. Anything written under ``~/.mempalace`` is therefore SHARED
across every team on the host — a cross-tenant leak waiting to happen. The
team-scoped data lives in Postgres; the only legitimate ``~/.mempalace`` writes
left on the server-reachable path are by-design-local artifacts (the local
chroma/jsonl files, host-local locks, personal config) that the postgres path
either never reaches or gates behind ``backend == "chroma"``.

Earlier tests proved parity *within* a vault but never proved the *negative*
property: that no host-global write path survives on the server path. This guard
encodes that negative proof. It statically scans every module reachable from the
MCP server for writes whose target resolves to a ``$HOME``-anchored path
(``~``/``expanduser``/``Path.home()``/``$HOME``/``.mempalace``) and fails if it
finds one that is not on the maintained allowlist below.

When a future change adds a new ``~/.mempalace`` write reachable from the server,
this test goes RED. The fix is to vault the write (route it to Postgres) — or,
if it is genuinely a by-design-local artifact, to add it to ``ALLOWLIST`` with a
one-line reason. A real cross-tenant leak must be FIXED, never silently
allowlisted.

The scanner is exposed as :func:`scan_source_for_host_global_writes` so the
self-test can feed it a synthetic leak and assert the guard actually fires —
proving this is not a vacuous always-pass.

Known blind spots (accepted limits, not present in the tree today): the scanner
resolves ``$HOME`` taint through names, ``self``/class attributes, ``os.path``
transforms, or-chains and intra-module helper returns, but it does NOT inspect
f-string targets (``ast.JoinedStr``, e.g. ``open(f"{home}/.mempalace/x", "w")``)
nor ``shutil.copy*`` destinations. A future leak using either shape would slip
past; extend ``_WRITE_ATTR_*`` / ``_HomePathResolver`` if one ever appears.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────
# Server-reachable module set
#
# The leak class is "reachable from the MCP server". We compute the transitive
# import closure from the two server entrypoints so a future commit that pulls a
# new module into the server path gets scanned automatically — the guard does
# not depend on a hand-maintained module list staying in sync. We then add the
# C-chain identity modules (entity_registry*) explicitly: they are the seam
# targets the server will route through, and the host-global JSON they own is
# exactly the kind of artifact this guard must keep honest, even before the
# static import edge exists.
# ─────────────────────────────────────────────────────────────────────────

_SERVER_ENTRYPOINTS = ("mcp_server", "mcp_fastmcp")
_EXTRA_SCANNED_MODULES = ("entity_registry", "entity_registry_postgres")

_PKG_DIR = Path(__file__).resolve().parent.parent / "mempalace"


def _module_local_imports(path: Path) -> set[str]:
    """Top-level module names imported from within the ``mempalace`` package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1 and node.module:
                deps.add(node.module.split(".")[0])
            elif node.level >= 1 and node.module is None:
                for alias in node.names:
                    deps.add(alias.name.split(".")[0])
            elif node.module and node.module.startswith("mempalace."):
                deps.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mempalace."):
                    deps.add(alias.name.split(".")[1])
    return deps


def _server_reachable_modules() -> list[str]:
    """Transitive in-package import closure from the server entrypoints."""
    available = {p.stem: p for p in _PKG_DIR.glob("*.py")}
    seen: set[str] = set()
    stack = list(_SERVER_ENTRYPOINTS) + list(_EXTRA_SCANNED_MODULES)
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = available.get(mod)
        if path is None:  # subpackage (e.g. backends) — no single-file module
            continue
        for dep in _module_local_imports(path):
            if dep not in seen and dep in available:
                stack.append(dep)
    return sorted(m for m in seen if m in available)


# ─────────────────────────────────────────────────────────────────────────
# The scanner
# ─────────────────────────────────────────────────────────────────────────


class HostGlobalWrite(NamedTuple):
    """One write whose target resolves to a ``$HOME``-anchored path."""

    module: str
    lineno: int
    target: str  # the unparsed write-target expression, e.g. "_WAL_FILE"
    call: str  # the unparsed write call, for the failure message


# Names/functions that resolve to the user-configurable PALACE data root rather
# than the host-global ``~/.mempalace`` config dir. Writing INTO the palace
# directory is the app's whole job and follows the configured palace_path, not
# ``$HOME`` directly — it is not the cross-tenant leak class this guard targets.
_PALACE_ROOT_NAMES = frozenset(
    {
        "palace_path",
        "palace_dir",
        "_get_palace_path",
        "dest_palace",
        "source_palace",
        "stale_path",
        "temp_palace",
        "archive_path",
    }
)

# Write sinks: attribute-style calls (receiver is the write target).
_WRITE_ATTR_RECEIVER = frozenset({"write_text", "write_bytes", "touch", "mkdir"})
# Write sinks whose first positional arg is the destination.
_WRITE_ATTR_FIRST_ARG = frozenset({"makedirs"})
# Write sinks where both of the first two positional args are destinations.
_WRITE_ATTR_TWO_ARGS = frozenset({"replace", "rename", "move"})


def _is_home_literal(value: str) -> bool:
    return value.startswith("~") or ".mempalace" in value


class _HomePathResolver:
    """Resolves whether an AST expression points at a ``$HOME``-anchored path.

    Propagates the taint through module-level and local names, ``self.<attr>``
    and class-attribute bindings, ternaries / ``or`` chains, ``Path`` aliases,
    common ``os.path`` transforms, and intra-module functions that return a
    host-global path. Palace-root providers are explicitly NOT home-global.
    """

    def __init__(self, tree: ast.AST):
        self._path_aliases = {"Path"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
                for alias in node.names:
                    if alias.name == "Path":
                        self._path_aliases.add(alias.asname or "Path")
        self._home_names: set[str] = set()
        self._home_self_attrs: set[str] = set()
        self._home_class_attrs: set[str] = set()
        self._home_return_funcs: set[str] = set()
        self._build(tree)

    def is_home(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _is_home_literal(node.value)
        if isinstance(node, ast.Name):
            if node.id in _PALACE_ROOT_NAMES:
                return False
            return node.id in self._home_names
        if isinstance(node, ast.Attribute):
            if node.attr in ("palace_path", "palace_dir"):
                return False
            base = node.value
            if isinstance(base, ast.Name):
                if base.id == "self" and node.attr in self._home_self_attrs:
                    return True
                if base.id in ("cls", "self") and node.attr in self._home_class_attrs:
                    return True
            return self.is_home(base)
        if isinstance(node, ast.IfExp):
            return self.is_home(node.body) or self.is_home(node.orelse)
        if isinstance(node, ast.BoolOp):
            return any(self.is_home(v) for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            return self.is_home(node.left) or self.is_home(node.right)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
                key = node.slice
                if isinstance(key, ast.Constant) and key.value == "HOME":
                    return True
            return self.is_home(node.value)
        if isinstance(node, ast.Call):
            return self._call_is_home(node)
        return False

    def _call_is_home(self, node: ast.Call) -> bool:
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else None)
        )
        if name in _PALACE_ROOT_NAMES:
            return False
        if name == "expanduser" and node.args:
            arg = node.args[0]
            return (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and _is_home_literal(arg.value)
            )
        if name == "home":  # Path.home()
            return True
        if name == "getenv" and node.args:
            arg = node.args[0]
            return isinstance(arg, ast.Constant) and arg.value == "HOME"
        if name in ("join", "dirname", "abspath", "realpath", "normpath", "expanduser"):
            return any(self.is_home(a) for a in node.args)
        if name in self._path_aliases or name == "str":
            return any(self.is_home(a) for a in node.args)
        if name in self._home_return_funcs:
            return True
        return False

    def _build(self, tree: ast.AST) -> None:
        # Fixpoint: home-tainting can flow forward (a name tainted on one line
        # taints a derived name on a later line), so iterate to stability.
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign) and self.is_home(stmt.value):
                            for target in stmt.targets:
                                if (
                                    isinstance(target, ast.Name)
                                    and target.id not in self._home_class_attrs
                                ):
                                    self._home_class_attrs.add(target.id)
                                    changed = True
                if isinstance(node, ast.Assign) and self.is_home(node.value):
                    changed |= self._taint_targets(node.targets)
                if isinstance(node, ast.AnnAssign) and node.value is not None:
                    if isinstance(node.target, ast.Name) and self.is_home(node.value):
                        if (
                            node.target.id not in _PALACE_ROOT_NAMES
                            and node.target.id not in self._home_names
                        ):
                            self._home_names.add(node.target.id)
                            changed = True
                if isinstance(node, ast.FunctionDef):
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Return) and self.is_home(sub.value):
                            if (
                                node.name not in _PALACE_ROOT_NAMES
                                and node.name not in self._home_return_funcs
                            ):
                                self._home_return_funcs.add(node.name)
                                changed = True

    def _taint_targets(self, targets: list[ast.expr]) -> bool:
        changed = False
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id not in _PALACE_ROOT_NAMES and target.id not in self._home_names:
                    self._home_names.add(target.id)
                    changed = True
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == "self" and target.attr not in self._home_self_attrs:
                    self._home_self_attrs.add(target.attr)
                    changed = True
                if target.value.id == "cls" and target.attr not in self._home_class_attrs:
                    self._home_class_attrs.add(target.attr)
                    changed = True
        return changed


def _write_targets(node: ast.Call) -> list[ast.expr]:
    """Destination expressions of a write call, or [] if the call is not a write."""
    func = node.func
    name = (
        func.attr
        if isinstance(func, ast.Attribute)
        else (func.id if isinstance(func, ast.Name) else None)
    )
    if name == "open":
        # builtin open(path, "w"/"a"/...) or os.open(path, O_WRONLY|O_CREAT|...).
        is_os_open = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        )
        if is_os_open:
            flags = ast.unparse(node.args[1]) if len(node.args) >= 2 else ""
            is_write = any(f in flags for f in ("O_WRONLY", "O_CREAT", "O_APPEND", "O_RDWR"))
        elif (
            len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            mode = node.args[1].value
            is_write = any(c in mode for c in "wax+")
        else:
            is_write = False  # open(path) with no mode defaults to read
        return [node.args[0]] if (is_write and node.args) else []
    if name in _WRITE_ATTR_TWO_ARGS:
        return list(node.args[:2])
    if name in _WRITE_ATTR_FIRST_ARG:
        return list(node.args[:1])
    if name == "mkstemp":
        return [kw.value for kw in node.keywords if kw.arg == "dir"]
    if name in _WRITE_ATTR_RECEIVER and isinstance(func, ast.Attribute):
        return [func.value]
    return []


def scan_source_for_host_global_writes(source: str, module: str) -> list[HostGlobalWrite]:
    """Return every host-global write sink in ``source``.

    This is the guard's core, exposed as a plain callable so the self-test can
    feed it synthetic source and confirm the guard fires.
    """
    tree = ast.parse(source, filename=module)
    resolver = _HomePathResolver(tree)
    findings: list[HostGlobalWrite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for target in _write_targets(node):
            unwrapped = target
            if (
                isinstance(unwrapped, ast.Call)
                and isinstance(unwrapped.func, ast.Name)
                and unwrapped.func.id == "str"
                and unwrapped.args
            ):
                unwrapped = unwrapped.args[0]
            if resolver.is_home(unwrapped):
                findings.append(
                    HostGlobalWrite(
                        module=module,
                        lineno=node.lineno,
                        # Record the UNWRAPPED target (drop a str(...) wrapper) so
                        # the allowlist key is stable, e.g. "_WAL_FILE" rather than
                        # "str(_WAL_FILE)".
                        target=ast.unparse(unwrapped),
                        call=ast.unparse(node)[:90],
                    )
                )
                break  # one finding per write call is enough
    return findings


# ─────────────────────────────────────────────────────────────────────────
# Allowlist — by-design-local host-global writes
#
# Keyed by (module, target-expression) so it survives line-number drift but
# still pins the specific sink. EVERY entry is a write that is legitimately
# host-local: a local chroma/jsonl artifact (gated behind backend=="chroma" or
# only created lazily on the jsonl path), a host-local coordination file, or a
# personal/CLI-only config the postgres server path does not reach. A NEW
# (module, target) that is not here means a new host-global write — fix it
# (vault it), do not just add it here.
# ─────────────────────────────────────────────────────────────────────────

ALLOWLIST: dict[tuple[str, str], str] = {
    # WAL jsonl audit file — chroma/jsonl-only. The postgres deploy defaults the
    # WAL sink to the team-tagged central table; the jsonl directory/file are now
    # created lazily on the first jsonl write, so importing the module on postgres
    # leaves no ~/.mempalace/wal artifact.
    (
        "mcp_server.py",
        "_WAL_FILE",
    ): "local jsonl audit sink; created lazily, postgres uses the team-tagged table",
    (
        "mcp_server.py",
        "wal_dir",
    ): "parent dir of the local jsonl WAL; created lazily on the jsonl write path",
    # Local user config — written by the CLI / onboarding (config.save /
    # create_default_config / save_people_map); not reached from any MCP server
    # tool handler. Personal per-developer config, not team data.
    (
        "config.py",
        "self._config_dir",
    ): "personal ~/.mempalace config dir; CLI/onboarding only, not server-reachable writes",
    ("config.py", "self._config_file"): "personal ~/.mempalace/config.json; CLI/onboarding only",
    (
        "config.py",
        "self._people_map_file",
    ): "personal ~/.mempalace/people_map.json; CLI/onboarding only",
    # Derived cross-wing tunnel/hallway JSON — gated behind backend=="chroma" at
    # the call sites; the postgres path derives links into the per-team vault.
    (
        "hallways.py",
        "_HALLWAY_FILE",
    ): "chroma-only hallways.json; postgres derives hallways into the per-team vault",
    ("hallways.py", "directory"): "parent dir of the chroma-only hallways.json",
    # SQLite knowledge graph — the local/chroma KG backend. The central deploy
    # uses the per-team PostgresKnowledgeGraph; this SQLite file is local-only.
    (
        "knowledge_graph.py",
        "db_parent",
    ): "chroma/local SQLite KG dir; postgres uses the per-team PostgresKnowledgeGraph",
    # Miner known-entities registry — chroma-only. On postgres the miner tags
    # entities from the per-team vault (PostgresEntityIndex.known_entities), and
    # the known_entities.json read/write is gated behind backend=="chroma".
    (
        "miner.py",
        "registry_path",
    ): "CLI-only known_entities.json (mempalace init/mine); not reached from any MCP server tool handler; postgres tags entities from the per-team vault",
    ("miner.py", "registry_path.parent"): "parent dir of the CLI-only known_entities.json",
    # Host-local advisory locks coordinating concurrent mines on ONE machine.
    # Local process coordination, not team data; never cross-tenant.
    (
        "palace.py",
        "lock_dir",
    ): "host-local ~/.mempalace/locks dir for single-host mine coordination",
    ("palace.py", "lock_path"): "host-local advisory lock file for single-host mine coordination",
    # EntityRegistry disambiguation JSON — chroma/CLI/onboarding path. The
    # central deploy routes disambiguation through entity_registry_postgres
    # (per-team table); this JSON is the local backend's store.
    (
        "entity_registry.py",
        "self._path",
    ): "chroma/CLI entity_registry.json; postgres uses the per-team entity_registry_postgres table",
    (
        "entity_registry.py",
        "self._path.parent",
    ): "parent dir of the chroma/CLI entity_registry.json",
}


def _collect_unallowlisted_writes() -> tuple[list[HostGlobalWrite], list[str]]:
    """Scan every server-reachable module; return (leaks, scanned-module-names)."""
    modules = _server_reachable_modules()
    leaks: list[HostGlobalWrite] = []
    for module in modules:
        path = _PKG_DIR / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        for finding in scan_source_for_host_global_writes(source, path.name):
            if (finding.module, finding.target) not in ALLOWLIST:
                leaks.append(finding)
    return leaks, modules


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


def test_no_new_host_global_write_path_on_the_server_path():
    """No un-allowlisted host-global write may be reachable from the server.

    Goes RED when a future change adds a ``~/.mempalace`` write reachable from
    the MCP server. Fix it by vaulting the write (route it to Postgres), or — if
    it is genuinely a by-design-local artifact — add it to ``ALLOWLIST`` with a
    one-line reason. Never allowlist a real cross-tenant leak.
    """
    leaks, modules = _collect_unallowlisted_writes()
    assert modules, "no server-reachable modules discovered — scanner is broken"
    assert not leaks, (
        "New host-global write path(s) reachable from the MCP server. Vault the "
        "write (route it to Postgres) or, if by-design-local, add a "
        "(module, target) entry to ALLOWLIST with a reason:\n"
        + "\n".join(
            f"  {leak.module}:{leak.lineno}  target={leak.target!r}  ->  {leak.call}"
            for leak in leaks
        )
    )


def test_allowlist_has_no_stale_entries():
    """Every allowlist entry must still correspond to a real detected write.

    Keeps the allowlist honest: if a sink is removed or vaulted, its entry must
    be deleted rather than lingering as dead cover for a future leak.
    """
    detected: set[tuple[str, str]] = set()
    for module in _server_reachable_modules():
        path = _PKG_DIR / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        for finding in scan_source_for_host_global_writes(source, path.name):
            detected.add((finding.module, finding.target))
    stale = sorted(set(ALLOWLIST) - detected)
    assert not stale, (
        "Stale ALLOWLIST entries no longer match any detected write (delete them):\n"
        + "\n".join(f"  {module}: {target}" for module, target in stale)
    )


def test_server_entrypoints_are_in_the_scanned_closure():
    """The scan must actually cover the server entrypoints and the PG write path."""
    modules = set(_server_reachable_modules())
    for entry in _SERVER_ENTRYPOINTS:
        assert entry in modules, f"server entrypoint {entry} missing from scan closure"
    # The postgres write-path modules the spec calls out must be in scope.
    for pg_mod in ("link_store", "link_store_postgres", "knowledge_graph_postgres"):
        assert pg_mod in modules, f"postgres write-path module {pg_mod} missing from scan closure"


def test_guard_scanner_flags_an_injected_host_global_write():
    """Self-test: the scanner must go RED on a synthetic host-global write.

    Feeds the scanner a fake module containing an obvious ``~/.mempalace`` write
    and asserts it is flagged. Proves the guard genuinely fires rather than
    always passing because the scanner silently matches nothing.
    """
    injected = (
        "import os\n"
        "import json\n"
        "def leak(data):\n"
        '    with open(os.path.expanduser("~/.mempalace/leak.json"), "w") as f:\n'
        "        json.dump(data, f)\n"
    )
    findings = scan_source_for_host_global_writes(injected, "fake_leak.py")
    assert findings, "scanner failed to flag an obvious host-global write — guard is vacuous"
    assert any("leak.json" in f.call or "expanduser" in f.call for f in findings)
    # And the injected leak is NOT on the allowlist, so the guard would fail.
    assert all((f.module, f.target) not in ALLOWLIST for f in findings)


def test_guard_scanner_flags_other_host_global_write_shapes():
    """The scanner catches the common shapes, not just the literal open() form.

    Path.home() targets, write_text(), mkdir(), os.replace into a home path, and
    a home path threaded through a local variable are all real leak shapes a
    future PR could introduce — the guard must catch each.
    """
    cases = [
        # Path.home() / write_text via a local variable.
        (
            "from pathlib import Path\n"
            "def f(d):\n"
            "    p = Path.home() / '.mempalace' / 'x.json'\n"
            "    p.write_text(d)\n"
        ),
        # mkdir on an expanduser-derived directory.
        (
            "import os\n"
            "from pathlib import Path\n"
            "def f():\n"
            "    Path(os.path.expanduser('~/.mempalace/sub')).mkdir(parents=True)\n"
        ),
        # os.replace whose DESTINATION (second arg) is the home path.
        (
            "import os\n"
            "TARGET = os.path.join(os.path.expanduser('~'), '.mempalace', 'y.json')\n"
            "def f(tmp):\n"
            "    os.replace(tmp, TARGET)\n"
        ),
    ]
    for source in cases:
        findings = scan_source_for_host_global_writes(source, "shape_case.py")
        assert findings, f"scanner missed a host-global write shape:\n{source}"


def test_guard_scanner_ignores_reads_and_palace_relative_writes():
    """The scanner must not flag reads or palace-relative (configurable) writes.

    A read of a home path is fine; only writes leak. And writing INTO the
    configured palace directory follows palace_path (the user's chosen data
    root), not ``$HOME`` directly — that is not the cross-tenant leak class.
    Flagging these would make the guard noisy and untrustworthy.
    """
    read_only = (
        "import os, json\n"
        "def f():\n"
        '    with open(os.path.expanduser("~/.mempalace/x.json"), "r") as fh:\n'
        "        return json.load(fh)\n"
    )
    assert not scan_source_for_host_global_writes(read_only, "read_case.py")

    palace_relative = (
        "import os\n"
        "def f(palace_path):\n"
        '    with open(os.path.join(palace_path, "corrupt_ids.txt"), "w") as fh:\n'
        "        fh.write('x')\n"
    )
    assert not scan_source_for_host_global_writes(palace_relative, "palace_case.py")


def test_importing_mcp_server_creates_no_host_global_wal_artifact(tmp_path, monkeypatch):
    """Importing the server must not create ~/.mempalace/wal at module load.

    The WAL directory/file are created lazily on the first jsonl write, so a
    postgres deploy (where the audit log goes to the central team-tagged table)
    leaves no host-global ~/.mempalace/wal artifact behind merely by importing.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_WAL_SINK", "postgres")

    import mempalace.mcp_server as mcp_server

    importlib.reload(mcp_server)
    try:
        wal_dir = fake_home / ".mempalace" / "wal"
        assert not wal_dir.exists(), (
            "importing mcp_server created a host-global WAL directory; the WAL "
            "must be created lazily on the jsonl write path only"
        )
    finally:
        # Reload once more under the test's (conftest) HOME so the module's
        # cached paths don't leak the fake HOME into later tests.
        monkeypatch.undo()
        importlib.reload(mcp_server)
