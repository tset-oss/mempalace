"""Regression: importing ``mcp_server`` must not exit the process on a foreign argv.

``mcp_server`` parses ``sys.argv`` at import time (``_args = _parse_args()``). It is
also imported as a *library* by ``mcp_fastmcp`` for the central server, which is
started via the OUTER CLI as ``mempalace serve --transport streamable-http``. That
argv belongs to ``mempalace.cli``, not to ``mcp_server`` — and ``streamable-http`` is
not in ``mcp_server``'s ``--transport {stdio,http}`` choices. ``parse_known_args``
tolerates unknown flags but still calls argparse's process-exiting error path on an
invalid *value* of a *known* flag, which used to kill ``mempalace serve`` at import.

These tests pin the fix: a foreign argv falls back to defaults instead of exiting,
while the legacy ``mempalace-mcp`` entry still parses its own flags.
"""

import subprocess
import sys


def test_parse_args_survives_foreign_serve_argv(monkeypatch):
    import mempalace.mcp_server as m

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mempalace",
            "serve",
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ],
    )
    args = m._parse_args()  # must not raise SystemExit
    # Fell back to defaults: the central path drives the server itself.
    assert args.transport == "stdio"
    assert args.palace is None
    assert args.backend is None


def test_parse_args_legacy_mcp_entry_still_parses(monkeypatch):
    import mempalace.mcp_server as m

    monkeypatch.setattr(
        sys, "argv", ["mempalace-mcp", "--transport", "http", "--backend", "postgres"]
    )
    args = m._parse_args()
    assert args.transport == "http"
    assert args.backend == "postgres"


def test_import_under_central_serve_argv_does_not_exit():
    """The true reproduction: a FRESH process where the module-level
    ``_args = _parse_args()`` runs against the central-serve argv at import."""
    code = (
        "import sys;"
        "sys.argv=['mempalace','serve','--transport','streamable-http','--host','0.0.0.0','--port','8080'];"
        "import mempalace.mcp_server as m;"
        "assert m._args.transport=='stdio', m._args.transport;"
        "print('IMPORT_OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    # returncode 0 is the real signal (the bug exited the process at import).
    # mcp_server redirects stdout->stderr to keep the stdio JSON-RPC channel
    # clean, so the marker can land on either stream.
    assert r.returncode == 0, (
        f"import crashed rc={r.returncode}\nSTDOUT={r.stdout}\nSTDERR={r.stderr}"
    )
    assert "IMPORT_OK" in (r.stdout + r.stderr)
