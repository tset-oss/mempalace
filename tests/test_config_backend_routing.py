"""Config-driven backend selection + team-vault routing (G003).

Covers the wiring that makes a machine's *local* config choose the storage
backend and the primary team vault, without breaking the chroma default.

* Config resolution: MEMPALACE_BACKEND / MEMPALACE_TEAM / MEMPALACE_DATABASE_URL
  (env > config.json > default), via a tmp config dir.
* palace.get_collection routing: the PalaceRef handed to the backend carries the
  team namespace for server-mode backends and None for chroma — verified with a
  recording fake backend (no database needed).
* A live-Postgres integration that asserts palace.get_collection actually lands
  rows in the configured team's schema (skipped when no DB is reachable).
"""

from __future__ import annotations

import hashlib
import os

import pytest

import mempalace.palace as palace_mod
from mempalace.config import MempalaceConfig

DIM = 384


# --------------------------------------------------------------------------
# Config resolution
# --------------------------------------------------------------------------


def _cfg(tmp_path, file_config=None):
    import json

    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir(exist_ok=True)
    cfgfile = cfgdir / "config.json"
    if file_config is not None:
        cfgfile.write_text(json.dumps(file_config))
    elif cfgfile.exists():
        cfgfile.unlink()  # ensure a clean "no config file" read across repeat calls
    return MempalaceConfig(config_dir=str(cfgdir))


def test_backend_defaults_to_chroma(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_BACKEND", raising=False)
    assert _cfg(tmp_path).backend == "chroma"


def test_backend_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    assert _cfg(tmp_path, {"backend": "chroma"}).backend == "postgres"


def test_backend_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_BACKEND", raising=False)
    assert _cfg(tmp_path, {"backend": "postgres"}).backend == "postgres"


def test_team_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_TEAM", raising=False)
    assert _cfg(tmp_path).team is None
    assert _cfg(tmp_path, {"team": "Frontend"}).team == "frontend"
    monkeypatch.setenv("MEMPALACE_TEAM", "Backend")
    assert _cfg(tmp_path, {"team": "frontend"}).team == "backend"


def test_database_url_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_DATABASE_URL", raising=False)
    assert _cfg(tmp_path).database_url is None
    assert _cfg(tmp_path, {"database_url": "postgresql://x"}).database_url == "postgresql://x"
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", "postgresql://env")
    assert _cfg(tmp_path, {"database_url": "postgresql://x"}).database_url == "postgresql://env"


# --------------------------------------------------------------------------
# Routing (no database) — verify the PalaceRef namespace handed to the backend
# --------------------------------------------------------------------------


class _Recorder:
    """A fake backend that records the PalaceRef it is asked for."""

    def __init__(self):
        self.refs = []

    def get_collection(self, *, palace, collection_name, create):
        self.refs.append((palace, collection_name, create))
        return "FAKE_COLLECTION"


@pytest.fixture()
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(palace_mod, "get_backend", lambda name: rec)
    return rec


def test_chroma_routing_has_no_namespace(monkeypatch):
    # chroma resolves to the module-level _DEFAULT_BACKEND (not via get_backend),
    # so patch its get_collection directly.
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.setenv("MEMPALACE_TEAM", "frontend")  # must be ignored for chroma
    captured = {}

    def fake(*, palace, collection_name, create):
        captured["ref"] = palace
        captured["name"] = collection_name
        return "FAKE_COLLECTION"

    monkeypatch.setattr(palace_mod._DEFAULT_BACKEND, "get_collection", fake)
    palace_mod.get_collection("/p", collection_name="mempalace_drawers", create=False)
    assert captured["ref"].local_path == "/p"
    assert captured["ref"].namespace is None
    assert captured["name"] == "mempalace_drawers"


def test_postgres_routing_uses_team_namespace(recorder, monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", "frontend")
    palace_mod.get_collection("/p", collection_name="mempalace_drawers", create=False)
    ref = recorder.refs[0][0]
    assert ref.namespace == "frontend"


def test_postgres_team_override_beats_config(recorder, monkeypatch):
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", "frontend")
    palace_mod.get_collection("/p", collection_name="mempalace_drawers", create=False, team="backend")
    ref = recorder.refs[0][0]
    assert ref.namespace == "backend"


def test_closets_collection_threads_team(recorder, monkeypatch):
    # Regression: the hybrid-search closets pass must route to the SAME team as
    # the drawers pass, or a cross-team search mixes one team's closets with
    # another's drawers (breaking isolation).
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", "frontend")
    palace_mod.get_closets_collection("/p", create=False, team="backend")
    ref, name, _ = recorder.refs[0]
    assert ref.namespace == "backend"
    assert name == "mempalace_closets"


# --------------------------------------------------------------------------
# Live integration — palace.get_collection lands in the team schema
# --------------------------------------------------------------------------

psycopg = pytest.importorskip("psycopg")


def _dsn():
    return (
        os.environ.get("MEMPALACE_TEST_PG_URL")
        or os.environ.get("MEMPALACE_DATABASE_URL")
        or "postgresql://mempalace:mempalace@localhost:5432/mempalace"
    )


def _reachable():
    try:
        with psycopg.connect(_dsn(), connect_timeout=3) as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


def _fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * DIM
        for i, b in enumerate(hashlib.sha256((t or "").encode()).digest()):
            v[i] = b / 255.0
        out.append(v)
    return out


@pytest.mark.skipif(not _reachable(), reason="no reachable Postgres")
def test_get_collection_routes_to_team_schema(monkeypatch):
    import uuid

    from mempalace.backends import get_backend
    from mempalace.backends.postgres import team_schema

    team = "t" + uuid.uuid4().hex[:10]
    monkeypatch.setenv("MEMPALACE_BACKEND", "postgres")
    monkeypatch.setenv("MEMPALACE_TEAM", team)
    monkeypatch.setenv("MEMPALACE_DATABASE_URL", _dsn())
    # Inject the deterministic embedder onto the shared (registry-cached) backend
    # so we don't download the ONNX model in tests.
    be = get_backend("postgres")
    be._embedder = _fake_embed
    try:
        col = palace_mod.get_collection("/ignored", create=True)
        col.add(documents=["routed via palace"], ids=["1"], metadatas=[{"wing": "code"}])
        assert col.count() == 1
        # The row physically lives in the team's schema.
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT count(*) FROM "{team_schema(team)}"."mempalace_drawers"')
                assert cur.fetchone()[0] == 1
    finally:
        be._embedder = None
        with psycopg.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{team_schema(team)}" CASCADE')
            conn.commit()
