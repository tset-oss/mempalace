"""Unit tests for the injectable known-entity set on entity extraction.

DB-less and fast: exercises ``_extract_entities_for_metadata(content, known=...)``
so the central Postgres write path can pass a per-vault known set instead of the
file-backed ``~/.mempalace/known_entities.json`` (which on a server is the
server process's home, not the engineer's). See
``docs/design/vault-scoped-entities-hallways.md``.
"""

from __future__ import annotations

from mempalace.miner import _extract_entities_for_metadata


def _entities(out: str) -> set[str]:
    return {e for e in out.split(";") if e} if out else set()


def test_injected_known_set_is_matched_case_insensitively():
    # "Dana" appears once (the frequency tier needs >=2), and "zelda" is
    # lowercase — neither would be tagged by frequency alone. An injected known
    # set tags both, matching case-insensitively.
    content = "Dana reviewed the ingest pipeline with zelda today."
    ents = _entities(_extract_entities_for_metadata(content, known={"Dana", "zelda"}))
    assert "Dana" in ents
    assert "zelda" in ents


def test_empty_known_set_falls_back_to_frequency_tier():
    # No known entities -> only the capitalized-word frequency tier fires.
    # "Zorptin" (not in any stoplist/COCA) repeated thrice is tagged.
    content = "Zorptin shipped. Zorptin grew. Zorptin again."
    ents = _entities(_extract_entities_for_metadata(content, known=set()))
    assert "Zorptin" in ents


def test_none_known_consults_the_file_registry(monkeypatch):
    # known=None must preserve the legacy chroma/miner behaviour: load the
    # file-backed registry. We assert the loader is consulted.
    import mempalace.miner as miner

    seen = {"loaded": False}

    def fake_loader():
        seen["loaded"] = True
        return frozenset({"Aya"})

    monkeypatch.setattr(miner, "_load_known_entities", fake_loader)
    ents = _entities(_extract_entities_for_metadata("a note mentioning Aya here", known=None))
    assert seen["loaded"] is True
    assert "Aya" in ents
