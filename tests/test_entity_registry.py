"""Tests for mempalace.entity_registry."""

import logging
from unittest.mock import patch

import pytest

from mempalace.entity_registry import (
    COMMON_ENGLISH_WORDS,
    PERSON_CONTEXT_PATTERNS,
    EntityRegistry,
    _coerce_project_name,
)
from mempalace.mcp_server import _merge_seed_into_registry

# Shared mock result for Wikipedia person lookup tests
_MOCK_SAOIRSE_PERSON = {
    "inferred_type": "person",
    "confidence": 0.80,
    "wiki_summary": "Saoirse is an Irish given name.",
    "wiki_title": "Saoirse",
}


# ── COMMON_ENGLISH_WORDS ────────────────────────────────────────────────


def test_common_english_words_has_expected_entries():
    assert "ever" in COMMON_ENGLISH_WORDS
    assert "grace" in COMMON_ENGLISH_WORDS
    assert "will" in COMMON_ENGLISH_WORDS
    assert "may" in COMMON_ENGLISH_WORDS
    assert "monday" in COMMON_ENGLISH_WORDS


def test_common_english_words_is_lowercase():
    for word in COMMON_ENGLISH_WORDS:
        assert word == word.lower(), f"{word} should be lowercase"


# ── PERSON_CONTEXT_PATTERNS ─────────────────────────────────────────────


def test_person_context_patterns_is_nonempty():
    assert len(PERSON_CONTEXT_PATTERNS) > 0


# ── EntityRegistry creation and empty state ─────────────────────────────


def test_load_from_nonexistent_dir(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    assert registry.people == {}
    assert registry.projects == []
    assert registry.mode == "personal"
    assert registry.ambiguous_flags == []


def test_save_and_load_roundtrip(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="work",
        people=[{"name": "Alice", "relationship": "colleague", "context": "work"}],
        projects=["MemPalace"],
    )
    # Load again from same dir
    loaded = EntityRegistry.load(config_dir=tmp_path)
    assert loaded.mode == "work"
    assert "Alice" in loaded.people
    assert "MemPalace" in loaded.projects


def test_save_creates_file(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.save()
    assert (tmp_path / "entity_registry.json").exists()


def test_save_is_atomic_does_not_leave_tmp(tmp_path):
    # Atomic write must not leave the .tmp sidecar file after a successful save.
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.save()
    leftover = list(tmp_path.glob("entity_registry.json.tmp*"))
    assert leftover == [], f"atomic write leaked tmp file(s): {leftover}"


def test_save_preserves_previous_on_serialization_failure(tmp_path, monkeypatch):
    # If serialization fails mid-write, the previous registry must remain
    # intact — this is the whole point of atomic write vs truncating in place.
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Alice", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    registry.save()
    target = tmp_path / "entity_registry.json"
    original = target.read_text(encoding="utf-8")

    # Force os.replace to raise — simulates filesystem full / permission flip
    # AFTER the temp file is written but BEFORE the rename completes.
    import os as _os

    real_replace = _os.replace

    def boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError):
        registry.seed(
            mode="personal",
            people=[{"name": "Bob", "relationship": "friend", "context": "personal"}],
            projects=[],
        )
        registry.save()

    # Restore os.replace before reading so the assertion can rely on it.
    monkeypatch.setattr(_os, "replace", real_replace)
    assert target.read_text(encoding="utf-8") == original
    # The .tmp sidecar must also be cleaned up — otherwise it litters the
    # palace directory and a future diagnostic cannot distinguish stale
    # debris from an in-flight write.
    leftover = list(tmp_path.glob("entity_registry.json.tmp*"))
    assert leftover == [], f"atomic write leaked tmp on rename failure: {leftover}"


def test_save_cleans_tmp_on_write_failure(tmp_path, monkeypatch):
    # Failure BEFORE the rename (disk full, FUSE break, IO error during
    # write/fsync) must also clean up the .tmp sidecar. The existing
    # rename-failure test only covers the post-write path; this exercises
    # the gap between write and rename.
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Alice", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    registry.save()
    target = tmp_path / "entity_registry.json"
    original = target.read_text(encoding="utf-8")

    # Force os.fsync to raise — simulates IO error after the bytes are in
    # the kernel page cache but before they hit the platter.
    import os as _os

    real_fsync = _os.fsync

    def boom(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(_os, "fsync", boom)
    with pytest.raises(OSError):
        registry.seed(
            mode="personal",
            people=[{"name": "Bob", "relationship": "friend", "context": "personal"}],
            projects=[],
        )
        registry.save()
    monkeypatch.setattr(_os, "fsync", real_fsync)

    # Previous registry intact (rename never happened — atomic guarantee).
    assert target.read_text(encoding="utf-8") == original
    # And the .tmp sidecar is gone, not litter on disk.
    leftover = list(tmp_path.glob("entity_registry.json.tmp*"))
    assert leftover == [], f"atomic write leaked tmp on fsync failure: {leftover}"


# ── seed ────────────────────────────────────────────────────────────────


def test_seed_registers_people(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
            {"name": "Devon", "relationship": "friend", "context": "personal"},
        ],
        projects=["MemPalace"],
    )
    assert "Riley" in registry.people
    assert "Devon" in registry.people
    assert registry.people["Riley"]["relationship"] == "daughter"
    assert registry.people["Riley"]["source"] == "onboarding"
    assert registry.people["Riley"]["confidence"] == 1.0


def test_seed_registers_projects(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="work", people=[], projects=["Acme", "Widget"])
    assert registry.projects == ["Acme", "Widget"]


def test_seed_sets_mode(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="combo", people=[], projects=[])
    assert registry.mode == "combo"


def test_seed_flags_ambiguous_names(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Grace", "relationship": "friend", "context": "personal"},
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
        ],
        projects=[],
    )
    assert "grace" in registry.ambiguous_flags
    # Riley is not a common English word
    assert "riley" not in registry.ambiguous_flags


def test_seed_with_aliases(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Maxwell", "relationship": "friend", "context": "personal"}],
        projects=[],
        aliases={"Max": "Maxwell"},
    )
    assert "Maxwell" in registry.people
    assert "Max" in registry.people
    assert registry.people["Max"].get("canonical") == "Maxwell"


def test_seed_skips_empty_names(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "", "relationship": "", "context": "personal"}],
        projects=[],
    )
    assert len(registry.people) == 0


# ── lookup ──────────────────────────────────────────────────────────────


def test_lookup_known_person(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Riley")
    assert result["type"] == "person"
    assert result["confidence"] == 1.0
    assert result["name"] == "Riley"


def test_lookup_known_project(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="work", people=[], projects=["MemPalace"])
    result = registry.lookup("MemPalace")
    assert result["type"] == "project"
    assert result["confidence"] == 1.0


def test_lookup_unknown_word(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])
    result = registry.lookup("Xyzzy")
    assert result["type"] == "unknown"
    assert result["confidence"] == 0.0


def test_lookup_case_insensitive(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("riley")
    assert result["type"] == "person"


def test_lookup_alias(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Maxwell", "relationship": "friend", "context": "personal"}],
        projects=[],
        aliases={"Max": "Maxwell"},
    )
    result = registry.lookup("Max")
    assert result["type"] == "person"


# ── disambiguation ──────────────────────────────────────────────────────


def test_lookup_ambiguous_word_as_person(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Grace", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Grace", context="I went with Grace today")
    assert result["type"] == "person"


def test_lookup_ambiguous_word_as_concept(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Ever", "relationship": "friend", "context": "personal"}],
        projects=[],
    )
    result = registry.lookup("Ever", context="have you ever tried this")
    assert result["type"] == "concept"


# ── research — local-only by default ───────────────────────────────────


def test_research_local_only_by_default(tmp_path):
    """research() must NOT call Wikipedia unless allow_network=True."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        side_effect=AssertionError("network call should not happen"),
    ):
        result = registry.research("Saoirse")

    assert result["inferred_type"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["word"] == "Saoirse"
    assert "network lookup disabled" in result.get("note", "")


def test_research_with_allow_network(tmp_path):
    """research(allow_network=True) calls Wikipedia and caches result."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        return_value=dict(_MOCK_SAOIRSE_PERSON),
    ):
        result = registry.research("Saoirse", auto_confirm=True, allow_network=True)
    assert result["inferred_type"] == "person"


def test_research_caches_result(tmp_path):
    """Once cached via allow_network, subsequent calls use cache without network."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        return_value=dict(_MOCK_SAOIRSE_PERSON),
    ):
        result = registry.research("Saoirse", auto_confirm=True, allow_network=True)
    assert result["inferred_type"] == "person"

    # Second call should use cache, not call Wikipedia again
    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        side_effect=AssertionError("should not be called"),
    ):
        cached = registry.research("Saoirse")
    assert cached["inferred_type"] == "person"


def test_research_local_only_not_cached(tmp_path):
    """Local-only result for uncached word should NOT be persisted to cache."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    registry.research("Xander")  # local-only, no network
    assert "Xander" not in registry._data.get("wiki_cache", {})


def test_confirm_research_adds_to_people(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    with patch(
        "mempalace.entity_registry._wikipedia_lookup",
        return_value=dict(_MOCK_SAOIRSE_PERSON),
    ):
        registry.research("Saoirse", auto_confirm=False, allow_network=True)

    registry.confirm_research("Saoirse", entity_type="person", relationship="friend")
    assert "Saoirse" in registry.people
    assert registry.people["Saoirse"]["source"] == "wiki"


def test_wikipedia_404_returns_unknown(tmp_path):
    """A 404 from Wikipedia should return 'unknown', not assert 'person'."""
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])

    mock_result = {
        "inferred_type": "unknown",
        "confidence": 0.3,
        "wiki_summary": None,
        "wiki_title": None,
        "note": "not found in Wikipedia",
    }
    with patch("mempalace.entity_registry._wikipedia_lookup", return_value=mock_result):
        result = registry.research("Zzxqy", auto_confirm=False, allow_network=True)

    assert result["inferred_type"] == "unknown"
    assert result["confidence"] < 0.5


# ── extract_people_from_query ───────────────────────────────────────────


def test_extract_people_from_query(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[
            {"name": "Riley", "relationship": "daughter", "context": "personal"},
            {"name": "Devon", "relationship": "friend", "context": "personal"},
        ],
        projects=[],
    )
    found = registry.extract_people_from_query("What did Riley say about the weather?")
    assert "Riley" in found
    assert "Devon" not in found


# ── extract_unknown_candidates ──────────────────────────────────────────


def test_extract_unknown_candidates(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[])
    unknowns = registry.extract_unknown_candidates("Saoirse went to the store")
    assert "Saoirse" in unknowns


def test_extract_unknown_candidates_skips_known(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=[],
    )
    unknowns = registry.extract_unknown_candidates("Riley went to the store")
    assert "Riley" not in unknowns


# ── summary ─────────────────────────────────────────────────────────────


def test_summary(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Riley", "relationship": "daughter", "context": "personal"}],
        projects=["MemPalace"],
    )
    s = registry.summary()
    assert "personal" in s
    assert "Riley" in s
    assert "MemPalace" in s


# ── write-boundary project normalization ────────────────────────────────


def test_coerce_project_name_accepts_str():
    assert _coerce_project_name("  infra.eden  ") == "infra.eden"


def test_coerce_project_name_extracts_dict_name():
    assert _coerce_project_name({"name": "infra.eden"}) == "infra.eden"


def test_coerce_project_name_rejects_dict_without_name():
    with pytest.raises(ValueError) as exc:
        _coerce_project_name({"id": 7})
    assert "{'id': 7}" in str(exc.value)


def test_coerce_project_name_rejects_non_str_non_dict():
    with pytest.raises(ValueError) as exc:
        _coerce_project_name(42)
    assert "42" in str(exc.value)


# ── read-side resilience against already-persisted malformed tokens ──────
#
# The write boundary now blocks creating a registry with a dict in projects
# or a non-str alias, so these tests poison ``reg._data`` directly to simulate
# a legacy / poisoned persisted doc. The read path must degrade for the one bad
# token and keep resolving every other name.


def test_lookup_skips_malformed_project_and_resolves_good_one(tmp_path, caplog):
    registry = EntityRegistry.load(config_dir=tmp_path)
    # Malformed token first so its coercion runs before the matching "good"
    # entry is reached (the loop returns early on a match).
    registry._data["projects"] = [{"oops": 1}, "good"]
    with caplog.at_level(logging.WARNING):
        result = registry.lookup("good")
    assert result["type"] == "project"
    assert result["name"] == "good"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "project" in warnings[0].getMessage()


def test_lookup_resolves_person_with_malformed_alias(tmp_path, caplog):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["people"]["P"] = {
        "aliases": [{"x": 1}, "MB"],
        "confidence": 1.0,
        "source": "onboarding",
    }
    with caplog.at_level(logging.WARNING):
        result = registry.lookup("MB")
    assert result["type"] == "person"
    assert result["name"] == "P"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "alias" in warnings[0].getMessage()


def test_lookup_canonical_still_works_with_malformed_alias(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["people"]["P"] = {
        "aliases": [{"x": 1}, "MB"],
        "confidence": 1.0,
        "source": "onboarding",
    }
    result = registry.lookup("P")
    assert result["type"] == "person"
    assert result["name"] == "P"


def test_lookup_recoverable_dict_project_still_resolves(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["projects"] = [{"name": "infra.eden"}]
    result = registry.lookup("infra.eden")
    assert result["type"] == "project"
    assert result["name"] == "infra.eden"


def test_lookup_recoverable_dict_alias_still_resolves(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["people"]["P"] = {
        "aliases": [{"name": "infra.eden"}],
        "confidence": 1.0,
        "source": "onboarding",
    }
    result = registry.lookup("infra.eden")
    assert result["type"] == "person"
    assert result["name"] == "P"


def test_extract_people_from_query_skips_malformed_alias(tmp_path, caplog):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["people"]["P"] = {
        "aliases": [{"x": 1}, "MB"],
        "confidence": 1.0,
        "source": "onboarding",
    }
    with caplog.at_level(logging.WARNING):
        found = registry.extract_people_from_query("what did MB say today")
    assert "P" in found
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "alias" in warnings[0].getMessage()


# ── seed() write-boundary validation ────────────────────────────────────


def test_seed_coerces_dict_project_to_string(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(mode="personal", people=[], projects=[{"name": "x"}], aliases={})
    assert registry.projects == ["x"]
    assert all(isinstance(p, str) for p in registry.projects)


def test_seed_raises_on_unrecoverable_project(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    with pytest.raises(ValueError) as exc:
        registry.seed(mode="personal", people=[], projects=[{"id": 7}], aliases={})
    assert "{'id': 7}" in str(exc.value)


# ── _merge_seed_into_registry write-boundary validation ─────────────────


def test_merge_seed_coerces_dict_project_to_string(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], [{"name": "infra.eden"}], {})
    # The dict must be normalized to a plain string in the projects list...
    assert "infra.eden" in registry.projects
    assert all(isinstance(p, str) for p in registry.projects)
    # ...so the read side resolves it without crashing on '.lower()'.
    result = registry.lookup("infra.eden")
    assert result["type"] == "project"


def test_merge_seed_raises_on_unrecoverable_project(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    with pytest.raises(ValueError) as exc:
        _merge_seed_into_registry(registry, [], [{"id": 7}], {})
    assert "{'id': 7}" in str(exc.value)


def test_merge_seed_plain_string_project_unchanged(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], ["infra.eden"], {})
    assert "infra.eden" in registry.projects
    assert registry.lookup("infra.eden")["type"] == "project"


def test_coerce_project_name_rejects_empty_and_whitespace():
    # An empty/whitespace project must be rejected, not kept as "" — otherwise
    # lookup("") would resolve an empty token as a real project. This keeps the
    # string branch consistent with the dict-name branch (both reject empty).
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValueError):
            _coerce_project_name(bad)
    with pytest.raises(ValueError):
        _coerce_project_name({"name": "   "})


def test_merge_seed_dict_then_string_project_is_idempotent(tmp_path):
    # Re-seeding the same project via the dict shape and then the bare string
    # must not create a duplicate: both coerce to the same canonical string.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], [{"name": "infra.eden"}], {})
    _merge_seed_into_registry(registry, [], ["infra.eden"], {})
    assert registry.projects.count("infra.eden") == 1
    assert registry.projects == ["infra.eden"]


def test_seed_dict_project_survives_save_reload(tmp_path):
    # The bug was about a dict reaching persisted storage; assert the coerced
    # string round-trips through save()/load(), not just the in-memory list.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], [{"name": "infra.eden"}], {})
    reloaded = EntityRegistry.load(config_dir=tmp_path)
    assert reloaded.projects == ["infra.eden"]
    assert all(isinstance(p, str) for p in reloaded.projects)
    assert reloaded.lookup("infra.eden")["type"] == "project"


# ── _merge_seed_into_registry embedded-alias handling ───────────────────


def test_merge_seed_embedded_alias_resolves_to_canonical(tmp_path):
    # An alias embedded on a person entry must register exactly like a
    # standalone {alias: canonical} entry: lookup(alias) resolves to a person
    # and the alias record points to the canonical name.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [{"name": "Markus Burger", "aliases": ["MB"]}], [], {})
    assert registry.people["MB"].get("canonical") == "Markus Burger"
    result = registry.lookup("MB")
    assert result["type"] == "person"


def test_merge_seed_embedded_alias_is_idempotent(tmp_path):
    # Re-running the identical embedded-alias merge must not duplicate "MB" in
    # the person's own aliases list (additive de-dup).
    registry = EntityRegistry.load(config_dir=tmp_path)
    seed = [{"name": "Markus Burger", "aliases": ["MB"]}]
    _merge_seed_into_registry(registry, seed, [], {})
    _merge_seed_into_registry(registry, seed, [], {})
    assert registry.people["Markus Burger"]["aliases"].count("MB") == 1


def test_merge_seed_embedded_alias_does_not_clobber_prior(tmp_path):
    # Seeding a second alias for the same person must keep the first one too.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [{"name": "X", "aliases": ["A"]}], [], {})
    _merge_seed_into_registry(registry, [{"name": "X", "aliases": ["B"]}], [], {})
    aliases = registry.people["X"]["aliases"]
    assert "A" in aliases
    assert "B" in aliases


def test_merge_seed_standalone_alias_direction_unchanged(tmp_path):
    # The {alias: canonical} param direction is preserved: lookup("MB")
    # resolves to Markus Burger, same as before embedded aliases existed.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [{"name": "Markus Burger"}], [], {"MB": "Markus Burger"})
    assert registry.people["MB"].get("canonical") == "Markus Burger"
    assert registry.lookup("MB")["type"] == "person"


def test_merge_seed_rejects_non_str_embedded_alias(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    with pytest.raises(ValueError):
        _merge_seed_into_registry(registry, [{"name": "X", "aliases": [{"x": 1}]}], [], {})


def test_merge_seed_rejects_empty_embedded_alias(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    with pytest.raises(ValueError):
        _merge_seed_into_registry(registry, [{"name": "X", "aliases": [""]}], [], {})


def test_merge_seed_rejects_non_str_standalone_alias_value(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    with pytest.raises(ValueError) as exc:
        _merge_seed_into_registry(registry, [], [], {"MB": 7})
    assert "7" in str(exc.value)


# ── kind field / project-alias type resolution ──────────────────────────


def test_merge_seed_project_alias_resolves_project(tmp_path):
    # #9 — a project alias resolves to type=project and canonicalizes the name;
    # the canonical project never leaks into the people dict.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], ["mempalace"], {"mempalace-poc": "mempalace"})
    assert registry.lookup("mempalace-poc")["type"] == "project"
    assert registry.lookup("mempalace-poc")["name"] == "mempalace"
    assert registry.lookup("mempalace")["type"] == "project"
    assert "mempalace" not in registry.people  # canonical stays in projects[] only
    assert registry.people["mempalace-poc"]["kind"] == "project"
    assert registry.people["mempalace-poc"]["canonical"] == "mempalace"


def test_common_word_project_alias_resolves_project_even_with_context(tmp_path):
    # #9b — a project alias that is a common English word must NOT be flagged
    # ambiguous and must NOT be re-typed by _disambiguate when a context is given.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], ["graceful-svc"], {"grace": "graceful-svc"})
    assert "grace" not in registry.ambiguous_flags
    result = registry.lookup("grace", context="I had grace yesterday")
    assert result["type"] == "project"
    assert result["name"] == "graceful-svc"


def test_lookup_project_alias_via_alias_token_resolves_project(tmp_path):
    # #12 sibling — mirror of test_lookup_recoverable_dict_alias_still_resolves,
    # but the record is a PROJECT alias (kind=project) and the target IS in
    # projects[], so it resolves project (the original stays person — see below).
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["projects"] = ["infra.eden"]
    registry._data["people"]["infra.eden-alias"] = {
        "aliases": ["infra.eden"],
        "canonical": "infra.eden",
        "kind": "project",
        "confidence": 1.0,
        "source": "onboarding",
    }
    result = registry.lookup("infra.eden")
    assert result["type"] == "project"
    assert result["name"] == "infra.eden"


def test_lookup_kindless_record_defaults_person(tmp_path):
    # #13 — a record with no `kind` key (legacy / forward-compat shape) reads as
    # person, preserving historical behavior.
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry._data["people"]["Dana"] = {
        "aliases": [],
        "confidence": 1.0,
        "source": "onboarding",
    }
    result = registry.lookup("Dana")
    assert result["type"] == "person"
    assert result["name"] == "Dana"


def test_merge_seed_person_and_alias_have_kind_person(tmp_path):
    # #11 — a person and their alias both carry kind=person and resolve person.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [{"name": "Markus Burger"}], [], {"MB": "Markus Burger"})
    assert registry.people["Markus Burger"]["kind"] == "person"
    assert registry.people["MB"]["kind"] == "person"
    assert registry.lookup("MB")["type"] == "person"


def test_seed_path_flag_collision_keeps_person_flag(tmp_path):
    # #14 — a project alias that is a common word AND a real person sharing the
    # lowercase: the person keeps its ambiguous flag (the project alias must not
    # suppress it), the alias is kind=project, the person is kind=person.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(
        registry,
        [{"name": "Grace"}],
        ["graceful-svc"],
        {"grace": "graceful-svc"},
    )
    assert "grace" in registry.ambiguous_flags
    assert registry.people["grace"]["kind"] == "project"
    assert registry.people["Grace"]["kind"] == "person"


def test_merge_seed_idempotent_reseed_not_skipped_by_guard(tmp_path):
    # #14b — an idempotent re-seed of an existing alias must NOT be skipped by the
    # collision guard: the 2nd seed adds a new context that the else-branch merges.
    # An over-broad "skip on any key-existence" guard would leave contexts at
    # ["personal"] and fail this test.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(
        registry, [{"name": "Markus Burger", "context": "personal"}], [], {"MB": "Markus Burger"}
    )
    _merge_seed_into_registry(
        registry, [{"name": "Markus Burger", "context": "work"}], [], {"MB": "Markus Burger"}
    )
    rec = registry.people["MB"]
    assert set(rec["contexts"]) >= {"personal", "work"}
    assert rec["canonical"] == "Markus Burger"
    assert rec["aliases"].count("Markus Burger") == 1
    assert registry.lookup("MB")["type"] == "person"


def test_merge_seed_guard_fires_on_real_person_collision(tmp_path, caplog):
    # #14c — guard fires when an alias key equals an existing real person's
    # canonical with a different incoming canonical: the person is left untouched.
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [{"name": "Riley"}], [], {})
    with caplog.at_level(logging.WARNING):
        _merge_seed_into_registry(registry, [], [], {"Riley": "SomeOther"})
    assert "canonical" not in registry.people["Riley"]
    assert registry.lookup("Riley")["type"] == "person"
    assert any("skipping alias" in r.getMessage() for r in caplog.records)


def test_seed_stamps_kind_and_sorts_flags(tmp_path):
    # seed() (onboarding REPLACE path) stamps kind and emits a sorted flag list.
    registry = EntityRegistry.load(config_dir=tmp_path)
    registry.seed(
        mode="personal",
        people=[{"name": "Markus Burger"}],
        projects=["mempalace"],
        aliases={"MB": "Markus Burger"},
    )
    assert registry.people["Markus Burger"]["kind"] == "person"
    assert registry.people["MB"]["kind"] == "person"
    assert registry.lookup("MB")["type"] == "person"
    assert registry.lookup("mempalace")["type"] == "project"
    assert registry.ambiguous_flags == sorted(registry.ambiguous_flags)


def test_check_kind_invariant_clean_and_detects_violation(tmp_path):
    registry = EntityRegistry.load(config_dir=tmp_path)
    _merge_seed_into_registry(registry, [], ["mempalace"], {"mempalace-poc": "mempalace"})
    assert registry.check_kind_invariant() == []
    # Inject a violation: a project name as a NON-alias people record.
    registry._data["people"]["mempalace"] = {
        "aliases": [],
        "confidence": 1.0,
        "source": "x",
        "kind": "person",
    }
    assert registry.check_kind_invariant() == ["mempalace"]
