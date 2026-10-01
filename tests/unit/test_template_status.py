"""Unit tests for the local-copy check in ``template_status``.

Every test builds its own template directory in ``tmp_path`` and its own
digest history, so nothing here depends on what the repo currently ships
(``test_shipped_hashes.py`` covers the committed file).

Mocks
-----
``classify`` / ``scan_local_copies`` take both the directory and the hash
mapping as arguments, which keeps the tests hermetic without monkeypatching.
Where the module-level cache is involved, ``load_shipped_hashes(refresh=True)``
is used to reset it.
"""

import json

import pytest

from comfyui_prompt_tools import template_status as ts
from comfyui_prompt_tools.template_status import (
    CUSTOMIZED,
    IDENTICAL,
    LOCAL_OVERRIDE,
    OUTDATED_COPY,
    SHIPPED,
    STALE_SHIPPED_OVERRIDE,
    UNVERIFIED,
    ShippedHashes,
    build_report,
    classify,
    describe_resolved_source,
    describe_source,
    digest_text,
    load_shipped_hashes,
    normalise,
    report_local_copies,
    scan_local_copies,
)

CURRENT = "current shipped template\n"
PREVIOUS = "the version shipped last year\n"
RETIRED_OVERRIDE = "an override this package no longer ships\n"


@pytest.fixture
def prompts_dir(tmp_path):
    """A template directory holding one shipped ``mode_a.txt.example``."""
    (tmp_path / "mode_a.txt.example").write_text(CURRENT, encoding="utf-8")
    return tmp_path


@pytest.fixture
def shipped():
    """Digest history: ``mode_a`` has two versions, ``mode_a`` one retiree."""
    return ShippedHashes(
        templates={"mode_a": [digest_text(CURRENT), digest_text(PREVIOUS)]},
        retired={"mode_a": [digest_text(RETIRED_OVERRIDE)]},
    )


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_normalise_strips_trailing_newlines_and_crlf():
    """Normalisation matches what the loader feeds the LLM."""
    assert normalise("a\r\nb\r\n\n") == "a\nb"


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["x\ny", "x\ny\n", "x\r\ny\r\n", "x\ny\n\n\n"])
def test_digest_ignores_line_ending_and_trailer_noise(variant):
    """An editor rewriting line endings does not turn a copy into an edit."""
    assert digest_text(variant) == digest_text("x\ny")


@pytest.mark.unit
def test_digest_differs_for_real_content_change():
    assert digest_text("x\ny") != digest_text("x\nz")


# ---------------------------------------------------------------------------
# classify — one test per classification
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_no_local_copy_is_shipped(prompts_dir, shipped):
    """Only the ``.txt.example`` present → the shipped template is in use."""
    result = classify("mode_a", prompts_dir, shipped)
    assert result.status == SHIPPED
    assert result.path == prompts_dir / "mode_a.txt.example"
    assert result.is_local is False


@pytest.mark.unit
def test_unknown_template_reports_no_path(prompts_dir, shipped):
    """Neither file present → no path to report."""
    result = classify("nope", prompts_dir, shipped)
    assert result.status == SHIPPED
    assert result.path is None


@pytest.mark.unit
def test_copy_of_current_template_is_identical(prompts_dir, shipped):
    """A copy equal to the current shipped version needs no action."""
    (prompts_dir / "mode_a.txt").write_text(CURRENT, encoding="utf-8")
    result = classify("mode_a", prompts_dir, shipped)
    assert result.status == IDENTICAL
    assert result.is_local is True
    assert result.shadows_shipped_template is True


@pytest.mark.unit
def test_copy_of_older_template_is_outdated_copy(prompts_dir, shipped):
    """A copy equal to an *older* shipped version hides the current one."""
    (prompts_dir / "mode_a.txt").write_text(PREVIOUS, encoding="utf-8")
    assert classify("mode_a", prompts_dir, shipped).status == OUTDATED_COPY


@pytest.mark.unit
def test_edited_copy_is_customized(prompts_dir, shipped):
    """A copy matching no shipped version is a real local edit."""
    (prompts_dir / "mode_a.txt").write_text("my own wording\n", encoding="utf-8")
    assert classify("mode_a", prompts_dir, shipped).status == CUSTOMIZED


@pytest.mark.unit
def test_family_override_matching_a_retired_version_is_stale(prompts_dir, shipped):
    """A left-behind family override is reported, not treated as authored.

    It is keyed by the base template name, so the family part of the
    filename never has to appear in the committed digest history.
    """
    (prompts_dir / "mode_a.somefamily.txt").write_text(
        RETIRED_OVERRIDE, encoding="utf-8"
    )
    result = classify("mode_a.somefamily", prompts_dir, shipped)
    assert result.status == STALE_SHIPPED_OVERRIDE
    assert result.shadows_shipped_template is False


@pytest.mark.unit
def test_own_family_override_is_local_override(prompts_dir, shipped):
    """A family override the user wrote matches nothing and is left alone."""
    (prompts_dir / "mode_a.somefamily.txt").write_text(
        "hand-written override\n", encoding="utf-8"
    )
    assert classify("mode_a.somefamily", prompts_dir, shipped).status == LOCAL_OVERRIDE


@pytest.mark.unit
def test_family_override_with_own_example_classifies_normally(prompts_dir, shipped):
    """A family override that ships as ``.txt.example`` follows the main rule."""
    (prompts_dir / "mode_a.somefamily.txt.example").write_text(
        CURRENT, encoding="utf-8"
    )
    (prompts_dir / "mode_a.somefamily.txt").write_text(PREVIOUS, encoding="utf-8")
    shipped_with_override = ShippedHashes(
        templates={
            **shipped.templates,
            "mode_a.somefamily": [digest_text(CURRENT), digest_text(PREVIOUS)],
        },
        retired=shipped.retired,
    )
    result = classify("mode_a.somefamily", prompts_dir, shipped_with_override)
    assert result.status == OUTDATED_COPY


@pytest.mark.unit
def test_shared_rules_is_classified_like_any_template(tmp_path):
    """``_shared_rules`` ships as ``.txt.example``, so the main rule applies.

    A stale copy of it reaches every mode that substitutes
    ``{shared_rules}``, which is why it is not exempted from the check.
    """
    (tmp_path / "_shared_rules.txt.example").write_text(CURRENT, encoding="utf-8")
    (tmp_path / "_shared_rules.txt").write_text(PREVIOUS, encoding="utf-8")
    shipped = ShippedHashes(
        templates={"_shared_rules": [digest_text(CURRENT), digest_text(PREVIOUS)]},
        retired={},
    )
    assert classify("_shared_rules", tmp_path, shipped).status == OUTDATED_COPY


@pytest.mark.unit
def test_shipped_template_absent_from_history_is_unverified(prompts_dir):
    """A stale digest file must not make an untouched copy look customized."""
    empty = ShippedHashes(templates={"other": [digest_text(CURRENT)]}, retired={})
    (prompts_dir / "mode_a.txt").write_text(CURRENT, encoding="utf-8")
    assert classify("mode_a", prompts_dir, empty).status == UNVERIFIED


@pytest.mark.unit
def test_unreadable_copy_is_unverified(prompts_dir, shipped):
    """A file that is not valid UTF-8 is reported, not guessed at."""
    (prompts_dir / "mode_a.txt").write_bytes(b"\xff\xfe not utf-8")
    assert classify("mode_a", prompts_dir, shipped).status == UNVERIFIED


# ---------------------------------------------------------------------------
# scan_local_copies
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_scan_lists_only_local_copies_sorted(prompts_dir, shipped):
    """Examples are not local copies; results come back sorted by name."""
    (prompts_dir / "mode_a.txt").write_text(CURRENT, encoding="utf-8")
    (prompts_dir / "mode_a.somefamily.txt").write_text("own\n", encoding="utf-8")
    (prompts_dir / "mode_b.txt.example").write_text(CURRENT, encoding="utf-8")
    names = [s.basename for s in scan_local_copies(prompts_dir, shipped)]
    assert names == ["mode_a", "mode_a.somefamily"]


@pytest.mark.unit
def test_scan_of_missing_directory_is_empty(tmp_path, shipped):
    assert scan_local_copies(tmp_path / "gone", shipped) == []


# ---------------------------------------------------------------------------
# load_shipped_hashes — missing / broken file degrades with a warning
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_missing_hash_file_warns_and_yields_empty(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        result = load_shipped_hashes(tmp_path / "absent.json", refresh=True)
    assert not result
    assert "scripts/update_shipped_hashes.py" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        "{ not json",
        '[]',
        '{"retired_override_versions": {}}',
        '{"templates": {}}',
        '{"templates": {"mode_a": []}}',
        '{"templates": {"mode_a": "deadbeef"}}',
        '{"templates": {"mode_a": ["not-a-digest"]}}',
        '{"templates": {"mode_a": ["' + "z" * 64 + '"]}}',
    ],
    ids=[
        "unparseable", "not-an-object", "no-templates-section", "empty-templates",
        "empty-list", "list-expected", "short-digest", "non-hex-digest",
    ],
)
def test_broken_hash_file_warns_and_yields_empty(tmp_path, caplog, payload):
    path = tmp_path / "shipped_hashes.json"
    path.write_text(payload, encoding="utf-8")
    with caplog.at_level("WARNING"):
        result = load_shipped_hashes(path, refresh=True)
    assert not result
    assert "scripts/update_shipped_hashes.py" in caplog.text


@pytest.mark.unit
def test_valid_hash_file_round_trips(tmp_path):
    path = tmp_path / "shipped_hashes.json"
    path.write_text(
        json.dumps(
            {
                "templates": {"mode_a": [digest_text(CURRENT)]},
                "retired_override_versions": {"mode_a": [digest_text(PREVIOUS)]},
            }
        ),
        encoding="utf-8",
    )
    result = load_shipped_hashes(path, refresh=True)
    assert result.templates == {"mode_a": [digest_text(CURRENT)]}
    assert result.retired == {"mode_a": [digest_text(PREVIOUS)]}


@pytest.mark.unit
def test_copy_is_unverified_without_a_hash_file(prompts_dir):
    """No digest history → the check is skipped, nothing is mis-labelled."""
    (prompts_dir / "mode_a.txt").write_text("whatever\n", encoding="utf-8")
    result = classify("mode_a", prompts_dir, ts.EMPTY_HASHES)
    assert result.status == UNVERIFIED


@pytest.mark.unit
def test_nodes_still_load_without_a_hash_file(prompts_dir, monkeypatch, caplog):
    """The startup check never raises — the nodes must load regardless."""
    monkeypatch.setattr(ts, "SHIPPED_HASHES_FILE", prompts_dir / "absent.json")
    monkeypatch.setattr(ts, "_hashes_cache", None)
    monkeypatch.setattr(ts, "_warned_about_hashes", False)
    (prompts_dir / "mode_a.txt").write_text("whatever\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        report = report_local_copies(prompts_dir=prompts_dir, force=True)
    assert "unverified (1)" in report
    assert "scripts/update_shipped_hashes.py" in caplog.text


# ---------------------------------------------------------------------------
# build_report / report_local_copies
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_report_is_none_when_nothing_to_say(prompts_dir, shipped):
    """Identical copies and own overrides alone produce no report."""
    (prompts_dir / "mode_a.txt").write_text(CURRENT, encoding="utf-8")
    assert build_report(scan_local_copies(prompts_dir, shipped)) is None


@pytest.mark.unit
def test_report_names_every_offending_path_and_what_to_do(prompts_dir, shipped):
    """One block, one line per file, with the advice per classification."""
    (prompts_dir / "mode_a.txt").write_text(PREVIOUS, encoding="utf-8")
    (prompts_dir / "mode_a.somefamily.txt").write_text(
        RETIRED_OVERRIDE, encoding="utf-8"
    )
    report = build_report(scan_local_copies(prompts_dir, shipped))
    assert str(prompts_dir / "mode_a.txt") in report
    assert str(prompts_dir / "mode_a.somefamily.txt") in report
    assert "outdated-copy (1)" in report
    assert "stale-shipped-override (1)" in report
    assert "Delete them" in report
    assert "No file was changed or deleted" in report


@pytest.mark.unit
def test_report_counts_identical_copies_without_listing_them(prompts_dir, shipped):
    (prompts_dir / "mode_a.txt").write_text(CURRENT, encoding="utf-8")
    (prompts_dir / "mode_b.txt.example").write_text(CURRENT, encoding="utf-8")
    (prompts_dir / "mode_b.txt").write_text("edited\n", encoding="utf-8")
    shipped_both = ShippedHashes(
        templates={**shipped.templates, "mode_b": [digest_text(CURRENT)]},
        retired=shipped.retired,
    )
    report = build_report(scan_local_copies(prompts_dir, shipped_both))
    assert "1 copy is identical" in report
    assert str(prompts_dir / "mode_a.txt") not in report


@pytest.mark.unit
def test_report_warns_about_missing_image_placeholders(
    prompts_dir, shipped, monkeypatch
):
    """An edit-mode copy without ``{imgN}`` ignores the target_model setting."""
    monkeypatch.setattr(ts, "_edit_mode_templates", lambda: {"mode_a"})
    (prompts_dir / "mode_a.txt").write_text("no placeholders here\n", encoding="utf-8")
    report = build_report(scan_local_copies(prompts_dir, shipped))
    assert "no {imgN} placeholders" in report
    assert "target_model" in report


@pytest.mark.unit
def test_no_placeholder_warning_when_placeholders_are_present(
    prompts_dir, shipped, monkeypatch
):
    monkeypatch.setattr(ts, "_edit_mode_templates", lambda: {"mode_a"})
    (prompts_dir / "mode_a.txt").write_text("use {img1} and {img2}\n", encoding="utf-8")
    report = build_report(scan_local_copies(prompts_dir, shipped))
    assert "no {imgN} placeholders" not in report


@pytest.mark.unit
def test_no_placeholder_warning_for_a_describe_mode(prompts_dir, shipped, monkeypatch):
    """Describe modes carry no image reference, so the note would mislead."""
    monkeypatch.setattr(ts, "_edit_mode_templates", lambda: set())
    (prompts_dir / "mode_a.txt").write_text("no placeholders here\n", encoding="utf-8")
    report = build_report(scan_local_copies(prompts_dir, shipped))
    assert "no {imgN} placeholders" not in report


@pytest.mark.unit
def test_report_runs_once_per_process(prompts_dir, shipped, monkeypatch):
    """The block is logged on the first call only."""
    monkeypatch.setattr(ts, "_reported", False)
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    (prompts_dir / "mode_a.txt").write_text(PREVIOUS, encoding="utf-8")
    assert report_local_copies(prompts_dir=prompts_dir) is not None
    assert report_local_copies(prompts_dir=prompts_dir) is None


@pytest.mark.unit
def test_report_never_raises(prompts_dir, monkeypatch, caplog):
    """A failure inside the check must not stop the node load."""
    monkeypatch.setattr(ts, "_reported", False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ts, "scan_local_copies", boom)
    with caplog.at_level("WARNING"):
        assert report_local_copies(prompts_dir=prompts_dir) is None
    assert "disk on fire" in caplog.text


# ---------------------------------------------------------------------------
# describe_source — what the nodes put into debug_info
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_describe_source_names_the_shipped_example(prompts_dir, monkeypatch, shipped):
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    assert describe_source("mode_a", prompts_dir) == "mode_a.txt.example (shipped)"


@pytest.mark.unit
@pytest.mark.parametrize(
    "content,expected",
    [
        (CURRENT, "mode_a.txt (local copy, identical to shipped)"),
        (PREVIOUS, "mode_a.txt (local copy of an older shipped version)"),
        ("mine\n", "mode_a.txt (local copy, customized)"),
    ],
    ids=["identical", "outdated", "customized"],
)
def test_describe_source_names_the_local_copy(
    prompts_dir, monkeypatch, shipped, content, expected
):
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    (prompts_dir / "mode_a.txt").write_text(content, encoding="utf-8")
    assert describe_source("mode_a", prompts_dir) == expected


@pytest.mark.unit
def test_describe_source_handles_a_missing_template(prompts_dir, monkeypatch, shipped):
    """debug_info stays printable even for a template that is not there."""
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    assert describe_source("gone", prompts_dir) == "gone.txt (not found)"


@pytest.mark.unit
def test_describe_resolved_source_follows_the_family_cascade(
    prompts_dir, monkeypatch, shipped
):
    """The reported file is the one the run actually used."""
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    (prompts_dir / "mode_a.qwen3vl.txt").write_text("override\n", encoding="utf-8")
    result = describe_resolved_source(
        "mode_a", model_name="qwen3-vl:32b", prompts_dir=prompts_dir
    )
    assert result == "mode_a.qwen3vl.txt (local override)"


@pytest.mark.unit
def test_describe_resolved_source_without_an_override(prompts_dir, monkeypatch, shipped):
    """No override file → the base template is reported."""
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    result = describe_resolved_source(
        "mode_a", model_name="qwen3-vl:32b", prompts_dir=prompts_dir
    )
    assert result == "mode_a.txt.example (shipped)"


@pytest.mark.unit
def test_describe_resolved_source_ignores_an_unknown_family(
    prompts_dir, monkeypatch, shipped
):
    monkeypatch.setattr(ts, "_hashes_cache", shipped)
    result = describe_resolved_source(
        "mode_a", model_name="some-unmapped-tag", prompts_dir=prompts_dir
    )
    assert result == "mode_a.txt.example (shipped)"
