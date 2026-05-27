"""Unit tests for comfyui_prompt_tools.prompts.

Mocks
-----
The override-cascade tests in this module use the standard pytest
``tmp_path`` + ``monkeypatch`` pattern (Standards §4a): the loader's
``_PROMPTS_DIR`` module attribute is rebound to a temp directory seeded
with controlled fixture files, so cascade behaviour can be observed in
isolation without shipping override files in the repo.
"""

import pytest

from comfyui_prompt_tools.prompts import (
    AVAILABLE_MODES,
    MODE_TO_FILE,
    MODEL_FAMILY_PATTERNS,
    detect_family,
    get_system_prompt,
)


# ---------------------------------------------------------------------------
# Existing baseline tests (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_available_modes_count():
    """Should have exactly 10 modes (9 v0.4 + Z-Image Text-to-Image v0.5)."""
    assert len(AVAILABLE_MODES) == 10


@pytest.mark.unit
def test_all_modes_have_file_mapping():
    """Every visible mode label must map to a filename."""
    for mode in AVAILABLE_MODES:
        assert mode in MODE_TO_FILE
        assert MODE_TO_FILE[mode]  # non-empty


@pytest.mark.unit
def test_each_mode_loads_non_empty_prompt():
    """Every mode should produce a non-empty system prompt."""
    for mode in AVAILABLE_MODES:
        prompt = get_system_prompt(mode)
        assert prompt
        assert len(prompt) > 30  # sanity: not just one word


@pytest.mark.unit
def test_shared_rules_are_substituted():
    """Modes with {shared_rules} should get the placeholder replaced."""
    prompt = get_system_prompt("FLUX Kontext (Scene Edit)")
    # The marker text only exists in _shared_rules.txt
    assert "NEVER refuse" in prompt
    assert "{shared_rules}" not in prompt


@pytest.mark.unit
def test_custom_prompt_has_no_shared_rules():
    """Custom mode is intentionally minimal — no shared rules block."""
    prompt = get_system_prompt("Custom System Prompt")
    assert "{shared_rules}" not in prompt
    # And it's short on purpose
    assert len(prompt) < 200


@pytest.mark.unit
def test_unknown_mode_raises():
    with pytest.raises(KeyError):
        get_system_prompt("Bogus Mode That Does Not Exist")


@pytest.mark.unit
def test_zimage_text_to_image_mode_registered():
    """Z-Image Text-to-Image must be a registered mode and load a Z-Image-
    flavoured default template (added in v0.5)."""
    assert "Z-Image Text-to-Image" in AVAILABLE_MODES
    assert MODE_TO_FILE["Z-Image Text-to-Image"] == "zimage_text_to_image"
    prompt = get_system_prompt("Z-Image Text-to-Image")
    assert "Z-Image" in prompt
    # Z-Image style guarantees: dense natural language, no tag syntax,
    # no negation guidance, length budget mentioned.
    assert "natural language" in prompt.lower()
    assert "negation" in prompt.lower()
    # Shared rules block is wired in
    assert "{shared_rules}" not in prompt


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name, expected_family",
    [
        ("Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M", "cydonia"),
        ("qwen3-vl:8b", "qwen3vl"),
        ("qwen3-vl:32b", "qwen3vl"),
        ("qwen3.2", "qwen3"),
        ("qwen3", "qwen3"),
        ("qwen3-prompt:latest", "qwen3"),
        ("gemma2:2b", "gemma"),
        ("gemma3", "gemma"),
        ("huihui_ai/qwen2-abliterated", "abliterated"),
        ("llama3.2:3b", "llama"),
        ("llama4", "llama"),
    ],
)
def test_detect_family_known_models(model_name, expected_family):
    assert detect_family(model_name) == expected_family


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name",
    [
        None,
        "",
        "some/unknown-model",
        "mistral:7b",
        "fermi/cydonia-foo",  # lowercase 'f' — case-sensitive, no match
        "QWEN3",              # uppercase — case-sensitive, no match
    ],
)
def test_detect_family_unknown_or_empty(model_name):
    assert detect_family(model_name) is None


@pytest.mark.unit
def test_pattern_order_qwen3_vl_beats_qwen3():
    """qwen3-vl pattern is listed before qwen3 and must win for vl tags."""
    assert detect_family("qwen3-vl:8b") == "qwen3vl"
    assert detect_family("qwen3-vl") == "qwen3vl"


@pytest.mark.unit
def test_family_patterns_list_is_module_level():
    """MODEL_FAMILY_PATTERNS must be importable and non-empty so users can
    monkey-patch it from a future config layer."""
    assert isinstance(MODEL_FAMILY_PATTERNS, list)
    assert len(MODEL_FAMILY_PATTERNS) >= 1


# ---------------------------------------------------------------------------
# Override cascade
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_prompts_dir(tmp_path, monkeypatch):
    """Redirect the prompts loader at a temp dir seeded with a default file.

    Mocks ``_PROMPTS_DIR`` (the module attribute used by both ``_read_file``
    and ``_resolve_template``) so cascade tests can observe override-vs-
    default resolution without touching the real prompt library.
    """
    from comfyui_prompt_tools import prompts as prompts_module

    (tmp_path / "_shared_rules.txt").write_text("SHARED", encoding="utf-8")
    (tmp_path / "flux_kontext_couple_scene.txt").write_text(
        "DEFAULT couple scene prompt content here.", encoding="utf-8"
    )
    monkeypatch.setattr(prompts_module, "_PROMPTS_DIR", tmp_path)
    return tmp_path


@pytest.mark.unit
def test_cascade_uses_override_when_present(isolated_prompts_dir):
    """An override file `<file>.<family>.txt` must win over the default."""
    (isolated_prompts_dir / "flux_kontext_couple_scene.cydonia.txt").write_text(
        "CYDONIA override prompt content here.", encoding="utf-8"
    )
    result = get_system_prompt(
        "FLUX Kontext (Couple Scene)",
        "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
    )
    assert "CYDONIA override" in result
    assert "DEFAULT couple scene" not in result


@pytest.mark.unit
def test_cascade_falls_back_to_default_when_no_override(isolated_prompts_dir):
    """Family is detected but no override file exists -> default loads."""
    result = get_system_prompt(
        "FLUX Kontext (Couple Scene)",
        "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
    )
    assert "DEFAULT couple scene" in result


@pytest.mark.unit
def test_cascade_falls_back_when_family_unknown(isolated_prompts_dir):
    """Unknown model name -> no family -> default."""
    result = get_system_prompt(
        "FLUX Kontext (Couple Scene)", "some/unknown-model"
    )
    assert "DEFAULT couple scene" in result


@pytest.mark.unit
def test_cascade_falls_back_when_model_name_is_none(isolated_prompts_dir):
    """Explicit ``model_name=None`` skips the override probe entirely."""
    # Even with an override file present, model_name=None must yield default.
    (isolated_prompts_dir / "flux_kontext_couple_scene.cydonia.txt").write_text(
        "CYDONIA override prompt content here.", encoding="utf-8"
    )
    result = get_system_prompt("FLUX Kontext (Couple Scene)", None)
    assert "DEFAULT couple scene" in result


@pytest.mark.unit
def test_model_name_none_equals_omitted():
    """Explicit ``model_name=None`` must equal the zero-argument call."""
    mode = "FLUX Kontext (Couple Scene)"
    assert get_system_prompt(mode) == get_system_prompt(mode, None)


@pytest.mark.unit
def test_override_with_shared_rules_substitution(isolated_prompts_dir):
    """Override files honour the {shared_rules} placeholder."""
    (isolated_prompts_dir / "flux_kontext_couple_scene.cydonia.txt").write_text(
        "Override start.\n{shared_rules}\nOverride end.", encoding="utf-8"
    )
    result = get_system_prompt(
        "FLUX Kontext (Couple Scene)",
        "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
    )
    assert "Override start" in result
    assert "Override end" in result
    assert "SHARED" in result
    assert "{shared_rules}" not in result


@pytest.mark.unit
def test_unknown_mode_raises_keyerror_even_with_model_name(isolated_prompts_dir):
    """KeyError for unregistered modes must not be masked by the cascade."""
    with pytest.raises(KeyError):
        get_system_prompt("Bogus Mode", "Fermi/Cydonia-24B")


# ---------------------------------------------------------------------------
# Cydonia override cascade — public v1.1 baseline
# ---------------------------------------------------------------------------
#
# The v1.0 internal release shipped pre-written `*.cydonia.txt` overrides
# for a handful of narrative modes. The public v1.1 release does NOT ship
# any override files — users who run a Cydonia model can drop their own
# `<mode>.cydonia.txt` (or `<mode>.cydonia.txt.example`) into
# `comfyui_prompt_tools/system_prompts/` to activate the per-model cascade.
#
# These tests therefore assert the inverse property: with no override file
# shipped, the cascade must transparently fall back to the default for
# every mode. The mechanics of the cascade itself (override picked up when
# present, fallback when absent) are covered exhaustively in
# `test_prompts_example_fallback.py` with isolated fixtures.

CYDONIA_TAG = "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M"


@pytest.mark.unit
@pytest.mark.parametrize("mode", AVAILABLE_MODES)
def test_cydonia_tag_falls_back_to_default_in_shipped_library(mode):
    """No Cydonia override files ship in the public release, so the
    cascade must fall back to the default for every mode when called
    with the Cydonia tag.

    A failure here means an override file has leaked back into the
    shipped prompt library — re-run the public-release audit.

    Mocks: none — exercises the real shipped prompt library.
    """
    default = get_system_prompt(mode)
    cydonia = get_system_prompt(mode, CYDONIA_TAG)
    assert default.strip(), f"empty default for {mode!r}"
    assert default == cydonia, (
        f"Mode {mode!r} produced different output for the Cydonia tag — "
        f"a `<mode>.cydonia.txt` override file appears to have leaked "
        f"into the shipped prompt library"
    )
    # shared_rules is substituted, never leaked as a placeholder
    assert "{shared_rules}" not in default


@pytest.mark.unit
def test_pony_override_keeps_score_tag_anchor():
    """The Random Character (Pony) Cydonia override still instructs the
    LLM to emit the Pony quality-tag anchor. Tag syntax is non-negotiable
    for Pony even when the surrounding tone is narrative.

    Mocks: none — pure file load.
    """
    cydonia = get_system_prompt("Random Character (Pony)", CYDONIA_TAG)
    for anchor in ("score_9", "score_8_up", "1girl", "1boy"):
        assert anchor in cydonia, (
            f"Pony Cydonia override dropped the {anchor!r} anchor — Pony "
            f"will not render correctly without it"
        )


@pytest.mark.unit
def test_qwen_couple_override_keeps_three_image_anchors():
    """The Qwen Image Edit (Couple Scene) Cydonia override still references
    all three image tokens — Qwen Image Edit cannot resolve identities
    without them.

    Mocks: none — pure file load.
    """
    cydonia = get_system_prompt("Qwen Image Edit (Couple Scene)", CYDONIA_TAG)
    for token in ("image 1", "image 2", "image 3"):
        assert token in cydonia, (
            f"Qwen Cydonia override dropped the {token!r} reference token"
        )
