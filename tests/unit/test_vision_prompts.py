"""Unit tests for comfyui_prompt_tools.vision_prompts.

Baseline tests run against the real ``system_prompts/`` directory (no mocks).
Cascade tests redirect ``prompts._PROMPTS_DIR`` to a temp directory via
``monkeypatch`` so per-model override resolution can be observed in
isolation (Standards §4a — patch at the module attribute used by the
shared ``render_template`` helper).
"""

import pytest

from comfyui_prompt_tools.vision_prompts import (
    AVAILABLE_VISION_MODES,
    VISION_MODE_TO_FILE,
    get_vision_system_prompt,
)


class TestVisionModeRegistry:
    def test_outfit_transfer_in_registry(self):
        """Outfit Transfer is registered as a known vision mode.
        Mocks: none.
        """
        assert "Outfit Transfer" in VISION_MODE_TO_FILE

    def test_available_modes_matches_registry(self):
        """AVAILABLE_VISION_MODES lists exactly the keys in VISION_MODE_TO_FILE.
        Mocks: none.
        """
        assert set(AVAILABLE_VISION_MODES) == set(VISION_MODE_TO_FILE.keys())

    def test_unknown_mode_raises_key_error(self):
        """get_vision_system_prompt raises KeyError for an unregistered mode.
        Mocks: none.
        """
        with pytest.raises(KeyError, match="Unknown vision prompt mode"):
            get_vision_system_prompt("Hair Transfer")


class TestOutfitTransferPrompt:
    def test_prompt_loads_non_empty(self):
        """Outfit Transfer system prompt loads and is non-empty.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outfit Transfer")
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0

    def test_shared_rules_placeholder_is_substituted(self):
        """The {shared_rules} placeholder is replaced — not present in the output.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outfit Transfer")
        assert "{shared_rules}" not in prompt

    def test_shared_rules_content_is_injected(self):
        """Content from _shared_rules.txt appears verbatim in the rendered prompt.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outfit Transfer")
        # A stable phrase from _shared_rules.txt
        assert "Output ONLY the enhanced prompt" in prompt

    def test_prompt_contains_flux_reference(self):
        """Rendered prompt contains the FLUX.2 model reference from the template.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outfit Transfer")
        assert "FLUX.2" in prompt

    def test_prompt_describes_two_images(self):
        """Rendered prompt names both input images by position.

        The wording that tells the vision LLM which input is which is fixed
        text ("the first image" / "the second image") and does not depend on
        the target model.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outfit Transfer")
        assert "The first image shows the IDENTITY SOURCE" in prompt
        assert "The second image shows the OUTFIT SOURCE" in prompt

    def test_outfit_transfer_contains_image_role_tokens(self):
        """The outfit-transfer template must reference 'image 1' and 'image 2'
        as literal tokens — this is the core multi-reference convention.
        Mocks: nothing — pure file load.
        """
        prompt = get_vision_system_prompt("Outfit Transfer")
        assert "image 1" in prompt
        assert "image 2" in prompt


# ---------------------------------------------------------------------------
# Per-model override cascade
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_prompts_dir(tmp_path, monkeypatch):
    """Redirect the shared prompts loader at a temp dir seeded with a
    minimal default vision-prompt file.

    Mocks: ``prompts._PROMPTS_DIR`` (the module attribute used by the
    shared ``render_template`` helper) is rebound to ``tmp_path`` so the
    cascade can be observed without touching the real prompt library.
    """
    from comfyui_prompt_tools import prompts as prompts_module

    (tmp_path / "_shared_rules.txt").write_text("SHARED", encoding="utf-8")
    (tmp_path / "vision_outfit_transfer.txt").write_text(
        "DEFAULT outfit transfer vision prompt.", encoding="utf-8"
    )
    monkeypatch.setattr(prompts_module, "_PROMPTS_DIR", tmp_path)
    return tmp_path


class TestVisionOverrideCascade:
    def test_cascade_uses_override_when_present(self, isolated_prompts_dir):
        """An ``<file>.<family>.txt`` override wins over the default.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        (isolated_prompts_dir / "vision_outfit_transfer.qwen3vl.txt").write_text(
            "QWEN3VL vision override prompt.", encoding="utf-8"
        )
        result = get_vision_system_prompt(
            "Outfit Transfer",
            "qwen3-vl:32b",
        )
        assert "QWEN3VL vision override" in result
        assert "DEFAULT outfit transfer" not in result

    def test_cascade_falls_back_to_default_when_no_override(
        self, isolated_prompts_dir
    ):
        """Family detected but no override file exists -> default loads.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        result = get_vision_system_prompt(
            "Outfit Transfer",
            "qwen3-vl:32b",
        )
        assert "DEFAULT outfit transfer" in result

    def test_cascade_falls_back_when_family_unknown(self, isolated_prompts_dir):
        """Unknown model name -> no family detected -> default loads.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        result = get_vision_system_prompt("Outfit Transfer", "some/unknown-model")
        assert "DEFAULT outfit transfer" in result

    def test_cascade_falls_back_when_model_name_is_none(
        self, isolated_prompts_dir
    ):
        """``model_name=None`` skips the override probe even if a matching
        override file exists.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        (isolated_prompts_dir / "vision_outfit_transfer.qwen3vl.txt").write_text(
            "QWEN3VL vision override prompt.", encoding="utf-8"
        )
        result = get_vision_system_prompt("Outfit Transfer", None)
        assert "DEFAULT outfit transfer" in result

    def test_model_name_none_equals_omitted(self):
        """Explicit ``model_name=None`` must equal the zero-argument call.
        Mocks: none — pure equality check against the real prompt library.
        """
        assert get_vision_system_prompt("Outfit Transfer") == (
            get_vision_system_prompt("Outfit Transfer", None)
        )

    def test_override_with_shared_rules_substitution(self, isolated_prompts_dir):
        """Override files honour the ``{shared_rules}`` placeholder.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        (isolated_prompts_dir / "vision_outfit_transfer.qwen3vl.txt").write_text(
            "Override head.\n{shared_rules}\nOverride tail.", encoding="utf-8"
        )
        result = get_vision_system_prompt(
            "Outfit Transfer",
            "qwen3-vl:32b",
        )
        assert "Override head" in result
        assert "Override tail" in result
        assert "SHARED" in result
        assert "{shared_rules}" not in result

    def test_unknown_mode_raises_keyerror_even_with_model_name(
        self, isolated_prompts_dir
    ):
        """KeyError for unregistered vision modes must not be masked by
        the cascade.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        with pytest.raises(KeyError, match="Unknown vision prompt mode"):
            get_vision_system_prompt("Hair Transfer", "qwen3-vl:32b")


class TestDescribePicturePrompt:
    """v0.6: holistic 'Describe Picture' mode for video-pipeline use."""

    def test_describe_picture_in_registry(self):
        """Describe Picture is registered as a known vision mode.
        Mocks: none.
        """
        assert "Describe Picture" in VISION_MODE_TO_FILE
        assert VISION_MODE_TO_FILE["Describe Picture"] == "vision_describe_picture"

    def test_describe_picture_loads_non_empty(self):
        """Describe Picture system prompt loads and is non-empty.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Describe Picture")
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 100  # holistic template is several sentences

    def test_describe_picture_shared_rules_substituted(self):
        """The {shared_rules} placeholder is replaced in the rendered output.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Describe Picture")
        assert "{shared_rules}" not in prompt

    def test_describe_picture_template_fingerprints(self):
        """Stable fingerprint strings from the template body — surface
        accidental file truncation in CI.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Describe Picture")
        assert "4–6 sentences" in prompt
        assert "downstream" in prompt.lower()


class TestOutpaintZoomOutPrompt:
    """v1.2: 'Outpaint (Zoom Out)' extends a crop into a full-body shot.

    The mode exists because the image model invents everything outside the
    original crop and follows the prompt literally — an unspecified lower
    garment comes back as bare legs. The template therefore has to force a
    COMPLETE outfit and a plausible build, not a description of what is
    visible. These tests pin that contract.
    """

    #: What `{img1}` has to become per image-capable target model.
    EXPECTED_IMAGE_REF = {
        "Generic": "image 1",
        "FLUX.2": "image 1",
        "FLUX Kontext": "image 1",
        "Krea 2": "Picture 1",
        "Qwen-Image-Edit 2511": "Picture 1",
        "Qwen-Image 2.1": "<image1>",
    }

    def test_mode_is_registered(self):
        """The label appears in the derived list and maps to its basename.
        Mocks: none.
        """
        assert "Outpaint (Zoom Out)" in VISION_MODE_TO_FILE
        assert VISION_MODE_TO_FILE["Outpaint (Zoom Out)"] == (
            "vision_outpaint_zoom_out"
        )
        assert "Outpaint (Zoom Out)" in AVAILABLE_VISION_MODES

    def test_prompt_loads_non_empty_with_shared_rules_substituted(self):
        """The template renders and the ``{shared_rules}`` placeholder is gone.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outpaint (Zoom Out)")
        assert len(prompt.strip()) > 100
        assert "{shared_rules}" not in prompt
        assert "Output ONLY the enhanced prompt" in prompt

    def test_template_fingerprints(self):
        """Stable strings from the template body — a truncated file surfaces.

        Each one carries a decision from the briefing: the crop-to-full-body
        job, the complete outfit down to the footwear, the ban on numeric
        measurements, and the scene continuing outward.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outpaint (Zoom Out)")
        for fingerprint in (
            "FULL-BODY photograph",
            "70–130 words",
            "DESCRIBE THE WHOLE PERSON",
            "OUTFIT, COMPLETE",
            "never numbers",
            "head to feet",
        ):
            assert fingerprint in prompt, f"missing fingerprint {fingerprint!r}"

    def test_instruction_to_the_llm_names_the_input_positionally(self):
        """Which input is which is fixed prose, independent of the target model.
        Mocks: none.
        """
        prompt = get_vision_system_prompt("Outpaint (Zoom Out)")
        assert "The first image shows only a CROP of a person" in prompt

    @pytest.mark.parametrize("label", sorted(EXPECTED_IMAGE_REF))
    def test_image_reference_renders_per_target_model(self, label):
        """``{img1}`` becomes the wording the selected image model expects.
        Mocks: none.
        """
        rendered = get_vision_system_prompt(
            "Outpaint (Zoom Out)", target_model=label
        )
        assert self.EXPECTED_IMAGE_REF[label] in rendered

    @pytest.mark.parametrize("label", sorted(EXPECTED_IMAGE_REF))
    def test_no_unrendered_placeholder_survives(self, label):
        """Not a single ``{img`` may reach the LLM — it would be echoed verbatim.
        Mocks: none.
        """
        rendered = get_vision_system_prompt(
            "Outpaint (Zoom Out)", target_model=label
        )
        assert "{img" not in rendered

    def test_every_image_capable_target_model_is_covered(self):
        """Guard: a new target model must be added to the table above.
        Mocks: none.
        """
        from comfyui_prompt_tools import catalog as catalog_mod

        assert set(catalog_mod.image_target_model_labels()) == set(
            self.EXPECTED_IMAGE_REF
        )
