"""Tests for the expanded 15-mode vision registry and VisionPromptHelper routing.

Covers:
- inventory: every registered mode has a system_prompt file on disk
- describe-modes carry length guidance for snippet output
- edit-modes carry FLUX/Qwen edit-instruction language
- VisionPromptHelper picks the right system_prompt per mode
- two-image vs single-image dispatch via TWO_IMAGE_MODES

Mocks: ``comfyui_prompt_tools.nodes.vision_prompt_helper.tensor_to_b64`` is
patched to return a canned string; the engine HTTP layer is patched via the
shared ``mock_openai_urlopen`` / ``mock_ollama_urlopen`` fixtures from
conftest. Patch targets follow Standards §4a (consuming-module path).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from comfyui_prompt_tools.nodes.vision_prompt_helper import VisionPromptHelper
from comfyui_prompt_tools.vision_prompts import (
    AVAILABLE_VISION_MODES,
    TWO_IMAGE_MODES,
    VISION_MODE_TO_FILE,
    get_vision_system_prompt,
    mode_uses_two_images,
)


EDIT_MODES = [
    "Outfit Transfer",
    "Hair Change",
    "Body Reshape",
    "Background Change",
    "Pose Change",
    "Combined Edit",
]
DESCRIBE_MODES = [
    "Describe Picture",
    "Describe Face",
    "Describe Hair",
    "Describe Body",
    "Describe Pose",
    "Describe Outfit",
    "Describe Background",
    "Describe Lighting",
    "Describe Composition",
]


class TestVisionModeInventory:
    def test_all_15_modes_have_system_prompt_file(self):
        """Every registered mode loads a non-empty rendered prompt from disk.
        Mocks: none — pure file load.
        """
        assert len(AVAILABLE_VISION_MODES) == 15
        for mode in AVAILABLE_VISION_MODES:
            rendered = get_vision_system_prompt(mode)
            assert rendered.strip(), f"Empty prompt for mode {mode!r}"

    def test_registry_partitions_into_edit_and_describe(self):
        """The 15 modes split into the documented edit / describe groups.
        Mocks: none.
        """
        assert set(AVAILABLE_VISION_MODES) == set(EDIT_MODES) | set(DESCRIBE_MODES)
        assert set(EDIT_MODES).isdisjoint(set(DESCRIBE_MODES))

    def test_two_image_modes_only_outfit_transfer(self):
        """Outfit Transfer is the sole two-image mode in v0.4.
        Mocks: none.
        """
        assert TWO_IMAGE_MODES == frozenset({"Outfit Transfer"})
        assert mode_uses_two_images("Outfit Transfer")
        assert not mode_uses_two_images("Hair Change")
        assert not mode_uses_two_images("Describe Face")


class TestDescribeModePrompts:
    @pytest.mark.parametrize("mode", DESCRIBE_MODES)
    def test_describe_modes_output_short_snippets(self, mode):
        """Describe-mode prompts instruct the LLM to output a short snippet
        (length range and 'image 1' anchor present).
        Mocks: none — string check on the rendered template.
        """
        rendered = get_vision_system_prompt(mode)
        assert "image 1" in rendered.lower()
        # Each describe template documents a word range like '20-50 words' or
        # '20-60 words'. Looking for the keyword 'words' is sufficient — the
        # exact range varies per aspect.
        assert "words" in rendered.lower()


class TestEditModePrompts:
    @pytest.mark.parametrize("mode", EDIT_MODES)
    def test_edit_modes_output_edit_instructions(self, mode):
        """Edit-mode prompts instruct an edit operation referencing image 1.
        Mocks: none — string check on the rendered template.
        """
        rendered = get_vision_system_prompt(mode)
        assert "image 1" in rendered.lower()


class TestVisionHelperRouting:
    """VisionPromptHelper sends the per-mode system_prompt to the engine.

    Each test uses the conftest mock_openai_urlopen fixture (canned 'test
    output' response). The image encoder is patched to a constant b64 string
    so we don't need real tensors.
    Mocks:
      - openai_client.urlopen via mock_openai_urlopen (HTTP)
      - vision_prompt_helper.tensor_to_b64 (returns canned 'IMG')
    """

    @pytest.fixture(autouse=True)
    def _stub_image_encoder(self):
        with patch(
            "comfyui_prompt_tools.nodes.vision_prompt_helper.tensor_to_b64",
            return_value="IMG",
        ) as m:
            yield m

    @pytest.mark.parametrize("mode", EDIT_MODES + DESCRIBE_MODES)
    def test_routes_to_correct_system_prompt(self, mode, mock_openai_urlopen):
        """The system_prompt sent to the engine matches the per-mode template."""
        helper = VisionPromptHelper()
        # Two-image modes need image_2 wired; pass any non-None placeholder.
        kwargs = {
            "intent": "test",
            "mode": mode,
            "engine": "vllm",
            "base_url": "http://x:8000/v1",
            "model": "qwen-vl",
            "temperature": 0.5,
            "jpeg_quality": 85,
            "max_edge": 1280,
            "image_1": object(),
        }
        if mode_uses_two_images(mode):
            kwargs["image_2"] = object()

        out, debug = helper.generate(**kwargs)
        assert out == "test output"

        # Check the HTTP payload contains the rendered system prompt for this mode.
        # We compare a stable substring to be robust to whitespace.
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        rendered = get_vision_system_prompt(mode)
        # Use the first 80 chars of the rendered prompt as a fingerprint.
        fingerprint = rendered.strip().split("\n", 1)[0][:80]
        assert fingerprint in body, (
            f"system prompt for mode {mode!r} did not reach the request body"
        )

    def test_two_image_mode_sends_two_images(self, mock_openai_urlopen):
        """Outfit Transfer encodes both inputs into the multipart payload."""
        helper = VisionPromptHelper()
        helper.generate(
            intent="t",
            mode="Outfit Transfer",
            engine="vllm",
            base_url="http://x:8000/v1",
            model="m",
            temperature=0.5,
            jpeg_quality=85,
            max_edge=1280,
            image_1=object(),
            image_2=object(),
        )
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        # Two inline images = two '"type": "image_url"' content parts
        assert body.count('"type": "image_url"') == 2

    def test_single_image_mode_sends_one_image(self, mock_openai_urlopen):
        """Describe Face uses only image_1, even if image_2 is provided."""
        helper = VisionPromptHelper()
        helper.generate(
            intent="t",
            mode="Describe Face",
            engine="vllm",
            base_url="http://x:8000/v1",
            model="m",
            temperature=0.5,
            jpeg_quality=85,
            max_edge=1280,
            image_1=object(),
            image_2=object(),  # silently ignored for single-image modes
        )
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        assert body.count('"type": "image_url"') == 1

    def test_two_image_mode_without_image_2_returns_error(
        self, mock_openai_urlopen
    ):
        """Outfit Transfer without image_2 returns an ERROR string and never
        contacts the engine."""
        helper = VisionPromptHelper()
        out, debug = helper.generate(
            intent="t",
            mode="Outfit Transfer",
            engine="vllm",
            base_url="http://x:8000/v1",
            model="m",
            temperature=0.5,
            jpeg_quality=85,
            max_edge=1280,
            image_1=object(),
            image_2=None,
        )
        assert out.startswith("ERROR:")
        assert "image_2" in out
        # Engine should not have been called
        assert mock_openai_urlopen.call_count == 0


class TestVisionPromptHelperInputs:
    def test_input_types_includes_engine_dropdown(self):
        """The migrated VisionPromptHelper exposes the engine selector field
        and the new image_1 / image_2 contract.
        Mocks: none.
        """
        spec = VisionPromptHelper.INPUT_TYPES()
        for field in ("engine", "base_url", "model", "temperature", "image_1"):
            assert field in spec["required"], f"missing required field {field!r}"
        assert "image_2" in spec["optional"]
        # Old fields are gone (workflow rebuild required, as agreed for v0.4)
        assert "backend" not in spec["required"]
        assert "api_url" not in spec["required"]
        assert "person_image" not in spec["required"]
        assert "outfit_image" not in spec["required"]

    def test_default_mode_is_outfit_transfer(self):
        """Outfit Transfer remains the default to ease migration of existing
        workflows that wired up two images.
        Mocks: none.
        """
        spec = VisionPromptHelper.INPUT_TYPES()
        assert spec["required"]["mode"][1]["default"] == "Outfit Transfer"


class TestEditModeIdentityMarkers:
    """Pre-release fix: edit-mode prompts used to rely on referential identity
    anchors ("the same woman from image 1"), which empirically lost identity
    in the generated image. The new prompts instruct the LLM to extract
    DESCRIPTIVE markers (ethnicity, eye shape, hair, freckles, glasses,
    facial hair, tattoos, piercings, body build) and emit them in the
    output. These smoke tests verify the new behaviour end-to-end:

    1. ``test_outfit_transfer_includes_identity_markers`` and
       ``test_hair_change_preserves_face_markers`` simulate the vision LLM
       returning a canned response in the new style, and assert the markers
       reach the node output unchanged.
    2. ``test_user_intent_overrides_identity_markers`` checks both that the
       rendered system prompt instructs the LLM to honour intent-driven
       removals positively (e.g. "remove glasses" → omit glasses, no
       negation), and that a canned response without the marker passes
       through cleanly.

    Mocks (Standards §4a):
    - ``VisionPromptHelper._call_engine`` is patched on the instance via
      ``patch.object(helper, "_call_engine", ...)`` — the patch target is
      the consuming module's instance, not the source class in
      base_prompt_node. Each test supplies its own canned response so the
      assertions can pin specific marker strings.
    - ``vision_prompt_helper.tensor_to_b64`` is stubbed via the autouse
      fixture to avoid synthesising real image tensors.
    """

    @pytest.fixture(autouse=True)
    def _stub_image_encoder(self):
        with patch(
            "comfyui_prompt_tools.nodes.vision_prompt_helper.tensor_to_b64",
            return_value="IMG",
        ) as m:
            yield m

    def _generate(self, helper, *, mode, intent, image_2=None):
        kwargs = {
            "intent": intent,
            "mode": mode,
            "engine": "vllm",
            "base_url": "http://x:8000/v1",
            "model": "qwen-vl",
            "temperature": 0.5,
            "jpeg_quality": 85,
            "max_edge": 1280,
            "image_1": object(),
        }
        if image_2 is not None:
            kwargs["image_2"] = image_2
        return helper.generate(**kwargs)

    def test_outfit_transfer_includes_identity_markers(self):
        """A canned outfit_transfer response with descriptive identity
        markers reaches the node output unchanged. Documents the contract
        between the new system prompt and downstream consumers.
        Mocks: instance-level patch.object on VisionPromptHelper._call_engine,
        plus the autouse tensor_to_b64 stub.
        """
        canned = (
            "A photorealistic waist-up fashion photograph of an East Asian "
            "woman in her late 20s with almond brown eyes, light freckles "
            "across the nose, shoulder-length wavy black hair, slender build. "
            "Preserving exact identity from image 1. She is wearing the "
            "garment from image 2: a relaxed-fit linen blazer with notched "
            "lapels and rolled sleeves. Natural daylight, sharp focus, "
            "neutral concrete background."
        )
        helper = VisionPromptHelper()
        with patch.object(helper, "_call_engine", return_value=canned):
            out, _debug = self._generate(
                helper,
                mode="Outfit Transfer",
                intent="streetwear",
                image_2=object(),
            )
        for marker in (
            "East Asian", "late 20s", "almond brown eyes",
            "freckles", "wavy black hair",
        ):
            assert marker in out, f"identity marker {marker!r} dropped from output"
        # The image-1 / image-2 routing tokens must still survive
        assert "image 1" in out and "image 2" in out

    def test_hair_change_preserves_face_markers(self):
        """A canned hair_change response keeps the face/identity markers
        but lets the hair description be replaced by the intent. Documents
        that the new prompt explicitly skips hair from the identity-anchor
        block (since hair is the changed aspect).
        Mocks: instance-level patch.object on VisionPromptHelper._call_engine,
        plus the autouse tensor_to_b64 stub.
        """
        canned = (
            "An edit of image 1: a Northern European man in his mid-30s with "
            "blue-grey eyes, fair complexion with a small scar on the left "
            "brow, short stubble, athletic build. Change the hairstyle to a "
            "tousled medium-length cut with a soft side parting. Keep the "
            "face, skin tone, eye colour, distinctive marks, stubble, outfit, "
            "pose, and background exactly as in image 1."
        )
        helper = VisionPromptHelper()
        with patch.object(helper, "_call_engine", return_value=canned):
            out, _debug = self._generate(
                helper, mode="Hair Change", intent="medium length, side part",
            )
        for marker in (
            "Northern European", "mid-30s", "blue-grey eyes",
            "small scar", "stubble", "athletic build",
        ):
            assert marker in out, f"face marker {marker!r} dropped from output"
        # Hair description IS in the output (the new style), but the prompt
        # template must instruct the LLM NOT to extract the existing hair.
        rendered = get_vision_system_prompt("Hair Change")
        assert "Do NOT extract the existing hair" in rendered

    def test_user_intent_overrides_identity_markers(self):
        """All six edit-mode prompts must instruct the LLM to drop identity
        markers when the intent removes them, using POSITIVE phrasing (omit,
        not negate). The flow test verifies that a canned response without
        the dropped marker passes through unchanged.
        Mocks: instance-level patch.object on VisionPromptHelper._call_engine,
        plus the autouse tensor_to_b64 stub.
        """
        # Template-level: every edit mode template carries the override block
        # with the positive-omit instruction.
        for mode in EDIT_MODES:
            rendered = get_vision_system_prompt(mode)
            assert "USER INTENT OVERRIDES" in rendered, (
                f"{mode!r} template missing USER INTENT OVERRIDES block"
            )
            assert "remove glasses" in rendered, (
                f"{mode!r} template missing 'remove glasses' example"
            )
            # Positive phrasing — the template tells the LLM to omit, not negate
            assert "omit" in rendered.lower(), (
                f"{mode!r} template should instruct positive omission"
            )

        # Flow-level: with intent "remove glasses", the LLM is expected to
        # produce an output that does not mention glasses. We feed a canned
        # response respecting that instruction and verify it survives.
        canned = (
            "A photorealistic waist-up fashion photograph of an East Asian "
            "woman in her late 20s with almond brown eyes, light freckles. "
            "Preserving exact identity from image 1. She is wearing the "
            "garment from image 2: a charcoal wool overcoat. Studio lighting, "
            "sharp focus."
        )
        helper = VisionPromptHelper()
        with patch.object(helper, "_call_engine", return_value=canned):
            out, _debug = self._generate(
                helper,
                mode="Outfit Transfer",
                intent="remove glasses, streetwear",
                image_2=object(),
            )
        assert "glasses" not in out.lower()
        # Negation must not have leaked in either
        assert "no glasses" not in out.lower()
        assert "without glasses" not in out.lower()


class TestVisionHelperModelPropagation:
    """The selected vision model must reach get_vision_system_prompt as
    model_name so the per-model override cascade can activate.

    Mocks: ``comfyui_prompt_tools.nodes.vision_prompt_helper.
    get_vision_system_prompt`` is patched at the consuming-module path
    (Standards §4a). The image encoder is stubbed via the autouse fixture.
    """

    @pytest.fixture(autouse=True)
    def _stub_image_encoder(self):
        with patch(
            "comfyui_prompt_tools.nodes.vision_prompt_helper.tensor_to_b64",
            return_value="IMG",
        ) as m:
            yield m

    def test_vision_helper_propagates_model_name(self, mock_openai_urlopen):
        """Calling .generate() with a recognised model passes
        ``model_name=<that model>`` to the vision-prompt loader.
        Mocks: nodes.vision_prompt_helper.get_vision_system_prompt
        (returns canned string).
        """
        helper = VisionPromptHelper()
        with patch(
            "comfyui_prompt_tools.nodes.vision_prompt_helper.get_vision_system_prompt",
            return_value="STUB VISION SYS PROMPT",
        ) as mock_loader:
            helper.generate(
                intent="t",
                mode="Describe Face",
                engine="vllm",
                base_url="http://x:8000/v1",
                model="qwen3-vl:32b",
                temperature=0.5,
                jpeg_quality=85,
                max_edge=1280,
                image_1=object(),
            )
        assert mock_loader.call_count == 1
        args, kwargs = mock_loader.call_args
        assert args == ("Describe Face",)
        assert kwargs == {
            "model_name": "qwen3-vl:32b"
        }

    def test_vision_helper_custom_prompt_does_not_call_loader(
        self, mock_openai_urlopen
    ):
        """A non-empty custom_system_prompt bypasses the loader entirely.
        Mocks: nodes.vision_prompt_helper.get_vision_system_prompt
        (asserted never called).
        """
        helper = VisionPromptHelper()
        with patch(
            "comfyui_prompt_tools.nodes.vision_prompt_helper.get_vision_system_prompt"
        ) as mock_loader:
            helper.generate(
                intent="t",
                mode="Describe Face",
                engine="vllm",
                base_url="http://x:8000/v1",
                model="qwen3-vl:32b",
                temperature=0.5,
                jpeg_quality=85,
                max_edge=1280,
                image_1=object(),
                custom_system_prompt="user-supplied vision prompt",
            )
        assert mock_loader.call_count == 0
