"""Saved workflows map widget values by POSITION — guard that mapping.

The ComfyUI frontend does not store widget names in a saved workflow. A
node is persisted as a flat ``widgets_values`` array and, on load, the
values are handed to the node's widgets *in order*. Insert a new input
anywhere but the end and every value after it lands on the wrong widget.

That is not hypothetical. The first draft of the prompt-catalog package put
the new ``target_model`` input before ``custom_system_prompt`` on
VisionPromptHelper. Existing nodes store 9 values ending in the custom
prompt, so on load that prompt was handed to ``target_model`` and
``custom_system_prompt`` came up empty — the node then failed validation
with ``Unknown target model: ''``, or silently sent the user's whole system
prompt as a target-model name.

The API-level tests missed it because an API prompt is a dict keyed by
input NAME; only the frontend's positional mapping is affected. So these
tests model that mapping explicitly.

The workflow fixtures below are synthetic — hand-written to the shape
ComfyUI writes, with placeholder URLs and model names only.

Mocks: none. ``INPUT_TYPES`` is read directly.
"""

from __future__ import annotations

import pytest

from comfyui_prompt_tools.catalog import resolve_image_target_model
from comfyui_prompt_tools.nodes.prompt_composer import PromptComposer
from comfyui_prompt_tools.nodes.prompt_helper import PromptHelper
from comfyui_prompt_tools.nodes.vision_prompt_helper import VisionPromptHelper

pytestmark = pytest.mark.unit

#: Input types the frontend renders as a widget (and therefore persists in
#: ``widgets_values``). Everything else — IMAGE, LATENT, … — is a link slot.
_WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}


def widget_order(node_cls) -> list[str]:
    """Return the input names in ``widgets_values`` order for ``node_cls``.

    Models the frontend: required inputs first, then optional, each in
    declaration order. A spec is a widget when its type is a list (combo)
    or one of the primitive widget types, and when it is not declared
    ``forceInput`` — that flag turns the input into a link-only socket, so
    it never occupies a ``widgets_values`` slot.

    Verified against the ``widgets_values`` ComfyUI itself wrote into the
    workflows under ``examples/``: VisionPromptHelper had 9 entries with
    ``image_1``/``image_2`` as slots, PromptComposer 7 with ``input_1..5``
    excluded, PromptHelper 15.
    """
    spec = node_cls.INPUT_TYPES()
    names: list[str] = []
    for section in ("required", "optional"):
        for name, declaration in (spec.get(section) or {}).items():
            input_type = declaration[0]
            options = declaration[1] if len(declaration) > 1 else {}
            if isinstance(options, dict) and options.get("forceInput"):
                continue
            if isinstance(input_type, list) or input_type in _WIDGET_TYPES:
                names.append(name)
    return names


def map_saved_values(node_cls, widgets_values: list) -> dict[str, object]:
    """Zip a saved ``widgets_values`` array onto the current widget order.

    This is what the frontend effectively does. Values beyond the current
    widget count are dropped; widgets beyond the saved count keep their
    default and are absent from the result.
    """
    return dict(zip(widget_order(node_cls), widgets_values))


# ---------------------------------------------------------------------------
# Synthetic pre-v1.2.0 workflow nodes
# ---------------------------------------------------------------------------
#: A VisionPromptHelper as saved before ``target_model`` existed: 9 values,
#: the last one the user's custom system prompt. image_1 / image_2 are link
#: slots and contribute nothing.
OLD_VISION_NODE = {
    "type": "VisionPromptHelper",
    "inputs": [{"name": "image_1"}, {"name": "image_2"}],
    # The widget order as of v1.1.4 — history, not derived from the code.
    "old_widget_order": [
        "intent", "mode", "engine", "base_url", "model", "temperature",
        "jpeg_quality", "max_edge", "custom_system_prompt",
    ],
    "widgets_values": [
        "transfer the outfit, keep the face",   # intent
        "Outfit Transfer",                      # mode
        "ollama",                               # engine
        "http://localhost:11434",               # base_url
        "[ollama] some-vision-model",            # model
        0.7,                                    # temperature
        85,                                     # jpeg_quality
        1280,                                   # max_edge
        "",                                     # custom_system_prompt
    ],
}

#: The same node, but with a custom system prompt actually filled in — the
#: case where the misalignment did visible damage instead of just failing.
OLD_VISION_NODE_WITH_CUSTOM_PROMPT = {
    **OLD_VISION_NODE,
    "widgets_values": OLD_VISION_NODE["widgets_values"][:-1]
    + ["You are a describer. Output one sentence."],
}

#: A PromptHelper as saved before this package: 15 values, old mode label.
OLD_HELPER_NODE = {
    "type": "PromptHelper",
    "inputs": [],
    "old_widget_order": [
        "prompt", "mode", "engine", "base_url", "model", "temperature",
        "keep_alive", "custom_system_prompt", "ethnicity_pool", "age_pool",
        "mood_pool", "hair_pool", "lighting_pool", "setting_pool",
        "outfit_style_pool",
    ],
    "widgets_values": [
        "a woman in a cafe",                    # prompt
        "Z-Image Text-to-Image",                # mode (former label)
        "ollama",                               # engine
        "http://localhost:11434",               # base_url
        "[ollama] some-text-model",              # model
        0.3,                                    # temperature
        "30s",                                  # keep_alive
        "",                                     # custom_system_prompt
        "European\nEast Asian",                 # ethnicity_pool
        "22-40",                                # age_pool
        "confident\nserene",                    # mood_pool
        "long straight\nbob cut",               # hair_pool
        "",                                     # lighting_pool
        "",                                     # setting_pool
        "",                                     # outfit_style_pool
    ],
}

#: A PromptComposer as saved before this package: 7 values, old style label.
#: input_1..input_5 are forceInput slots and contribute nothing.
OLD_COMPOSER_NODE = {
    "type": "PromptComposer",
    "inputs": [{"name": f"input_{i}"} for i in range(1, 6)],
    "old_widget_order": [
        "engine", "base_url", "model", "temperature", "user_instruction",
        "output_style", "lora_keywords",
    ],
    "widgets_values": [
        "ollama",                               # engine
        "http://localhost:11434",               # base_url
        "[ollama] some-text-model",              # model
        0.3,                                    # temperature
        "she walks toward the camera",           # user_instruction
        "Wan 2.2 motion",                       # output_style (former label)
        "",                                     # lora_keywords
    ],
}


# ---------------------------------------------------------------------------
# The widget order itself
# ---------------------------------------------------------------------------
class TestWidgetOrder:
    def test_vision_helper_appends_target_model_last(self):
        """``target_model`` must be the final widget.

        This is the whole fix: appended, a stored 9-value array still maps
        1:1 and ``target_model`` simply has no saved value.
        """
        assert widget_order(VisionPromptHelper)[-1] == "target_model"

    def test_vision_helper_widget_order(self):
        assert widget_order(VisionPromptHelper) == [
            "intent",
            "mode",
            "engine",
            "base_url",
            "model",
            "temperature",
            "jpeg_quality",
            "max_edge",
            "custom_system_prompt",
            "target_model",
        ]

    def test_image_inputs_are_slots_not_widgets(self):
        order = widget_order(VisionPromptHelper)
        assert "image_1" not in order
        assert "image_2" not in order

    def test_composer_force_input_slots_are_not_widgets(self):
        order = widget_order(PromptComposer)
        for i in range(1, 6):
            assert f"input_{i}" not in order
        assert order == [
            "engine",
            "base_url",
            "model",
            "temperature",
            "user_instruction",
            "output_style",
            "lora_keywords",
        ]

    def test_prompt_helper_widget_order(self):
        assert widget_order(PromptHelper) == [
            "prompt",
            "mode",
            "engine",
            "base_url",
            "model",
            "temperature",
            "keep_alive",
            "custom_system_prompt",
            "ethnicity_pool",
            "age_pool",
            "mood_pool",
            "hair_pool",
            "lighting_pool",
            "setting_pool",
            "outfit_style_pool",
        ]


# ---------------------------------------------------------------------------
# Positional mapping of saved workflows
# ---------------------------------------------------------------------------
class TestSavedWorkflowMapping:
    def test_old_vision_node_keeps_custom_system_prompt_in_place(self):
        """The regression: the 9th stored value is the custom system prompt.

        Before the fix it landed in ``target_model``.
        """
        mapped = map_saved_values(VisionPromptHelper, OLD_VISION_NODE["widgets_values"])
        assert mapped["custom_system_prompt"] == ""
        assert "target_model" not in mapped

    def test_old_vision_node_with_custom_prompt_is_not_shifted(self):
        values = OLD_VISION_NODE_WITH_CUSTOM_PROMPT["widgets_values"]
        mapped = map_saved_values(VisionPromptHelper, values)
        assert mapped["custom_system_prompt"] == (
            "You are a describer. Output one sentence."
        )
        assert "target_model" not in mapped

    def test_old_vision_node_maps_every_value_to_its_own_input(self):
        mapped = map_saved_values(VisionPromptHelper, OLD_VISION_NODE["widgets_values"])
        assert mapped == {
            "intent": "transfer the outfit, keep the face",
            "mode": "Outfit Transfer",
            "engine": "ollama",
            "base_url": "http://localhost:11434",
            "model": "[ollama] some-vision-model",
            "temperature": 0.7,
            "jpeg_quality": 85,
            "max_edge": 1280,
            "custom_system_prompt": "",
        }

    def test_old_helper_node_maps_every_value_to_its_own_input(self):
        mapped = map_saved_values(PromptHelper, OLD_HELPER_NODE["widgets_values"])
        assert mapped["prompt"] == "a woman in a cafe"
        assert mapped["mode"] == "Z-Image Text-to-Image"
        assert mapped["keep_alive"] == "30s"
        assert mapped["custom_system_prompt"] == ""
        assert mapped["age_pool"] == "22-40"
        assert mapped["outfit_style_pool"] == ""
        assert len(mapped) == len(OLD_HELPER_NODE["widgets_values"])

    def test_old_composer_node_maps_every_value_to_its_own_input(self):
        mapped = map_saved_values(PromptComposer, OLD_COMPOSER_NODE["widgets_values"])
        assert mapped["user_instruction"] == "she walks toward the camera"
        assert mapped["output_style"] == "Wan 2.2 motion"
        assert mapped["lora_keywords"] == ""
        assert len(mapped) == len(OLD_COMPOSER_NODE["widgets_values"])

    @pytest.mark.parametrize(
        "node_cls,node",
        [
            (VisionPromptHelper, OLD_VISION_NODE),
            (PromptHelper, OLD_HELPER_NODE),
            (PromptComposer, OLD_COMPOSER_NODE),
        ],
    )
    def test_saved_arrays_are_never_longer_than_the_widget_list(self, node_cls, node):
        """A stored array longer than the widget list means an input was
        removed — the remaining values would all shift."""
        assert len(node["widgets_values"]) <= len(widget_order(node_cls))

    @pytest.mark.parametrize(
        "node_cls,node",
        [
            (VisionPromptHelper, OLD_VISION_NODE),
            (PromptHelper, OLD_HELPER_NODE),
            (PromptComposer, OLD_COMPOSER_NODE),
        ],
    )
    def test_pre_v120_widget_order_is_a_prefix_of_the_current_one(
        self, node_cls, node
    ):
        """The invariant that keeps saved workflows loadable.

        A new input may only be APPENDED. If the pre-v1.2.0 order is still
        an exact prefix of today's, every stored value lands on the input it
        was written for and the new inputs simply have no stored value. Any
        insertion, removal or reordering breaks the prefix — and this test.
        """
        old_order = node["old_widget_order"]
        current = widget_order(node_cls)
        assert current[: len(old_order)] == old_order
        assert len(node["widgets_values"]) == len(old_order)


# ---------------------------------------------------------------------------
# What the node does with the absent / empty value
# ---------------------------------------------------------------------------
class TestUnsetTargetModel:
    def test_absent_target_model_resolves_to_generic(self):
        """A pre-v1.2.0 node has no stored value at all."""
        assert resolve_image_target_model(None).id == "generic"

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_blank_target_model_resolves_to_generic(self, value):
        """A blank string is what a positional shift or a hand-edited
        workflow can deliver — treat it as "not set", not as invalid."""
        assert resolve_image_target_model(value).id == "generic"

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_validate_inputs_accepts_an_unset_target_model(self, value):
        assert VisionPromptHelper.VALIDATE_INPUTS("Outfit Transfer", value) is True

    def test_target_model_default_is_generic(self):
        spec = VisionPromptHelper.INPUT_TYPES()["optional"]["target_model"]
        assert spec[1]["default"] == "Generic"
