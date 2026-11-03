"""Tests for PromptComposer's ``base_prompt`` / ``base_prompt_rest`` inputs.

Covers both modes end to end through ``compose()``, the byte-identical
behaviour with an empty base prompt, error handling for an unreadable edit
list, the LoRA-token guarantee and the override cascade of the new template.
The pure edit helpers have their own file (``test_base_prompt_edits.py``).

Mocks:
- ``PromptComposer._call_engine`` replaced with a ``MagicMock`` returning a
  canned reply, so no HTTP is issued and the exact system prompt / user
  message can be inspected. ``_resolve_engine`` runs for real (it only
  builds a client object).
- ``prompts._PROMPTS_DIR`` redirected to ``tmp_path`` for the cascade tests.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from comfyui_prompt_tools.base_prompt_edits import apply_edits, parse_edit_list
from comfyui_prompt_tools.catalog import PROMPTS_DIR
from comfyui_prompt_tools.nodes.prompt_composer import (
    BASE_PROMPT_ADAPT,
    BASE_PROMPT_KEEP,
    BASE_PROMPT_REPLACE_TEMPLATE,
    BASE_PROMPT_REST_CHOICES,
    PromptComposer,
    _build_base_prompt_directive,
    _build_lora_keywords_directive,
    _load_composer_system_prompt,
    _remove_back_references,
)
from comfyui_prompt_tools.prompts import render_template

pytestmark = pytest.mark.unit

BASE = (
    "a woman with long red curly hair in a yellow summer dress walking "
    "through a foggy forest at dawn, 35mm film"
)
STYLE = "FLUX.2 natural language"  # former label, resolves via alias


def _kwargs(**overrides):
    base = {
        "engine": "vllm",
        "base_url": "http://x:8000/v1",
        "model": "some-model",
        "temperature": 0.3,
        "user_instruction": "",
        "output_style": STYLE,
    }
    base.update(overrides)
    return base


def _run(reply, **overrides):
    """Run ``compose`` with a canned LLM reply; return (output, debug, mock)."""
    mock = MagicMock(return_value=reply)
    with patch.object(PromptComposer, "_call_engine", mock):
        out, debug = PromptComposer().compose(**_kwargs(**overrides))
    return out, debug, mock


def _sent(mock):
    """``(system_prompt, user_message)`` of the single engine call."""
    assert mock.call_count == 1
    kwargs = mock.call_args.kwargs
    return kwargs["system_prompt"], kwargs["user_message"]


def _edit_list(replacements=(), additions=()):
    return json.dumps(
        {"replacements": list(replacements), "additions": list(additions)}
    )


# ---------------------------------------------------------------------------
# Empty base prompt = behaviour of main, byte for byte
# ---------------------------------------------------------------------------
#: User message exactly as main (5c0c10d) built it for these inputs —
#: captured from that commit, not derived from the current code.
MAIN_USER_MESSAGE = (
    "USER INSTRUCTION:\nwarm dinner\n\nINPUT DESCRIPTIONS:\n[1] face a\n"
    "[2] outfit b\n[3] bg c\n\nCompose ONE prompt that fuses these inputs "
    "with the user instruction, in the required output style.\n/no_think"
)


class TestEmptyBasePromptUnchanged:
    INPUTS = {"input_1": "face a", "input_3": "  ", "input_4": "outfit b", "input_5": "bg c"}

    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"base_prompt": ""},
            {"base_prompt": "   \n "},
            {"base_prompt": None},
            {"base_prompt": "", "base_prompt_rest": BASE_PROMPT_ADAPT},
        ],
        ids=["absent", "empty", "whitespace", "none", "empty-adapt"],
    )
    def test_payload_is_byte_identical_to_main(self, extra):
        out, debug, mock = _run(
            "fused prompt",
            user_instruction="warm dinner",
            lora_keywords="lora_a, <lora:x:0.8>",
            **self.INPUTS,
            **extra,
        )
        system_prompt, user_message = _sent(mock)
        assert user_message == MAIN_USER_MESSAGE
        assert system_prompt == _load_composer_system_prompt(
            STYLE, model_name="some-model"
        ) + _build_lora_keywords_directive(["lora_a", "<lora:x:0.8>"])
        assert out == "fused prompt, lora_a, <lora:x:0.8>"
        assert "Base prompt" not in debug

    def test_nothing_to_compose_error_unchanged(self):
        out, debug, mock = _run("x", base_prompt="")
        assert out.startswith("ERROR: no user_instruction and no inputs")
        assert mock.call_count == 0


# ---------------------------------------------------------------------------
# keep verbatim
# ---------------------------------------------------------------------------
class TestKeepVerbatim:
    def test_default_mode_is_keep_verbatim(self):
        reply = _edit_list([{"find": "long red curly hair", "replace": "short platinum bob"}])
        out, debug, mock = _run(reply, base_prompt=BASE, input_1="short platinum bob")
        assert out == (
            "a woman with short platinum bob in a yellow summer dress walking "
            "through a foggy forest at dawn, 35mm film"
        )
        assert "Base prompt: verbatim | Replaced: 1 | Added: 0 | Skipped: 0" in debug

    def test_uses_the_replace_template_not_the_style(self):
        _, debug, mock = _run(
            _edit_list(), base_prompt=BASE, base_prompt_rest=BASE_PROMPT_KEEP,
            input_1="x",
        )
        system_prompt, _ = _sent(mock)
        assert system_prompt == render_template(BASE_PROMPT_REPLACE_TEMPLATE, "some-model")
        assert f"Template: {BASE_PROMPT_REPLACE_TEMPLATE}.txt" in debug
        assert "(not used)" in debug

    def test_user_message_carries_base_inputs_and_instruction(self):
        _, _, mock = _run(
            _edit_list(),
            base_prompt=f"  {BASE}\n",
            user_instruction="keep the background",
            input_2="short platinum bob",
            input_4="black coat",
        )
        _, user_message = _sent(mock)
        assert user_message == (
            f"BASE PROMPT:\n{BASE}\n\nINPUT DESCRIPTIONS:\n[1] short platinum bob\n"
            "[2] black coat\n\nUSER INSTRUCTION:\nkeep the background\n\n"
            "Return the JSON edit list only.\n/no_think"
        )

    def test_instruction_without_inputs_still_calls_llm(self):
        _, _, mock = _run(_edit_list(), base_prompt=BASE, user_instruction="make it night")
        _, user_message = _sent(mock)
        assert "INPUT DESCRIPTIONS:\n(none)" in user_message

    def test_no_inputs_no_instruction_returns_base_without_llm(self):
        out, debug, mock = _run("unused", base_prompt=f" {BASE} ")
        assert out == BASE
        assert mock.call_count == 0
        assert "LLM not called" in debug

    def test_counts_and_skip_breakdown(self):
        reply = _edit_list(
            [
                {"find": "long red curly hair", "replace": "a bob"},
                {"find": "purple hat", "replace": "x"},
                {"find": "red curly", "replace": "y"},
            ],
            [{"after": "summer dress", "insert": "and silver earrings"}],
        )
        out, debug, _ = _run(reply, base_prompt=BASE, input_1="bob", input_2="earrings")
        assert "silver earrings, walking" in out
        assert (
            "Replaced: 1 | Added: 1 | Skipped: 2 "
            "(not found: 1, overlap: 1, invalid: 0)"
        ) in debug

    @pytest.mark.parametrize(
        "reply",
        [
            "<think>hmm</think>\n```json\n"
            + _edit_list([{"find": "35mm film", "replace": "digital"}])
            + "\n```",
            "Sure! " + _edit_list([{"find": "35mm film", "replace": "digital"}]),
        ],
        ids=["think+fence", "prose"],
    )
    def test_noisy_reply_is_read(self, reply):
        out, _, _ = _run(reply, base_prompt=BASE, input_1="digital")
        assert out.endswith("at dawn, digital")

    @pytest.mark.parametrize(
        "reply",
        [
            "a woman with a short bob in a yellow dress",  # a free rewrite
            '{"replacements": [{"find": "x"',               # truncated JSON
            '{"replacements": "nope"}',                     # wrong shape
        ],
    )
    def test_unreadable_reply_is_an_error_not_a_rewrite(self, reply):
        out, debug, _ = _run(reply, base_prompt=BASE, input_1="bob")
        assert out.startswith("ERROR: unreadable edit list from the LLM")
        assert "a short bob" not in out
        assert "Base prompt: verbatim" in debug
        assert "Error:" in debug

    def test_lora_tokens_requested_as_additions_and_guaranteed(self):
        reply = _edit_list(additions=[{"after": "", "insert": "lora_a"}])
        out, debug, mock = _run(
            reply, base_prompt=BASE, input_1="bob",
            lora_keywords="lora_a, <lora:x:0.8>",
        )
        system_prompt, _ = _sent(mock)
        assert "MANDATORY TOKENS" in system_prompt
        assert "lora_a, <lora:x:0.8>" in system_prompt
        assert out == f"{BASE}, lora_a, <lora:x:0.8>"
        assert "LoRA tokens: 2 (1 appended)" in debug

    def test_lora_tokens_appended_without_llm_call(self):
        out, debug, mock = _run("unused", base_prompt=BASE, lora_keywords="lora_a")
        assert mock.call_count == 0
        assert out == f"{BASE}, lora_a"
        assert "LoRA tokens: 1 (1 appended)" in debug

    def test_no_lora_directive_without_keywords(self):
        _, _, mock = _run(_edit_list(), base_prompt=BASE, input_1="bob")
        system_prompt, _ = _sent(mock)
        assert "MANDATORY TOKENS" not in system_prompt

    def test_engine_error_is_reported(self):
        from comfyui_prompt_tools.engines import OpenAIError

        mock = MagicMock(side_effect=OpenAIError("boom"))
        with patch.object(PromptComposer, "_call_engine", mock):
            out, debug = PromptComposer().compose(
                **_kwargs(base_prompt=BASE, input_1="bob")
            )
        assert out.startswith("ERROR:") and "boom" in out

    def test_empty_reply_is_reported(self):
        out, _, _ = _run("", base_prompt=BASE, input_1="bob")
        assert out.startswith("ERROR: empty LLM output")


# ---------------------------------------------------------------------------
# adapt to output style
# ---------------------------------------------------------------------------
class TestAdaptToOutputStyle:
    def test_style_template_plus_base_prompt_directive(self):
        out, debug, mock = _run(
            '"A woman with a platinum bob walks through fog."',
            base_prompt=BASE,
            base_prompt_rest=BASE_PROMPT_ADAPT,
            input_1="short platinum bob",
        )
        system_prompt, user_message = _sent(mock)
        style_prompt = _load_composer_system_prompt(STYLE, model_name="some-model")
        assert system_prompt.startswith(style_prompt)
        assert "BASE PROMPT MODE" in system_prompt[len(style_prompt):]
        assert user_message == (
            "USER INSTRUCTION:\n(no specific instruction)\n\n"
            f"BASE PROMPT:\n{BASE}\n\n"
            "INPUT DESCRIPTIONS:\n[1] short platinum bob\n\n"
            "Compose ONE prompt from the base prompt: replace what the inputs "
            "cover, add what is missing, keep the rest, in the required output "
            "style. Describe only the final result — never mention what was "
            "replaced.\nBefore you answer, check every sentence: if it names a "
            "feature from the base prompt that an input replaced (in any "
            "wording), delete that part — end the sentence right after the new "
            "feature.\n/no_think"
        )
        # strip_llm_noise still applies to the free-text reply
        assert out == "A woman with a platinum bob walks through fog."
        assert "Base prompt: adapted" in debug
        assert "Template: composer_flux2.txt" in debug

    def test_directive_replaces_per_garment_type(self):
        directive = _build_base_prompt_directive()
        assert "SAME feature at the same level of detail" in directive
        assert "per garment type" in directive
        for garment in ("top", "dress", "trousers", "skirt", "jacket", "coat",
                        "shoes", "accessories"):
            assert garment in directive
        assert "a jacket does not replace a dress" in directive
        assert "A dress covers the top and the bottom" in directive
        assert "explicitly describes a complete outfit" in directive
        assert "an eye colour does not replace the hairstyle" in directive

    def test_directive_forbids_mentioning_replaced_features(self):
        directive = _build_base_prompt_directive()
        assert "Describe only the final result" in directive
        assert "must not appear in your output in any form" in directive
        for phrase in ("replacing", "instead of", "previously", "no longer",
                       "rather than"):
            assert f'"{phrase}"' in directive

    def test_user_message_repeats_final_result_rule(self):
        _, _, mock = _run("x", base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT,
                          input_1="short platinum bob")
        _, user_message = _sent(mock)
        assert "never mention what was replaced" in user_message
        assert "check every sentence" in user_message
        assert user_message.endswith("/no_think")

    def test_back_reference_removed_and_reported(self):
        out, debug, _ = _run(
            "She has a short platinum blonde bob, replacing the previous long "
            "red hair. She walks through fog.",
            base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT,
            input_1="short platinum bob",
        )
        assert out == "She has a short platinum blonde bob. She walks through fog."
        assert debug.endswith(
            "Base prompt: adapted | Back-references removed: 1"
        )

    def test_no_back_reference_no_debug_field(self):
        _, debug, _ = _run(
            "A bob, contrasting with the misty surroundings.",
            base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT, input_1="bob",
        )
        assert "Back-references" not in debug

    def test_removal_runs_before_lora_safety_net(self):
        out, debug, _ = _run(
            "A bob, replacing the previous red hair.",
            base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT,
            input_1="bob", lora_keywords="lora_a",
        )
        assert out == "A bob., lora_a"
        assert "Back-references removed: 1 | LoRA tokens: 1 (1 appended)" in debug

    def test_keep_verbatim_is_not_filtered(self):
        base = "a woman in a red coat, replacing the previous blue one"
        out, debug, mock = _run("unused", base_prompt=base)
        assert out == base
        assert "Back-references" not in debug

    def test_empty_base_prompt_is_not_filtered(self):
        out, debug, _ = _run(
            "A bob, replacing the previous red hair.", user_instruction="x"
        )
        assert out == "A bob, replacing the previous red hair."
        assert "Back-references" not in debug

    def test_base_prompt_only_restyles(self):
        _, _, mock = _run("restyled", base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT)
        _, user_message = _sent(mock)
        assert "INPUT DESCRIPTIONS" not in user_message
        assert f"BASE PROMPT:\n{BASE}" in user_message

    def test_lora_directive_and_guarantee(self):
        out, debug, mock = _run(
            "restyled", base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT,
            lora_keywords="lora_a",
        )
        system_prompt, _ = _sent(mock)
        assert system_prompt.endswith(_build_lora_keywords_directive(["lora_a"]))
        assert out == "restyled, lora_a"
        assert "LoRA tokens: 1 (1 appended)" in debug

    def test_unknown_style_is_an_error(self):
        out, _, mock = _run(
            "x", base_prompt=BASE, base_prompt_rest=BASE_PROMPT_ADAPT,
            output_style="No Such Style",
        )
        assert out.startswith("ERROR:")
        assert mock.call_count == 0


class TestBasePromptRestValues:
    def test_choices(self):
        assert BASE_PROMPT_REST_CHOICES == ["keep verbatim", "adapt to output style"]

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_means_keep_verbatim(self, value):
        _, debug, _ = _run(
            _edit_list(), base_prompt=BASE, base_prompt_rest=value, input_1="x"
        )
        assert "Base prompt: verbatim" in debug

    def test_unknown_value_is_an_error(self):
        out, _, mock = _run("x", base_prompt=BASE, base_prompt_rest="rewrite all")
        assert out.startswith("ERROR: unknown base_prompt_rest")
        assert mock.call_count == 0


# ---------------------------------------------------------------------------
# INPUT_TYPES
# ---------------------------------------------------------------------------
class TestInputTypes:
    def test_base_prompt_spec(self):
        spec = PromptComposer.INPUT_TYPES()["optional"]["base_prompt"]
        assert spec == ("STRING", {"multiline": True, "default": ""})

    def test_base_prompt_rest_spec(self):
        spec = PromptComposer.INPUT_TYPES()["optional"]["base_prompt_rest"]
        assert spec[0] == ["keep verbatim", "adapt to output style"]
        assert spec[1]["default"] == "keep verbatim"

    def test_new_inputs_come_last(self):
        optional = list(PromptComposer.INPUT_TYPES()["optional"])
        assert optional[-3:] == ["lora_keywords", "base_prompt", "base_prompt_rest"]


# ---------------------------------------------------------------------------
# Back-reference safety net (pure function)
# ---------------------------------------------------------------------------
class TestRemoveBackReferences:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # the two failure shapes from the live run
            (
                "She has a short platinum blonde bob with blunt bangs, replacing "
                "the previous long red hair. She wears a jacket.",
                "She has a short platinum blonde bob with blunt bangs. She wears a jacket.",
            ),
            (
                "She has short platinum blonde hair styled in a bob, contrasting "
                "with her previous long red curls. She wears a jacket.",
                "She has short platinum blonde hair styled in a bob. She wears a jacket.",
            ),
            (
                "She wears a fitted black leather biker jacket over a white "
                "t-shirt, replacing the previous summer attire.",
                "She wears a fitted black leather biker jacket over a white t-shirt.",
            ),
        ],
    )
    def test_live_run_failures_are_removed(self, text, expected):
        assert _remove_back_references(text) == (expected, 1)

    @pytest.mark.parametrize(
        "intro",
        ["replacing", "contrasting with", "instead of", "rather than",
         "in place of", "unlike", "no longer", "which replaces"],
    )
    def test_every_intro(self, intro):
        text = f"A bob, {intro} the original red curls. Fog."
        assert _remove_back_references(text) == ("A bob. Fog.", 1)

    @pytest.mark.parametrize(
        "word",
        ["previous", "previously", "former", "formerly", "original",
         "originally", "old", "earlier", "prior", "before"],
    )
    def test_every_back_reference_word(self, word):
        text = f"A bob, replacing the {word} red curls. Fog."
        assert _remove_back_references(text) == ("A bob. Fog.", 1)

    @pytest.mark.parametrize(
        "text",
        [
            "She has a bob, contrasting with the misty surroundings.",
            "She carries an old leather bag.",
            "The room is furnished in the original 1970s style.",
            "Rather than posing she walks.",
            "Rather than posing, she walks through the old town.",
            "A bob, replacing the gloomy mood with warmth.",
            "She stands before an old oak, unlike the others.",
            "Before sunrise, the original mist hangs low.",
        ],
    )
    def test_legitimate_text_is_untouched(self, text):
        assert _remove_back_references(text) == (text, 0)

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("A bob, replacing the previous hair!", "A bob!"),
            ("A bob, replacing the previous hair?", "A bob?"),
            ("A bob, replacing the previous hair", "A bob"),
            ("A bob, replacing the previous hair...", "A bob..."),
            ("Shot at f/1.8, replacing the earlier 35mm look.", "Shot at f/1.8."),
        ],
    )
    def test_sentence_end_punctuation_is_kept(self, text, expected):
        assert _remove_back_references(text) == (expected, 1)

    def test_several_occurrences(self):
        text = (
            "A bob, replacing the previous red hair. A coat, instead of the "
            "earlier dress! Fog, unlike the original sunny morning"
        )
        assert _remove_back_references(text) == ("A bob. A coat! Fog", 3)

    def test_line_breaks_are_kept(self):
        text = "Shot 1: a bob, replacing the old curls.\nShot 2: she turns."
        assert _remove_back_references(text) == (
            "Shot 1: a bob.\nShot 2: she turns.", 1
        )

    def test_clause_followed_by_more_description_is_kept(self):
        """Only a clause that ends the sentence is removed — cutting up to the
        sentence end here would also delete "and a green jacket"."""
        text = "A bob, replacing her previous red hair, and a green jacket."
        assert _remove_back_references(text) == (text, 0)

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("She walks. Her hair is no longer the earlier red. Fog.", "She walks. Fog."),
            ("Instead of her previous red hair. She walks.", "She walks."),
            ("She walks.\nNo longer the original curls.\nFog.", "She walks.\n\nFog."),
        ],
    )
    def test_whole_back_reference_sentence_is_removed(self, text, expected):
        assert _remove_back_references(text) == (expected, 1)

    @pytest.mark.parametrize(
        "text",
        [
            # content before the intro: the sentence carries the new feature
            "She has a short bob instead of her previous long red hair.",
            # a comma: more than a back-reference
            "Instead of the original summer attire, she wears a biker jacket.",
            # intro but no back-reference word (rule needs both)
            "Her hair is no longer red.",
        ],
    )
    def test_sentence_with_content_or_without_both_markers_is_kept(self, text):
        assert _remove_back_references(text) == (text, 0)

    def test_empty_text(self):
        assert _remove_back_references("") == ("", 0)


# ---------------------------------------------------------------------------
# Template + override cascade
# ---------------------------------------------------------------------------
class TestReplaceTemplate:
    def test_shipped_template_renders(self):
        text = render_template(BASE_PROMPT_REPLACE_TEMPLATE)
        assert '"replacements"' in text and '"additions"' in text
        assert "{shared_rules}" not in text
        assert "{img" not in text

    def test_not_a_composer_style(self):
        from comfyui_prompt_tools.catalog import load_catalog

        templates = {e.template for e in load_catalog().composer_styles}
        assert BASE_PROMPT_REPLACE_TEMPLATE not in templates


class TestReplaceTemplateRules:
    """The shipped template states the same-feature rules (read from the
    ``.example`` file directly, so a local copy cannot mask a change)."""

    @pytest.fixture(scope="class")
    def text(self):
        path = PROMPTS_DIR / f"{BASE_PROMPT_REPLACE_TEMPLATE}.txt.example"
        return path.read_text(encoding="utf-8")

    def test_same_feature_same_level_of_detail(self, text):
        assert "SAME FEATURE MEANS THE SAME THING AT THE SAME LEVEL OF DETAIL" in text

    def test_clothing_per_garment_type(self, text):
        assert "Clothing counts per garment type" in text
        for garment in ("top", "dress", "trousers or skirt", "jacket or coat",
                        "shoes", "accessories"):
            assert garment in text
        assert "A new garment replaces only a garment of the SAME type" in text
        assert "a jacket does not replace a dress" in text

    def test_dress_covers_top_and_bottom(self, text):
        assert "A dress covers the top and the bottom" in text

    def test_complete_outfit_may_replace_the_outfit(self, text):
        assert "explicitly describes a complete outfit" in text

    def test_face_hair_background_follow_the_same_rule(self, text):
        assert "eye colour does not replace the hairstyle" in text
        assert "a new location does not replace the time of day" in text

    def test_never_mention_the_old_feature(self, text):
        assert "Never mention what was there before" in text
        for phrase in ("replacing", "instead of", "previously", "no longer",
                       "rather than"):
            assert phrase in text

    def test_jacket_example_keeps_the_dress(self, text):
        assert "a cropped white denim jacket worn over the dress" in text
        assert "so it is added and the dress stays" in text

    def test_examples_match_apply_edits(self, text):
        """Every worked example's Result is what the program really makes of
        its Answer — the model must not learn from a wrong result."""
        examples = re.findall(
            r"^Base prompt: (.+)\n(?:.+\n)*?Answer: (\{.+\})\nResult: (.+)$",
            text,
            flags=re.MULTILINE,
        )
        assert len(examples) == 2
        for base, answer, result in examples:
            assert apply_edits(base, *parse_edit_list(answer)).text == result

    @pytest.fixture
    def prompts_dir(self, tmp_path, monkeypatch):
        from comfyui_prompt_tools import prompts as prompts_module

        (tmp_path / f"{BASE_PROMPT_REPLACE_TEMPLATE}.txt.example").write_text(
            "SHIPPED replace prompt", encoding="utf-8"
        )
        monkeypatch.setattr(prompts_module, "_PROMPTS_DIR", tmp_path)
        return tmp_path

    def test_example_is_the_fallback(self, prompts_dir):
        _, _, mock = _run(_edit_list(), base_prompt=BASE, input_1="x")
        assert _sent(mock)[0] == "SHIPPED replace prompt"

    def test_local_copy_wins(self, prompts_dir):
        (prompts_dir / f"{BASE_PROMPT_REPLACE_TEMPLATE}.txt").write_text(
            "LOCAL replace prompt", encoding="utf-8"
        )
        _, _, mock = _run(_edit_list(), base_prompt=BASE, input_1="x")
        assert _sent(mock)[0] == "LOCAL replace prompt"

    def test_family_override_wins(self, prompts_dir):
        (prompts_dir / f"{BASE_PROMPT_REPLACE_TEMPLATE}.txt").write_text(
            "LOCAL replace prompt", encoding="utf-8"
        )
        (prompts_dir / f"{BASE_PROMPT_REPLACE_TEMPLATE}.qwen3vl.txt").write_text(
            "FAMILY replace prompt", encoding="utf-8"
        )
        _, _, mock = _run(
            _edit_list(), base_prompt=BASE, input_1="x", model="qwen3-vl:8b"
        )
        assert _sent(mock)[0] == "FAMILY replace prompt"

    def test_missing_template_is_an_error(self, tmp_path, monkeypatch):
        from comfyui_prompt_tools import prompts as prompts_module

        monkeypatch.setattr(prompts_module, "_PROMPTS_DIR", tmp_path)
        out, _, mock = _run(_edit_list(), base_prompt=BASE, input_1="x")
        assert out.startswith("ERROR:")
        assert mock.call_count == 0
