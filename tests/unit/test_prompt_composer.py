"""Tests for the PromptComposer node.

PromptComposer fuses N upstream description snippets (typically from
describe-mode VisionPromptHelpers) with a user instruction into one final
prompt via an LLM call. These tests verify:

- single-input flow reaches the engine
- five-input flow reaches the engine
- empty / None inputs are skipped (no gaps in the numbered list)
- the output_style dropdown routes to the matching composer_<style>.txt prompt
- the user_instruction text reaches the request body verbatim

Mocks:
- ``mock_openai_urlopen`` fixture from conftest patches
  ``comfyui_prompt_tools.engines.openai_client.urllib.request.urlopen`` so no
  real HTTP is issued. Patch target follows Standards §4a (urlopen patched
  at the import path inside the engine source module).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from comfyui_prompt_tools.engines import OpenAIError
from comfyui_prompt_tools.nodes.prompt_composer import (
    OUTPUT_STYLES,
    PromptComposer,
    _build_lora_keywords_directive,
    _ensure_verbatim_tokens,
    _load_composer_system_prompt,
    _parse_lora_keywords,
)


def _compose_kwargs(**overrides):
    """Default kwargs for PromptComposer.compose() with sensible test values."""
    base = {
        "engine":           "vllm",
        "base_url":         "http://x:8000/v1",
        "model":            "qwen-7b",
        "temperature":      0.7,
        "user_instruction": "warm romantic dinner",
        "output_style":     "FLUX.2 natural language",
    }
    base.update(overrides)
    return base


class TestComposerSingleInput:
    def test_composer_with_one_input_only(self, mock_openai_urlopen):
        """A single wired input flows through to the engine and returns the
        mocked response.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        composer = PromptComposer()
        out, debug = composer.compose(
            **_compose_kwargs(input_1="From image 1: asian woman, almond eyes, oval face")
        )
        assert out == "test output"
        # Debug summary lists 1 non-empty input
        assert "Inputs: 1" in debug
        # Single request issued
        assert mock_openai_urlopen.call_count == 1
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        # The snippet content reaches the user message (enumerated as [1])
        assert "almond eyes" in body
        assert "[1]" in body


class TestComposerFiveInputs:
    def test_composer_with_five_inputs(self, mock_openai_urlopen):
        """All five wired inputs reach the engine, enumerated 1..5.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        composer = PromptComposer()
        out, debug = composer.compose(
            **_compose_kwargs(
                input_1="face snippet alpha",
                input_2="hair snippet bravo",
                input_3="outfit snippet charlie",
                input_4="background snippet delta",
                input_5="lighting snippet echo",
            )
        )
        assert out == "test output"
        assert "Inputs: 5" in debug
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        for marker in ("alpha", "bravo", "charlie", "delta", "echo"):
            assert marker in body, f"input snippet {marker!r} missing from body"
        # All five enumerator labels present
        for idx in ("[1]", "[2]", "[3]", "[4]", "[5]"):
            assert idx in body


class TestComposerSkipsEmptyInputs:
    def test_composer_skips_empty_inputs(self, mock_openai_urlopen):
        """Empty / whitespace-only / None inputs are dropped before the call;
        the LLM never sees enumerator gaps. Non-empty entries are renumbered
        contiguously starting at [1].
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        composer = PromptComposer()
        out, debug = composer.compose(
            **_compose_kwargs(
                input_1="real snippet first",
                input_2="",            # empty string
                input_3="   ",         # whitespace-only
                input_4=None,          # unwired
                input_5="real snippet last",
            )
        )
        assert out == "test output"
        # Only 2 inputs survive
        assert "Inputs: 2" in debug
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        # Real snippets present and renumbered contiguously
        assert "real snippet first" in body
        assert "real snippet last" in body
        assert "[1]" in body
        assert "[2]" in body
        # No third enumerator since only 2 inputs survived
        assert "[3]" not in body


class TestComposerRoutesToOutputStyle:
    @pytest.mark.parametrize("style", OUTPUT_STYLES)
    def test_composer_routes_to_correct_output_style(
        self, style, mock_openai_urlopen
    ):
        """Each output_style label loads its own composer_<style>.txt and
        ships that system_prompt to the engine.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        composer = PromptComposer()
        composer.compose(
            **_compose_kwargs(
                output_style=style,
                input_1="snippet",
            )
        )
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        rendered = _load_composer_system_prompt(style)
        # Use the first line of the rendered template as a fingerprint
        fingerprint = rendered.strip().split("\n", 1)[0][:80]
        assert fingerprint in body, (
            f"composer system prompt for {style!r} did not reach the request body"
        )


class TestComposerIncludesUserInstruction:
    def test_composer_includes_user_instruction_in_call(self, mock_openai_urlopen):
        """The user_instruction string reaches the engine request body verbatim
        and is labelled clearly so the LLM can distinguish it from the inputs.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        composer = PromptComposer()
        instruction = "two people sharing dessert at a candlelit bistro"
        composer.compose(
            **_compose_kwargs(
                user_instruction=instruction,
                input_1="background snippet",
            )
        )
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        assert instruction in body
        assert "USER INSTRUCTION" in body


class TestComposerErrorPaths:
    """Coverage for the defensive error branches in compose().

    These run without the urlopen fixture because the goal is to verify
    the node short-circuits BEFORE hitting the network. The engine error
    test uses an instance-level patch to inject a controlled exception.
    """

    def test_composer_short_circuits_when_no_instruction_and_no_inputs(self):
        """No user_instruction and no wired inputs returns an ERROR string
        immediately and never resolves an engine.
        Mocks: none — pure short-circuit path.
        """
        composer = PromptComposer()
        out, debug = composer.compose(
            **_compose_kwargs(user_instruction="", input_1=None)
        )
        assert out.startswith("ERROR:")
        assert "nothing to compose" in out
        assert "Inputs: 0" in debug

    def test_composer_handles_invalid_engine_choice(self):
        """An unknown engine label returns an ERROR string with the
        ValueError message and never contacts the network.
        Mocks: none — _resolve_engine raises ValueError synchronously.
        """
        composer = PromptComposer()
        out, debug = composer.compose(
            **_compose_kwargs(engine="nonexistent_xyz", input_1="snippet")
        )
        assert out.startswith("ERROR:")
        assert "Unknown engine" in out
        assert "nonexistent_xyz" in debug

    def test_composer_returns_error_string_on_engine_failure(self):
        """When the engine raises OpenAIError, compose() catches it and
        returns an ERROR tuple so ShowText displays the failure to the user.
        Mocks: BasePromptNode._call_engine patched to raise OpenAIError
        (instance-level patch on the composer instance to keep the mock
        scoped to this test, per Standards §4a).
        """
        composer = PromptComposer()
        with patch.object(
            composer, "_call_engine", side_effect=OpenAIError("502 bad gateway")
        ):
            out, debug = composer.compose(
                **_compose_kwargs(input_1="snippet")
            )
        assert out.startswith("ERROR:")
        assert "502 bad gateway" in out
        assert "Error: 502 bad gateway" in debug


class TestComposerOverrideCascade:
    """Cascade behaviour for ``_load_composer_system_prompt``.

    Composer styles use ``composer_<style>`` file bases; the per-model
    override file naming follows the same ``<file>.<family>.txt`` pattern
    as text and vision modes.
    """

    @pytest.fixture
    def isolated_prompts_dir(self, tmp_path, monkeypatch):
        """Redirect ``prompts._PROMPTS_DIR`` at a temp dir seeded with a
        minimal composer default file.

        Mocks: ``prompts._PROMPTS_DIR`` rebound to ``tmp_path`` so the
        cascade can be observed without touching the real prompt library.
        """
        from comfyui_prompt_tools import prompts as prompts_module

        (tmp_path / "_shared_rules.txt").write_text("SHARED", encoding="utf-8")
        (tmp_path / "composer_flux2.txt").write_text(
            "DEFAULT flux2 composer prompt.", encoding="utf-8"
        )
        monkeypatch.setattr(prompts_module, "_PROMPTS_DIR", tmp_path)
        return tmp_path

    def test_cascade_uses_override_when_present(self, isolated_prompts_dir):
        """An ``composer_<style>.<family>.txt`` override wins over the default.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        (isolated_prompts_dir / "composer_flux2.cydonia.txt").write_text(
            "CYDONIA composer override.", encoding="utf-8"
        )
        result = _load_composer_system_prompt(
            "FLUX.2 natural language",
            "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
        )
        assert "CYDONIA composer override" in result
        assert "DEFAULT flux2" not in result

    def test_cascade_falls_back_to_default_when_no_override(
        self, isolated_prompts_dir
    ):
        """Family detected but no override file exists -> default loads.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        result = _load_composer_system_prompt(
            "FLUX.2 natural language",
            "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
        )
        assert "DEFAULT flux2 composer" in result

    def test_cascade_falls_back_when_model_name_is_none(
        self, isolated_prompts_dir
    ):
        """``model_name=None`` skips the override probe.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        (isolated_prompts_dir / "composer_flux2.cydonia.txt").write_text(
            "CYDONIA composer override.", encoding="utf-8"
        )
        result = _load_composer_system_prompt("FLUX.2 natural language", None)
        assert "DEFAULT flux2 composer" in result

    def test_model_name_none_equals_omitted(self):
        """Explicit ``model_name=None`` must equal the zero-argument call.
        Mocks: none — equality check against the real prompt library.
        """
        for style in OUTPUT_STYLES:
            assert _load_composer_system_prompt(style) == (
                _load_composer_system_prompt(style, None)
            )

    def test_unknown_style_raises_keyerror_even_with_model_name(self):
        """KeyError for an unregistered style must not be masked by the
        cascade — the style-to-file lookup happens before the loader.
        Mocks: none.
        """
        with pytest.raises(KeyError, match="Unknown output style"):
            _load_composer_system_prompt("Bogus Style", "Fermi/Cydonia-24B")

    def test_override_with_shared_rules_substitution(self, isolated_prompts_dir):
        """Override files honour the ``{shared_rules}`` placeholder.
        Mocks: ``prompts._PROMPTS_DIR`` redirected to ``tmp_path``.
        """
        (isolated_prompts_dir / "composer_flux2.cydonia.txt").write_text(
            "Override head.\n{shared_rules}\nOverride tail.", encoding="utf-8"
        )
        result = _load_composer_system_prompt(
            "FLUX.2 natural language",
            "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
        )
        assert "Override head" in result
        assert "Override tail" in result
        assert "SHARED" in result
        assert "{shared_rules}" not in result


class TestComposerModelPropagation:
    """The selected model must reach _load_composer_system_prompt as
    model_name so the per-model override cascade can activate.

    Mocks: ``comfyui_prompt_tools.nodes.prompt_composer.
    _load_composer_system_prompt`` is patched at the consuming-module
    path (Standards §4a).
    """

    def test_composer_propagates_model_name(self, mock_openai_urlopen):
        """compose() passes the selected model into the composer-prompt
        loader as ``model_name``.
        Mocks: nodes.prompt_composer._load_composer_system_prompt
        (returns canned string).
        """
        composer = PromptComposer()
        with patch(
            "comfyui_prompt_tools.nodes.prompt_composer._load_composer_system_prompt",
            return_value="STUB COMPOSER SYS PROMPT",
        ) as mock_loader:
            composer.compose(
                **_compose_kwargs(
                    model="Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M",
                    output_style="FLUX.2 natural language",
                    input_1="snippet",
                )
            )
        assert mock_loader.call_count == 1
        args, kwargs = mock_loader.call_args
        assert args == ("FLUX.2 natural language",)
        assert kwargs == {
            "model_name": "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M"
        }


class TestWan22MotionStyle:
    """v0.6: Wan 2.2 motion composer style for image-to-video pipelines."""

    def test_wan22_in_output_styles(self):
        """Wan 2.2 motion is registered in OUTPUT_STYLES.
        Mocks: none.
        """
        assert "Wan 2.2 motion" in OUTPUT_STYLES

    def test_wan22_style_loads_non_empty_prompt(self):
        """Wan 2.2 motion style resolves to a non-empty composer prompt.
        Mocks: none.
        """
        prompt = _load_composer_system_prompt("Wan 2.2 motion")
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 100

    def test_wan22_style_shared_rules_substituted(self):
        """The {shared_rules} placeholder is replaced in the rendered output.
        Mocks: none.
        """
        prompt = _load_composer_system_prompt("Wan 2.2 motion")
        assert "{shared_rules}" not in prompt

    def test_wan22_style_template_fingerprints(self):
        """Stable fingerprint strings from the template body — surface
        accidental file truncation in CI.
        Mocks: none.
        """
        prompt = _load_composer_system_prompt("Wan 2.2 motion")
        assert "Wan 2.2" in prompt
        assert "60–100 words" in prompt
        # Wan-specific instruction: no negation
        assert "negation" in prompt.lower()

    def test_wan22_style_routes_to_correct_file(self, mock_openai_urlopen):
        """Selecting 'Wan 2.2 motion' in compose() routes through the
        wan22 system prompt file (asserts via request body inspection).
        Mocks: openai_client.urllib.request.urlopen via mock_openai_urlopen.
        """
        composer = PromptComposer()
        out, debug = composer.compose(
            **_compose_kwargs(
                output_style="Wan 2.2 motion",
                user_instruction="she turns her head slowly to the right and smiles",
                input_1="A young woman with shoulder-length dark hair stands in front of a window, wearing a light grey shirt, soft daylight from the left.",
            )
        )
        assert out == "test output"
        assert "Style: Wan 2.2 motion" in debug


# ==========================================================================
# v0.8: LoRA-trigger keywords field on PromptComposer
# ==========================================================================


class TestParseLoraKeywords:
    """Helper: _parse_lora_keywords splits and cleans the raw input."""

    def test_empty_string_returns_empty_list(self):
        assert _parse_lora_keywords("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _parse_lora_keywords("   ") == []

    def test_single_token(self):
        assert _parse_lora_keywords("ohwx_man") == ["ohwx_man"]

    def test_multiple_tokens(self):
        assert _parse_lora_keywords("ohwx_man, cinematic, slow motion") == [
            "ohwx_man", "cinematic", "slow motion"
        ]

    def test_per_token_whitespace_stripped(self):
        assert _parse_lora_keywords("  alpha ,   bravo  , charlie ") == [
            "alpha", "bravo", "charlie"
        ]

    def test_empty_entries_dropped(self):
        """Doubled commas or trailing commas produce empty fragments
        that must not appear in the parsed list."""
        assert _parse_lora_keywords("alpha,, bravo, ,charlie,") == [
            "alpha", "bravo", "charlie"
        ]

    def test_order_preserved(self):
        """Order matters for some LoRAs (leading-token-trained), so
        the parser must not reorder."""
        assert _parse_lora_keywords("zulu, alpha, mike") == [
            "zulu", "alpha", "mike"
        ]

    def test_preserves_lora_syntax(self):
        """LoRA syntax like <lora:name:0.8> and (token:1.2) must
        pass through untouched as a single token."""
        result = _parse_lora_keywords("ohwx_man, <lora:cinematic_v2:0.8>, (style:1.2)")
        assert result == ["ohwx_man", "<lora:cinematic_v2:0.8>", "(style:1.2)"]


class TestBuildLoraKeywordsDirective:
    """Helper: _build_lora_keywords_directive renders the system-prompt
    suffix that instructs the LLM to keep the tokens verbatim."""

    def test_empty_list_returns_empty_string(self):
        """Caller can concatenate unconditionally — empty input must
        not produce a directive (would confuse the LLM)."""
        assert _build_lora_keywords_directive([]) == ""

    def test_non_empty_contains_mandatory_tokens_phrase(self):
        directive = _build_lora_keywords_directive(["ohwx_man"])
        assert "MANDATORY TOKENS" in directive

    def test_non_empty_contains_verbatim_phrase(self):
        directive = _build_lora_keywords_directive(["ohwx_man"])
        assert "verbatim" in directive

    def test_all_tokens_appear_in_directive(self):
        tokens = ["ohwx_man", "<lora:cinematic:0.8>", "slow motion"]
        directive = _build_lora_keywords_directive(tokens)
        for tok in tokens:
            assert tok in directive

    def test_starts_with_double_newline(self):
        """The directive must visually separate from the upstream
        system prompt body."""
        directive = _build_lora_keywords_directive(["x"])
        assert directive.startswith("\n\n")


class TestEnsureVerbatimTokens:
    """Helper: _ensure_verbatim_tokens appends missing tokens as a
    safety net so LoRA triggers never get silently lost."""

    def test_no_required_tokens_returns_output_unchanged(self):
        out, missing = _ensure_verbatim_tokens("hello world", [])
        assert out == "hello world"
        assert missing == []

    def test_all_tokens_present_returns_output_unchanged(self):
        out, missing = _ensure_verbatim_tokens(
            "a cinematic shot of ohwx_man at sunset",
            ["ohwx_man", "cinematic"],
        )
        assert out == "a cinematic shot of ohwx_man at sunset"
        assert missing == []

    def test_some_tokens_missing_get_appended(self):
        out, missing = _ensure_verbatim_tokens(
            "a cinematic shot at sunset",
            ["ohwx_man", "cinematic", "slow motion"],
        )
        assert "ohwx_man" in out
        assert "slow motion" in out
        assert set(missing) == {"ohwx_man", "slow motion"}

    def test_all_tokens_missing_all_appended(self):
        out, missing = _ensure_verbatim_tokens(
            "a peaceful garden scene",
            ["alpha", "bravo", "charlie"],
        )
        for tok in ["alpha", "bravo", "charlie"]:
            assert tok in out
        assert set(missing) == {"alpha", "bravo", "charlie"}

    def test_appended_tokens_use_comma_separator(self):
        """The appended suffix joins tokens with ', ' so it lands
        cleanly in a comma-tag CLIPTextEncode."""
        out, _missing = _ensure_verbatim_tokens(
            "scene description.",
            ["alpha", "bravo"],
        )
        assert out.endswith("alpha, bravo")

    def test_handles_trailing_comma_in_output(self):
        """If the LLM ends with a trailing comma, we don't want to
        produce ',, ' in the result."""
        out, _missing = _ensure_verbatim_tokens(
            "scene description,",
            ["alpha"],
        )
        assert ",," not in out
        assert out.endswith("alpha")

    def test_handles_trailing_whitespace_in_output(self):
        out, _missing = _ensure_verbatim_tokens(
            "scene description.   ",
            ["alpha"],
        )
        # No weird double-space before the comma
        assert "  ," not in out


class TestComposerLoraKeywordsIntegration:
    """Integration: lora_keywords flows from compose() through the
    request body and post-check."""

    def test_empty_lora_keywords_no_directive_in_system_prompt(
        self, mock_openai_urlopen
    ):
        """Empty lora_keywords must not add the MANDATORY TOKENS
        directive — would noise up the system prompt."""
        composer = PromptComposer()
        composer.compose(
            **_compose_kwargs(input_1="snippet", lora_keywords="")
        )
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        assert "MANDATORY TOKENS" not in body

    def test_non_empty_lora_keywords_directive_in_system_prompt(
        self, mock_openai_urlopen
    ):
        """A populated lora_keywords field appends the directive and
        token list to the system prompt the engine receives."""
        composer = PromptComposer()
        composer.compose(
            **_compose_kwargs(
                input_1="snippet",
                lora_keywords="ohwx_man, cinematic",
            )
        )
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        assert "MANDATORY TOKENS" in body
        assert "ohwx_man" in body
        assert "cinematic" in body

    def test_compose_accepts_lora_keywords_kwarg(self, mock_openai_urlopen):
        """compose() signature accepts the new kwarg without breakage —
        guards against accidental signature regressions."""
        composer = PromptComposer()
        out, _debug = composer.compose(
            **_compose_kwargs(
                input_1="snippet",
                lora_keywords="ohwx_man",
            )
        )
        assert not out.startswith("ERROR:")

    def test_missing_tokens_appended_to_output(self):
        """The mock 'test output' does not contain the LoRA triggers,
        so the post-check must append both."""
        composer = PromptComposer()
        # Use instance-level patch for a controlled non-mocked-urlopen flow
        with patch.object(
            composer, "_call_engine", return_value="a peaceful garden scene"
        ):
            out, _debug = composer.compose(
                **_compose_kwargs(
                    input_1="snippet",
                    lora_keywords="ohwx_man, <lora:cinematic_v2:0.8>",
                )
            )
        assert "ohwx_man" in out
        assert "<lora:cinematic_v2:0.8>" in out
        assert out.startswith("a peaceful garden scene")

    def test_present_tokens_not_duplicated(self):
        """If the LLM already wove the token in, the post-check must
        not duplicate it."""
        composer = PromptComposer()
        with patch.object(
            composer,
            "_call_engine",
            return_value="ohwx_man stands at a cinematic sunset",
        ):
            out, _debug = composer.compose(
                **_compose_kwargs(
                    input_1="snippet",
                    lora_keywords="ohwx_man, cinematic",
                )
            )
        # Count occurrences — should be exactly 1 each
        assert out.count("ohwx_man") == 1
        assert out.count("cinematic") == 1

    def test_debug_info_reports_token_count_when_keywords_given(self):
        """debug_info surfaces LoRA-token statistics so users can
        see at a glance whether the post-check kicked in."""
        composer = PromptComposer()
        with patch.object(
            composer, "_call_engine", return_value="scene without triggers"
        ):
            _out, debug = composer.compose(
                **_compose_kwargs(
                    input_1="snippet",
                    lora_keywords="ohwx_man, cinematic",
                )
            )
        assert "LoRA tokens: 2" in debug
        assert "(2 appended)" in debug

    def test_debug_info_no_token_section_when_empty(self, mock_openai_urlopen):
        """No LoRA-tokens section in debug_info when the field is empty."""
        composer = PromptComposer()
        _out, debug = composer.compose(
            **_compose_kwargs(input_1="snippet", lora_keywords="")
        )
        assert "LoRA tokens" not in debug

    def test_partial_compliance_only_missing_appended(self):
        """If the LLM included one of two tokens, only the missing one
        is appended."""
        composer = PromptComposer()
        with patch.object(
            composer,
            "_call_engine",
            return_value="ohwx_man at the beach",  # has ohwx_man, lacks cinematic
        ):
            out, debug = composer.compose(
                **_compose_kwargs(
                    input_1="snippet",
                    lora_keywords="ohwx_man, cinematic",
                )
            )
        assert "cinematic" in out
        assert out.count("ohwx_man") == 1
        assert "LoRA tokens: 2" in debug
        assert "(1 appended)" in debug
