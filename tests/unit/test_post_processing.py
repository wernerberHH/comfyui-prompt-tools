"""Unit tests for comfyui_prompt_tools.post_processing."""

import pytest

from comfyui_prompt_tools.post_processing import strip_llm_noise


@pytest.mark.unit
def test_strips_leading_quote():
    assert strip_llm_noise('"hello"') == "hello"
    assert strip_llm_noise("'world'") == "world"


@pytest.mark.unit
def test_strips_here_is_preamble():
    assert strip_llm_noise("Here is your enhanced prompt") == "your enhanced prompt"
    assert strip_llm_noise("Here's the answer") == "the answer"


@pytest.mark.unit
def test_strips_sure_preamble():
    assert strip_llm_noise("Sure, this works") == "this works"


@pytest.mark.unit
def test_strips_enhanced_prompt_label():
    assert strip_llm_noise("Enhanced prompt: foo bar") == "foo bar"


@pytest.mark.unit
def test_clean_input_unchanged():
    """A clean prompt should pass through (modulo trim)."""
    text = "Generate an image of a sunset over mountains"
    assert strip_llm_noise(text) == text


@pytest.mark.unit
def test_strips_outer_whitespace():
    assert strip_llm_noise("   hello   ") == "hello"


@pytest.mark.unit
def test_case_insensitive_prefix_match():
    """Lower-case prefix should also be stripped."""
    assert strip_llm_noise("here is the result") == "the result"
