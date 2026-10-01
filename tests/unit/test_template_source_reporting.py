"""The nodes report which system-prompt file a run actually used.

A local ``system_prompts/<name>.txt`` silently wins over the shipped
``.txt.example`` (see :mod:`comfyui_prompt_tools.template_status`). These
tests cover the per-run half of that report:

- ``PromptComposer.debug_info`` and ``VisionPromptHelper.debug_info`` gain a
  ``Template:`` field — the existing fields stay untouched.
- ``PromptHelper`` has no ``debug_info`` output, so its summary log line
  carries the same field instead.

Mocks
-----
- ``mock_openai_urlopen`` from conftest keeps the engine off the network.
- ``vision_prompt_helper.tensor_to_b64`` is stubbed so no real tensors are
  needed, following ``test_vision_modes.py``.
- ``template_status._hashes_cache`` / ``_PROMPTS_DIR`` are redirected at a
  temp directory where a copy's classification is set up explicitly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from comfyui_prompt_tools import prompts as prompts_mod
from comfyui_prompt_tools import template_status as ts
from comfyui_prompt_tools.nodes.prompt_composer import PromptComposer
from comfyui_prompt_tools.nodes.prompt_helper import PromptHelper
from comfyui_prompt_tools.nodes.vision_prompt_helper import VisionPromptHelper

COMPOSER_STYLE = "FLUX.2 natural language"
COMPOSER_TEMPLATE = "composer_flux2"
VISION_MODE = "Describe Face"
VISION_TEMPLATE = "vision_describe_face"
HELPER_MODE = "FLUX – Text-to-Image"
HELPER_TEMPLATE = "flux_text_to_image"

SHIPPED_BODY = "shipped template body\n"


@pytest.fixture
def local_copy(tmp_path, monkeypatch):
    """A template dir where every template is shipped, and a copy can be added.

    Returns a callable that writes ``<basename>.txt`` with the given content
    and registers the shipped digest, so the classification is unambiguous.
    """
    monkeypatch.setattr(prompts_mod, "_PROMPTS_DIR", tmp_path)
    monkeypatch.setattr(ts, "PROMPTS_DIR", tmp_path)
    templates: dict[str, list[str]] = {}
    monkeypatch.setattr(
        ts, "_hashes_cache", ts.ShippedHashes(templates=templates, retired={})
    )

    def setup(basename: str, local_content: str | None = None) -> None:
        (tmp_path / f"{basename}.txt.example").write_text(
            SHIPPED_BODY, encoding="utf-8"
        )
        templates[basename] = [ts.digest_text(SHIPPED_BODY)]
        if local_content is not None:
            (tmp_path / f"{basename}.txt").write_text(
                local_content, encoding="utf-8"
            )

    return setup


def _compose_kwargs(**overrides):
    base = {
        "engine":           "vllm",
        "base_url":         "http://x:8000/v1",
        "model":            "qwen-7b",
        "temperature":      0.7,
        "user_instruction": "warm romantic dinner",
        "output_style":     COMPOSER_STYLE,
        "input_1":          "snippet",
    }
    base.update(overrides)
    return base


def _vision_kwargs(**overrides):
    base = {
        "intent":       "describe the face",
        "mode":         VISION_MODE,
        "engine":       "vllm",
        "base_url":     "http://x:8000/v1",
        "model":        "qwen-7b",
        "temperature":  0.7,
        "jpeg_quality": 90,
        "max_edge":     1024,
        "image_1":      object(),
    }
    base.update(overrides)
    return base


def _helper_kwargs(**overrides):
    base = {
        "prompt":      "a woman in a red dress",
        "mode":        HELPER_MODE,
        "engine":      "vllm",
        "base_url":    "http://x:8000/v1",
        "model":       "qwen-7b",
        "temperature": 0.7,
    }
    base.update(overrides)
    return base


@pytest.fixture
def stub_tensor_to_b64():
    """Avoid synthesising real image tensors for VisionPromptHelper."""
    with patch(
        "comfyui_prompt_tools.nodes.vision_prompt_helper.tensor_to_b64",
        return_value="IMG",
    ):
        yield


# ---------------------------------------------------------------------------
# PromptComposer
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_composer_debug_info_names_the_shipped_template(
    local_copy, mock_openai_urlopen
):
    local_copy(COMPOSER_TEMPLATE)
    _out, debug = PromptComposer().compose(**_compose_kwargs())
    assert f"Template: {COMPOSER_TEMPLATE}.txt.example (shipped)" in debug


@pytest.mark.unit
def test_composer_debug_info_flags_a_customized_copy(
    local_copy, mock_openai_urlopen
):
    local_copy(COMPOSER_TEMPLATE, "my own composer wording\n")
    _out, debug = PromptComposer().compose(**_compose_kwargs())
    assert f"Template: {COMPOSER_TEMPLATE}.txt (local copy, customized)" in debug


@pytest.mark.unit
def test_composer_debug_info_keeps_its_existing_fields(
    local_copy, mock_openai_urlopen
):
    """The format is extended, not rebuilt."""
    local_copy(COMPOSER_TEMPLATE)
    _out, debug = PromptComposer().compose(**_compose_kwargs())
    for field in ("Engine:", "Model:", "Style:", "Inputs:", "Latency:"):
        assert field in debug
    assert debug.index("Latency:") < debug.index("Template:")


# ---------------------------------------------------------------------------
# VisionPromptHelper
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_vision_debug_info_names_the_shipped_template(
    local_copy, stub_tensor_to_b64, mock_openai_urlopen
):
    local_copy(VISION_TEMPLATE)
    _out, debug = VisionPromptHelper().generate(**_vision_kwargs())
    assert f"Template: {VISION_TEMPLATE}.txt.example (shipped)" in debug


@pytest.mark.unit
def test_vision_debug_info_flags_a_customized_copy(
    local_copy, stub_tensor_to_b64, mock_openai_urlopen
):
    local_copy(VISION_TEMPLATE, "my own vision wording\n")
    _out, debug = VisionPromptHelper().generate(**_vision_kwargs())
    assert f"Template: {VISION_TEMPLATE}.txt (local copy, customized)" in debug


@pytest.mark.unit
def test_vision_debug_info_reports_the_custom_input_as_the_source(
    local_copy, stub_tensor_to_b64, mock_openai_urlopen
):
    """A system prompt typed into the node comes from no file at all."""
    local_copy(VISION_TEMPLATE)
    _out, debug = VisionPromptHelper().generate(
        **_vision_kwargs(custom_system_prompt="straight from the node")
    )
    assert "Template: (custom_system_prompt input)" in debug


@pytest.mark.unit
def test_vision_debug_info_keeps_its_existing_fields(
    local_copy, stub_tensor_to_b64, mock_openai_urlopen
):
    local_copy(VISION_TEMPLATE)
    _out, debug = VisionPromptHelper().generate(**_vision_kwargs())
    for field in ("Engine:", "Model:", "Mode:", "Target:", "Images:",
                  "InTokens:", "Latency:"):
        assert field in debug
    assert debug.index("Latency:") < debug.index("Template:")


# ---------------------------------------------------------------------------
# PromptHelper — no debug_info output, so the log line carries the field
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_prompt_helper_outputs_are_unchanged(
    local_copy, mock_openai_urlopen
):
    """Still two outputs: adding a third would change the node signature."""
    assert PromptHelper.RETURN_NAMES == ("enhanced_prompt", "original_prompt")
    local_copy(HELPER_TEMPLATE)
    result = PromptHelper().enhance(**_helper_kwargs())
    assert len(result) == 2


@pytest.mark.unit
def test_prompt_helper_log_names_the_shipped_template(
    local_copy, mock_openai_urlopen, capsys
):
    local_copy(HELPER_TEMPLATE)
    PromptHelper().enhance(**_helper_kwargs())
    out = capsys.readouterr().out
    assert f"Template: {HELPER_TEMPLATE}.txt.example (shipped)" in out


@pytest.mark.unit
def test_prompt_helper_log_flags_a_customized_copy(
    local_copy, mock_openai_urlopen, capsys
):
    local_copy(HELPER_TEMPLATE, "my own helper wording\n")
    PromptHelper().enhance(**_helper_kwargs())
    out = capsys.readouterr().out
    assert f"Template: {HELPER_TEMPLATE}.txt (local copy, customized)" in out


@pytest.mark.unit
def test_prompt_helper_log_keeps_its_existing_fields(
    local_copy, mock_openai_urlopen, capsys
):
    local_copy(HELPER_TEMPLATE)
    PromptHelper().enhance(**_helper_kwargs())
    out = capsys.readouterr().out
    assert "[PromptHelper] Mode:" in out
    assert "Engine:" in out and "Model:" in out


@pytest.mark.unit
def test_prompt_helper_log_reports_the_custom_input_as_the_source(
    local_copy, mock_openai_urlopen, capsys
):
    local_copy("custom_system_prompt")
    PromptHelper().enhance(
        **_helper_kwargs(
            mode="Custom System Prompt",
            custom_system_prompt="straight from the node",
        )
    )
    out = capsys.readouterr().out
    assert "Template: (custom_system_prompt input)" in out
