"""Tests for PromptHelper engine selection and mode coverage.

PromptHelper now inherits from BasePromptNode. These tests verify that:
- the engine dropdown routes to the right engine class
- random modes still work when vllm is selected
- all 9 modes from AVAILABLE_MODES remain wired

Mocks: mock_ollama_urlopen / mock_openai_urlopen fixtures from conftest patch
``urllib.request.urlopen`` inside the engine source modules. Patch target
follows Standards §4a — engine-source-module urlopen, not the helper.
"""

from __future__ import annotations

from unittest.mock import patch

from comfyui_prompt_tools.nodes.prompt_helper import PromptHelper
from comfyui_prompt_tools.prompts import AVAILABLE_MODES


def _enhance_kwargs(**overrides):
    """Default kwargs for PromptHelper.enhance() with sensible test values."""
    base = {
        "prompt": "a girl",
        "mode": "FLUX Kontext (Scene Edit)",
        "engine": "ollama",
        "base_url": "http://x",
        "model": "m",
        "temperature": 0.7,
        "keep_alive": "30s",
    }
    base.update(overrides)
    return base


class TestPromptHelperEngineSelection:
    def test_prompt_helper_with_ollama_engine(self, mock_ollama_urlopen):
        """engine='ollama' issues a request against the Ollama /api/chat URL.
        Mocks: ollama_client.urllib.request.urlopen (canned 'test output').
        """
        helper = PromptHelper()
        out = helper.enhance(**_enhance_kwargs(engine="ollama", base_url="http://x"))
        assert out[0] == "test output"
        url = mock_ollama_urlopen.call_args[0][0].full_url
        assert url.endswith("/api/chat")

    def test_prompt_helper_with_vllm_engine(self, mock_openai_urlopen):
        """engine='vllm' issues a request against /v1/chat/completions.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        helper = PromptHelper()
        out = helper.enhance(
            **_enhance_kwargs(engine="vllm", base_url="http://x:8000")
        )
        assert out[0] == "test output"
        url = mock_openai_urlopen.call_args[0][0].full_url
        assert url.endswith("/v1/chat/completions")

    def test_prompt_helper_random_mode_still_works_with_vllm(
        self, mock_openai_urlopen
    ):
        """Random modes route through vllm, slot pools are honoured.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        helper = PromptHelper()
        out = helper.enhance(
            **_enhance_kwargs(
                mode="Random Character (Z-Image)",
                engine="vllm",
                base_url="http://x:8000",
                ethnicity_pool="japanese",
                age_pool="25-35",
                mood_pool="serious",
                hair_pool="long black",
            )
        )
        assert out[0] == "test output"
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        # User message should contain the slot values
        assert "japanese" in body
        assert "serious" in body

    def test_prompt_helper_keeps_all_modes_listed(self):
        """The mode dropdown exposes exactly the 10 documented v0.5 modes."""
        modes = AVAILABLE_MODES
        assert len(modes) == 10
        expected = {
            "FLUX Kontext (Scene Edit)",
            "FLUX Kontext (Couple Scene)",
            "Qwen Image Edit (Couple Scene)",
            "FLUX Text-to-Image",
            "Z-Image Text-to-Image",
            "SDXL Photorealistic",
            "SDXL Pony/Illustrious",
            "Random Character (Z-Image)",
            "Random Character (Pony)",
            "Custom System Prompt",
        }
        assert set(modes) == expected

    def test_input_types_includes_engine_dropdown(self):
        """The migrated INPUT_TYPES exposes the engine selector field."""
        spec = PromptHelper.INPUT_TYPES()
        assert "engine" in spec["required"]
        assert "base_url" in spec["required"]
        # ollama_url is gone — user rebuilds workflow once for v0.4
        assert "ollama_url" not in spec["required"]


class TestPromptHelperModelPropagation:
    """The selected model must reach get_system_prompt as model_name so the
    per-model override cascade can activate.

    Mocks: ``comfyui_prompt_tools.nodes.prompt_helper.get_system_prompt``
    patched at the consuming-module path (Standards §4a) so we can inspect
    call kwargs without depending on real prompt files.
    """

    def test_standard_mode_propagates_model_name(self, mock_openai_urlopen):
        """A non-Random, non-Custom mode passes ``model_name=<model>`` to
        the loader.
        Mocks: nodes.prompt_helper.get_system_prompt (returns canned string).
        """
        helper = PromptHelper()
        with patch(
            "comfyui_prompt_tools.nodes.prompt_helper.get_system_prompt",
            return_value="STUB SYS PROMPT",
        ) as mock_loader:
            helper.enhance(
                **_enhance_kwargs(
                    mode="FLUX Kontext (Scene Edit)",
                    engine="vllm",
                    base_url="http://x:8000",
                    model="qwen3-vl:32b",
                )
            )
        assert mock_loader.call_count == 1
        _, kwargs = mock_loader.call_args
        assert kwargs == {
            "model_name": "qwen3-vl:32b"
        }

    def test_random_mode_propagates_model_name(self, mock_openai_urlopen):
        """Random Character modes also propagate the model into the cascade
        (e.g. so a model family can have a creative override for Random Pony).
        Mocks: nodes.prompt_helper.get_system_prompt (returns canned string).
        """
        helper = PromptHelper()
        with patch(
            "comfyui_prompt_tools.nodes.prompt_helper.get_system_prompt",
            return_value="STUB SYS PROMPT",
        ) as mock_loader:
            helper.enhance(
                **_enhance_kwargs(
                    mode="Random Character (Pony)",
                    engine="vllm",
                    base_url="http://x:8000",
                    model="qwen3-vl:32b",
                    ethnicity_pool="japanese",
                    age_pool="25-35",
                )
            )
        assert mock_loader.call_count == 1
        _, kwargs = mock_loader.call_args
        assert kwargs == {
            "model_name": "qwen3-vl:32b"
        }

    def test_custom_mode_does_not_call_loader(self, mock_openai_urlopen):
        """Custom System Prompt bypasses the loader entirely — the user
        supplies the prompt directly, so no cascade applies.
        Mocks: nodes.prompt_helper.get_system_prompt (asserted never called).
        """
        helper = PromptHelper()
        with patch(
            "comfyui_prompt_tools.nodes.prompt_helper.get_system_prompt"
        ) as mock_loader:
            helper.enhance(
                **_enhance_kwargs(
                    mode="Custom System Prompt",
                    engine="vllm",
                    base_url="http://x:8000",
                    model="qwen3-vl:32b",
                    custom_system_prompt="user-supplied system prompt",
                )
            )
        assert mock_loader.call_count == 0
