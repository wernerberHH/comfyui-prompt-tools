"""Tests for BasePromptNode engine factory and call helpers.

Mocks: only ``OllamaChatEngine.chat`` is patched (in
test_call_engine_handles_connection_error_gracefully) to raise OllamaError.
The mock_openai_urlopen / mock_ollama_urlopen fixtures from conftest patch
``urllib.request.urlopen`` inside the engine modules to keep HTTP off-network.
The resolve-engine tests instantiate the engine classes directly and inspect
their attributes — no patching needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from comfyui_prompt_tools.engines import (
    OllamaChatEngine,
    OllamaError,
    OpenAIChatEngine,
)
from comfyui_prompt_tools.nodes.base_prompt_node import (
    BasePromptNode,
    ENGINE_CHOICES,
)


class TestBuildEngineInputs:
    def test_contains_engine_dropdown(self):
        """The engine field is the choice list and not a free text."""
        inputs = BasePromptNode._build_engine_inputs()
        assert inputs["engine"][0] == ENGINE_CHOICES

    def test_contains_url_model_temperature(self):
        """The fragment exposes base_url, model, temperature."""
        inputs = BasePromptNode._build_engine_inputs()
        for field in ("base_url", "model", "temperature"):
            assert field in inputs

    def test_build_engine_inputs_combines_urls_from_all_engines(self):
        """The base_url Combo lists URLs from every engine in ENGINE_CHOICES,
        not just the default one. ComfyUI's INPUT_TYPES is static, so vLLM
        users must still see vLLM URLs even when the default engine is ollama.

        Mocks: ``load_endpoints`` patched at the consuming-module import path
        (base_prompt_node) per Standards §4a, returning a config with both
        engines populated.
        """
        fake_config = {
            "engines": {
                "ollama": [
                    {"url": "http://ollama-host:11434", "models": ["qwen3-vl:8b"]},
                ],
                "vllm": [
                    {"url": "http://vllm-host:8000/v1", "models": ["Qwen/Q2.5-7B"]},
                ],
            }
        }
        with patch(
            "comfyui_prompt_tools.nodes.base_prompt_node.load_endpoints",
            return_value=fake_config,
        ):
            inputs = BasePromptNode._build_engine_inputs()
        url_choices, url_meta = inputs["base_url"]
        assert isinstance(url_choices, list)
        assert "http://ollama-host:11434" in url_choices
        assert "http://vllm-host:8000/v1" in url_choices
        # Default = first URL of the first engine (ollama before vllm in
        # ENGINE_CHOICES) → ollama URL.
        assert url_meta["default"] == "http://ollama-host:11434"

    def test_build_engine_inputs_combines_models_from_all_engines(self):
        """The model Combo lists models from every (engine, url) pair, sorted
        and deduplicated, so vLLM models surface even with the ollama default.

        Mocks: ``load_endpoints`` patched at the consuming-module import path
        (base_prompt_node) per Standards §4a.
        """
        fake_config = {
            "engines": {
                "ollama": [
                    {
                        "url": "http://ollama-host:11434",
                        "models": ["qwen3-vl:8b", "llama3.2:3b"],
                    },
                ],
                "vllm": [
                    {
                        "url": "http://vllm-host:8000/v1",
                        "models": ["Qwen/Qwen2.5-7B-Instruct", "qwen3-vl:8b"],
                    },
                ],
            }
        }
        with patch(
            "comfyui_prompt_tools.nodes.base_prompt_node.load_endpoints",
            return_value=fake_config,
        ):
            inputs = BasePromptNode._build_engine_inputs()
        model_choices, model_meta = inputs["model"]
        # v0.10.3: models are prefixed with "[engine] " so users can see
        # which backend each model belongs to. Order: engines in
        # ENGINE_CHOICES order; within each engine, models alphabetic.
        # ollama -> llama3.2:3b, qwen3-vl:8b
        # vllm   -> Qwen/Qwen2.5-7B-Instruct, qwen3-vl:8b
        assert model_choices == [
            "[ollama] llama3.2:3b",
            "[ollama] qwen3-vl:8b",
            "[vllm] Qwen/Qwen2.5-7B-Instruct",
            "[vllm] qwen3-vl:8b",
        ]
        # Default = first model of the first (engine, url) pair, prefixed
        # with that engine. First entry in the ollama models list (NOT the
        # alphabetic first) for backward-compat with the previous default
        # behaviour.
        assert model_meta["default"] == "[ollama] qwen3-vl:8b"

    def test_build_engine_inputs_falls_back_to_string_when_config_empty(self):
        """An empty config degrades both fields to plain STRING text inputs
        with the hardcoded ollama defaults. Keeps the node usable without
        pyyaml or an endpoints.yaml file.

        Mocks: ``load_endpoints`` patched at the consuming-module import path
        (base_prompt_node) per Standards §4a, returning ``{}``.
        """
        with patch(
            "comfyui_prompt_tools.nodes.base_prompt_node.load_endpoints",
            return_value={},
        ):
            inputs = BasePromptNode._build_engine_inputs()
        # Both fields are STRING type, not a list of choices
        assert inputs["base_url"][0] == "STRING"
        assert inputs["model"][0] == "STRING"
        # Defaults match the module's hardcoded fallbacks
        from comfyui_prompt_tools.nodes.base_prompt_node import (
            DEFAULT_OLLAMA_MODEL,
            DEFAULT_OLLAMA_URL,
        )
        assert inputs["base_url"][1]["default"] == DEFAULT_OLLAMA_URL
        assert inputs["model"][1]["default"] == DEFAULT_OLLAMA_MODEL


class TestResolveEngine:
    def test_resolve_engine_ollama_returns_ollama_engine(self):
        """ollama choice returns an OllamaChatEngine with the given base_url."""
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://localhost:11434",
            model="qwen3-vl:8b",
        )
        assert isinstance(engine, OllamaChatEngine)
        assert engine.base_url == "http://localhost:11434"
        assert engine.model == "qwen3-vl:8b"

    def test_resolve_engine_ollama_strips_trailing_v1(self):
        """An ollama URL accidentally containing /v1 is normalised away."""
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://localhost:11434/v1",
            model="x",
        )
        assert isinstance(engine, OllamaChatEngine)
        assert engine.base_url == "http://localhost:11434"

    def test_resolve_engine_ollama_forwards_keep_alive(self):
        """keep_alive is forwarded to OllamaChatEngine when supplied."""
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://x",
            model="m",
            keep_alive="60s",
        )
        assert engine.keep_alive == "60s"

    def test_resolve_engine_vllm_returns_openai_engine(self):
        """vllm choice returns an OpenAIChatEngine with /v1 ensured."""
        engine = BasePromptNode._resolve_engine(
            engine="vllm",
            base_url="http://localhost:8000",
            model="Qwen/Qwen3-7B",
        )
        assert isinstance(engine, OpenAIChatEngine)
        assert engine.api_url == "http://localhost:8000/v1"
        assert engine.model == "Qwen/Qwen3-7B"

    def test_resolve_engine_vllm_keeps_existing_v1(self):
        """A vllm URL that already ends in /v1 is left untouched."""
        engine = BasePromptNode._resolve_engine(
            engine="vllm",
            base_url="http://localhost:8000/v1",
            model="x",
        )
        assert engine.api_url == "http://localhost:8000/v1"

    def test_resolve_engine_invalid_raises(self):
        """Unknown engine choice raises ValueError listing the valid ones."""
        with pytest.raises(ValueError, match="Unknown engine"):
            BasePromptNode._resolve_engine(
                engine="nonexistent_xyz",
                base_url="http://localhost",
                model="x",
            )


class TestCallEngine:
    def test_call_engine_passes_images_through_for_vision_models(
        self, mock_openai_urlopen
    ):
        """images_b64 are forwarded into the OpenAI multipart payload.
        Mocks: openai_client.urllib.request.urlopen (canned 'test output').
        """
        engine = BasePromptNode._resolve_engine(
            engine="vllm",
            base_url="http://x/v1",
            model="qwen-vl",
        )
        out = BasePromptNode._call_engine(
            engine,
            system_prompt="sys",
            user_message="msg",
            images_b64=["base64data"],
        )
        assert out == "test output"
        body = mock_openai_urlopen.call_args[0][0].data.decode()
        assert "image_url" in body
        assert "base64data" in body

    def test_call_engine_passes_images_through_for_ollama_vision(
        self, mock_ollama_urlopen
    ):
        """images_b64 are forwarded into the Ollama 'images' array.
        Mocks: ollama_client.urllib.request.urlopen (canned 'test output').
        """
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://x",
            model="qwen3-vl:8b",
        )
        out = BasePromptNode._call_engine(
            engine,
            system_prompt="sys",
            user_message="msg",
            images_b64=["b64a", "b64b"],
        )
        assert out == "test output"
        body = mock_ollama_urlopen.call_args[0][0].data.decode()
        assert '"images"' in body
        assert "b64a" in body and "b64b" in body

    def test_call_engine_handles_connection_error_gracefully(self):
        """When chat raises OllamaError, _call_engine propagates the type
        unchanged so subclasses can render a user-facing ERROR string.
        Mocks: OllamaChatEngine.chat (instance-level patch raising OllamaError).
        """
        engine = OllamaChatEngine(base_url="http://x", model="m")
        with patch.object(engine, "chat", side_effect=OllamaError("boom")):
            with pytest.raises(OllamaError, match="boom"):
                BasePromptNode._call_engine(engine, "sys", "msg")


# ==========================================================================
# v0.10.3: Engine-prefix strip in _resolve_engine
# ==========================================================================


class TestResolveEngineStripsPrefix:
    """Models flowing in from the dropdown carry a "[engine] " prefix; the
    resolve step must remove it so the engine client sees the bare ID."""

    def test_prefix_stripped_for_ollama(self):
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://localhost:11434",
            model="[ollama] qwen3-vl:8b",
            timeout=180,
        )
        # OllamaChatEngine stores model on the instance
        assert engine.model == "qwen3-vl:8b"

    def test_prefix_stripped_for_openai(self):
        engine = BasePromptNode._resolve_engine(
            engine="openai",
            base_url="https://api.openai.com/v1",
            model="[openai] gpt-4o-mini",
            timeout=180,
        )
        assert engine.model == "gpt-4o-mini"

    def test_no_prefix_passes_through(self):
        """Bare model names (no prefix) keep working — important for
        backward-compat with existing workflows."""
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://localhost:11434",
            model="qwen3-vl:8b",
            timeout=180,
        )
        assert engine.model == "qwen3-vl:8b"

    def test_mismatched_prefix_engine_field_wins(self):
        """If the user picks engine=ollama but a [openai]-prefixed model,
        the engine field wins and the prefix is silently stripped. The
        request will then fail at runtime (model not on ollama) — that is
        the documented behaviour."""
        engine = BasePromptNode._resolve_engine(
            engine="ollama",
            base_url="http://localhost:11434",
            model="[openai] gpt-4o-mini",
            timeout=180,
        )
        assert engine.model == "gpt-4o-mini"
        # And it's an ollama engine, not openai
        from comfyui_prompt_tools.engines import OllamaChatEngine
        assert isinstance(engine, OllamaChatEngine)
