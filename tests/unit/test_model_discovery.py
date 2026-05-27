"""Tests for engines/model_discovery.py.

Mocks urlopen at the import path inside the engine source module per
Standards §4a. Covers parser handling for all three response shapes,
URL building, and the discover_all batch flow.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from comfyui_prompt_tools.engines.model_discovery import (
    DiscoveryError,
    _build_discovery_url,
    _parse_response,
    discover_all,
    discover_models,
)


def _mock_response(payload: dict):
    """Build an urlopen-context-manager-compatible mock."""
    raw = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    return mock_resp


class TestParseResponse:
    def test_openai_parser(self):
        payload = {
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-4o-mini"},
            ]
        }
        assert _parse_response("openai", payload) == ["gpt-4o", "gpt-4o-mini"]

    def test_ollama_parser(self):
        payload = {
            "models": [
                {"name": "qwen3-vl:8b"},
                {"name": "gemma:7b"},
            ]
        }
        assert _parse_response("ollama", payload) == ["gemma:7b", "qwen3-vl:8b"]

    def test_parser_dedup_and_sort(self):
        payload = {
            "data": [
                {"id": "b"}, {"id": "a"}, {"id": "b"}, {"id": "c"},
            ]
        }
        assert _parse_response("openai", payload) == ["a", "b", "c"]

    def test_parser_skips_malformed_entries(self):
        payload = {"data": [{"id": "ok"}, "not-a-dict", {"no_id": True}]}
        assert _parse_response("openai", payload) == ["ok"]

    def test_unknown_parser_raises(self):
        with pytest.raises(DiscoveryError, match="Unknown parser"):
            _parse_response("invalid", {})


class TestBuildDiscoveryUrl:
    def test_openai_discovery_url(self):
        url = _build_discovery_url("openai", "https://api.openai.com/v1")
        assert url == "https://api.openai.com/v1/models"

    def test_ollama_discovery_url(self):
        url = _build_discovery_url("ollama", "http://localhost:11434")
        assert url == "http://localhost:11434/api/tags"

    def test_gemini_discovery_uses_openai_compat_endpoint(self):
        """Gemini discovery hits the OpenAI-compatible /v1beta/openai/models
        path so we can authenticate with a Bearer token. The native
        /v1beta/models endpoint rejects Bearer auth (Google wants
        ?key= or x-goog-api-key)."""
        url = _build_discovery_url(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        assert url == "https://generativelanguage.googleapis.com/v1beta/openai/models"

    def test_invalid_url_raises(self):
        with pytest.raises(DiscoveryError, match="Invalid base_url"):
            _build_discovery_url("openai", "not-a-url")


class TestDiscoverModels:
    def test_ollama_discovery_success(self):
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            return_value=_mock_response({
                "models": [{"name": "qwen3-vl:8b"}, {"name": "gemma:7b"}]
            }),
        ):
            result = discover_models("ollama", base_url="http://localhost:11434")
        assert result == ["gemma:7b", "qwen3-vl:8b"]

    def test_openai_discovery_with_api_key_sends_auth_header(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return _mock_response({"data": [{"id": "gpt-4o"}]})

        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = discover_models("openai", api_key="sk-test")
        assert result == ["gpt-4o"]
        # Header capitalisation varies by urllib version → normalise
        norm = {k.lower(): v for k, v in captured["headers"].items()}
        assert norm.get("authorization") == "Bearer sk-test"

    def test_cloud_provider_skipped_without_api_key(self):
        # No env var, no override, no patch needed: just resolves to None
        # and the function returns [] without an HTTP call.
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen"
        ) as mock_urlopen:
            result = discover_models("openai", api_key=None)
        assert result == []
        mock_urlopen.assert_not_called()

    def test_local_provider_works_without_api_key(self):
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            return_value=_mock_response({"models": [{"name": "x"}]}),
        ) as mock_urlopen:
            result = discover_models("ollama")
        assert result == ["x"]
        mock_urlopen.assert_called_once()

    def test_401_raises_discovery_error_with_helpful_message(self):
        err = urllib.error.HTTPError(
            "http://x", 401, "Unauthorized", {}, BytesIO(b"")
        )
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(DiscoveryError, match="authentication failed"):
                discover_models("openai", api_key="bad-key")

    def test_unreachable_raises_discovery_error(self):
        err = urllib.error.URLError("connection refused")
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(DiscoveryError, match="cannot reach"):
                discover_models("ollama")

    def test_malformed_json_raises_discovery_error(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            with pytest.raises(DiscoveryError, match="not valid JSON"):
                discover_models("ollama")

    def test_unknown_provider_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown provider"):
            discover_models("nonexistent")


class TestDiscoverAll:
    def test_discover_all_continues_on_per_provider_failures(self):
        """One provider failing must not abort the batch."""
        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            if "openai.com" in req.full_url:
                raise urllib.error.URLError("unreachable")
            return _mock_response({"models": [{"name": "x"}]})

        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = discover_all()
        # ollama succeeded, openai failed → omitted from result
        # claude/gemini skipped (no key), vllm tries localhost:8000
        assert "ollama" in result
        assert "openai" not in result


# ==========================================================================
# v0.10.3: chat-model filtering
# ==========================================================================


class TestChatModelFilter:
    """_filter_chat_models drops non-chat entries from each provider's list."""

    def test_openai_filter_drops_tts_and_transcribe(self):
        models = [
            "gpt-4o-mini",
            "gpt-4o-mini-tts",
            "gpt-4o-mini-transcribe",
            "gpt-4o",
        ]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o-mini", "gpt-4o"]

    def test_openai_filter_drops_legacy_completion(self):
        models = ["gpt-4o", "babbage-002", "davinci-002", "gpt-3.5-turbo"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o"]

    def test_openai_filter_drops_image_audio_realtime(self):
        models = [
            "gpt-4o", "gpt-4o-audio-preview", "gpt-4o-realtime-preview",
            "chatgpt-image-latest", "gpt-image-1", "dall-e-3",
        ]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o"]

    def test_openai_filter_drops_embeddings_whisper(self):
        models = ["gpt-4o", "text-embedding-3-small", "whisper-1"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o"]

    def test_openai_filter_drops_search_preview(self):
        models = ["gpt-4o", "gpt-4o-mini-search-preview", "gpt-4o-search-preview"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o"]

    def test_openai_filter_drops_codex_anywhere_in_name(self):
        """v0.10.4: 'codex' is matched as substring, so it catches both
        legacy 'codex-mini-latest' and newer 'gpt-5-codex' variants."""
        models = ["codex-mini-latest", "gpt-5-codex", "gpt-4o-mini"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o-mini"]

    def test_openai_filter_drops_sora(self):
        """v0.10.4: 'sora' video-gen models are filtered as substring."""
        models = ["sora-2", "sora-2-pro", "gpt-4o"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o"]

    def test_shared_filter_drops_embedding_substring(self):
        """v0.10.4: 'embedding' is a substring, covering OpenAI's
        text-embedding-* and Gemini's gemini-embedding-* alike."""
        models = ["gpt-4o", "text-embedding-3-small", "gemini-embedding-001"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", models) == ["gpt-4o"]

    def test_gemini_filter_drops_nano_banana(self):
        """v0.10.4: Gemini Flash Image (codename 'nano-banana') is image-gen."""
        models = ["gemini-2.5-flash", "gemini-2.5-flash-image-preview", "nano-banana"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("gemini", models) == ["gemini-2.5-flash"]

    def test_gemini_filter_drops_deep_research(self):
        """v0.10.4: deep-research is a specialised mode, not free chat."""
        models = ["gemini-2.5-pro", "gemini-2.5-pro-deep-research"]
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("gemini", models) == ["gemini-2.5-pro"]

    def test_ollama_filter_passes_everything(self):
        """Ollama lists are user-curated locally — no provider blacklist."""
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        models = ["qwen3-vl:8b", "gemma:7b"]
        assert _filter_chat_models("ollama", models) == models

    def test_empty_input_returns_empty(self):
        from comfyui_prompt_tools.engines.model_discovery import _filter_chat_models
        assert _filter_chat_models("openai", []) == []

    def test_discover_models_applies_filter(self):
        """End-to-end: discover_models() runs the filter automatically."""
        import json
        from unittest.mock import patch, MagicMock
        payload = {"data": [
            {"id": "gpt-4o-mini"},
            {"id": "gpt-4o-mini-tts"},
            {"id": "babbage-002"},
            {"id": "gpt-4.1"},
        ]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        with patch(
            "comfyui_prompt_tools.engines.model_discovery.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            from comfyui_prompt_tools.engines.model_discovery import discover_models
            result = discover_models("openai", api_key="sk-test")
        assert result == ["gpt-4.1", "gpt-4o-mini"]
