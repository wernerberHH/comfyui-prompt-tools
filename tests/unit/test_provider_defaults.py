"""Tests for engines/provider_defaults.py.

Covers the static PROVIDERS map shape, PROVIDER_CHOICES content, and
the normalize_url helper for the three normalisation rules.
"""

from __future__ import annotations

import pytest

from comfyui_prompt_tools.engines.provider_defaults import (
    ENGINE_TOOLTIP,
    PROVIDER_CHOICES,
    PROVIDERS,
    normalize_url,
)


class TestProvidersMap:
    def test_all_five_providers_registered(self):
        assert set(PROVIDERS.keys()) == {
            "ollama", "vllm", "openai", "claude", "gemini"
        }

    def test_provider_choices_matches_providers(self):
        assert set(PROVIDER_CHOICES) == set(PROVIDERS.keys())

    def test_provider_choices_is_list_with_stable_order(self):
        """Order matters for the dropdown default — first entry is used
        when nothing else picks."""
        assert isinstance(PROVIDER_CHOICES, list)
        assert PROVIDER_CHOICES[0] == "ollama"

    @pytest.mark.parametrize("provider", ["ollama", "vllm", "openai", "claude", "gemini"])
    def test_every_provider_has_required_fields(self, provider):
        spec = PROVIDERS[provider]
        for key in (
            "default_url", "auth_env_var", "url_normalization",
            "discovery_endpoint", "discovery_parser",
        ):
            assert key in spec, f"{provider} missing {key}"

    def test_local_providers_have_no_auth_env_var(self):
        assert PROVIDERS["ollama"]["auth_env_var"] is None
        assert PROVIDERS["vllm"]["auth_env_var"] is None

    def test_cloud_providers_have_standard_env_var(self):
        assert PROVIDERS["openai"]["auth_env_var"] == "OPENAI_API_KEY"
        assert PROVIDERS["claude"]["auth_env_var"] == "OPENROUTER_API_KEY"
        assert PROVIDERS["gemini"]["auth_env_var"] == "GEMINI_API_KEY"

    def test_default_urls_are_https_for_cloud(self):
        for p in ("openai", "claude", "gemini"):
            assert PROVIDERS[p]["default_url"].startswith("https://"), (
                f"{p} default_url should be HTTPS for Cloud safety"
            )


class TestEngineTooltip:
    def test_tooltip_mentions_all_five_engines(self):
        for engine in PROVIDER_CHOICES:
            assert engine in ENGINE_TOOLTIP

    def test_tooltip_mentions_each_cloud_env_var(self):
        assert "OPENAI_API_KEY" in ENGINE_TOOLTIP
        assert "OPENROUTER_API_KEY" in ENGINE_TOOLTIP
        assert "GEMINI_API_KEY" in ENGINE_TOOLTIP


class TestNormalizeUrl:
    """The three rules: no_v1, with_v1, as_is."""

    # no_v1 (ollama)
    def test_no_v1_strips_trailing_v1(self):
        assert normalize_url("ollama", "http://x:11434/v1") == "http://x:11434"

    def test_no_v1_leaves_url_without_v1_alone(self):
        assert normalize_url("ollama", "http://x:11434") == "http://x:11434"

    def test_no_v1_strips_trailing_slash(self):
        assert normalize_url("ollama", "http://x:11434/") == "http://x:11434"

    # with_v1 (vllm, openai, claude)
    def test_with_v1_appends_v1_when_missing(self):
        assert normalize_url("vllm", "http://x:8000") == "http://x:8000/v1"

    def test_with_v1_leaves_url_with_v1_alone(self):
        assert normalize_url("openai", "https://api.openai.com/v1") == (
            "https://api.openai.com/v1"
        )

    def test_with_v1_handles_trailing_slash(self):
        assert normalize_url("claude", "https://openrouter.ai/api/v1/") == (
            "https://openrouter.ai/api/v1"
        )

    # as_is (gemini)
    def test_as_is_preserves_full_path(self):
        url = "https://generativelanguage.googleapis.com/v1beta/openai"
        assert normalize_url("gemini", url) == url

    def test_as_is_still_strips_trailing_slash(self):
        url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert normalize_url("gemini", url) == url.rstrip("/")

    # edge cases
    def test_empty_url_returns_empty(self):
        assert normalize_url("ollama", "") == ""

    def test_unknown_provider_passes_through(self):
        """Unknown provider must not crash — just strip slashes."""
        assert normalize_url("nonexistent", "http://x/") == "http://x"
