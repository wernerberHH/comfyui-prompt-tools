"""Tests for comfyui_prompt_tools/web_api.py.

Covers the pure-function helpers (status, save, test, discover) by
redirecting the module-level path constants into tmp_path. The aiohttp
route layer is not exercised here — it's a thin wrapper around these
helpers and would need a ComfyUI server to run.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from comfyui_prompt_tools import web_api
from comfyui_prompt_tools.engines import api_key_resolver
from comfyui_prompt_tools.engines.model_discovery import DiscoveryError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip env vars that would interfere with resolver-based tests."""
    for var in (
        "COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI",
        "COMFYUI_PROMPT_TOOLS_API_KEY_CLAUDE",
        "COMFYUI_PROMPT_TOOLS_API_KEY_GEMINI",
        "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect both api_keys.yaml and endpoints.yaml into tmp_path."""
    api_keys = tmp_path / "config" / "api_keys.yaml"
    endpoints = tmp_path / "config" / "endpoints.yaml"
    api_keys.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(web_api, "_API_KEYS_PATH",   api_keys)
    monkeypatch.setattr(web_api, "_ENDPOINTS_PATH",  endpoints)
    monkeypatch.setattr(api_key_resolver, "_PROJECT_CONFIG_PATH", api_keys)
    monkeypatch.setattr(api_key_resolver, "_USER_CONFIG_PATH",    api_keys)
    monkeypatch.setattr(api_key_resolver, "_warned_perms", set())
    return {"api_keys": api_keys, "endpoints": endpoints}


def _read_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text()) or {}


# ----- get_provider_status -------------------------------------------------

class TestGetProviderStatus:
    def test_returns_all_five_providers(self, isolated_paths):
        status = web_api.get_provider_status()
        assert set(status.keys()) == {"ollama", "vllm", "openai", "claude", "gemini"}

    def test_local_providers_marked_needs_key_false(self, isolated_paths):
        status = web_api.get_provider_status()
        assert status["ollama"]["needs_key"] is False
        assert status["vllm"]["needs_key"] is False

    def test_cloud_providers_marked_needs_key_true(self, isolated_paths):
        status = web_api.get_provider_status()
        for p in ("openai", "claude", "gemini"):
            assert status[p]["needs_key"] is True

    def test_configured_false_when_no_key(self, isolated_paths):
        status = web_api.get_provider_status()
        assert status["openai"]["configured"] is False

    def test_configured_true_after_key_saved(self, isolated_paths):
        web_api.save_provider_config({"openai": {"api_key": "sk-x"}})
        status = web_api.get_provider_status()
        assert status["openai"]["configured"] is True

    def test_status_never_contains_the_key_itself(self, isolated_paths):
        """Security: the key must never be returned to the client."""
        web_api.save_provider_config({"openai": {"api_key": "sk-very-secret"}})
        status = web_api.get_provider_status()
        flat = repr(status)
        assert "sk-very-secret" not in flat

    def test_default_url_always_present(self, isolated_paths):
        status = web_api.get_provider_status()
        for p, s in status.items():
            assert s["default_url"].startswith("http")


# ----- save_provider_config ------------------------------------------------

class TestSaveProviderConfig:
    def test_save_writes_yaml(self, isolated_paths):
        r = web_api.save_provider_config({"openai": {"api_key": "sk-x"}})
        assert r["ok"] is True
        assert "openai" in r["written"]
        data = _read_yaml(isolated_paths["api_keys"])
        assert data["providers"]["openai"]["api_key"] == "sk-x"

    def test_save_merges_with_existing(self, isolated_paths):
        web_api.save_provider_config({"openai": {"api_key": "sk-x"}})
        web_api.save_provider_config({"claude": {"api_key": "sk-or-y"}})
        data = _read_yaml(isolated_paths["api_keys"])
        assert data["providers"]["openai"]["api_key"] == "sk-x"
        assert data["providers"]["claude"]["api_key"] == "sk-or-y"

    def test_empty_string_removes_field(self, isolated_paths):
        web_api.save_provider_config({"openai": {"api_key": "sk-x"}})
        web_api.save_provider_config({"openai": {"api_key": ""}})
        data = _read_yaml(isolated_paths["api_keys"])
        # api_key removed; provider entry gone (only had that field)
        assert "openai" not in (data.get("providers") or {})

    def test_url_override_saved(self, isolated_paths):
        web_api.save_provider_config({"openai": {
            "api_key": "k", "url": "https://proxy/v1"
        }})
        data = _read_yaml(isolated_paths["api_keys"])
        assert data["providers"]["openai"]["url"] == "https://proxy/v1"

    def test_url_can_be_set_without_key(self, isolated_paths):
        """Just changing the URL must not require also re-entering the key."""
        web_api.save_provider_config({"openai": {"url": "https://proxy/v1"}})
        data = _read_yaml(isolated_paths["api_keys"])
        assert data["providers"]["openai"]["url"] == "https://proxy/v1"
        assert "api_key" not in data["providers"]["openai"]

    def test_unknown_provider_rejected(self, isolated_paths):
        r = web_api.save_provider_config({"nonexistent": {"api_key": "x"}})
        assert r["ok"] is False
        assert "Unknown providers" in r["error"]

    def test_empty_update_is_noop(self, isolated_paths):
        r = web_api.save_provider_config({})
        assert r["ok"] is True
        assert r["written"] == []


# ----- test_provider_connection -------------------------------------------

class TestTestProviderConnection:
    def test_success_returns_model_count(self, isolated_paths):
        with patch(
            "comfyui_prompt_tools.web_api.resolve_api_key", return_value="sk-test"
        ), patch(
            "comfyui_prompt_tools.web_api.discover_models",
            return_value=["gpt-4o", "gpt-4o-mini"],
        ):
            r = web_api.test_provider_connection("openai")
        assert r["ok"] is True
        assert r["model_count"] == 2

    def test_discovery_error_surfaced(self, isolated_paths):
        with patch(
            "comfyui_prompt_tools.web_api.resolve_api_key", return_value="sk-test"
        ), patch(
            "comfyui_prompt_tools.web_api.discover_models",
            side_effect=DiscoveryError("authentication failed"),
        ):
            r = web_api.test_provider_connection("openai")
        assert r["ok"] is False
        assert "authentication failed" in r["error"]

    def test_unknown_provider_rejected(self, isolated_paths):
        r = web_api.test_provider_connection("nonexistent")
        assert r["ok"] is False
        assert "Unknown provider" in r["error"]

    def test_no_api_key_reports_not_ok(self, isolated_paths):
        """v0.10.4 Bug #16: providers without a configured key must
        return ok=False instead of being shown as green by the UI."""
        with patch(
            "comfyui_prompt_tools.web_api.resolve_api_key", return_value=None
        ):
            r = web_api.test_provider_connection("openai")
        assert r["ok"] is False
        assert "no API key" in r["error"]

    def test_empty_discovery_reports_not_ok(self, isolated_paths):
        """v0.10.4 Bug #16: when discovery succeeds but returns zero
        models, the UI must not show green."""
        with patch(
            "comfyui_prompt_tools.web_api.resolve_api_key", return_value="sk-test"
        ), patch(
            "comfyui_prompt_tools.web_api.discover_models", return_value=[],
        ):
            r = web_api.test_provider_connection("openai")
        assert r["ok"] is False
        assert "0 models" in r["error"]

    def test_keyless_provider_does_not_require_key(self, isolated_paths):
        """Providers with auth_env_var=None (ollama, vllm) must still
        be testable without a key — they aren't gated."""
        with patch(
            "comfyui_prompt_tools.web_api.discover_models",
            return_value=["llama3:8b"],
        ):
            r = web_api.test_provider_connection("ollama")
        assert r["ok"] is True
        assert r["model_count"] == 1


# ----- run_discovery ------------------------------------------------------

class TestRunDiscovery:
    def test_discovery_writes_endpoints_yaml(self, isolated_paths):
        with patch(
            "comfyui_prompt_tools.web_api.discover_models",
            return_value=["m1", "m2"],
        ):
            r = web_api.run_discovery("ollama")
        assert r["ok"] is True
        data = _read_yaml(isolated_paths["endpoints"])
        ollama_entries = data["engines"]["ollama"]
        assert ollama_entries[0]["models"] == ["m1", "m2"]

    def test_discovery_preserves_user_url(self, isolated_paths):
        """User-set URLs in endpoints.yaml must not be overwritten."""
        isolated_paths["endpoints"].write_text(
            "engines:\n  ollama:\n    - url: 'http://custom:11434'\n      models: []\n"
        )
        with patch(
            "comfyui_prompt_tools.web_api.discover_models",
            return_value=["new-model"],
        ):
            web_api.run_discovery("ollama")
        data = _read_yaml(isolated_paths["endpoints"])
        assert data["engines"]["ollama"][0]["url"] == "http://custom:11434"
        assert data["engines"]["ollama"][0]["models"] == ["new-model"]

    def test_discovery_makes_backup(self, isolated_paths):
        isolated_paths["endpoints"].write_text("engines: {}\n")
        with patch(
            "comfyui_prompt_tools.web_api.discover_models",
            return_value=[],
        ):
            web_api.run_discovery("ollama")
        backup = isolated_paths["endpoints"].with_suffix(".yaml.backup")
        assert backup.is_file()

    def test_discovery_continues_on_per_provider_failure(self, isolated_paths):
        """Failing provider must not abort the others."""
        def side_effect(provider, *args, **kwargs):
            if provider == "openai":
                raise DiscoveryError("auth failed")
            return ["m"]

        with patch(
            "comfyui_prompt_tools.web_api.discover_models",
            side_effect=side_effect,
        ):
            r = web_api.run_discovery(None)
        assert r["ok"] is True
        assert r["results"]["openai"]["ok"] is False
        assert r["results"]["ollama"]["ok"] is True

    def test_unknown_single_provider_rejected(self, isolated_paths):
        r = web_api.run_discovery("nonexistent")
        assert r["ok"] is False
