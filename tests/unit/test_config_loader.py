"""Tests for the endpoints.yaml config loader.

The loader reads ``config/endpoints.yaml`` with fallback to
``endpoints.yaml.example``. It must never raise — every failure path
returns an empty dict / empty list so the node UI degrades gracefully.

These tests use ``tmp_path`` plus ``monkeypatch`` against the module-level
path constants to keep the real ``config/`` directory untouched. The cache
is reset per test via the autouse fixture below.

Mocks: monkeypatching of ``config_loader._USER_FILE`` /
``config_loader._EXAMPLE_FILE`` / ``config_loader._cache`` is done in this
file because it is path-state setup, not behaviour mocking. The loader's
public functions are exercised end-to-end against real YAML files written
to ``tmp_path`` — no mock of pyyaml itself.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from comfyui_prompt_tools import config_loader
from comfyui_prompt_tools.config_loader import (
    get_models_for_engine_url,
    get_urls_for_engine,
    load_endpoints,
)


@pytest.fixture(autouse=True)
def _redirect_config_paths(tmp_path: Path, monkeypatch):
    """Point the loader at tmp_path for every test and clear the cache.

    Each test writes whatever ``endpoints.yaml`` / ``endpoints.yaml.example``
    it needs to ``tmp_path`` and the loader picks them up.
    """
    user_file = tmp_path / "endpoints.yaml"
    example_file = tmp_path / "endpoints.yaml.example"
    monkeypatch.setattr(config_loader, "_USER_FILE", user_file)
    monkeypatch.setattr(config_loader, "_EXAMPLE_FILE", example_file)
    monkeypatch.setattr(config_loader, "_cache", None)
    yield (user_file, example_file)


_EXAMPLE_YAML = textwrap.dedent(
    """
    engines:
      ollama:
        - url: "http://example-ollama:11434"
          models:
            - "qwen3-vl:8b"
            - "llama3.2:3b"
      vllm:
        - url: "http://example-vllm:8000/v1"
          models:
            - "Qwen/Qwen2.5-7B-Instruct"
    """
).strip()


_USER_YAML = textwrap.dedent(
    """
    engines:
      ollama:
        - url: "http://my-ollama:11434"
          models: ["custom-model:7b"]
    """
).strip()


class TestLoadEndpoints:
    def test_loads_example_when_no_user_config(self, _redirect_config_paths):
        """Without endpoints.yaml the .example file is the source of truth.
        Mocks: none — real YAML parsed via pyyaml.
        """
        user_file, example_file = _redirect_config_paths
        example_file.write_text(_EXAMPLE_YAML, encoding="utf-8")

        config = load_endpoints(refresh=True)
        assert "engines" in config
        assert "ollama" in config["engines"]
        # Example URL surfaces through
        assert config["engines"]["ollama"][0]["url"] == "http://example-ollama:11434"

    def test_loads_user_config_overrides_example(self, _redirect_config_paths):
        """When both files exist, endpoints.yaml wins over .example entirely.
        Mocks: none — real YAML parsed via pyyaml.
        """
        user_file, example_file = _redirect_config_paths
        example_file.write_text(_EXAMPLE_YAML, encoding="utf-8")
        user_file.write_text(_USER_YAML, encoding="utf-8")

        config = load_endpoints(refresh=True)
        # User config's ollama entry replaces the example's
        assert config["engines"]["ollama"][0]["url"] == "http://my-ollama:11434"
        # User config has no vllm key — it should NOT be backfilled from .example
        assert "vllm" not in config["engines"]

    def test_returns_empty_dict_on_malformed_yaml(self, _redirect_config_paths):
        """Garbage YAML in the user file degrades to {} without raising.
        Mocks: none — pyyaml raises YAMLError, the loader swallows it.
        """
        user_file, example_file = _redirect_config_paths
        user_file.write_text("engines: [unterminated", encoding="utf-8")

        config = load_endpoints(refresh=True)
        # Malformed user file → fall through to example. Example absent → {}.
        assert config == {}

    def test_returns_empty_dict_when_no_files_present(self, _redirect_config_paths):
        """Both files absent → empty dict (never raises).
        Mocks: none.
        """
        # Neither file written.
        config = load_endpoints(refresh=True)
        assert config == {}


class TestGetUrlsForEngine:
    def test_get_urls_for_engine_returns_list(self, _redirect_config_paths):
        """ollama returns its configured URL list.
        Mocks: none.
        """
        user_file, example_file = _redirect_config_paths
        example_file.write_text(_EXAMPLE_YAML, encoding="utf-8")

        urls = get_urls_for_engine("ollama")
        assert urls == ["http://example-ollama:11434"]

    def test_missing_engine_returns_empty_list(self, _redirect_config_paths):
        """An unknown engine returns an empty list, not an error.
        Mocks: none.
        """
        user_file, example_file = _redirect_config_paths
        example_file.write_text(_EXAMPLE_YAML, encoding="utf-8")

        assert get_urls_for_engine("claude") == []


class TestGetModelsForEngineUrl:
    def test_get_models_for_engine_url_returns_list(self, _redirect_config_paths):
        """The model list for a known (engine, url) pair is returned in order.
        Mocks: none.
        """
        user_file, example_file = _redirect_config_paths
        example_file.write_text(_EXAMPLE_YAML, encoding="utf-8")

        models = get_models_for_engine_url("ollama", "http://example-ollama:11434")
        assert models == ["qwen3-vl:8b", "llama3.2:3b"]

    def test_unknown_url_returns_empty_list(self, _redirect_config_paths):
        """A URL that's not in the config returns an empty list.
        Mocks: none.
        """
        user_file, example_file = _redirect_config_paths
        example_file.write_text(_EXAMPLE_YAML, encoding="utf-8")

        assert get_models_for_engine_url("ollama", "http://nowhere:11434") == []


class TestShippedExample:
    """Smoke test against the shipped ``config/endpoints.yaml.example``.

    Verifies that the example file in the repo parses cleanly and exposes
    a reasonable out-of-the-box setup (localhost URLs for ollama and vllm,
    at least one model per local engine, cloud providers present with
    empty model lists ready for discovery). This guards against accidental
    edits that would break the autocomplete dropdown in the node UI.

    Mocks: ``config_loader._EXAMPLE_FILE`` is repointed at the real
    committed example file (the autouse fixture would otherwise redirect
    it at an empty tmp_path). ``_USER_FILE`` stays redirected at an empty
    tmp_path so the user-file branch is not taken in this test.
    """

    @pytest.fixture
    def _shipped_example_loaded(self, _redirect_config_paths, monkeypatch):
        """Repoint ``_EXAMPLE_FILE`` at the real committed example and
        return the parsed config dict."""
        repo_root = Path(__file__).resolve().parents[2]
        real_example = repo_root / "config" / "endpoints.yaml.example"
        assert real_example.is_file(), (
            f"shipped example missing at {real_example}"
        )
        monkeypatch.setattr(config_loader, "_EXAMPLE_FILE", real_example)
        monkeypatch.setattr(config_loader, "_cache", None)
        return load_endpoints(refresh=True)

    def test_shipped_example_parses_cleanly(self, _shipped_example_loaded):
        """The committed example file must parse to a non-empty config."""
        assert _shipped_example_loaded, "shipped example parsed to empty dict"
        assert "engines" in _shipped_example_loaded

    def test_shipped_example_exposes_localhost_ollama(
        self, _shipped_example_loaded
    ):
        """Localhost Ollama must be a configured URL for the out-of-the-box
        single-host experience."""
        urls = get_urls_for_engine("ollama")
        assert "http://localhost:11434" in urls, (
            "shipped example must list http://localhost:11434 for the "
            "documented single-host setup"
        )

    def test_shipped_example_exposes_localhost_vllm(
        self, _shipped_example_loaded
    ):
        """Localhost vLLM must be a configured URL for the out-of-the-box
        single-host experience."""
        urls = get_urls_for_engine("vllm")
        assert "http://localhost:8000/v1" in urls, (
            "shipped example must list http://localhost:8000/v1 for the "
            "documented single-host setup"
        )

    def test_shipped_example_localhost_ollama_has_models(
        self, _shipped_example_loaded
    ):
        """The localhost Ollama entry must list at least one model so the
        node dropdown is not empty on a fresh install."""
        models = get_models_for_engine_url("ollama", "http://localhost:11434")
        assert models, (
            "shipped example must list at least one model for "
            "http://localhost:11434 so the dropdown is populated"
        )

    def test_shipped_example_cloud_providers_present(
        self, _shipped_example_loaded
    ):
        """Cloud provider entries (openai, claude, gemini) must be present
        so users can fill in API keys and run discovery without editing
        the file structure."""
        engines = _shipped_example_loaded.get("engines", {})
        for provider in ("openai", "claude", "gemini"):
            assert provider in engines, (
                f"shipped example missing cloud provider section: {provider}"
            )
