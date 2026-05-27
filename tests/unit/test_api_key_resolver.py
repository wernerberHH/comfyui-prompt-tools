"""Tests for engines/api_key_resolver.py.

Covers the 4-step resolution priority and the URL-override lookup.
File-system access is isolated via tmp_path + monkeypatching of the
two config-file path constants.
"""

from __future__ import annotations

import pytest

from comfyui_prompt_tools.engines import api_key_resolver as resolver
from comfyui_prompt_tools.engines.api_key_resolver import (
    resolve_api_key,
    resolve_url_override,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear all env vars that the resolver might read between tests."""
    for var in (
        "COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI",
        "COMFYUI_PROMPT_TOOLS_API_KEY_CLAUDE",
        "COMFYUI_PROMPT_TOOLS_API_KEY_GEMINI",
        "COMFYUI_PROMPT_TOOLS_API_KEY_VLLM",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect both config-file paths into tmp_path/{project,user}."""
    project = tmp_path / "project" / "api_keys.yaml"
    user = tmp_path / "user" / "api_keys.yaml"
    project.parent.mkdir()
    user.parent.mkdir()
    monkeypatch.setattr(resolver, "_PROJECT_CONFIG_PATH", project)
    monkeypatch.setattr(resolver, "_USER_CONFIG_PATH", user)
    # Reset warn cache between tests
    monkeypatch.setattr(resolver, "_warned_perms", set())
    return {"project": project, "user": user}


def _write_yaml(path, providers_dict):
    """Helper: write a providers-shaped YAML to path."""
    import yaml
    path.write_text(yaml.safe_dump({"providers": providers_dict}))


class TestCustomEnvVar:
    def test_custom_env_var_takes_precedence(self, monkeypatch, isolated_config):
        monkeypatch.setenv("COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI", "from-custom-env")
        monkeypatch.setenv("OPENAI_API_KEY", "from-standard-env")
        _write_yaml(isolated_config["project"], {"openai": {"api_key": "from-file"}})
        assert resolve_api_key("openai") == "from-custom-env"

    def test_custom_env_var_per_provider(self, monkeypatch, isolated_config):
        monkeypatch.setenv("COMFYUI_PROMPT_TOOLS_API_KEY_CLAUDE", "claude-key")
        monkeypatch.setenv("COMFYUI_PROMPT_TOOLS_API_KEY_GEMINI", "gemini-key")
        assert resolve_api_key("claude") == "claude-key"
        assert resolve_api_key("gemini") == "gemini-key"

    def test_empty_custom_env_var_falls_through(self, monkeypatch, isolated_config):
        monkeypatch.setenv("COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI", "")
        monkeypatch.setenv("OPENAI_API_KEY", "from-standard-env")
        assert resolve_api_key("openai") == "from-standard-env"


class TestStandardEnvVar:
    def test_openai_api_key(self, monkeypatch, isolated_config):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert resolve_api_key("openai") == "sk-test"

    def test_openrouter_api_key_for_claude(self, monkeypatch, isolated_config):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert resolve_api_key("claude") == "sk-or-test"

    def test_gemini_api_key(self, monkeypatch, isolated_config):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
        assert resolve_api_key("gemini") == "AIza-test"

    def test_local_providers_have_no_standard_env_var(self, isolated_config):
        # No env var, no config file → None
        assert resolve_api_key("ollama") is None
        assert resolve_api_key("vllm") is None


class TestProjectConfigFile:
    def test_project_file_used_when_no_env_var(self, isolated_config):
        _write_yaml(isolated_config["project"], {"openai": {"api_key": "from-project"}})
        assert resolve_api_key("openai") == "from-project"

    def test_project_file_beats_user_file(self, isolated_config):
        _write_yaml(isolated_config["project"], {"openai": {"api_key": "from-project"}})
        _write_yaml(isolated_config["user"], {"openai": {"api_key": "from-user"}})
        assert resolve_api_key("openai") == "from-project"


class TestUserConfigFile:
    def test_user_file_used_when_no_project_file(self, isolated_config):
        _write_yaml(isolated_config["user"], {"openai": {"api_key": "from-user"}})
        assert resolve_api_key("openai") == "from-user"


class TestMissingKey:
    def test_returns_none_when_nothing_configured(self, isolated_config):
        assert resolve_api_key("openai") is None

    def test_returns_none_for_provider_not_in_file(self, isolated_config):
        _write_yaml(isolated_config["project"], {"claude": {"api_key": "x"}})
        assert resolve_api_key("openai") is None

    def test_handles_malformed_yaml_gracefully(self, isolated_config):
        isolated_config["project"].write_text("not: valid: yaml: ::")
        # Must not raise — fall through to None
        assert resolve_api_key("openai") is None


class TestPriority:
    def test_full_priority_chain(self, monkeypatch, isolated_config):
        """Custom env > standard env > project file > user file."""
        # All four configured — custom env should win
        monkeypatch.setenv("COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI", "lvl1-custom")
        monkeypatch.setenv("OPENAI_API_KEY", "lvl2-standard")
        _write_yaml(isolated_config["project"], {"openai": {"api_key": "lvl3-project"}})
        _write_yaml(isolated_config["user"], {"openai": {"api_key": "lvl4-user"}})
        assert resolve_api_key("openai") == "lvl1-custom"

        # Remove custom — standard wins
        monkeypatch.delenv("COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI")
        assert resolve_api_key("openai") == "lvl2-standard"

        # Remove standard — project file wins
        monkeypatch.delenv("OPENAI_API_KEY")
        assert resolve_api_key("openai") == "lvl3-project"

        # Remove project file — user file wins
        isolated_config["project"].unlink()
        assert resolve_api_key("openai") == "lvl4-user"


class TestUrlOverride:
    def test_url_override_from_project_file(self, isolated_config):
        _write_yaml(isolated_config["project"], {
            "openai": {"api_key": "k", "url": "https://custom-proxy/v1"}
        })
        assert resolve_url_override("openai") == "https://custom-proxy/v1"

    def test_no_override_when_file_omits_url(self, isolated_config):
        _write_yaml(isolated_config["project"], {"openai": {"api_key": "k"}})
        assert resolve_url_override("openai") is None

    def test_no_override_returns_none(self, isolated_config):
        assert resolve_url_override("openai") is None

    def test_project_file_beats_user_file_for_url(self, isolated_config):
        _write_yaml(isolated_config["project"], {
            "openai": {"url": "https://project-url/v1"}
        })
        _write_yaml(isolated_config["user"], {
            "openai": {"url": "https://user-url/v1"}
        })
        assert resolve_url_override("openai") == "https://project-url/v1"
