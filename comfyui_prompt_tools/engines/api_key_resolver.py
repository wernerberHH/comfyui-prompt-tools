"""API-key resolver for OpenAI-compatible Cloud providers.

Looks up the API key for a given provider in priority order:

1. **Custom env var**: ``COMFYUI_PROMPT_TOOLS_API_KEY_<PROVIDER_UPPER>``
   (e.g. ``COMFYUI_PROMPT_TOOLS_API_KEY_OPENAI``). This slot is
   project-specific and never collides with other tools.
2. **Standard env var** as declared in ``PROVIDERS[provider]["auth_env_var"]``
   (e.g. ``OPENAI_API_KEY``). Familiar to most developers.
3. **Project-local config file**: ``<repo>/config/api_keys.yaml``
   (gitignored).
4. **User-global config file**: ``~/.config/comfyui-prompt-tools/api_keys.yaml``.
5. ``None`` if nothing matches → caller may still proceed (a local vLLM
   without auth accepts requests without an Authorization header).

Config-file schema::

    providers:
      openai:
        api_key: "sk-..."
        url: "https://api.openai.com/v1"      # optional override
      claude:
        api_key: "sk-or-..."
      gemini:
        api_key: "AIza..."

The resolver never raises on missing files or malformed YAML — it just
returns ``None``. File-permission warnings (e.g. world-readable
``api_keys.yaml``) are logged once.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Optional

from .provider_defaults import PROVIDERS

logger = logging.getLogger(__name__)


_PROJECT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "api_keys.yaml"
)
_USER_CONFIG_PATH = (
    Path.home() / ".config" / "comfyui-prompt-tools" / "api_keys.yaml"
)


_warned_perms: set[Path] = set()


def _custom_env_var_name(provider: str) -> str:
    """Return the per-provider custom env-var name."""
    return f"COMFYUI_PROMPT_TOOLS_API_KEY_{provider.upper()}"


def _read_yaml(path: Path) -> dict:
    """Return parsed YAML dict, or ``{}`` on any failure."""
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _check_permissions(path: Path) -> None:
    """Warn once if api_keys.yaml is world-readable (Unix only)."""
    if path in _warned_perms or not path.is_file():
        return
    try:
        mode = path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            logger.warning(
                "%s is readable by group/others. Set permissions 600: "
                "chmod 600 %s",
                path, path,
            )
    except OSError:
        pass
    _warned_perms.add(path)


def _key_from_config_file(provider: str, path: Path) -> Optional[str]:
    """Extract the api_key for a provider from a config file."""
    _check_permissions(path)
    data = _read_yaml(path)
    providers = data.get("providers") or {}
    entry = providers.get(provider) or {}
    if not isinstance(entry, dict):
        return None
    key = entry.get("api_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def resolve_api_key(provider: str) -> Optional[str]:
    """Resolve the API key for ``provider`` in priority order.

    See module docstring for the full resolution sequence.
    """
    # 1. Custom env var (project-specific, never collides)
    custom = os.environ.get(_custom_env_var_name(provider))
    if custom and custom.strip():
        return custom.strip()

    # 2. Standard env var as declared in PROVIDERS
    spec = PROVIDERS.get(provider)
    if spec is not None:
        std_name = spec.get("auth_env_var")
        if std_name:
            std_val = os.environ.get(std_name)
            if std_val and std_val.strip():
                return std_val.strip()

    # 3. Project-local config file
    key = _key_from_config_file(provider, _PROJECT_CONFIG_PATH)
    if key:
        return key

    # 4. User-global config file
    key = _key_from_config_file(provider, _USER_CONFIG_PATH)
    if key:
        return key

    # 5. Not found
    return None


def resolve_url_override(provider: str) -> Optional[str]:
    """Return a URL override for ``provider`` from the config files, if any.

    URL override priority is the same as for the key (project-local
    first, then user-global). Returns ``None`` when no override is
    configured — the caller should fall back to ``PROVIDERS[provider]
    ["default_url"]``.
    """
    for path in (_PROJECT_CONFIG_PATH, _USER_CONFIG_PATH):
        data = _read_yaml(path)
        entry = (data.get("providers") or {}).get(provider) or {}
        if isinstance(entry, dict):
            url = entry.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
    return None
