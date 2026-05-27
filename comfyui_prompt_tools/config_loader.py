"""Endpoints config loader for engine URL / model autocomplete.

The loader reads ``config/endpoints.yaml`` (user-edited, gitignored) and
falls back to ``config/endpoints.yaml.example`` (committed) if the user
file is missing. On any error — missing pyyaml, missing file, malformed
YAML — the loader returns an empty dict so the node UI degrades gracefully
to plain text fields.

Schema::

    engines:
      ollama:
        - url: "http://host:11434"
          models: ["qwen3-vl:8b", ...]
      vllm:
        - url: "http://host:8000/v1"
          models: ["Qwen/Qwen2.5-7B-Instruct", ...]

Public surface:

- ``load_endpoints() -> dict`` — full parsed config (cached on first call).
- ``get_urls_for_engine(engine) -> list[str]``
- ``get_models_for_engine_url(engine, url) -> list[str]``

Never raises. Failures are logged once and the loader returns ``{}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_USER_FILE = _CONFIG_DIR / "endpoints.yaml"
_EXAMPLE_FILE = _CONFIG_DIR / "endpoints.yaml.example"

_cache: Optional[dict] = None
_warned_missing_yaml = False


def _try_import_yaml():
    """Return the pyyaml module, or None and log once if absent."""
    global _warned_missing_yaml
    try:
        import yaml  # type: ignore[import-not-found]
        return yaml
    except ImportError:
        if not _warned_missing_yaml:
            logger.warning(
                "pyyaml not installed — endpoint autocomplete disabled. "
                "Install with: pip install pyyaml"
            )
            _warned_missing_yaml = True
        return None


def _read_yaml_file(path: Path) -> dict:
    """Parse a YAML file into a dict. Return ``{}`` on any failure."""
    yaml = _try_import_yaml()
    if yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def load_endpoints(refresh: bool = False) -> dict:
    """Load the endpoints config. Cached after first call.

    Order of resolution:

    1. ``config/endpoints.yaml`` (user-edited, gitignored)
    2. ``config/endpoints.yaml.example`` (committed fallback)
    3. ``{}`` if neither parses

    Pass ``refresh=True`` to force a re-read (mainly for tests).
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    if _USER_FILE.is_file():
        data = _read_yaml_file(_USER_FILE)
        if data:
            _cache = data
            return _cache

    if _EXAMPLE_FILE.is_file():
        data = _read_yaml_file(_EXAMPLE_FILE)
        if data:
            _cache = data
            return _cache

    _cache = {}
    return _cache


def get_urls_for_engine(engine: str) -> list[str]:
    """Return the list of configured URLs for the named engine.

    Empty list if the engine has no entries or the config is empty.
    """
    config = load_endpoints()
    engines = config.get("engines") or {}
    entries = engines.get(engine) or []
    urls: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            url = entry.get("url")
            if isinstance(url, str) and url.strip():
                urls.append(url.strip())
    return urls


def get_models_for_engine_url(engine: str, url: str) -> list[str]:
    """Return the model list configured for the (engine, url) pair.

    Matches ``url`` exactly against the configured ``url`` field. Returns
    empty list if engine/url not found or models list is missing.
    """
    config = load_endpoints()
    engines = config.get("engines") or {}
    entries = engines.get(engine) or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("url") == url:
            models = entry.get("models") or []
            return [m for m in models if isinstance(m, str) and m.strip()]
    return []
