"""HTTP routes for the Settings UI (Phase 3b / v0.10.0).

Routes registered on the ComfyUI PromptServer:

  GET  /comfyui-prompt-tools/status
       -> per-provider status without ever revealing the actual key
  POST /comfyui-prompt-tools/save
       -> write api_key + optional url for one or more providers
       to config/api_keys.yaml (chmod 600)
  POST /comfyui-prompt-tools/test
       -> 1-line live test of a provider's chat endpoint
       (uses the just-saved key; never trusts user-supplied keys
       in the request body for security against CSRF-style abuse)
  POST /comfyui-prompt-tools/discover
       -> run model discovery for one provider or all, merge results
       into config/endpoints.yaml (with .backup of the previous file)

Routes are registered defensively: if PromptServer is not importable
(e.g. during pytest of the package without ComfyUI installed), the
registration is silently skipped and the module remains importable.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

# Module-level imports of the engine internals — needed so tests can
# monkey-patch them via web_api.X and so import-time errors surface
# immediately instead of being deferred into request handling.
from .engines import PROVIDERS
from .engines.api_key_resolver import resolve_api_key, resolve_url_override
from .engines.model_discovery import DiscoveryError, discover_models

logger = logging.getLogger(__name__)


# Paths — same anchors as api_key_resolver.py / config_loader.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_API_KEYS_PATH = _REPO_ROOT / "config" / "api_keys.yaml"
_ENDPOINTS_PATH = _REPO_ROOT / "config" / "endpoints.yaml"


# ----- Pure-function helpers (testable without aiohttp) -------------------


def get_provider_status() -> dict[str, dict[str, Any]]:
    """Return per-provider status: configured + url-override flags.

    Never returns the key itself. The UI uses this to render a check/cross
    badge per provider and to pre-fill the URL override field.
    """
    out: dict[str, dict[str, Any]] = {}
    for provider, spec in PROVIDERS.items():
        key = resolve_api_key(provider)
        url_override = resolve_url_override(provider)
        out[provider] = {
            "configured":     bool(key),
            "needs_key":      spec["auth_env_var"] is not None,
            "default_url":    spec["default_url"],
            "url_override":   url_override,
        }
    return out


def save_provider_config(updates: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Write api_key + optional url for one or more providers.

    ``updates`` shape::

        {
            "openai": {"api_key": "sk-...", "url": "https://..."},
            "claude": {"api_key": "sk-or-..."},
        }

    Empty strings are treated as "remove this field". Returns a summary
    of what was written. Sets file permissions to 600 on POSIX.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {"ok": False, "error": "pyyaml not installed on the server"}

    # Validate provider names
    bad = [p for p in updates if p not in PROVIDERS]
    if bad:
        return {"ok": False, "error": f"Unknown providers: {bad}"}

    # Load existing config (if any) so we merge, not overwrite
    existing: dict = {}
    if _API_KEYS_PATH.is_file():
        try:
            existing = yaml.safe_load(_API_KEYS_PATH.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            return {"ok": False, "error": f"Existing YAML is malformed: {exc}"}
        if not isinstance(existing, dict):
            existing = {}

    providers = existing.get("providers") or {}
    if not isinstance(providers, dict):
        providers = {}

    written: list[str] = []
    for provider, fields in updates.items():
        if not isinstance(fields, dict):
            continue
        entry = providers.get(provider) or {}
        if not isinstance(entry, dict):
            entry = {}

        # api_key: explicit empty string removes the field
        if "api_key" in fields:
            key = (fields["api_key"] or "").strip()
            if key:
                entry["api_key"] = key
            else:
                entry.pop("api_key", None)

        # url: explicit empty string removes the override
        if "url" in fields:
            url = (fields["url"] or "").strip()
            if url:
                entry["url"] = url
            else:
                entry.pop("url", None)

        if entry:
            providers[provider] = entry
        else:
            providers.pop(provider, None)
        written.append(provider)

    existing["providers"] = providers
    _API_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _API_KEYS_PATH.write_text(
        yaml.safe_dump(existing, sort_keys=False), encoding="utf-8"
    )

    # Tighten permissions on POSIX
    try:
        os.chmod(_API_KEYS_PATH, 0o600)
    except OSError:
        pass

    # Resolver caches nothing currently, but if it ever does we'd
    # invalidate here.

    return {"ok": True, "written": written, "path": str(_API_KEYS_PATH)}


def test_provider_connection(provider: str) -> dict[str, Any]:
    """One-shot live test: discovery call against the provider.

    Discovery is a cheap, harmless GET that exercises the same auth
    path as chat. If it succeeds we know the key + URL work.

    A provider with no API key configured returns ok=False (rather than
    letting discover_models silently return an empty list, which the UI
    would otherwise paint green).
    """
    if provider not in PROVIDERS:
        return {"ok": False, "error": f"Unknown provider: {provider}"}

    spec = PROVIDERS[provider]
    if spec["auth_env_var"] is not None and not resolve_api_key(provider):
        return {"ok": False, "error": "no API key configured"}

    try:
        models = discover_models(provider)
    except DiscoveryError as exc:
        return {"ok": False, "error": str(exc)}

    if not models:
        # Auth succeeded but the provider returned an empty list. Unusual
        # but never a "green" result for the UI.
        return {"ok": False, "error": "discovery returned 0 models"}

    return {
        "ok":            True,
        "model_count":   len(models),
        "sample_models": models[:3],
    }


def run_discovery(provider: Optional[str] = None) -> dict[str, Any]:
    """Run discovery and merge results into config/endpoints.yaml.

    The previous endpoints.yaml (if any) is copied to endpoints.yaml.backup
    before writing. Only the ``models`` lists for the affected providers
    are touched — user-customised URLs and other providers are preserved.

    When ``provider`` is ``None``, discovery runs for all providers.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {"ok": False, "error": "pyyaml not installed"}

    targets = [provider] if provider else list(PROVIDERS.keys())
    if provider and provider not in PROVIDERS:
        return {"ok": False, "error": f"Unknown provider: {provider}"}

    # Load existing endpoints.yaml (or the .example as fallback)
    source_path = _ENDPOINTS_PATH if _ENDPOINTS_PATH.is_file() else (
        _ENDPOINTS_PATH.parent / "endpoints.yaml.example"
    )
    if source_path.is_file():
        try:
            doc = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            return {"ok": False, "error": f"Existing YAML malformed: {exc}"}
        if not isinstance(doc, dict):
            doc = {}
    else:
        doc = {}

    engines_block = doc.get("engines") or {}
    if not isinstance(engines_block, dict):
        engines_block = {}

    results: dict[str, dict[str, Any]] = {}
    for prov in targets:
        try:
            models = discover_models(prov)
        except DiscoveryError as exc:
            results[prov] = {"ok": False, "error": str(exc), "count": 0}
            continue
        except KeyError as exc:
            results[prov] = {"ok": False, "error": str(exc), "count": 0}
            continue

        # Merge models into existing entry — preserve user-set URL
        entries = engines_block.get(prov) or []
        if not isinstance(entries, list) or not entries:
            entries = [{
                "url":    PROVIDERS[prov]["default_url"],
                "models": models,
            }]
        else:
            first = entries[0] if isinstance(entries[0], dict) else {}
            first["models"] = models
            entries[0] = first
        engines_block[prov] = entries
        results[prov] = {"ok": True, "count": len(models)}

    doc["engines"] = engines_block

    # Backup before write
    if _ENDPOINTS_PATH.is_file():
        try:
            shutil.copy2(_ENDPOINTS_PATH, _ENDPOINTS_PATH.with_suffix(".yaml.backup"))
        except OSError as exc:
            logger.warning("Backup failed: %s", exc)

    _ENDPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ENDPOINTS_PATH.write_text(
        yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
    )

    # Invalidate the config_loader cache so the next node load sees fresh data
    from . import config_loader
    config_loader._cache = None  # type: ignore[attr-defined]

    return {
        "ok":      True,
        "results": results,
        "path":    str(_ENDPOINTS_PATH),
    }


# ----- aiohttp route registration (skipped if ComfyUI absent) -------------


def register_routes() -> bool:
    """Hook into the ComfyUI PromptServer; return True if registered."""
    try:
        from server import PromptServer  # type: ignore[import-not-found]
        from aiohttp import web  # type: ignore[import-not-found]
    except ImportError:
        logger.info("ComfyUI server not available; web routes skipped")
        return False

    routes = PromptServer.instance.routes

    @routes.get("/comfyui-prompt-tools/status")
    async def _status(_request):
        return web.json_response(get_provider_status())

    @routes.post("/comfyui-prompt-tools/save")
    async def _save(request):
        try:
            data = await request.json()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        result = save_provider_config(data or {})
        return web.json_response(result)

    @routes.post("/comfyui-prompt-tools/test")
    async def _test(request):
        try:
            data = await request.json()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        provider = (data or {}).get("provider")
        if not isinstance(provider, str):
            return web.json_response(
                {"ok": False, "error": "missing 'provider'"}, status=400
            )
        return web.json_response(test_provider_connection(provider))

    @routes.post("/comfyui-prompt-tools/discover")
    async def _discover(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        provider = (data or {}).get("provider")
        return web.json_response(run_discovery(provider))

    logger.info("ComfyUI Prompt Tools: HTTP routes registered")
    return True
