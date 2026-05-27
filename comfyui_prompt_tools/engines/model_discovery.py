"""On-demand model discovery for the five engine providers.

Each provider exposes a list-models endpoint. We hit it, parse the
response according to the provider's known schema, and return a sorted
list of model IDs. The caller (CLI or future Settings UI) merges this
into ``config/endpoints.yaml``.

Discovery is intentionally **passive**: it never modifies state by
itself. Returns the list, the caller decides what to do with it.

Failure modes are explicit:

- ``DiscoveryError`` on transport / auth / parse failure (with a
  human-readable message)
- providers without an API key (when one is required) return an empty
  list with a warning logged — they are silently skipped in batch
  runs

Discovery responses by parser type::

    openai-style    {"data": [{"id": "model-id", ...}, ...]}
    ollama-style    {"models": [{"name": "model:tag", ...}, ...]}
    gemini-style    {"models": [{"name": "models/<id>", ...}, ...]}
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .api_key_resolver import resolve_api_key, resolve_url_override
from .provider_defaults import PROVIDERS, normalize_url

logger = logging.getLogger(__name__)

_DISCOVERY_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Per-provider chat-model blacklist
# ---------------------------------------------------------------------------
# Some providers (notably OpenAI) expose ~100 entries on /v1/models, most of
# which are not chat-capable (TTS, transcription, image gen, embeddings,
# legacy completion, realtime audio, search-preview, etc.). For prompt
# generation we only want chat-capable models. Each pattern is a regex
# checked against the model ID with re.search(); a match excludes the model.
#
# claude (via OpenRouter) and gemini lists are already curated by the
# providers themselves, so no blacklist is needed there by default.
#
# Provider-specific patterns are merged with the shared list; an empty list
# means "no provider-specific filtering".

_SHARED_BLACKLIST: list[str] = [
    r"-tts(?:$|-)",            # text-to-speech
    r"-transcribe(?:$|-)",     # speech-to-text
    r"-audio(?:$|-)",          # audio-only models
    r"-realtime(?:$|-)",       # realtime audio
    r"-image(?:$|-)",          # image gen
    r"-search-preview",        # web-search preview models
    r"-moderation(?:$|-)",     # content moderation
    r"embedding",              # substring: text-embedding-*, gemini-embedding-*, ...
    r"^whisper",               # ASR
    r"^dall-e",                # image gen
    r"^tts-",                  # TTS family
    r"^omni-moderation",       # moderation
]

_OPENAI_BLACKLIST: list[str] = [
    r"^babbage",               # legacy completion
    r"^davinci",               # legacy completion
    r"codex",                  # substring: codex-mini-latest, gpt-5-codex, ...
    r"sora",                   # substring: sora-2, sora-2-pro, ... (video gen)
    r"^gpt-3\.5",              # too old for prompt-gen quality
    r"^gpt-image",             # image gen
    r"^chatgpt-image",         # image gen
    r"^chat-latest$",          # alias, unstable
]

# Gemini lists are mostly curated, but a few entries are not chat:
# - "nano-banana" is the codename for Gemini Flash Image (image gen)
# - "deep-research" is a specialised mode, not free-form chat
# preview models are intentionally NOT filtered — Google ships active
# models as *-preview, so filtering would drop the newest releases.
_GEMINI_BLACKLIST: list[str] = [
    r"nano-banana",            # Gemini Flash Image (codename)
    r"deep-research",          # specialised research mode
]

PROVIDER_BLACKLISTS: dict[str, list[str]] = {
    "openai": _OPENAI_BLACKLIST,
    "claude": [],
    "gemini": _GEMINI_BLACKLIST,
    "ollama": [],
    "vllm":   [],
}


def _filter_chat_models(provider: str, models: list[str]) -> list[str]:
    """Drop entries that match the blacklist for this provider."""
    patterns = _SHARED_BLACKLIST + PROVIDER_BLACKLISTS.get(provider, [])
    if not patterns:
        return models
    compiled = [re.compile(p) for p in patterns]
    kept = [m for m in models if not any(c.search(m) for c in compiled)]
    dropped = len(models) - len(kept)
    if dropped:
        logger.info("%s: filtered %d non-chat models", provider, dropped)
    return kept


class DiscoveryError(Exception):
    """Raised when model discovery for a provider fails."""


def _build_discovery_url(provider: str, base_url: str) -> str:
    """Compute the absolute URL to hit for discovery.

    Strategy: take the scheme+host of ``base_url`` and append the
    provider's discovery endpoint path. Ignores any path component of
    ``base_url`` because chat path != discovery path for some providers
    (notably Gemini).
    """
    spec = PROVIDERS[provider]
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise DiscoveryError(
            f"Invalid base_url for {provider}: {base_url!r}"
        )
    root = f"{parsed.scheme}://{parsed.netloc}"
    return root + spec["discovery_endpoint"]


def _parse_response(parser: str, payload: dict) -> list[str]:
    """Extract model IDs from a discovery response according to parser type."""
    if parser == "openai":
        data = payload.get("data") or []
        return sorted({
            entry["id"] for entry in data
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        })
    if parser == "ollama":
        models = payload.get("models") or []
        return sorted({
            entry["name"] for entry in models
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        })
    raise DiscoveryError(f"Unknown parser type: {parser!r}")


def discover_models(
    provider: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """Discover available models for ``provider``.

    Parameters
    ----------
    provider:
        Key into :data:`PROVIDERS`. Raises ``KeyError`` for unknown
        providers.
    base_url:
        Override for the discovery base URL. Falls back to a config-file
        override, then to ``PROVIDERS[provider]["default_url"]``.
    api_key:
        Override for the API key. Falls back to ``resolve_api_key()``.

    Returns
    -------
    list[str]
        Sorted, deduplicated model IDs. Empty list on (a) provider
        skipped because no key available, (b) empty response.

    Raises
    ------
    DiscoveryError
        Transport, authentication, or parse failure.
    """
    if provider not in PROVIDERS:
        raise KeyError(f"Unknown provider: {provider!r}")
    spec = PROVIDERS[provider]

    if base_url is None:
        base_url = resolve_url_override(provider) or spec["default_url"]
    base_url = normalize_url(provider, base_url)

    if api_key is None:
        api_key = resolve_api_key(provider)

    # Providers that normally require auth are skipped when no key is set
    if spec["auth_env_var"] is not None and not api_key:
        logger.warning(
            "Skipping discovery for %s: no API key configured", provider
        )
        return []

    url = _build_discovery_url(provider, base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_DISCOVERY_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise DiscoveryError(
                f"{provider}: authentication failed (401). Check API key."
            ) from exc
        raise DiscoveryError(
            f"{provider}: HTTP {exc.code} from {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise DiscoveryError(
            f"{provider}: cannot reach {url}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DiscoveryError(
            f"{provider}: response from {url} is not valid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise DiscoveryError(
            f"{provider}: unexpected response shape from {url}"
        )

    models = _parse_response(spec["discovery_parser"], payload)
    return _filter_chat_models(provider, models)


def discover_all() -> dict[str, list[str]]:
    """Run discovery for every provider; skip those that fail.

    Returns a ``{provider: [model_ids]}`` map. Failed providers are
    logged but do not raise — the result simply omits them. Useful for
    a bulk "refresh everything" UI button.
    """
    out: dict[str, list[str]] = {}
    for provider in PROVIDERS:
        try:
            models = discover_models(provider)
            if models:
                out[provider] = models
            else:
                logger.info("%s: 0 models (skipped or empty response)", provider)
        except DiscoveryError as exc:
            logger.warning("%s discovery failed: %s", provider, exc)
    return out
