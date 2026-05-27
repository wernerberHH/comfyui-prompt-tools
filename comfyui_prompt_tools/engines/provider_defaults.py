"""Provider defaults for the five supported engine choices.

Each provider is a key into the ``PROVIDERS`` map; the entry carries
everything subclasses need to know about its conventions: default URL,
expected env-var name for the API key, URL normalisation rule (so the
user can paste the URL in any sensible shape), discovery endpoint, and
discovery response shape.

Authentication semantics:

- ``auth_env_var = None`` means the provider does not normally need an
  API key (local Ollama / lab-internal vLLM). A user-supplied key is
  still honoured if present, but the resolver does not search env vars.
- ``auth_env_var = "X_API_KEY"`` means the resolver checks ``$X_API_KEY``
  as a known standard fallback after the custom env-var slot.

URL normalisation:

- ``"no_v1"``: strip trailing ``/v1`` (Ollama, which does not use ``/v1``).
- ``"with_v1"``: append ``/v1`` if not present (OpenAI-compat style).
- ``"as_is"``: leave the URL untouched (Gemini's path is non-standard).

Discovery parser:

- ``"openai"``: response has ``{"data": [{"id": "model-id", ...}, ...]}``.
- ``"ollama"``: response has ``{"models": [{"name": "model:tag", ...}, ...]}``.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class ProviderSpec(TypedDict):
    """Static spec for one engine provider.

    Note: ``discovery_endpoint`` is the path relative to the host root
    (not the chat ``base_url``), because Gemini's discovery and chat
    paths live on different prefixes.
    """

    default_url:          str
    auth_env_var:         Optional[str]
    url_normalization:    str
    discovery_endpoint:   str
    discovery_parser:     str


PROVIDERS: dict[str, ProviderSpec] = {
    "ollama": {
        "default_url":         "http://localhost:11434",
        "auth_env_var":        None,
        "url_normalization":   "no_v1",
        "discovery_endpoint":  "/api/tags",
        "discovery_parser":    "ollama",
    },
    "vllm": {
        "default_url":         "http://localhost:8000/v1",
        "auth_env_var":        None,
        "url_normalization":   "with_v1",
        "discovery_endpoint":  "/v1/models",
        "discovery_parser":    "openai",
    },
    "openai": {
        "default_url":         "https://api.openai.com/v1",
        "auth_env_var":        "OPENAI_API_KEY",
        "url_normalization":   "with_v1",
        "discovery_endpoint":  "/v1/models",
        "discovery_parser":    "openai",
    },
    "claude": {
        "default_url":         "https://openrouter.ai/api/v1",
        "auth_env_var":        "OPENROUTER_API_KEY",
        "url_normalization":   "with_v1",
        "discovery_endpoint":  "/v1/models",
        "discovery_parser":    "openai",
    },
    "gemini": {
        "default_url":         "https://generativelanguage.googleapis.com/v1beta/openai/",
        "auth_env_var":        "GEMINI_API_KEY",
        "url_normalization":   "as_is",
        # Discovery uses the OpenAI-compatible endpoint so we can authenticate
        # with a Bearer token. The native /v1beta/models endpoint rejects
        # Bearer auth — Google wants ?key= or x-goog-api-key there.
        "discovery_endpoint":  "/v1beta/openai/models",
        "discovery_parser":    "openai",
    },
}


PROVIDER_CHOICES: list[str] = list(PROVIDERS.keys())


ENGINE_TOOLTIP = (
    "ollama: local Ollama server (no API key needed)\n"
    "vllm: local OpenAI-compatible server (vLLM, sglang, LM Studio, ...)\n"
    "openai: ChatGPT — needs OPENAI_API_KEY\n"
    "claude: Anthropic via OpenRouter — needs OPENROUTER_API_KEY\n"
    "gemini: Google Gemini — needs GEMINI_API_KEY"
)


def normalize_url(provider: str, raw_url: str) -> str:
    """Apply the provider's URL normalisation rule.

    Strips trailing slashes first, then applies the per-provider rule.
    Unknown providers pass through unchanged.
    """
    url = (raw_url or "").strip().rstrip("/")
    if not url:
        return url
    spec = PROVIDERS.get(provider)
    if spec is None:
        return url
    rule = spec["url_normalization"]
    if rule == "no_v1":
        if url.endswith("/v1"):
            url = url[:-3].rstrip("/")
    elif rule == "with_v1":
        if not url.endswith("/v1"):
            url = f"{url}/v1"
    # "as_is" → no change
    return url
