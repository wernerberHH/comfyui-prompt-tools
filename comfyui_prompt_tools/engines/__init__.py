"""Backend engines for talking to LLM APIs."""

from .api_key_resolver import resolve_api_key, resolve_url_override
from .ollama_client import OllamaChatEngine, OllamaError
from .openai_client import OpenAIChatEngine, OpenAIError
from .provider_defaults import (
    ENGINE_TOOLTIP,
    PROVIDER_CHOICES,
    PROVIDERS,
    normalize_url,
)

__all__ = [
    "OllamaChatEngine",
    "OllamaError",
    "OpenAIChatEngine",
    "OpenAIError",
    "PROVIDERS",
    "PROVIDER_CHOICES",
    "ENGINE_TOOLTIP",
    "normalize_url",
    "resolve_api_key",
    "resolve_url_override",
]
