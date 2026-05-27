"""Base class for prompt nodes that drive an LLM engine.

Provides the engine-selection input fragment, an engine factory, and a
unified chat-call helper. Subclasses inherit from ``BasePromptNode`` and
splice ``_build_engine_inputs()`` into their ``INPUT_TYPES["required"]``.

Engine choice is a string from ``ENGINE_CHOICES`` (``ollama`` or ``vllm``).
The factory normalises the URL: Ollama wants no ``/v1`` suffix, vLLM does.
"""

from __future__ import annotations

from typing import Optional, Union

from ..config_loader import load_endpoints
from ..engines import (
    ENGINE_TOOLTIP,
    OllamaChatEngine,
    OpenAIChatEngine,
    PROVIDERS,
    PROVIDER_CHOICES,
    normalize_url,
    resolve_api_key,
    resolve_url_override,
)

# All five engine choices. "ollama" is the only non-OpenAI-compatible
# backend; the other four share OpenAIChatEngine with different defaults.
ENGINE_CHOICES = PROVIDER_CHOICES

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b"
DEFAULT_VLLM_MODEL = ""  # no default — user picks the model in the node UI
DEFAULT_TIMEOUT = 180

EngineType = Union[OllamaChatEngine, OpenAIChatEngine]


class BasePromptNode:
    """Mixin/base for nodes that talk to an LLM engine.

    Subclasses splice ``_build_engine_inputs()`` into their
    ``INPUT_TYPES["required"]``, then call ``_resolve_engine(...)`` and
    ``_call_engine(...)`` from the node's main function.
    """

    @staticmethod
    def _build_engine_inputs() -> dict:
        """Return the engine-related ``INPUT_TYPES`` fragment.

        Order matters for the ComfyUI UI; engine first so users see the
        choice before the URL/model that depend on it.

        Autocomplete behaviour (Phase 5, fixed pre-merge): ComfyUI's
        ``INPUT_TYPES`` is built once at registration time and cannot react
        to the user's engine choice. So instead of filtering URLs by the
        currently-selected engine (which used to hardcode "ollama"), we
        flatten every URL across every entry in :data:`ENGINE_CHOICES` into
        one Combo, and every model across every (engine, url) pair into
        another. The user picks ``engine`` + a matching ``base_url`` + a
        matching ``model`` themselves — wrong combinations surface as a
        connection error at run time rather than silently routing to the
        wrong server.

        Defaults: first URL of the first engine that has any entries in the
        config; first model registered for that (engine, url) pair. When
        the config is empty or missing, both fields degrade to plain
        STRING text inputs (``DEFAULT_OLLAMA_URL`` / ``DEFAULT_OLLAMA_MODEL``)
        so the node still works without pyyaml.

        Users who need a URL or model that is not in the config simply edit
        ``config/endpoints.yaml`` and add it — the dropdown picks it up on
        the next ComfyUI reload.
        """
        engines = (load_endpoints().get("engines") or {})

        all_urls: list[str] = []
        # Models are prefixed with "[engine] " so the user can tell which
        # backend each model belongs to. The prefix is stripped again
        # before the value reaches the engine client (see _resolve_engine).
        # Order: engines follow ENGINE_CHOICES; within each engine, models
        # are sorted alphabetically — this gives a stable, readable dropdown.
        prefixed_models: list[str] = []
        default_url: Optional[str] = None
        default_url_models: list[str] = []
        default_engine_for_default_url: Optional[str] = None

        for engine_name in ENGINE_CHOICES:
            entries = engines.get(engine_name) or []
            engine_models: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                url = entry.get("url")
                if isinstance(url, str) and url.strip():
                    url = url.strip()
                    if url not in all_urls:
                        all_urls.append(url)
                    if default_url is None:
                        default_url = url
                        default_engine_for_default_url = engine_name
                        default_url_models = [
                            m for m in (entry.get("models") or [])
                            if isinstance(m, str) and m.strip()
                        ]
                for m in (entry.get("models") or []):
                    if isinstance(m, str) and m.strip():
                        engine_models.add(m.strip())
            for m in sorted(engine_models):
                prefixed_models.append(f"[{engine_name}] {m}")

        sorted_models = prefixed_models

        # Default model is the first model of the default URL, prefixed
        # with the engine that owns that URL.
        if default_url_models and default_engine_for_default_url:
            default_url_models = [
                f"[{default_engine_for_default_url}] {m}"
                for m in default_url_models
            ]

        if all_urls:
            base_url_field = (all_urls, {"default": default_url})
        else:
            base_url_field = ("STRING", {"default": DEFAULT_OLLAMA_URL})

        default_model = default_url_models[0] if default_url_models else (
            sorted_models[0] if sorted_models else None
        )
        if sorted_models and default_model is not None:
            model_field = (sorted_models, {"default": default_model})
        else:
            model_field = ("STRING", {"default": DEFAULT_OLLAMA_MODEL})

        return {
            "engine":      (ENGINE_CHOICES, {
                "default": "ollama",
                "tooltip": ENGINE_TOOLTIP,
            }),
            "base_url":    base_url_field,
            "model":       model_field,
            "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
        }

    @staticmethod
    def _resolve_engine(
        engine: str,
        base_url: str,
        model: str,
        timeout: int = DEFAULT_TIMEOUT,
        keep_alive: Optional[str] = None,
    ) -> EngineType:
        """Build an engine instance for the given choice.

        ``keep_alive`` is forwarded to Ollama only; vLLM ignores it.
        Trailing slashes are stripped, then the URL is normalised: Ollama
        gets ``/v1`` removed, vLLM gets it appended if missing.

        Raises ``ValueError`` for unknown engine choices.
        """
        if engine not in PROVIDERS:
            raise ValueError(
                f"Unknown engine: {engine!r}. Must be one of {ENGINE_CHOICES}."
            )

        # Strip the "[engine] " display prefix that the dropdown adds (v0.10.3).
        # The prefix is purely a UI affordance — the wire format must be the
        # bare model identifier. Engine routing is determined by the engine
        # field, not by the prefix; a mismatch (e.g. engine=ollama with a
        # "[openai] gpt-4o" model) silently strips the prefix and routes via
        # the explicitly selected engine — which is the documented behaviour.
        import re as _re
        model = _re.sub(r"^\[[^\]]+\]\s+", "", (model or "").strip())

        # If user left base_url empty, fall back via config-file override
        # then provider default. Otherwise honour the supplied URL.
        raw_url = (base_url or "").strip()
        if not raw_url:
            raw_url = (
                resolve_url_override(engine)
                or PROVIDERS[engine]["default_url"]
            )
        url = normalize_url(engine, raw_url)

        if engine == "ollama":
            kwargs: dict = {"base_url": url, "model": model, "timeout": timeout}
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            return OllamaChatEngine(**kwargs)

        # vllm, openai, claude, gemini all use OpenAIChatEngine; the
        # only difference is the (resolved) API key.
        api_key = resolve_api_key(engine)
        return OpenAIChatEngine(
            api_url=url, model=model, timeout=timeout, api_key=api_key
        )

    @staticmethod
    def _call_engine(
        engine_obj: EngineType,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        images_b64: Optional[list[str]] = None,
    ) -> str:
        """Invoke ``engine_obj.chat`` with a uniform argument shape.

        Errors from the underlying engine (``OllamaError`` /
        ``OpenAIError``) propagate unchanged so subclasses can render a
        user-facing message via ``ShowText``.
        """
        return engine_obj.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            images_b64=images_b64,
        )
