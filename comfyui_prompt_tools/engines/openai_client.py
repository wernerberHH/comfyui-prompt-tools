"""OpenAI-compatible Chat API client (vLLM backend).

Thin wrapper around the ``/chat/completions`` endpoint following the
OpenAI API spec, used for vision-capable models served via vLLM, Ollama or any other OpenAI-compatible backend.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .ollama_client import _strip_thinking_blocks


class OpenAIError(Exception):
    """Raised for connection or response errors from an OpenAI-compatible API."""


def _extract_content(result: object, api_url: str) -> str:
    """Defensively pull the assistant content from an OpenAI-style response.

    Real-world failure modes seen against vLLM / OpenAI / toolproxy:
      - top-level ``result`` not a dict (rare, but observed when an
        upstream proxy returns plain text on error)
      - ``choices`` missing, ``None``, or ``[]`` (toolproxy swallows
        400-errors from vLLM and returns ``{"choices": null}``)
      - ``message`` missing or not a dict
      - ``content`` is ``None`` (content-filtered or refusal)
      - ``content`` is not a string

    Each case raises ``OpenAIError`` with an actionable message instead
    of the original ``TypeError: 'NoneType' object is not subscriptable``.
    """
    if not isinstance(result, dict):
        raise OpenAIError(
            f"Unexpected response shape from {api_url}: "
            f"{type(result).__name__}, not a JSON object."
        )

    # Bubble up an upstream error field if the response carries one —
    # often the only diagnostic we get when a proxy mangles the body.
    upstream_err = result.get("error") or result.get("message")

    choices = result.get("choices")
    if not choices:
        suffix = f" Upstream error: {upstream_err}" if upstream_err else ""
        raise OpenAIError(
            f"Response from {api_url} has no choices "
            f"(proxy or upstream may have stripped them).{suffix}"
        )

    first = choices[0]
    if not isinstance(first, dict):
        raise OpenAIError(
            f"choices[0] from {api_url} is not an object: "
            f"{type(first).__name__}."
        )

    message = first.get("message")
    if not isinstance(message, dict):
        raise OpenAIError(
            f"Response from {api_url} is missing 'message' object in choices[0]."
        )

    content = message.get("content")
    if content is None:
        finish_reason = first.get("finish_reason")
        raise OpenAIError(
            f"Response from {api_url} has content=null "
            f"(finish_reason={finish_reason!r}). The upstream model may "
            "have refused or been content-filtered."
        )
    if not isinstance(content, str):
        raise OpenAIError(
            f"Response content from {api_url} is not a string: "
            f"{type(content).__name__}."
        )

    return content.strip()


@dataclass
class OpenAIChatEngine:
    """Send chat completions to a vLLM or OpenAI-compatible server.

    Works against any OpenAI-compatible ``/v1/chat/completions`` endpoint:
    local vLLM, sglang, LM Studio, ChatGPT, OpenRouter (Claude),
    Google Gemini, etc.

    Parameters
    ----------
    api_url:
        Base URL including ``/v1``, e.g. ``http://localhost:8000/v1``.
    model:
        Model identifier, e.g. ``Qwen/Qwen2.5-7B-Instruct`` or ``gpt-4o-mini``.
    timeout:
        Per-request timeout in seconds.
    api_key:
        Optional API key. When set, sent as ``Authorization: Bearer <key>``.
        Required for Cloud providers, ignored if ``None`` (lets local
        unauthenticated vLLM keep working).
    """

    api_url: str
    model: str
    timeout: int = 180
    api_key: Optional[str] = None

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        num_ctx: int = 16384,
        images_b64: Optional[list[str]] = None,
    ) -> str:
        """Run a single chat turn and return the trimmed assistant content.

        When ``images_b64`` is provided and non-empty, the user message is
        encoded as a multipart content array with inline image_url entries.
        When absent, the user message is sent as a plain string.

        ``num_ctx`` is ignored — context length is managed server-side on vLLM.
        """
        if images_b64:
            user_content: object = [{"type": "text", "text": user_message}] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
                for b64 in images_b64
            ]
        else:
            user_content = user_message

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": num_predict,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.api_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Surface auth errors with actionable advice
            if exc.code == 401:
                raise OpenAIError(
                    f"Authentication failed (401) for {self.api_url}. "
                    "Check the API key — either set the provider env var "
                    "or update config/api_keys.yaml."
                ) from exc
            if exc.code == 403:
                raise OpenAIError(
                    f"Access forbidden (403) for {self.api_url}. "
                    "API key valid but lacks permission for this model."
                ) from exc
            raise OpenAIError(
                f"HTTP {exc.code} from {self.api_url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIError(f"Cannot reach API at {self.api_url}: {exc}") from exc

        content = _extract_content(result, self.api_url)
        return _strip_thinking_blocks(content)
