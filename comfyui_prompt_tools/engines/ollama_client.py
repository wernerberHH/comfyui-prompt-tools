"""Ollama Chat API client.

Thin wrapper around the ``/api/chat`` endpoint that hides JSON construction
and HTTP plumbing from the node code.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


class OllamaError(Exception):
    """Raised for connection or response errors from Ollama."""


@dataclass
class OllamaChatEngine:
    """Send chat completions to an Ollama server.

    Parameters
    ----------
    base_url:
        Server URL, e.g. ``http://localhost:11434``.
    model:
        Ollama model tag, e.g. ``qwen3-vl:32b`` or ``gemma4``.
    keep_alive:
        How long Ollama should keep the model loaded after the call.
    timeout:
        Per-request timeout in seconds.
    """

    base_url: str
    model: str
    keep_alive: str = "30s"
    timeout: int = 180

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        num_ctx: int = 4096,
        images_b64: Optional[list[str]] = None,
    ) -> str:
        """Run a single chat turn and return the trimmed assistant content.

        ``images_b64`` is a list of base64-encoded image bytes (no data: prefix);
        Ollama supports vision via the ``images`` array on the user message.
        """

        user_msg: dict = {"role": "user", "content": user_message}
        if images_b64:
            user_msg["images"] = images_b64

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_msg,
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

        content = result.get("message", {}).get("content", "").strip()
        return _strip_thinking_blocks(content)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking_blocks(text: str) -> str:
    """Remove Qwen3-style ``<think>...</think>`` blocks.

    If a stray opening ``<think>`` remains (token cut-off mid-stream), drop
    everything from that point onward.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0].strip()
    return cleaned
