"""Unit tests for OllamaChatEngine payload construction.

All tests inspect the JSON body sent to urlopen without making real HTTP calls.
"""

import json

import pytest

from comfyui_prompt_tools.engines.ollama_client import OllamaChatEngine


def _engine() -> OllamaChatEngine:
    return OllamaChatEngine(base_url="http://localhost:11434", model="qwen3-vl:32b")


def _sent_payload(mock_urlopen) -> dict:
    """Extract and decode the JSON payload from the captured urlopen call."""
    req = mock_urlopen.call_args[0][0]
    return json.loads(req.data)


class TestOllamaPayload:
    def test_no_images_key_when_images_b64_none(self, mock_ollama_urlopen):
        """User message has no 'images' key when images_b64 is None.
        Mocks: urllib.request.urlopen (returns fake Ollama response).
        """
        _engine().chat("sys", "hello")
        payload = _sent_payload(mock_ollama_urlopen)
        user_msg = payload["messages"][1]
        assert "images" not in user_msg

    def test_no_images_key_when_images_b64_empty(self, mock_ollama_urlopen):
        """User message has no 'images' key when images_b64 is an empty list.
        Mocks: urllib.request.urlopen (returns fake Ollama response).
        """
        _engine().chat("sys", "hello", images_b64=[])
        payload = _sent_payload(mock_ollama_urlopen)
        user_msg = payload["messages"][1]
        assert "images" not in user_msg

    def test_images_array_on_user_message(self, mock_ollama_urlopen):
        """images_b64 values appear under 'images' on the user message.
        Mocks: urllib.request.urlopen (returns fake Ollama response).
        """
        _engine().chat("sys", "describe", images_b64=["aGVsbG8=", "d29ybGQ="])
        payload = _sent_payload(mock_ollama_urlopen)
        user_msg = payload["messages"][1]
        assert user_msg["images"] == ["aGVsbG8=", "d29ybGQ="]

    def test_options_temperature_and_num_predict(self, mock_ollama_urlopen):
        """temperature and num_predict are forwarded inside the options dict.
        Mocks: urllib.request.urlopen (returns fake Ollama response).
        """
        _engine().chat("sys", "x", temperature=0.4, num_predict=1024)
        payload = _sent_payload(mock_ollama_urlopen)
        assert payload["options"]["temperature"] == pytest.approx(0.4)
        assert payload["options"]["num_predict"] == 1024

    def test_stream_is_false(self, mock_ollama_urlopen):
        """stream is always set to False for synchronous usage.
        Mocks: urllib.request.urlopen (returns fake Ollama response).
        """
        _engine().chat("sys", "x")
        payload = _sent_payload(mock_ollama_urlopen)
        assert payload["stream"] is False
