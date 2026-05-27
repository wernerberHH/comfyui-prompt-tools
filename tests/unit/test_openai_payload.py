"""Unit tests for OpenAIChatEngine payload construction.

All tests inspect the JSON body sent to urlopen without making real HTTP calls.
"""

import json

import pytest

from comfyui_prompt_tools.engines.openai_client import (
    OpenAIChatEngine,
    OpenAIError,
    _extract_content,
)


def _engine() -> OpenAIChatEngine:
    return OpenAIChatEngine(api_url="http://localhost:8090/v1", model="test-model")


def _sent_payload(mock_urlopen) -> dict:
    """Extract and decode the JSON payload from the captured urlopen call."""
    req = mock_urlopen.call_args[0][0]
    return json.loads(req.data)


class TestOpenAIPayload:
    def test_plain_string_user_content_when_no_images(self, mock_openai_urlopen):
        """User content is a plain string when images_b64 is None.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("sys", "hello")
        payload = _sent_payload(mock_openai_urlopen)
        user_msg = payload["messages"][1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], str)
        assert user_msg["content"] == "hello"

    def test_plain_string_user_content_when_images_empty(self, mock_openai_urlopen):
        """User content is a plain string when images_b64 is an empty list.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("sys", "hello", images_b64=[])
        payload = _sent_payload(mock_openai_urlopen)
        assert isinstance(payload["messages"][1]["content"], str)

    def test_multipart_content_when_images_provided(self, mock_openai_urlopen):
        """User content is a multipart array when images_b64 is non-empty.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("sys", "describe", images_b64=["aGVsbG8=", "d29ybGQ="])
        payload = _sent_payload(mock_openai_urlopen)
        content = payload["messages"][1]["content"]
        assert isinstance(content, list)
        # first entry is the text part
        assert content[0] == {"type": "text", "text": "describe"}
        # remaining entries are image_url parts
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[2]["type"] == "image_url"

    def test_image_count_matches_images_b64_length(self, mock_openai_urlopen):
        """Number of image_url entries equals the number of supplied b64 strings.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("sys", "x", images_b64=["a", "b", "c"])
        payload = _sent_payload(mock_openai_urlopen)
        content = payload["messages"][1]["content"]
        image_entries = [p for p in content if p["type"] == "image_url"]
        assert len(image_entries) == 3

    def test_model_and_temperature_in_payload(self, mock_openai_urlopen):
        """model and temperature fields are forwarded correctly.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("sys", "x", temperature=0.3)
        payload = _sent_payload(mock_openai_urlopen)
        assert payload["model"] == "test-model"
        assert payload["temperature"] == pytest.approx(0.3)

    def test_max_tokens_maps_from_num_predict(self, mock_openai_urlopen):
        """num_predict is sent as max_tokens in the OpenAI payload.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("sys", "x", num_predict=512)
        payload = _sent_payload(mock_openai_urlopen)
        assert payload["max_tokens"] == 512

    def test_system_prompt_in_first_message(self, mock_openai_urlopen):
        """System prompt appears as the first message with role=system.
        Mocks: urllib.request.urlopen (returns fake OpenAI response).
        """
        _engine().chat("be helpful", "question")
        payload = _sent_payload(mock_openai_urlopen)
        assert payload["messages"][0] == {"role": "system", "content": "be helpful"}


# ==========================================================================
# v0.9: API-key handling on OpenAIChatEngine
# ==========================================================================

import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

from comfyui_prompt_tools.engines.openai_client import (
    OpenAIChatEngine,
    OpenAIError,
)


def _capture_request(payload_response: dict = None):
    """Return (mock_urlopen, captured) — captured["request"] is the
    urllib Request the engine built (so the test can inspect headers)."""
    if payload_response is None:
        payload_response = {"choices": [{"message": {"content": "ok"}}]}
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["request"] = req
        mock_resp = MagicMock()
        mock_resp.read.return_value = __import__("json").dumps(
            payload_response
        ).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        return mock_resp

    return fake_urlopen, captured


class TestOpenAIClientAuthHeader:
    def test_no_api_key_no_authorization_header(self):
        """Backward compat: when api_key is None (default), no Authorization
        header is sent. Local vLLM without auth keeps working."""
        engine = OpenAIChatEngine(
            api_url="http://localhost:8000/v1", model="x"
        )
        fake_urlopen, captured = _capture_request()
        with patch(
            "comfyui_prompt_tools.engines.openai_client.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            engine.chat(system_prompt="s", user_message="u")
        headers = {k.lower(): v for k, v in captured["request"].header_items()}
        assert "authorization" not in headers

    def test_api_key_adds_bearer_authorization_header(self):
        engine = OpenAIChatEngine(
            api_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="sk-test-123",
        )
        fake_urlopen, captured = _capture_request()
        with patch(
            "comfyui_prompt_tools.engines.openai_client.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            engine.chat(system_prompt="s", user_message="u")
        headers = {k.lower(): v for k, v in captured["request"].header_items()}
        assert headers.get("authorization") == "Bearer sk-test-123"

    def test_401_raises_openai_error_with_actionable_message(self):
        engine = OpenAIChatEngine(
            api_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="bad-key",
        )
        err = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            401, "Unauthorized", {}, BytesIO(b""),
        )
        with patch(
            "comfyui_prompt_tools.engines.openai_client.urllib.request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(OpenAIError, match="Authentication failed"):
                engine.chat(system_prompt="s", user_message="u")

    def test_403_raises_openai_error_with_permission_message(self):
        engine = OpenAIChatEngine(
            api_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="valid-but-restricted",
        )
        err = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            403, "Forbidden", {}, BytesIO(b""),
        )
        with patch(
            "comfyui_prompt_tools.engines.openai_client.urllib.request.urlopen",
            side_effect=err,
        ):
            with pytest.raises(OpenAIError, match="Access forbidden"):
                engine.chat(system_prompt="s", user_message="u")


class TestOpenAIResponseExtraction:
    """v0.10.4 Bug #17: defensive parsing of the chat-completion response.

    Previously the final line of chat() was a chained subscript:
        result["choices"][0]["message"]["content"].strip()
    which crashed with TypeError when any of those slots was None or
    missing. The toolproxy in particular swallows 400-errors from vLLM
    and returns {"choices": null}, producing 'NoneType is not subscriptable'.
    """

    API = "http://test/v1"

    def test_success_strips_content(self):
        result = {"choices": [{"message": {"content": "  hello  "}}]}
        assert _extract_content(result, self.API) == "hello"

    def test_non_dict_result_raises(self):
        with pytest.raises(OpenAIError, match="not a JSON object"):
            _extract_content("plaintext error from proxy", self.API)

    def test_choices_null_raises(self):
        """The toolproxy bug — choices is None, not absent."""
        result = {"id": "", "object": "", "choices": None}
        with pytest.raises(OpenAIError, match="no choices"):
            _extract_content(result, self.API)

    def test_choices_missing_raises(self):
        with pytest.raises(OpenAIError, match="no choices"):
            _extract_content({}, self.API)

    def test_choices_empty_raises(self):
        with pytest.raises(OpenAIError, match="no choices"):
            _extract_content({"choices": []}, self.API)

    def test_upstream_error_field_is_surfaced(self):
        result = {"choices": None, "error": "model unavailable"}
        with pytest.raises(OpenAIError, match="model unavailable"):
            _extract_content(result, self.API)

    def test_choices_first_not_dict_raises(self):
        with pytest.raises(OpenAIError, match="not an object"):
            _extract_content({"choices": ["string"]}, self.API)

    def test_message_missing_raises(self):
        with pytest.raises(OpenAIError, match="missing 'message'"):
            _extract_content({"choices": [{}]}, self.API)

    def test_content_null_raises_with_finish_reason(self):
        result = {"choices": [{
            "message": {"content": None},
            "finish_reason": "content_filter",
        }]}
        with pytest.raises(OpenAIError, match="content=null") as exc:
            _extract_content(result, self.API)
        assert "content_filter" in str(exc.value)

    def test_content_non_string_raises(self):
        result = {"choices": [{"message": {"content": 42}}]}
        with pytest.raises(OpenAIError, match="not a string"):
            _extract_content(result, self.API)
