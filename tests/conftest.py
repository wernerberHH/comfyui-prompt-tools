"""Shared pytest fixtures for comfyui-prompt-tools tests."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Repo root contains both __init__.py (ComfyUI loader) and comfyui_prompt_tools/.
# Tests import from comfyui_prompt_tools directly.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_mock_response(body: bytes) -> MagicMock:
    """Return a context-manager mock that yields a response reading ``body``."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    return mock_resp


_OPENAI_RESPONSE = json.dumps(
    {
        "choices": [{"message": {"content": "test output"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
).encode()

_OLLAMA_RESPONSE = json.dumps(
    {"message": {"content": "test output"}}
).encode()


@pytest.fixture
def mock_openai_urlopen():
    """Patches urllib.request.urlopen inside openai_client to prevent real HTTP.

    Yields the patch mock so tests can inspect call_args for payload assertions.
    """
    with patch(
        "comfyui_prompt_tools.engines.openai_client.urllib.request.urlopen",
        return_value=_make_mock_response(_OPENAI_RESPONSE),
    ) as m:
        yield m


@pytest.fixture
def mock_ollama_urlopen():
    """Patches urllib.request.urlopen inside ollama_client to prevent real HTTP.

    Yields the patch mock so tests can inspect call_args for payload assertions.
    """
    with patch(
        "comfyui_prompt_tools.engines.ollama_client.urllib.request.urlopen",
        return_value=_make_mock_response(_OLLAMA_RESPONSE),
    ) as m:
        yield m
