"""Unit tests for comfyui_prompt_tools.image_io.tensor_to_b64.

No mocks — pure logic tested against synthetic tensors.
Requires torch and Pillow (available in any ComfyUI environment).
"""

import base64
import io

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from comfyui_prompt_tools.image_io import tensor_to_b64  # noqa: E402


def _make_tensor(h: int, w: int, batch: int = 1) -> "torch.Tensor":
    """Return a solid-colour float32 tensor of shape [B, H, W, 3]."""
    return torch.full((batch, h, w, 3), 0.5, dtype=torch.float32)


class TestTensorToB64:
    def test_returns_nonempty_string(self):
        """Verifies the function returns a non-empty string for a valid tensor.
        Mocks: none.
        """
        result = tensor_to_b64(_make_tensor(64, 64))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_b64_decodes_to_valid_jpeg(self):
        """Verifies the returned string decodes to a valid JPEG bytestream.
        Mocks: none.
        """
        result = tensor_to_b64(_make_tensor(64, 64))
        raw = base64.b64decode(result)
        img = Image.open(io.BytesIO(raw))
        assert img.format == "JPEG"

    def test_no_resize_when_within_max_edge(self):
        """Verifies that an image below max_edge is not scaled up.
        Mocks: none.
        """
        result = tensor_to_b64(_make_tensor(100, 80), max_edge=256)
        raw = base64.b64decode(result)
        img = Image.open(io.BytesIO(raw))
        assert img.width == 80
        assert img.height == 100

    def test_long_edge_clamped_to_max_edge(self):
        """Verifies that the long edge is scaled to exactly max_edge.
        Mocks: none.
        """
        result = tensor_to_b64(_make_tensor(400, 200), max_edge=200)
        raw = base64.b64decode(result)
        img = Image.open(io.BytesIO(raw))
        assert img.height == 200
        assert img.width == 100  # aspect ratio preserved: 400:200 → 200:100

    def test_aspect_ratio_preserved_landscape(self):
        """Verifies aspect ratio is preserved when resizing a landscape image.
        Mocks: none.
        """
        result = tensor_to_b64(_make_tensor(200, 400), max_edge=200)
        raw = base64.b64decode(result)
        img = Image.open(io.BytesIO(raw))
        assert img.width == 200
        assert img.height == 100

    def test_batch_dim_ignored_uses_first_image(self):
        """Verifies that only the first image in a batch tensor is encoded.
        Mocks: none.
        """
        # Batch of 3 images — should not raise; returns one encoded image
        t = torch.rand(3, 64, 64, 3)
        result = tensor_to_b64(t)
        raw = base64.b64decode(result)
        img = Image.open(io.BytesIO(raw))
        assert img.size == (64, 64)
