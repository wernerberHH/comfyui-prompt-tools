"""Image I/O helpers for vision nodes.

Converts ComfyUI image tensors (shape [B, H, W, C], values 0–1, float32)
to base64-encoded JPEG strings suitable for vision model APIs.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def tensor_to_b64(
    tensor: "torch.Tensor",
    max_edge: int = 1280,
    jpeg_quality: int = 85,
) -> str:
    """Convert a ComfyUI image tensor to a base64-encoded JPEG string.

    Takes the first image from a batch tensor of shape [B, H, W, C] with
    float values in [0, 1]. The long edge is clamped to ``max_edge`` before
    encoding; aspect ratio is preserved.

    Returns a plain base64 string without the ``data:`` URI prefix.
    """
    import torch
    from PIL import Image

    if tensor.ndim == 4:
        img_tensor = tensor[0]
    else:
        img_tensor = tensor

    pixels = (img_tensor.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
    image = Image.fromarray(pixels)

    h, w = image.height, image.width
    long_edge = max(h, w)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        image = image.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
