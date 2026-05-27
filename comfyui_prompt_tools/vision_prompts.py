"""Loader for vision-mode system prompt templates.

Mirrors the structure of :mod:`prompts` but for vision nodes.
System prompt files live in the same ``system_prompts/`` directory and follow
the same ``{shared_rules}`` substitution and per-model override cascade
(see :mod:`prompts` module docstring for cascade details).

Modes split into two families:

- **Edit modes**: produce a FLUX-Kontext / Qwen-Image-Edit instruction that
  transforms the input image. ``Outfit Transfer`` consumes two reference
  images; the other edit modes consume one.
- **Describe modes**: extract a single aspect of the input image as a
  short snippet for downstream prompt composition.
"""

from __future__ import annotations

from typing import Dict, Optional

from .prompts import render_template

VISION_MODE_TO_FILE: Dict[str, str] = {
    # ---- Edit modes ------------------------------------------------
    "Outfit Transfer":     "vision_outfit_transfer",
    "Hair Change":         "vision_hair_change",
    "Body Reshape":        "vision_body_reshape",
    "Background Change":   "vision_background_change",
    "Pose Change":         "vision_pose_change",
    "Combined Edit":       "vision_combined_edit",
    # ---- Describe modes -------------------------------------------
    "Describe Picture":     "vision_describe_picture",
    "Describe Face":        "vision_describe_face",
    "Describe Hair":        "vision_describe_hair",
    "Describe Body":        "vision_describe_body",
    "Describe Pose":        "vision_describe_pose",
    "Describe Outfit":      "vision_describe_outfit",
    "Describe Background":  "vision_describe_background",
    "Describe Lighting":    "vision_describe_lighting",
    "Describe Composition": "vision_describe_composition",
}

AVAILABLE_VISION_MODES = list(VISION_MODE_TO_FILE.keys())

# Modes that consume a second reference image. All other modes use only
# image_1; passing image_2 is silently ignored. Outfit Transfer is the
# only two-image mode in v0.4.
TWO_IMAGE_MODES = frozenset({"Outfit Transfer"})


def get_vision_system_prompt(mode: str, model_name: Optional[str] = None) -> str:
    """Return the rendered system prompt for the given vision mode label.

    If ``model_name`` is provided and maps to a known family via
    :func:`prompts.detect_family`, a model-specific override file
    (``<mode_id>.<family>.txt``) takes precedence over the default.

    Raises ``KeyError`` for unknown modes.
    """
    if mode not in VISION_MODE_TO_FILE:
        raise KeyError(f"Unknown vision prompt mode: {mode}")
    return render_template(VISION_MODE_TO_FILE[mode], model_name)


def mode_uses_two_images(mode: str) -> bool:
    """True if the mode expects a second reference image."""
    return mode in TWO_IMAGE_MODES
