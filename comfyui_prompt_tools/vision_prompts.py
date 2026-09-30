"""Loader for vision-mode system prompt templates.

Mirrors the structure of :mod:`prompts` but for vision nodes.
System prompt files live in the same ``system_prompts/`` directory and follow
the same ``{shared_rules}`` substitution and per-model override cascade
(see :mod:`prompts` module docstring for cascade details). Which modes exist,
what they are called and which template backs them comes from the prompt
catalog (see :mod:`catalog`).

Modes split into two families, declared as ``kind`` in the catalog:

- **Edit modes**: produce an image-editing instruction that transforms the
  input image. Their templates address the images the *output* prompt has to
  name through ``{img1}`` / ``{img2}`` placeholders, rendered with the
  ``image_ref`` wording of the caller's ``target_model``. ``images: 2`` marks
  the modes that consume a second reference image.
- **Describe modes**: extract a single aspect of the input image as a
  short snippet for downstream prompt composition. Their output carries no
  image reference at all.

Instructions *to the vision LLM* about which input is which always read
"the first image" / "the second image" — that wording is independent of the
target model.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from .catalog import load_catalog, resolve_vision_mode
from .prompts import render_template

# Derived views on the catalog, kept for callers that used to read the
# hard-coded dictionaries. The catalog is the source of truth; these are
# snapshots taken at import time.
#: Mapping: visible mode label -> template basename (without .txt)
VISION_MODE_TO_FILE: Dict[str, str] = {
    entry.label: entry.template for entry in load_catalog().vision_modes
}

AVAILABLE_VISION_MODES = list(VISION_MODE_TO_FILE.keys())

#: Modes that consume a second reference image (``images: 2`` in the
#: catalog). All other modes use only image_1; passing image_2 is silently
#: ignored.
TWO_IMAGE_MODES: FrozenSet[str] = frozenset(
    entry.label for entry in load_catalog().vision_modes if entry.uses_two_images
)


def get_vision_system_prompt(
    mode: str,
    model_name: Optional[str] = None,
    target_model: Optional[str] = None,
) -> str:
    """Return the rendered system prompt for the given vision mode label.

    ``mode`` may be the current label or any former label kept as an alias
    in the catalog.

    If ``model_name`` is provided and maps to a known family via
    :func:`prompts.detect_family`, a model-specific override file
    (``<mode_id>.<family>.txt``) takes precedence over the default.

    ``target_model`` selects the image-reference wording used for the
    ``{imgN}`` placeholders in edit-mode templates; ``None`` means the
    neutral ``Generic`` wording.

    Raises ``KeyError`` for unknown modes or an unknown ``target_model``.
    """
    entry = resolve_vision_mode(mode)
    return render_template(entry.template, model_name, target_model)


def mode_uses_two_images(mode: str) -> bool:
    """True if the mode expects a second reference image.

    Accepts the current label or a former label kept as an alias. Unknown
    labels are reported as single-image so the node's own validation, not
    this helper, produces the error message.
    """
    try:
        return resolve_vision_mode(mode).uses_two_images
    except KeyError:
        return False
