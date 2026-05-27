"""Loader for system prompt templates.

System prompts live in ``system_prompts/<mode_id>.txt`` where ``mode_id`` is the
filename-safe identifier of the mode. The visible label (with parentheses,
slashes etc.) is mapped to the file name via :data:`MODE_TO_FILE`.

Each prompt file may contain the placeholder ``{shared_rules}`` which gets
replaced by the contents of ``_shared_rules.txt``.

User vs. template fallback
--------------------------
Each prompt file resolves through a two-step lookup mirroring the pattern in
:mod:`config_loader`:

1. ``system_prompts/<name>.txt`` — user-editable, gitignored (survives ``git pull``)
2. ``system_prompts/<name>.txt.example`` — committed template (shipped fallback)

The committed templates ship as ``.txt.example`` so a ``git pull`` never
clobbers user-edited copies. The loader returns the user file when present
and otherwise falls back to the example. ``FileNotFoundError`` is raised
only if neither file exists.

Per-model override cascade
--------------------------
:func:`get_system_prompt` accepts an optional ``model_name`` and resolves in
this order:

1. ``system_prompts/<mode_id>.<family>.txt`` — model-family override
2. ``system_prompts/<mode_id>.txt`` — default
3. ``KeyError`` for an unregistered mode

Each step honours the user/example fallback above. The family is detected by
:func:`detect_family` from the model tag.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROMPTS_DIR = Path(__file__).parent / "system_prompts"

# Mapping: visible mode label -> filename (without .txt)
MODE_TO_FILE: Dict[str, str] = {
    "FLUX Kontext (Scene Edit)":      "flux_kontext_scene_edit",
    "FLUX Kontext (Couple Scene)":    "flux_kontext_couple_scene",
    "Qwen Image Edit (Couple Scene)": "qwen_image_edit_couple_scene",
    "FLUX Text-to-Image":             "flux_text_to_image",
    "Z-Image Text-to-Image":          "zimage_text_to_image",
    "SDXL Photorealistic":            "sdxl_photorealistic",
    "SDXL Pony/Illustrious":          "sdxl_pony_illustrious",
    "Random Character (Z-Image)":     "random_character_zimage",
    "Random Character (Pony)":        "random_character_pony",
    "Custom System Prompt":           "custom_system_prompt",
}

AVAILABLE_MODES = list(MODE_TO_FILE.keys())

# Ordered (regex, family) tuples. First match wins. Case-sensitive — Ollama
# and vLLM model tags are case-sensitive, so we match them literally.
# Add a new family by inserting a line in the desired priority position.
MODEL_FAMILY_PATTERNS: List[Tuple[str, str]] = [
    (r"^Fermi/Cydonia",         "cydonia"),
    (r"^qwen3-vl",              "qwen3vl"),
    (r"^qwen3(\.|$|-prompt)",   "qwen3"),
    (r"^gemma",                 "gemma"),
    (r"^huihui_ai/",            "abliterated"),
    (r"^llama",                 "llama"),
]


def detect_family(model_name: Optional[str]) -> Optional[str]:
    """Map an Ollama/vLLM model tag to a family name, or ``None`` if unknown.

    Walks :data:`MODEL_FAMILY_PATTERNS` in order and returns the first match.
    Returns ``None`` for ``None``, empty string, or unmatched tags. Matching
    is case-sensitive.
    """
    if not model_name:
        return None
    for pattern, family in MODEL_FAMILY_PATTERNS:
        if re.search(pattern, model_name):
            return family
    return None


def _resolve_prompt_path(name: str) -> Optional[Path]:
    """Return the on-disk path for prompt ``name`` (without ``.txt`` suffix).

    Order of resolution mirrors :mod:`config_loader`:

    1. ``<name>.txt`` — user-editable copy (gitignored)
    2. ``<name>.txt.example`` — committed template (shipped fallback)
    3. ``None`` if neither exists

    The user file always wins so locally edited prompts survive ``git pull``.
    """
    user_path = _PROMPTS_DIR / f"{name}.txt"
    if user_path.is_file():
        return user_path
    example_path = _PROMPTS_DIR / f"{name}.txt.example"
    if example_path.is_file():
        return example_path
    return None


def _prompt_file_exists(name: str) -> bool:
    """Return ``True`` if either a user copy or example template exists."""
    return _resolve_prompt_path(name) is not None


def _read_file(name: str) -> str:
    """Read the prompt file ``name`` (without ``.txt`` suffix).

    Prefers ``<name>.txt`` (user-editable), falls back to ``<name>.txt.example``
    (committed template). Raises ``FileNotFoundError`` if neither exists.
    """
    path = _resolve_prompt_path(name)
    if path is None:
        raise FileNotFoundError(
            f"System prompt file not found: "
            f"{_PROMPTS_DIR / f'{name}.txt'} (also no .example template)"
        )
    return path.read_text(encoding="utf-8").rstrip("\n")


def _resolve_template(file_base: str, model_name: Optional[str]) -> str:
    """Return the raw template, preferring a per-model override if present."""
    family = detect_family(model_name)
    if family is not None:
        override_name = f"{file_base}.{family}"
        if _prompt_file_exists(override_name):
            return _read_file(override_name)
    return _read_file(file_base)


def render_template(file_base: str, model_name: Optional[str] = None) -> str:
    """Return a fully rendered system-prompt template for ``file_base``.

    Applies the per-model override cascade (see module docstring) and
    substitutes the ``{shared_rules}`` placeholder. This is the shared
    entry point for :func:`get_system_prompt`, the vision-prompt loader,
    and the composer-prompt loader so that all three honour the same
    cascade and substitution rules.

    Raises ``FileNotFoundError`` if neither the override nor the default
    file exists.
    """
    template = _resolve_template(file_base, model_name)
    if "{shared_rules}" in template:
        template = template.replace("{shared_rules}", _read_file("_shared_rules"))
    return template


def get_system_prompt(mode: str, model_name: Optional[str] = None) -> str:
    """Return the rendered system prompt for the given mode label.

    If ``model_name`` is provided and matches a family in
    :data:`MODEL_FAMILY_PATTERNS`, a model-specific override file
    (``<mode_id>.<family>.txt``) takes precedence over the default. Falls
    back to the default if no override exists or the family is unknown.

    Raises ``KeyError`` for unknown modes.
    """
    if mode not in MODE_TO_FILE:
        raise KeyError(f"Unknown prompt mode: {mode}")
    return render_template(MODE_TO_FILE[mode], model_name)
