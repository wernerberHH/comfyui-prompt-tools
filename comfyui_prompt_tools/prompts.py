"""Loader for system prompt templates.

System prompts live in ``system_prompts/<template>.txt`` where ``template`` is
the basename registered for the mode in the prompt catalog (see
:mod:`catalog`). The visible label — and every former label kept as an alias —
resolves to that basename through the catalog, so renaming a label or adding a
mode is a data change, not a code change.

Each prompt file may contain the placeholder ``{shared_rules}`` which gets
replaced by the contents of ``_shared_rules.txt``, and ``{img1}`` / ``{img2}``
placeholders which :func:`catalog.render_image_refs` turns into the image
reference wording of the selected target model.

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
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .catalog import (
    PROMPTS_DIR,
    load_catalog,
    render_image_refs,
    resolve_helper_mode,
    resolve_target_model,
    resolve_template_path,
)

#: Module attribute so tests can redirect the prompt library at a temp dir.
_PROMPTS_DIR = PROMPTS_DIR
logger = logging.getLogger(__name__)

# Derived views on the catalog, kept for callers that used to read the
# hard-coded dictionaries. The catalog is the source of truth; these are
# snapshots taken at import time.
#: Mapping: visible mode label -> template basename (without .txt)
MODE_TO_FILE: Dict[str, str] = {
    entry.label: entry.template for entry in load_catalog().prompt_helper_modes
}

AVAILABLE_MODES = list(MODE_TO_FILE.keys())

# Built-in family patterns: ordered (regex, family) tuples, first match wins.
# Case-sensitive — Ollama and vLLM model tags are case-sensitive, so we match
# them literally. These are the neutral shipped defaults; site-specific
# mappings live in config/model_families.yaml (gitignored) and are PREPENDED
# at lookup time so they win over these. Add a new default by inserting a line
# in the desired priority position.
MODEL_FAMILY_PATTERNS: List[Tuple[str, str]] = [
    (r"^qwen3-vl",              "qwen3vl"),
    (r"^qwen3(\.|$|-prompt)",   "qwen3"),
    (r"^gemma",                 "gemma"),
    (r"^llama",                 "llama"),
]

_FAMILIES_FILE = Path(__file__).resolve().parent.parent / "config" / "model_families.yaml"

_custom_family_cache: Optional[List[Tuple[str, str]]] = None
_warned_missing_yaml = False


def _load_custom_family_patterns(refresh: bool = False) -> List[Tuple[str, str]]:
    """Return user-defined (regex, family) tuples from config/model_families.yaml.

    The file is gitignored so site-specific model routing stays private. A
    missing file, missing pyyaml, or malformed YAML all yield an empty list —
    the built-in :data:`MODEL_FAMILY_PATTERNS` still apply. Cached after the
    first call; pass ``refresh=True`` to force a re-read (mainly for tests).

    Schema::

        families:
          - { pattern: "^Some/Model", family: somefamily }
    """
    global _custom_family_cache, _warned_missing_yaml
    if _custom_family_cache is not None and not refresh:
        return _custom_family_cache

    patterns: List[Tuple[str, str]] = []
    if _FAMILIES_FILE.is_file():
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            if not _warned_missing_yaml:
                logger.warning(
                    "pyyaml not installed — custom model-family mappings "
                    "disabled. Install with: pip install pyyaml"
                )
                _warned_missing_yaml = True
            yaml = None
        if yaml is not None:
            try:
                with _FAMILIES_FILE.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("Failed to read %s: %s", _FAMILIES_FILE, exc)
                data = None
            if isinstance(data, dict):
                for entry in data.get("families") or []:
                    if not isinstance(entry, dict):
                        continue
                    pattern = entry.get("pattern")
                    family = entry.get("family")
                    if (isinstance(pattern, str) and pattern.strip()
                            and isinstance(family, str) and family.strip()):
                        patterns.append((pattern, family))

    _custom_family_cache = patterns
    return patterns


def detect_family(model_name: Optional[str]) -> Optional[str]:
    """Map an Ollama/vLLM model tag to a family name, or ``None`` if unknown.

    Walks the custom patterns from config/model_families.yaml (gitignored,
    prepended so they win) followed by the built-in
    :data:`MODEL_FAMILY_PATTERNS`, returning the first match. Returns ``None``
    for ``None``, empty string, or unmatched tags. Matching is case-sensitive.
    """
    if not model_name:
        return None
    for pattern, family in _load_custom_family_patterns() + MODEL_FAMILY_PATTERNS:
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
    return resolve_template_path(name, _PROMPTS_DIR)


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


def render_template(
    file_base: str,
    model_name: Optional[str] = None,
    target_model: Optional[str] = None,
) -> str:
    """Return a fully rendered system-prompt template for ``file_base``.

    Applies the per-model override cascade (see module docstring), then
    substitutes ``{shared_rules}`` and finally the ``{imgN}`` image
    references. This is the shared entry point for
    :func:`get_system_prompt`, the vision-prompt loader, and the
    composer-prompt loader so that all three honour the same cascade and
    substitution rules.

    ``target_model`` is a target-model label from the catalog; ``None``
    means the neutral ``Generic`` wording.

    Raises ``FileNotFoundError`` if neither the override nor the default
    file exists, and ``KeyError`` for an unknown ``target_model``.
    """
    template = _resolve_template(file_base, model_name)
    if "{shared_rules}" in template:
        template = template.replace("{shared_rules}", _read_file("_shared_rules"))
    return render_image_refs(template, resolve_target_model(target_model))


def get_system_prompt(mode: str, model_name: Optional[str] = None) -> str:
    """Return the rendered system prompt for the given mode label.

    ``mode`` may be the current label or any former label kept as an alias
    in the catalog, so saved workflows keep resolving after a rename.

    If ``model_name`` is provided and matches a family in
    :data:`MODEL_FAMILY_PATTERNS`, a model-specific override file
    (``<mode_id>.<family>.txt``) takes precedence over the default. Falls
    back to the default if no override exists or the family is unknown.

    Image references render with the neutral ``Generic`` wording — the
    PromptHelper has no target-model selector of its own.

    Raises ``KeyError`` for unknown modes.
    """
    return render_template(resolve_helper_mode(mode).template, model_name)
