"""Loader for the prompt catalog — the data source behind every dropdown.

The catalog replaces the label→file dictionaries that used to live in
:mod:`prompts`, :mod:`vision_prompts` and :mod:`nodes.prompt_composer`.
Adding a target model, a PromptHelper mode, a composer style or a vision
mode is an edit to ``config/catalog.yaml.example`` (plus a template file)
and needs no code change.

User vs. template fallback
--------------------------
Mirrors :mod:`config_loader` and :mod:`prompts`:

1. ``config/catalog.yaml.example`` — committed, the complete shipped truth
2. ``config/catalog.yaml`` — user file, gitignored, OVERLAID on top

The overlay is matched by ``id`` per section: an existing ``id`` gets its
listed fields replaced, an unknown ``id`` is appended at the end of its
section, and ``enabled: false`` removes the entry from its list. A user
therefore never has to restate the whole catalog to change one field.

Fail fast
---------
Unlike :mod:`config_loader`, a broken catalog is not survivable — without it
there are no dropdown values at all. :func:`load_catalog` therefore raises
:class:`CatalogError` with a message naming the offending section, entry and
field instead of degrading silently.

Image references
----------------
Target models address their reference images differently (``image 1``,
``Picture 1``, ``<image1>``). Templates write ``{img1}`` / ``{img2}`` and
:func:`render_image_refs` substitutes the ``image_ref`` pattern of the chosen
target model. That is the only place the wording is decided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Directory holding the system-prompt templates. Defined here because the
#: catalog validates that every ``template`` actually exists on disk;
#: :mod:`prompts` imports it from here so the path has exactly one home.
PROMPTS_DIR = Path(__file__).resolve().parent / "system_prompts"

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_EXAMPLE_FILE = _CONFIG_DIR / "catalog.yaml.example"
_USER_FILE = _CONFIG_DIR / "catalog.yaml"

#: Schema versions this loader understands.
SUPPORTED_VERSIONS = (1,)

#: The target model used when no selection is made: the neutral wording that
#: predates the catalog. PromptHelper and PromptComposer always render with
#: it, and it is the VisionPromptHelper default.
DEFAULT_TARGET_MODEL_ID = "generic"

_SECTIONS = ("target_models", "prompt_helper_modes", "composer_styles", "vision_modes")

_VISION_KINDS = ("edit", "describe")

_IMG_PLACEHOLDER_RE = re.compile(r"\{img(\d+)\}")


class CatalogError(RuntimeError):
    """Raised when the catalog is missing, unparseable or inconsistent."""


def resolve_template_path(basename: str, prompts_dir: Path = PROMPTS_DIR) -> Optional[Path]:
    """Return the on-disk path of template ``basename``, or ``None``.

    1. ``<basename>.txt`` — user-editable copy (gitignored)
    2. ``<basename>.txt.example`` — committed template (shipped fallback)

    This is the single implementation of the two-step lookup;
    :mod:`prompts` calls it with its own (test-patchable) directory.
    """
    user_path = prompts_dir / f"{basename}.txt"
    if user_path.is_file():
        return user_path
    example_path = prompts_dir / f"{basename}.txt.example"
    if example_path.is_file():
        return example_path
    return None


# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetModel:
    """An image / video model the prompts are written for."""

    id: str
    label: str
    image_ref: Optional[str] = None
    aliases: Tuple[str, ...] = ()

    @property
    def takes_images(self) -> bool:
        """True if the model addresses reference images inside the prompt."""
        return self.image_ref is not None


@dataclass(frozen=True)
class PromptEntry:
    """A PromptHelper mode or a PromptComposer output style."""

    id: str
    label: str
    template: str
    target_model: str
    aliases: Tuple[str, ...] = ()
    uses_random_pools: bool = False


@dataclass(frozen=True)
class VisionMode:
    """A VisionPromptHelper mode."""

    id: str
    label: str
    template: str
    kind: str
    images: int
    aliases: Tuple[str, ...] = ()

    @property
    def is_edit(self) -> bool:
        return self.kind == "edit"

    @property
    def uses_two_images(self) -> bool:
        return self.images >= 2


@dataclass(frozen=True)
class Catalog:
    """A validated catalog with label/alias lookup per section."""

    version: int
    target_models: Tuple[TargetModel, ...]
    prompt_helper_modes: Tuple[PromptEntry, ...]
    composer_styles: Tuple[PromptEntry, ...]
    vision_modes: Tuple[VisionMode, ...]
    _index: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    # -- lookup ------------------------------------------------------------
    def _lookup(self, section: str, what: str, value: Any) -> Any:
        if not isinstance(value, str):
            raise KeyError(f"Unknown {what}: {value!r} (expected a string)")
        entry = self._index[section].get(value)
        if entry is None:
            known = ", ".join(repr(e.label) for e in getattr(self, section))
            raise KeyError(f"Unknown {what}: {value!r}. Known values: {known}")
        return entry

    def target_model(self, value: str) -> TargetModel:
        """Resolve a target-model label or alias. Raises ``KeyError``."""
        return self._lookup("target_models", "target model", value)

    def helper_mode(self, value: str) -> PromptEntry:
        """Resolve a PromptHelper mode label or alias. Raises ``KeyError``."""
        return self._lookup("prompt_helper_modes", "prompt mode", value)

    def composer_style(self, value: str) -> PromptEntry:
        """Resolve a composer style label or alias. Raises ``KeyError``."""
        return self._lookup("composer_styles", "output style", value)

    def vision_mode(self, value: str) -> VisionMode:
        """Resolve a vision mode label or alias. Raises ``KeyError``."""
        return self._lookup("vision_modes", "vision prompt mode", value)

    def image_target_model(self, value: str) -> TargetModel:
        """Resolve a target model and require that it takes reference images."""
        model = self.target_model(value)
        if not model.takes_images:
            offered = ", ".join(repr(m.label) for m in self.image_target_models)
            raise KeyError(
                f"Target model {value!r} takes no reference images "
                f"(no image_ref in the catalog). Known values: {offered}"
            )
        return model

    # -- derived lists -----------------------------------------------------
    @property
    def image_target_models(self) -> Tuple[TargetModel, ...]:
        """Target models that address reference images inside the prompt."""
        return tuple(m for m in self.target_models if m.takes_images)

    @property
    def default_target_model(self) -> TargetModel:
        """The ``Generic`` entry — validated to exist and carry an image_ref."""
        for model in self.target_models:
            if model.id == DEFAULT_TARGET_MODEL_ID:
                return model
        raise CatalogError(  # pragma: no cover — guaranteed by _parse
            f"target_models has no entry with id {DEFAULT_TARGET_MODEL_ID!r}"
        )


# ---------------------------------------------------------------------------
# YAML reading
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    """Parse a catalog YAML file into a dict. Raises :class:`CatalogError`."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — declared dependency
        raise CatalogError(
            "pyyaml is required to read the prompt catalog. "
            "Install it with: pip install pyyaml"
        ) from exc
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _check_version(data: dict, path: Path) -> int:
    version = data.get("version")
    if version is None:
        raise CatalogError(f"{path}: missing top-level 'version'")
    if version not in SUPPORTED_VERSIONS:
        supported = ", ".join(str(v) for v in SUPPORTED_VERSIONS)
        raise CatalogError(
            f"{path}: unsupported catalog version {version!r} "
            f"(this build understands: {supported})"
        )
    return version


def _section_entries(data: dict, section: str, path: Path) -> List[dict]:
    """Return the raw entry list of ``section``, validating its shape."""
    if section not in data:
        raise CatalogError(f"{path}: missing section '{section}'")
    raw = data[section]
    if not isinstance(raw, list):
        raise CatalogError(
            f"{path}: section '{section}' must be a list, got {type(raw).__name__}"
        )
    entries: List[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise CatalogError(
                f"{path}: {section}[{i}] must be a mapping, got {type(entry).__name__}"
            )
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise CatalogError(f"{path}: {section}[{i}] has no usable 'id'")
        entries.append(entry)
    return entries


def _overlay(base: List[dict], extra: List[dict], section: str) -> List[dict]:
    """Merge user entries onto shipped entries, matched by ``id``.

    Same id replaces the listed fields, an unknown id is appended, and the
    caller drops whatever ends up with ``enabled: false``.
    """
    merged = [dict(e) for e in base]
    by_id = {e["id"]: e for e in merged}
    for entry in extra:
        entry_id = entry["id"]
        existing = by_id.get(entry_id)
        if existing is None:
            new_entry = dict(entry)
            merged.append(new_entry)
            by_id[entry_id] = new_entry
        else:
            existing.update(entry)
    seen: Dict[str, int] = {}
    for entry in merged:
        seen[entry["id"]] = seen.get(entry["id"], 0) + 1
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    if duplicates:
        raise CatalogError(
            f"section '{section}': duplicate id(s) {', '.join(duplicates)}"
        )
    return [e for e in merged if e.get("enabled", True)]


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------
def _where(section: str, entry: dict) -> str:
    return f"{section} entry {entry['id']!r}"


def _str_field(entry: dict, key: str, section: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{_where(section, entry)}: '{key}' must be a non-empty string")
    return value.strip()


def _aliases(entry: dict, section: str) -> Tuple[str, ...]:
    raw = entry.get("aliases") or []
    if not isinstance(raw, list):
        raise CatalogError(f"{_where(section, entry)}: 'aliases' must be a list")
    out: List[str] = []
    for alias in raw:
        if not isinstance(alias, str) or not alias.strip():
            raise CatalogError(
                f"{_where(section, entry)}: every alias must be a non-empty string"
            )
        out.append(alias.strip())
    return tuple(out)


def _template(entry: dict, section: str) -> str:
    basename = _str_field(entry, "template", section)
    if resolve_template_path(basename) is None:
        raise CatalogError(
            f"{_where(section, entry)}: template {basename!r} not found — expected "
            f"{PROMPTS_DIR / (basename + '.txt')} or its .txt.example fallback"
        )
    return basename


def _image_ref(entry: dict, section: str) -> Optional[str]:
    if "image_ref" not in entry or entry["image_ref"] is None:
        return None
    pattern = _str_field(entry, "image_ref", section)
    if "{n}" not in pattern:
        raise CatalogError(
            f"{_where(section, entry)}: image_ref {pattern!r} must contain '{{n}}' — "
            "that is where the image number goes"
        )
    return pattern


def _build_index(section: str, entries: List[Any]) -> Dict[str, Any]:
    """Map every label and alias of a section to its entry.

    Labels and aliases share one namespace per section: an old label must
    resolve exactly like the new one, so a collision between two entries is
    an error rather than a silent last-one-wins.
    """
    index: Dict[str, Any] = {}
    for entry in entries:
        for name in (entry.label, *entry.aliases):
            if name in index:
                raise CatalogError(
                    f"section '{section}': label/alias {name!r} is used by both "
                    f"{index[name].id!r} and {entry.id!r}"
                )
            index[name] = entry
    return index


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_cache: Optional[Catalog] = None


def _parse(data: dict, user_data: Optional[dict], path: Path) -> Catalog:
    version = _check_version(data, path)

    raw: Dict[str, List[dict]] = {}
    for section in _SECTIONS:
        base = _section_entries(data, section, path)
        extra = (
            _section_entries(user_data, section, _USER_FILE)
            if user_data is not None and section in user_data
            else []
        )
        raw[section] = _overlay(base, extra, section)

    target_models = tuple(
        TargetModel(
            id=entry["id"],
            label=_str_field(entry, "label", "target_models"),
            image_ref=_image_ref(entry, "target_models"),
            aliases=_aliases(entry, "target_models"),
        )
        for entry in raw["target_models"]
    )
    known_models = {m.id for m in target_models}

    def _prompt_entries(section: str) -> Tuple[PromptEntry, ...]:
        out: List[PromptEntry] = []
        for entry in raw[section]:
            target = _str_field(entry, "target_model", section)
            if target not in known_models:
                raise CatalogError(
                    f"{_where(section, entry)}: target_model {target!r} is not an id "
                    f"in 'target_models' ({', '.join(sorted(known_models))})"
                )
            out.append(
                PromptEntry(
                    id=entry["id"],
                    label=_str_field(entry, "label", section),
                    template=_template(entry, section),
                    target_model=target,
                    aliases=_aliases(entry, section),
                    uses_random_pools=bool(entry.get("uses_random_pools", False)),
                )
            )
        return tuple(out)

    vision_modes: List[VisionMode] = []
    for entry in raw["vision_modes"]:
        kind = _str_field(entry, "kind", "vision_modes")
        if kind not in _VISION_KINDS:
            raise CatalogError(
                f"{_where('vision_modes', entry)}: kind {kind!r} must be one of "
                f"{', '.join(_VISION_KINDS)}"
            )
        images = entry.get("images")
        if images not in (1, 2):
            raise CatalogError(
                f"{_where('vision_modes', entry)}: 'images' must be 1 or 2, got {images!r}"
            )
        vision_modes.append(
            VisionMode(
                id=entry["id"],
                label=_str_field(entry, "label", "vision_modes"),
                template=_template(entry, "vision_modes"),
                kind=kind,
                images=images,
                aliases=_aliases(entry, "vision_modes"),
            )
        )

    catalog = Catalog(
        version=version,
        target_models=target_models,
        prompt_helper_modes=_prompt_entries("prompt_helper_modes"),
        composer_styles=_prompt_entries("composer_styles"),
        vision_modes=tuple(vision_modes),
        _index={},
    )
    for section in _SECTIONS:
        catalog._index[section] = _build_index(section, list(getattr(catalog, section)))

    default_model = next(
        (m for m in target_models if m.id == DEFAULT_TARGET_MODEL_ID), None
    )
    if default_model is None:
        raise CatalogError(
            f"target_models must contain an entry with id {DEFAULT_TARGET_MODEL_ID!r} — "
            "it is the neutral default used by every node"
        )
    if not default_model.takes_images:
        raise CatalogError(
            f"target model {DEFAULT_TARGET_MODEL_ID!r} must define an 'image_ref' — "
            "it is the pattern used when no target model is selected"
        )
    return catalog


def load_catalog(refresh: bool = False) -> Catalog:
    """Return the validated catalog, cached after the first call.

    Reads ``config/catalog.yaml.example`` and overlays an optional
    ``config/catalog.yaml``. Pass ``refresh=True`` to re-read from disk
    (mainly for tests). Raises :class:`CatalogError` on any problem.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    if not _EXAMPLE_FILE.is_file():
        raise CatalogError(f"Prompt catalog not found: {_EXAMPLE_FILE}")
    data = _read_yaml(_EXAMPLE_FILE)
    user_data = _read_yaml(_USER_FILE) if _USER_FILE.is_file() else None
    if user_data is not None:
        _check_version(user_data, _USER_FILE)

    _cache = _parse(data, user_data, _EXAMPLE_FILE)
    return _cache


# ---------------------------------------------------------------------------
# Convenience accessors — what the nodes and loaders call
# ---------------------------------------------------------------------------
def helper_mode_labels() -> List[str]:
    """Dropdown values for the PromptHelper ``mode`` input."""
    return [e.label for e in load_catalog().prompt_helper_modes]


def composer_style_labels() -> List[str]:
    """Dropdown values for the PromptComposer ``output_style`` input."""
    return [e.label for e in load_catalog().composer_styles]


def vision_mode_labels() -> List[str]:
    """Dropdown values for the VisionPromptHelper ``mode`` input."""
    return [e.label for e in load_catalog().vision_modes]


def image_target_model_labels() -> List[str]:
    """Dropdown values for the VisionPromptHelper ``target_model`` input.

    Only models that address reference images inside the prompt text.
    """
    return [m.label for m in load_catalog().image_target_models]


def default_target_model_label() -> str:
    """Label of the neutral default target model (``Generic``)."""
    return load_catalog().default_target_model.label


def resolve_helper_mode(value: str) -> PromptEntry:
    """Label or alias -> PromptHelper mode entry. Raises ``KeyError``."""
    return load_catalog().helper_mode(value)


def resolve_composer_style(value: str) -> PromptEntry:
    """Label or alias -> composer style entry. Raises ``KeyError``."""
    return load_catalog().composer_style(value)


def resolve_vision_mode(value: str) -> VisionMode:
    """Label or alias -> vision mode entry. Raises ``KeyError``."""
    return load_catalog().vision_mode(value)


def resolve_target_model(value: Optional[str]) -> TargetModel:
    """Label or alias -> target model. ``None`` yields the default."""
    catalog = load_catalog()
    if value is None:
        return catalog.default_target_model
    return catalog.target_model(value)


def resolve_image_target_model(value: Optional[str]) -> TargetModel:
    """Like :func:`resolve_target_model` but rejects text-only models."""
    catalog = load_catalog()
    if value is None:
        return catalog.default_target_model
    return catalog.image_target_model(value)


def render_image_refs(text: str, target_model: TargetModel) -> str:
    """Replace every ``{imgN}`` placeholder with the model's image reference.

    ``{img1}`` becomes ``image 1`` for a model whose ``image_ref`` is
    ``"image {n}"`` and ``<image1>`` for ``"<image{n}>"``. Text without
    placeholders is returned unchanged.
    """
    pattern = target_model.image_ref
    if pattern is None:
        raise ValueError(
            f"Target model {target_model.label!r} has no image_ref — "
            "cannot render image references"
        )
    return _IMG_PLACEHOLDER_RE.sub(lambda m: pattern.replace("{n}", m.group(1)), text)
