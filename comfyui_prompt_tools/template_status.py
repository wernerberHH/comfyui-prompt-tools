"""Tell the user when a local system-prompt copy hides a shipped template.

The prompt loader prefers ``system_prompts/<name>.txt`` (user-editable,
gitignored) over ``system_prompts/<name>.txt.example`` (committed) — see
:mod:`prompts`. That is what makes local edits survive a ``git pull``, and
it is also a trap: a copy made from an older release keeps winning after an
update, so improved templates and new placeholders never reach the user.
Package updates cannot fix this on their own — the ComfyUI-Manager unpacks a
release over the node folder and only removes files that belonged to the
previous package, so a local ``.txt`` is left untouched by design.

This module reports the situation. It never changes, renames or deletes a
file; what to do with a local copy is the user's call.

Classification
--------------
``shipped_hashes.json`` (committed next to the templates) lists the digest of
every version of every template this package has ever shipped, newest first
— written by ``scripts/update_shipped_hashes.py``. It has two sections:
``templates`` for the templates that ship today, and
``retired_override_versions`` for versions shipped under a name the package no
longer uses, keyed by base template name only (the family-name part is
dropped so model tags stay out of the public repo). A local file is compared
against that history:

``identical``
    Same content as the current shipped template. Nothing to report.
``outdated-copy``
    Same content as an *older* shipped version: taken over unchanged and now
    hiding the current template. Safe to delete.
``customized``
    Matches no shipped version — a real local edit. Deliberate, but the
    shipped template stays hidden, so improvements have to be re-applied.
``stale-shipped-override``
    A model-family override (``<name>.<family>.txt``) with no ``.txt.example``
    of its own whose content matches one of its base template's
    ``retired_override_versions``. Left behind by an older release and now
    hiding the current base template for that family. Safe to delete.
``local-override``
    A file with no shipped counterpart at all — an override the user authored.
    Reported as a count only; there is nothing it could be out of date with.

``unverified``
    ``shipped_hashes.json`` is missing or unreadable, so nothing can be
    compared. The check is skipped with one warning and the nodes load
    normally.

Digests cover the *normalised* text — decoded as UTF-8, line endings
converted to ``\\n`` and trailing newlines stripped — which is exactly the
string the loader hands to the LLM (:func:`prompts._read_file` rstrips it).
An editor that rewrites line endings or appends a final newline therefore
does not turn an untouched copy into a ``customized`` one.

``_shared_rules.txt`` is deliberately classified like any other template: it
ships as ``.txt.example``, a local copy shadows it the same way, and a stale
copy of it affects every mode that substitutes ``{shared_rules}``. Only the
``{imgN}`` note below is withheld — it is not a mode template and addresses
no reference images.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from .catalog import PROMPTS_DIR, CatalogError, load_catalog

logger = logging.getLogger(__name__)

#: The committed digest history. Written by ``scripts/update_shipped_hashes.py``.
SHIPPED_HASHES_FILE = PROMPTS_DIR / "shipped_hashes.json"

#: Suffix of a committed template and of a local copy.
EXAMPLE_SUFFIX = ".txt.example"
USER_SUFFIX = ".txt"

# -- status values ----------------------------------------------------------
SHIPPED = "shipped"
IDENTICAL = "identical"
OUTDATED_COPY = "outdated-copy"
CUSTOMIZED = "customized"
STALE_SHIPPED_OVERRIDE = "stale-shipped-override"
LOCAL_OVERRIDE = "local-override"
UNVERIFIED = "unverified"

#: Statuses the startup report lists file by file, in reporting order.
REPORTED_STATUSES = (OUTDATED_COPY, CUSTOMIZED, STALE_SHIPPED_OVERRIDE)

_ADVICE: Dict[str, str] = {
    OUTDATED_COPY: (
        "unchanged copies of an older shipped template — they hide the "
        "current one. Delete them to pick up the shipped version."
    ),
    CUSTOMIZED: (
        "your own edits — the shipped template stays hidden. Re-apply your "
        "changes on top of the new .txt.example, or delete the copy to take "
        "the shipped version as it is."
    ),
    STALE_SHIPPED_OVERRIDE: (
        "model-family overrides left behind by an older release — they hide "
        "the current base template for that model family. Delete them unless "
        "you want that older wording."
    ),
}

#: Short wording for :func:`describe_source`, used in ``debug_info``.
_SOURCE_WORDING: Dict[str, str] = {
    IDENTICAL: "local copy, identical to shipped",
    OUTDATED_COPY: "local copy of an older shipped version",
    CUSTOMIZED: "local copy, customized",
    STALE_SHIPPED_OVERRIDE: "local override from an older release",
    LOCAL_OVERRIDE: "local override",
    UNVERIFIED: "local copy, unverified",
    SHIPPED: "shipped",
}


@dataclass(frozen=True)
class TemplateStatus:
    """How the template ``basename`` resolves on this installation."""

    basename: str
    #: The file the loader actually reads, or ``None`` if the template is gone.
    path: Optional[Path]
    status: str

    @property
    def is_local(self) -> bool:
        """True if a local ``.txt`` wins over a shipped template."""
        return self.status not in (SHIPPED,) and self.path is not None

    @property
    def shadows_shipped_template(self) -> bool:
        """True if a committed ``.txt.example`` exists that stays unread."""
        return self.status in (IDENTICAL, OUTDATED_COPY, CUSTOMIZED, UNVERIFIED)


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    """Return ``text`` as the loader would use it: ``\\n`` endings, no trailer."""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def digest_text(text: str) -> str:
    """SHA-256 of the normalised ``text``, as 64 lowercase hex characters."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def digest_file(path: Path) -> Optional[str]:
    """Digest of the template at ``path``, or ``None`` if it cannot be read."""
    try:
        return digest_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot hash system-prompt file %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# The shipped digest history
# ---------------------------------------------------------------------------
_HEX_DIGITS = set("0123456789abcdef")


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


#: Top-level keys of ``shipped_hashes.json``.
TEMPLATES_KEY = "templates"
RETIRED_KEY = "retired_override_versions"


@dataclass(frozen=True)
class ShippedHashes:
    """The committed digest history, split into its two sections."""

    #: ``{basename: [digest, ...]}`` for the templates shipping today.
    templates: Dict[str, List[str]]
    #: ``{base template: [digest, ...]}`` for retired override versions.
    retired: Dict[str, List[str]]

    def __bool__(self) -> bool:
        """False when nothing could be loaded — the check is then skipped."""
        return bool(self.templates or self.retired)


#: What :func:`load_shipped_hashes` returns when the file is unusable.
EMPTY_HASHES = ShippedHashes(templates={}, retired={})


def _parse_section(raw: object, key: str) -> Dict[str, List[str]]:
    """Validate one ``{name: [digest, ...]}`` section. Raises ``ValueError``."""
    if not isinstance(raw, dict):
        raise ValueError(f"'{key}' must be an object, got {type(raw).__name__}")
    out: Dict[str, List[str]] = {}
    for name, digests in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"'{key}': {name!r} is not a usable template name")
        if not isinstance(digests, list) or not digests:
            raise ValueError(
                f"'{key}': {name!r} must map to a non-empty list of digests"
            )
        for entry in digests:
            if not _is_digest(entry):
                raise ValueError(
                    f"'{key}': {name!r}: {entry!r} is not a sha256 hex digest"
                )
        out[name] = list(digests)
    return out


def _parse_shipped_hashes(data: object) -> ShippedHashes:
    """Validate the two-section mapping. Raises ``ValueError``."""
    if not isinstance(data, dict):
        raise ValueError(f"top level must be an object, got {type(data).__name__}")
    if TEMPLATES_KEY not in data:
        raise ValueError(f"missing '{TEMPLATES_KEY}' section")
    templates = _parse_section(data[TEMPLATES_KEY], TEMPLATES_KEY)
    if not templates:
        raise ValueError(f"'{TEMPLATES_KEY}' section is empty")
    retired = _parse_section(data.get(RETIRED_KEY, {}), RETIRED_KEY)
    return ShippedHashes(templates=templates, retired=retired)


_hashes_cache: Optional[ShippedHashes] = None
_warned_about_hashes = False


def load_shipped_hashes(
    path: Optional[Path] = None, refresh: bool = False
) -> ShippedHashes:
    """Return the committed digest history, newest version first per template.

    A missing, unparseable or malformed file yields :data:`EMPTY_HASHES` and
    one warning — the check degrades to ``unverified`` instead of breaking the
    node load. Cached after the first call; ``refresh=True`` re-reads.
    """
    global _hashes_cache, _warned_about_hashes
    if path is None:
        path = SHIPPED_HASHES_FILE
        if _hashes_cache is not None and not refresh:
            return _hashes_cache
    if refresh:
        _warned_about_hashes = False

    result = EMPTY_HASHES
    try:
        with path.open("r", encoding="utf-8") as f:
            result = _parse_shipped_hashes(json.load(f))
    except FileNotFoundError:
        _warn_about_hashes(path, "is missing")
    except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
        _warn_about_hashes(path, f"is unusable ({exc})")

    if path == SHIPPED_HASHES_FILE:
        _hashes_cache = result
    return result


def _warn_about_hashes(path: Path, what: str) -> None:
    """Warn once per process that the digest history cannot be used."""
    global _warned_about_hashes
    if _warned_about_hashes:
        return
    _warned_about_hashes = True
    logger.warning(
        "%s %s — skipping the check for local system-prompt copies that hide "
        "a shipped template. The nodes work normally. Regenerate the file "
        "with: python3 scripts/update_shipped_hashes.py",
        path,
        what,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(
    basename: str,
    prompts_dir: Optional[Path] = None,
    shipped: Optional[ShippedHashes] = None,
) -> TemplateStatus:
    """Classify how template ``basename`` resolves. See the module docstring."""
    directory = PROMPTS_DIR if prompts_dir is None else prompts_dir
    if shipped is None:
        shipped = load_shipped_hashes()

    user_path = directory / f"{basename}{USER_SUFFIX}"
    example_path = directory / f"{basename}{EXAMPLE_SUFFIX}"

    if not user_path.is_file():
        return TemplateStatus(
            basename, example_path if example_path.is_file() else None, SHIPPED
        )

    if not shipped:
        return TemplateStatus(basename, user_path, UNVERIFIED)

    actual = digest_file(user_path)
    if actual is None:
        return TemplateStatus(basename, user_path, UNVERIFIED)

    if example_path.is_file():
        history = shipped.templates.get(basename)
        if not history:
            # The template ships but is absent from the digest history — the
            # file is stale (the gate test in tests/unit keeps that from
            # reaching a release). Claiming "customized" would be a guess.
            return TemplateStatus(basename, user_path, UNVERIFIED)
        if actual == history[0]:
            return TemplateStatus(basename, user_path, IDENTICAL)
        if actual in history[1:]:
            return TemplateStatus(basename, user_path, OUTDATED_COPY)
        return TemplateStatus(basename, user_path, CUSTOMIZED)

    # No committed template under this name — a model-family override, or a
    # name the package retired. Matching a version it once shipped means the
    # file was left behind by an update, not authored by the user.
    retired = shipped.retired.get(_base_template(basename), ())
    if actual in retired:
        return TemplateStatus(basename, user_path, STALE_SHIPPED_OVERRIDE)
    return TemplateStatus(basename, user_path, LOCAL_OVERRIDE)


def _base_template(basename: str) -> str:
    """``vision_hair_change.somefamily`` -> ``vision_hair_change``."""
    return basename.split(".", 1)[0]


def _basenames(prompts_dir: Path) -> List[str]:
    """Every template name that has a local ``.txt`` in ``prompts_dir``."""
    names = {
        p.name[: -len(USER_SUFFIX)]
        for p in prompts_dir.glob(f"*{USER_SUFFIX}")
        if p.is_file()
    }
    return sorted(names)


def scan_local_copies(
    prompts_dir: Optional[Path] = None,
    shipped: Optional[ShippedHashes] = None,
) -> List[TemplateStatus]:
    """Classify every local ``.txt`` in the template directory, sorted by name."""
    directory = PROMPTS_DIR if prompts_dir is None else prompts_dir
    if not directory.is_dir():
        return []
    if shipped is None:
        shipped = load_shipped_hashes()
    return [
        classify(name, directory, shipped) for name in _basenames(directory)
    ]


# ---------------------------------------------------------------------------
# The {imgN} hint for image-edit modes
# ---------------------------------------------------------------------------
def _edit_mode_templates() -> Set[str]:
    """Template basenames backing a vision *edit* mode, or an empty set.

    A broken catalog is not this module's problem to report — the loaders
    fail loudly on their own — so it only costs us the extra hint.
    """
    try:
        return {m.template for m in load_catalog().vision_modes if m.is_edit}
    except CatalogError as exc:  # pragma: no cover — defensive
        logger.debug("Catalog unavailable, skipping {imgN} hints: %s", exc)
        return set()


def _is_edit_mode_template(basename: str, edit_templates: Set[str]) -> bool:
    """True for an edit-mode template or a model-family override of one."""
    return basename in edit_templates or _base_template(basename) in edit_templates


def _lacks_image_placeholders(path: Path) -> bool:
    """True if the file carries no ``{imgN}`` placeholder at all."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # pragma: no cover — defensive
        return False
    return "{img" not in text


# ---------------------------------------------------------------------------
# The startup report
# ---------------------------------------------------------------------------
def _plural(count: int, singular: str, plural: str) -> str:
    """Pick the word form matching ``count``."""
    return singular if count == 1 else plural


def build_report(statuses: List[TemplateStatus]) -> Optional[str]:
    """Render one log block for ``statuses``, or ``None`` if nothing to say."""
    by_status: Dict[str, List[TemplateStatus]] = {}
    for entry in statuses:
        by_status.setdefault(entry.status, []).append(entry)

    listed = [s for s in REPORTED_STATUSES if by_status.get(s)]
    unverified = by_status.get(UNVERIFIED, [])
    if not listed and not unverified:
        return None

    edit_templates = _edit_mode_templates() if by_status.get(CUSTOMIZED) else set()

    lines = [
        "Local system-prompt copies take precedence over the shipped "
        "templates, so these files hide the versions that came with the "
        "package:",
    ]
    for status in listed:
        entries = by_status[status]
        lines.append("")
        lines.append(f"  {status} ({len(entries)}) — {_ADVICE[status]}")
        for entry in entries:
            note = ""
            if (
                status == CUSTOMIZED
                and entry.path is not None
                and _is_edit_mode_template(entry.basename, edit_templates)
                and _lacks_image_placeholders(entry.path)
            ):
                note = (
                    "  [no {imgN} placeholders — the target_model setting "
                    "has no effect on this mode]"
                )
            lines.append(f"    {entry.path}{note}")

    if unverified:
        lines.append("")
        lines.append(
            f"  unverified ({len(unverified)}) — could not be compared against "
            "the shipped versions:"
        )
        for entry in unverified:
            lines.append(f"    {entry.path}")

    quiet = []
    if by_status.get(IDENTICAL):
        count = len(by_status[IDENTICAL])
        quiet.append(
            f"{count} {_plural(count, 'copy', 'copies')} "
            f"{_plural(count, 'is', 'are')} identical to the current shipped "
            "template (nothing to do)"
        )
    if by_status.get(LOCAL_OVERRIDE):
        count = len(by_status[LOCAL_OVERRIDE])
        quiet.append(
            f"{count} local {_plural(count, 'override', 'overrides')} "
            f"{_plural(count, 'has', 'have')} no shipped counterpart "
            "(left alone)"
        )
    if quiet:
        lines.append("")
        lines.append("  Also present: " + "; ".join(quiet) + ".")

    lines.append("")
    lines.append(
        "  No file was changed or deleted. See docs/system-prompt-overrides.md "
        "for how to clean up."
    )
    return "\n".join(lines)


_reported = False


def report_local_copies(
    prompts_dir: Optional[Path] = None, force: bool = False
) -> Optional[str]:
    """Log one summary block about local template copies, once per process.

    Returns the block that was logged, or ``None`` when there was nothing to
    report. Never raises: a problem here must not stop the nodes from loading.
    Pass ``force=True`` to run again (tests).
    """
    global _reported
    if _reported and not force:
        return None
    _reported = True

    try:
        statuses = scan_local_copies(prompts_dir)
        report = build_report(statuses)
    except Exception as exc:  # noqa: BLE001 — never block the node load
        logger.warning("Could not check for local system-prompt copies: %s", exc)
        return None

    if report is None:
        logger.debug("No local system-prompt copies hide a shipped template.")
        return None
    logger.warning("%s", report)
    return report


# ---------------------------------------------------------------------------
# Per-run source line for debug_info
# ---------------------------------------------------------------------------
def describe_source(basename: str, prompts_dir: Optional[Path] = None) -> str:
    """Return ``"<filename> (<wording>)"`` for the template actually in use.

    Example: ``vision_outfit_transfer.txt (local copy, customized)`` or
    ``vision_outfit_transfer.txt.example (shipped)``. Returns the expected
    filename with ``(not found)`` when neither file exists, so ``debug_info``
    stays printable even then.
    """
    try:
        entry = classify(basename, prompts_dir)
    except Exception as exc:  # noqa: BLE001 — debug_info must never raise
        logger.debug("Could not describe template %r: %s", basename, exc)
        return f"{basename}{USER_SUFFIX} (unknown)"
    if entry.path is None:
        return f"{basename}{USER_SUFFIX} (not found)"
    return f"{entry.path.name} ({_SOURCE_WORDING[entry.status]})"


def describe_resolved_source(
    basename: str,
    model_name: Optional[str] = None,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Like :func:`describe_source` but following the model-family cascade.

    The nodes render a template through :func:`prompts.render_template`, which
    prefers ``<basename>.<family>`` when such a file exists. This reports the
    file that cascade actually picks, so ``debug_info`` never names a template
    the run did not use.
    """
    from .prompts import detect_family  # local: prompts imports nothing here

    try:
        family = detect_family(model_name)
    except Exception as exc:  # noqa: BLE001 — debug_info must never raise
        logger.debug("Family detection failed for %r: %s", model_name, exc)
        family = None
    if family is not None:
        directory = PROMPTS_DIR if prompts_dir is None else prompts_dir
        override = f"{basename}.{family}"
        if (
            (directory / f"{override}{USER_SUFFIX}").is_file()
            or (directory / f"{override}{EXAMPLE_SUFFIX}").is_file()
        ):
            return describe_source(override, prompts_dir)
    return describe_source(basename, prompts_dir)
