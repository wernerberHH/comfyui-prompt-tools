#!/usr/bin/env python3
"""Regenerate ``system_prompts/shipped_hashes.json`` from the git history.

The file lists, per template, the digest of every version this package has
ever shipped — newest first. The startup check in
:mod:`comfyui_prompt_tools.template_status` uses it to tell an untouched copy
of an older template (safe to delete) from a real local edit (keep, but the
shipped improvements have to be re-applied).

Two sections::

    {
      "templates": {"<basename>": ["<newest digest>", ...]},
      "retired_override_versions": {"<base template>": ["<digest>", ...]}
    }

``templates`` covers the templates that ship today, keyed by the basename the
loader resolves (``<basename>.txt`` / ``<basename>.txt.example``), current
version first.

``retired_override_versions`` covers versions that were shipped under a name
the package no longer uses — in practice model-family overrides
(``<base>.<family>.txt``) that were later externalised. They are keyed by the
**base template name only**: the family-name part is dropped on purpose, so
the model tags this project keeps out of the public repo stay out of it. That
is enough for the check, which only has to answer "did this exact content
ever ship?" for a file whose name it already knows.

Usage::

    python3 scripts/update_shipped_hashes.py           # rewrite the file
    python3 scripts/update_shipped_hashes.py --check    # only report drift
    python3 scripts/update_shipped_hashes.py --stdout   # print, write nothing

Run it after changing, adding or removing a ``system_prompts/*.txt.example``
and commit the result together with the template. ``tests/unit/
test_shipped_hashes.py`` fails until you do.

What counts as "shipped"
------------------------
Every blob that ever lived at ``system_prompts/<name>.txt.example`` *or*
``system_prompts/<name>.txt`` on any branch. The second name matters: the
templates were tracked as plain ``.txt`` before v1.1 moved them to
``.txt.example``, and some model-family overrides (``<name>.<family>.txt``)
were shipped and later externalised. Those versions are exactly the ones
still sitting in user installations, so they belong in the history.

The current working-tree content of each ``.txt.example`` is always the first
entry of its ``templates`` list, so editing a template and running this script
leaves a consistent file even before the commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "comfyui_prompt_tools" / "system_prompts"
#: Path relative to the repo root — what git wants to hear.
PROMPTS_PREFIX = "comfyui_prompt_tools/system_prompts"
OUTPUT_FILE = PROMPTS_DIR / "shipped_hashes.json"

EXAMPLE_SUFFIX = ".txt.example"
USER_SUFFIX = ".txt"

#: Top-level keys of the generated file.
TEMPLATES_KEY = "templates"
RETIRED_KEY = "retired_override_versions"


class ScriptError(RuntimeError):
    """Something the user has to fix before the script can do its job."""


# ---------------------------------------------------------------------------
# Digest — must match comfyui_prompt_tools.template_status.digest_text
# ---------------------------------------------------------------------------
def digest_bytes(raw: bytes) -> Optional[str]:
    """SHA-256 over the normalised text, or ``None`` if it is not UTF-8."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    """Run a read-only git command in the repo and return its stdout."""
    try:
        completed = subprocess.run(
            ("git", "-C", str(REPO_ROOT), *args),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ScriptError("git is not installed — cannot read the history") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise ScriptError(f"git {' '.join(args)} failed: {detail}") from exc
    return completed.stdout.decode("utf-8", "replace")


def _template_basename(path: str) -> Optional[str]:
    """``…/system_prompts/x.txt.example`` -> ``x``; ``None`` for other files."""
    if not path.startswith(f"{PROMPTS_PREFIX}/"):
        return None
    name = path[len(PROMPTS_PREFIX) + 1 :]
    if "/" in name:
        return None
    if name.endswith(EXAMPLE_SUFFIX):
        return name[: -len(EXAMPLE_SUFFIX)]
    if name.endswith(USER_SUFFIX):
        return name[: -len(USER_SUFFIX)]
    return None


def _historical_blobs() -> List[Tuple[str, str]]:
    """Return ``(basename, blob_sha)`` pairs, newest commit first.

    Walks every commit on every ref that touched the template directory and
    lists the templates present in it. Duplicates are expected and filtered
    by the caller; the order is what makes "newest first" work.
    """
    revs = [
        line.strip()
        for line in _git("rev-list", "--all", "--", PROMPTS_PREFIX).splitlines()
        if line.strip()
    ]
    pairs: List[Tuple[str, str]] = []
    for rev in revs:
        listing = _git("ls-tree", "-r", "-z", rev, "--", f"{PROMPTS_PREFIX}/")
        for record in listing.split("\0"):
            if not record:
                continue
            meta, _, path = record.partition("\t")
            fields = meta.split()
            if len(fields) < 3 or fields[1] != "blob":
                continue
            basename = _template_basename(path)
            if basename is not None:
                pairs.append((basename, fields[2]))
    return pairs


def _blob_digests(shas: Iterable[str]) -> Dict[str, str]:
    """Map each blob sha to the digest of its normalised content."""
    wanted = sorted(set(shas))
    if not wanted:
        return {}
    try:
        completed = subprocess.run(
            ("git", "-C", str(REPO_ROOT), "cat-file", "--batch"),
            input=("\n".join(wanted) + "\n").encode("utf-8"),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover — defensive
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise ScriptError(f"git cat-file --batch failed: {detail}") from exc

    out: Dict[str, str] = {}
    stream = completed.stdout
    pos = 0
    while pos < len(stream):
        newline = stream.find(b"\n", pos)
        if newline == -1:  # pragma: no cover — defensive
            break
        header = stream[pos:newline].decode("utf-8", "replace").split()
        pos = newline + 1
        if len(header) != 3:  # pragma: no cover — missing object
            continue
        sha, _kind, size_text = header
        size = int(size_text)
        digest = digest_bytes(stream[pos : pos + size])
        pos += size + 1  # trailing newline after the payload
        if digest is not None:
            out[sha] = digest
    return out


# ---------------------------------------------------------------------------
# Building the mapping
# ---------------------------------------------------------------------------
def _working_tree_digests() -> Dict[str, str]:
    """Digest of every ``*.txt.example`` currently on disk, keyed by basename."""
    if not PROMPTS_DIR.is_dir():
        raise ScriptError(f"template directory not found: {PROMPTS_DIR}")
    out: Dict[str, str] = {}
    for path in sorted(PROMPTS_DIR.glob(f"*{EXAMPLE_SUFFIX}")):
        digest = digest_bytes(path.read_bytes())
        if digest is None:
            raise ScriptError(f"{path} is not valid UTF-8")
        out[path.name[: -len(EXAMPLE_SUFFIX)]] = digest
    return out


def _base_template(basename: str) -> str:
    """``flux_kontext_couple_scene.somefamily`` -> ``flux_kontext_couple_scene``.

    Dropping the family part is what keeps model tags out of the public repo.
    """
    return basename.split(".", 1)[0]


def build_payload() -> Dict[str, Dict[str, List[str]]]:
    """Return the two-section mapping written to ``shipped_hashes.json``.

    ``templates`` holds the current version of every shipped template first,
    followed by its older versions, newest first. Versions that only ever
    shipped under a name the package no longer uses go to
    ``retired_override_versions``, keyed by their base template name.
    """
    current = _working_tree_digests()
    pairs = _historical_blobs()
    digests = _blob_digests(sha for _, sha in pairs)

    templates: Dict[str, List[str]] = {
        basename: [digest] for basename, digest in current.items()
    }
    retired: Dict[str, List[str]] = {}
    for basename, sha in pairs:
        digest = digests.get(sha)
        if digest is None:
            continue
        if basename in current:
            history = templates[basename]
        else:
            history = retired.setdefault(_base_template(basename), [])
        if digest not in history:
            history.append(digest)

    return {
        TEMPLATES_KEY: {name: templates[name] for name in sorted(templates)},
        RETIRED_KEY: {name: retired[name] for name in sorted(retired)},
    }


def render(payload: Dict[str, Dict[str, List[str]]]) -> str:
    """Serialise the payload the way the committed file is formatted."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed file is out of date; write nothing",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the generated JSON instead of writing the file",
    )
    args = parser.parse_args(argv)

    try:
        payload = render(build_payload())
    except ScriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.stdout:
        sys.stdout.write(payload)
        return 0

    existing = (
        OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.is_file() else None
    )
    if args.check:
        if existing == payload:
            print(f"{OUTPUT_FILE.name} is up to date.")
            return 0
        print(
            f"{OUTPUT_FILE} is out of date — run: "
            "python3 scripts/update_shipped_hashes.py",
            file=sys.stderr,
        )
        return 1

    if existing == payload:
        print(f"{OUTPUT_FILE.name} unchanged.")
        return 0
    OUTPUT_FILE.write_text(payload, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
