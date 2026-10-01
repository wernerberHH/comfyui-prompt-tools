"""Gate: the committed ``shipped_hashes.json`` matches the shipped templates.

Changing a ``system_prompts/*.txt.example`` without regenerating the digest
history would silently break the startup check: the edited template's new
content is absent from the history, so an untouched local copy of the *old*
version can no longer be recognised as an ``outdated-copy``. These tests fail
until ``scripts/update_shipped_hashes.py`` has been run.

Unlike ``test_template_status.py`` these tests read the real repository — that
is the point.
"""

from pathlib import Path

import pytest

from comfyui_prompt_tools.template_status import (
    EXAMPLE_SUFFIX,
    RETIRED_KEY,
    SHIPPED_HASHES_FILE,
    TEMPLATES_KEY,
    digest_text,
    load_shipped_hashes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "comfyui_prompt_tools" / "system_prompts"

REGENERATE = (
    "Run: python3 scripts/update_shipped_hashes.py — and commit the result "
    "together with the template change."
)


def _shipped_templates() -> dict[str, Path]:
    """Every committed template, keyed by the basename the loader resolves."""
    return {
        path.name[: -len(EXAMPLE_SUFFIX)]: path
        for path in sorted(PROMPTS_DIR.glob(f"*{EXAMPLE_SUFFIX}"))
    }


@pytest.fixture(scope="module")
def shipped():
    """The committed digest history, read from the real file."""
    return load_shipped_hashes(SHIPPED_HASHES_FILE, refresh=True)


@pytest.mark.unit
def test_hash_file_is_committed_and_loadable(shipped):
    """A missing or malformed file degrades at runtime but fails the suite."""
    assert SHIPPED_HASHES_FILE.is_file(), (
        f"{SHIPPED_HASHES_FILE} is missing. {REGENERATE}"
    )
    assert shipped, f"{SHIPPED_HASHES_FILE} could not be parsed. {REGENERATE}"


@pytest.mark.unit
def test_there_are_templates_to_check():
    """Guards the parametrised test below against silently collecting nothing."""
    assert _shipped_templates(), f"no {EXAMPLE_SUFFIX} templates in {PROMPTS_DIR}"


@pytest.mark.unit
@pytest.mark.parametrize("basename", sorted(_shipped_templates()))
def test_current_template_version_is_the_first_entry(basename, shipped):
    """Each shipped template's current content heads its digest list."""
    path = _shipped_templates()[basename]
    history = shipped.templates.get(basename)
    assert history, (
        f"{path.name} has no entry in {SHIPPED_HASHES_FILE.name}. {REGENERATE}"
    )
    expected = digest_text(path.read_text(encoding="utf-8"))
    assert history[0] == expected, (
        f"{path.name} has changed since {SHIPPED_HASHES_FILE.name} was "
        f"generated. {REGENERATE}"
    )


@pytest.mark.unit
def test_hash_file_lists_no_template_that_is_not_shipped(shipped):
    """A removed template belongs in the retired section, not in ``templates``."""
    unknown = sorted(set(shipped.templates) - set(_shipped_templates()))
    assert not unknown, (
        f"{TEMPLATES_KEY} names templates that no longer ship: "
        f"{', '.join(unknown)}. {REGENERATE}"
    )


@pytest.mark.unit
def test_every_digest_list_is_free_of_duplicates(shipped):
    """Identical versions collapse to one entry, so the lists stay readable."""
    for section in (shipped.templates, shipped.retired):
        for name, digests in section.items():
            assert len(digests) == len(set(digests)), (
                f"{name} lists the same digest twice. {REGENERATE}"
            )


@pytest.mark.unit
def test_retired_section_carries_no_model_family_names(shipped):
    """Retired override versions are keyed by base template name only.

    The family part of a retired ``<base>.<family>.txt`` is dropped on
    purpose: this is a public repository and the model tags this project
    routes on are kept out of it (see ``config/model_families.yaml``, which
    is gitignored). A dot in a key here would put one back.
    """
    leaking = sorted(name for name in shipped.retired if "." in name)
    assert not leaking, (
        f"{RETIRED_KEY} keys must not contain a model-family suffix: "
        f"{', '.join(leaking)}"
    )


@pytest.mark.unit
def test_retired_keys_refer_to_a_known_template(shipped):
    """A retired override is only useful next to the template it overrode."""
    known = set(_shipped_templates())
    orphans = sorted(set(shipped.retired) - known)
    assert not orphans, (
        f"{RETIRED_KEY} names base templates that no longer ship: "
        f"{', '.join(orphans)}"
    )
