"""Unit tests for the .txt vs .txt.example fallback in prompts.py.

These cover the new behaviour introduced for v1.1 where committed templates
ship as ``<name>.txt.example`` and user-edited copies live as ``<name>.txt``
(gitignored).

Mocks
-----
We rebind ``comfyui_prompt_tools.prompts._PROMPTS_DIR`` to a ``tmp_path``
populated with the specific files we want to test, mirroring the existing
pattern in ``test_prompts_loader.py``.
"""

from pathlib import Path

import pytest

from comfyui_prompt_tools import prompts as prompts_mod
from comfyui_prompt_tools.prompts import (
    _read_file,
    _resolve_prompt_path,
    render_template,
)


@pytest.fixture
def isolated_prompts_dir(tmp_path, monkeypatch):
    """Rebind ``_PROMPTS_DIR`` to a clean temp directory."""
    monkeypatch.setattr(prompts_mod, "_PROMPTS_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_prompt_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_returns_user_txt_when_only_user_present(isolated_prompts_dir):
    """Only ``<name>.txt`` exists → return it."""
    (isolated_prompts_dir / "mode_a.txt").write_text("user")
    result = _resolve_prompt_path("mode_a")
    assert result == isolated_prompts_dir / "mode_a.txt"


@pytest.mark.unit
def test_resolve_returns_example_when_only_example_present(isolated_prompts_dir):
    """Only ``<name>.txt.example`` exists → return the example."""
    (isolated_prompts_dir / "mode_b.txt.example").write_text("template")
    result = _resolve_prompt_path("mode_b")
    assert result == isolated_prompts_dir / "mode_b.txt.example"


@pytest.mark.unit
def test_resolve_prefers_user_over_example(isolated_prompts_dir):
    """Both files present → user copy wins."""
    (isolated_prompts_dir / "mode_c.txt").write_text("user")
    (isolated_prompts_dir / "mode_c.txt.example").write_text("template")
    result = _resolve_prompt_path("mode_c")
    assert result == isolated_prompts_dir / "mode_c.txt"


@pytest.mark.unit
def test_resolve_returns_none_when_neither_present(isolated_prompts_dir):
    """No matching files → ``None``."""
    assert _resolve_prompt_path("nonexistent") is None


# ---------------------------------------------------------------------------
# _read_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_file_returns_user_content_when_user_exists(isolated_prompts_dir):
    """User copy contents are returned when present."""
    (isolated_prompts_dir / "mode_d.txt").write_text("user-content\n")
    (isolated_prompts_dir / "mode_d.txt.example").write_text("template-content\n")
    assert _read_file("mode_d") == "user-content"


@pytest.mark.unit
def test_read_file_falls_back_to_example(isolated_prompts_dir):
    """Without a user copy the example content is returned."""
    (isolated_prompts_dir / "mode_e.txt.example").write_text("template-content\n")
    assert _read_file("mode_e") == "template-content"


@pytest.mark.unit
def test_read_file_raises_when_neither_present(isolated_prompts_dir):
    """Missing user copy AND example → FileNotFoundError with helpful message."""
    with pytest.raises(FileNotFoundError) as exc:
        _read_file("missing")
    # Error message should mention both the .txt path and the .example fallback
    assert "missing.txt" in str(exc.value)
    assert ".example" in str(exc.value)


# ---------------------------------------------------------------------------
# Integration: render_template + shared_rules fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_template_uses_example_fallback(isolated_prompts_dir):
    """``render_template`` reads the .example file when no user copy exists."""
    (isolated_prompts_dir / "x.txt.example").write_text("hello\n")
    assert render_template("x") == "hello"


@pytest.mark.unit
def test_render_template_resolves_shared_rules_via_example(isolated_prompts_dir):
    """The ``{shared_rules}`` placeholder is filled from the .example fallback."""
    (isolated_prompts_dir / "y.txt.example").write_text("A {shared_rules} B\n")
    (isolated_prompts_dir / "_shared_rules.txt.example").write_text("RULES\n")
    assert render_template("y") == "A RULES B"


@pytest.mark.unit
def test_render_template_user_overrides_example(isolated_prompts_dir):
    """A user-edited .txt copy wins over a shipped .example template."""
    (isolated_prompts_dir / "z.txt.example").write_text("template\n")
    (isolated_prompts_dir / "z.txt").write_text("user-edited\n")
    assert render_template("z") == "user-edited"


# ---------------------------------------------------------------------------
# Per-model override cascade respects the example fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_override_falls_back_to_example(isolated_prompts_dir):
    """A model-family override only available as .example is still picked up."""
    (isolated_prompts_dir / "base.txt.example").write_text("default\n")
    (isolated_prompts_dir / "base.cydonia.txt.example").write_text("cydonia\n")
    # detect_family("Fermi/Cydonia-...") -> "cydonia"
    result = render_template("base", model_name="Fermi/Cydonia-24B")
    assert result == "cydonia"


@pytest.mark.unit
def test_override_user_copy_wins_over_example_default(isolated_prompts_dir):
    """User .txt override beats shipped .example default for the same family."""
    (isolated_prompts_dir / "base.txt.example").write_text("default\n")
    (isolated_prompts_dir / "base.cydonia.txt.example").write_text("template\n")
    (isolated_prompts_dir / "base.cydonia.txt").write_text("user-tuned\n")
    result = render_template("base", model_name="Fermi/Cydonia-24B")
    assert result == "user-tuned"


@pytest.mark.unit
def test_override_missing_falls_through_to_default(isolated_prompts_dir):
    """When no override file (neither .txt nor .example) exists, the default
    template is rendered."""
    (isolated_prompts_dir / "base.txt.example").write_text("default-template\n")
    # No base.cydonia.* file at all
    result = render_template("base", model_name="Fermi/Cydonia-24B")
    assert result == "default-template"
