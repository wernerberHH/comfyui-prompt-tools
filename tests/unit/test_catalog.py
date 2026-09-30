"""Tests for the prompt catalog — the data source behind every dropdown.

Covered here:

- the shipped ``config/catalog.yaml.example`` loads and is self-consistent
- the approved label table, including every former label kept as an alias
- one test per validation failure mode (the loader fails fast, not silently)
- the local ``config/catalog.yaml`` overlay (replace / append / disable)
- ``VALIDATE_INPUTS`` on all three nodes: old labels pass, unknown rejected
- ``{imgN}`` rendering per target model that takes reference images
- describe-mode templates carry no image reference at all

The validation tests drive ``catalog._parse`` with synthetic dicts so a
malformed catalog never has to be written to the repo's config directory.
Templates referenced by those fixtures are real basenames from
``system_prompts/`` because existence on disk is part of what is validated.

Mocks: ``catalog._EXAMPLE_FILE`` / ``catalog._USER_FILE`` are redirected at
``tmp_path`` for the overlay tests; the module-level cache is restored
afterwards so later tests still see the shipped catalog.
"""

from __future__ import annotations

import re

import pytest

from comfyui_prompt_tools import catalog as catalog_mod
from comfyui_prompt_tools.catalog import (
    CatalogError,
    load_catalog,
    render_image_refs,
    resolve_composer_style,
    resolve_helper_mode,
    resolve_image_target_model,
    resolve_target_model,
    resolve_vision_mode,
)
from comfyui_prompt_tools.nodes.prompt_composer import PromptComposer
from comfyui_prompt_tools.nodes.prompt_helper import PromptHelper
from comfyui_prompt_tools.nodes.vision_prompt_helper import VisionPromptHelper
from comfyui_prompt_tools.vision_prompts import get_vision_system_prompt

pytestmark = pytest.mark.unit

#: Any wording a downstream image model would read as an image reference.
IMAGE_REF_RE = re.compile(r"image \d|picture \d|<image\d>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _minimal() -> dict:
    """A valid two-section-per-list catalog using real template basenames."""
    return {
        "version": 1,
        "target_models": [
            {"id": "generic", "label": "Generic", "image_ref": "image {n}"},
            {"id": "qwen21", "label": "Qwen-Image 2.1", "image_ref": "<image{n}>"},
            {"id": "sdxl", "label": "SDXL"},
        ],
        "prompt_helper_modes": [
            {
                "id": "sdxl_photorealistic",
                "label": "SDXL – Photorealistic",
                "aliases": ["SDXL Photorealistic"],
                "template": "sdxl_photorealistic",
                "target_model": "sdxl",
            },
        ],
        "composer_styles": [
            {
                "id": "composer_sdxl",
                "label": "SDXL – Tags",
                "template": "composer_sdxl",
                "target_model": "sdxl",
            },
        ],
        "vision_modes": [
            {
                "id": "vision_describe_face",
                "label": "Describe Face",
                "template": "vision_describe_face",
                "kind": "describe",
                "images": 1,
            },
        ],
    }


def _parse(data, user_data=None):
    return catalog_mod._parse(data, user_data, catalog_mod._EXAMPLE_FILE)


@pytest.fixture
def isolated_catalog(tmp_path, monkeypatch):
    """Redirect the catalog files at ``tmp_path`` and restore the cache after.

    Yields a ``write(example_yaml, user_yaml=None)`` helper that writes the
    files and returns the freshly loaded catalog.
    """
    example = tmp_path / "catalog.yaml.example"
    user = tmp_path / "catalog.yaml"
    monkeypatch.setattr(catalog_mod, "_EXAMPLE_FILE", example)
    monkeypatch.setattr(catalog_mod, "_USER_FILE", user)

    def write(example_yaml: str, user_yaml: str | None = None):
        example.write_text(example_yaml, encoding="utf-8")
        if user_yaml is not None:
            user.write_text(user_yaml, encoding="utf-8")
        return load_catalog(refresh=True)

    try:
        yield write
    finally:
        monkeypatch.undo()
        load_catalog(refresh=True)


SHIPPED_SECTIONS = ("target_models", "prompt_helper_modes", "composer_styles", "vision_modes")


# ---------------------------------------------------------------------------
# The shipped catalog
# ---------------------------------------------------------------------------
class TestShippedCatalog:
    def test_shipped_catalog_loads(self):
        """``config/catalog.yaml.example`` parses and validates as shipped."""
        cat = load_catalog()
        assert cat.version == 1
        for section in SHIPPED_SECTIONS:
            assert getattr(cat, section), f"section {section} is empty"

    def test_every_template_resolves_on_disk(self):
        """Every registered template has a ``.txt`` or ``.txt.example`` file."""
        cat = load_catalog()
        for section in ("prompt_helper_modes", "composer_styles", "vision_modes"):
            for entry in getattr(cat, section):
                assert catalog_mod.resolve_template_path(entry.template) is not None

    def test_generic_is_the_default_and_keeps_todays_wording(self):
        """``Generic`` is the default target model and renders ``image 1``."""
        cat = load_catalog()
        assert cat.default_target_model.id == "generic"
        assert cat.default_target_model.image_ref == "image {n}"

    def test_only_image_capable_models_are_offered_to_the_vision_node(self):
        """The ``target_model`` dropdown lists exactly the models with an
        ``image_ref`` — a text-to-image model has nothing to address."""
        cat = load_catalog()
        offered = catalog_mod.image_target_model_labels()
        assert offered == [m.label for m in cat.target_models if m.image_ref]
        assert set(offered) < {m.label for m in cat.target_models}

    def test_every_mode_and_style_points_at_a_known_target_model(self):
        """A dangling ``target_model`` reference would silently mislabel a mode."""
        known = {m.id for m in load_catalog().target_models}
        for section in ("prompt_helper_modes", "composer_styles"):
            for entry in getattr(load_catalog(), section):
                assert entry.target_model in known

    def test_vision_modes_declare_kind_and_image_count(self):
        cat = load_catalog()
        for entry in cat.vision_modes:
            assert entry.kind in ("edit", "describe")
            assert entry.images in (1, 2)
        assert [e.label for e in cat.vision_modes if e.uses_two_images] == [
            "Outfit Transfer"
        ]


# ---------------------------------------------------------------------------
# The approved label table
# ---------------------------------------------------------------------------
#: (new label, former label) — frozen: both must keep resolving to one entry.
HELPER_LABEL_TABLE = [
    ("FLUX Kontext – Scene Edit", "FLUX Kontext (Scene Edit)"),
    ("FLUX Kontext – Couple Scene", "FLUX Kontext (Couple Scene)"),
    ("Qwen-Image-Edit – Couple Scene", "Qwen Image Edit (Couple Scene)"),
    ("FLUX – Text-to-Image", "FLUX Text-to-Image"),
    ("Z-Image – Text-to-Image", "Z-Image Text-to-Image"),
    ("SDXL – Photorealistic", "SDXL Photorealistic"),
    ("SDXL Pony – Illustrious", "SDXL Pony/Illustrious"),
    ("Z-Image – Random Character", "Random Character (Z-Image)"),
    ("SDXL Pony – Random Character", "Random Character (Pony)"),
    ("LTX-2.3 – Video with Audio", "LTX-2.3 Video (Audio-Video)"),
    ("Krea 2 – Text-to-Image", "Krea 2 Text-to-Image"),
    ("LTX-2.5 – Multi-Shot Video with Audio", "LTX-2.5 Video (Multi-Shot Audio-Video)"),
]

COMPOSER_LABEL_TABLE = [
    ("FLUX.2 – Natural Language", "FLUX.2 natural language"),
    ("SDXL – Tags", "SDXL tag-based"),
    ("Z-Image – Compact", "Z-Image compact"),
    ("Wan 2.2 – Motion", "Wan 2.2 motion"),
    ("LTX-2.3 – Video with Audio", "LTX-2.3 audio-video"),
    ("SDXL Pony – Photoreal", "Pony photoreal"),
    ("SDXL Pony – Anime/Illustrious", "Pony anime/illustrious"),
    ("Krea 2 – Natural Language", "Krea 2 natural language"),
    ("LTX-2.5 – Multi-Shot Video with Audio", "LTX-2.5 multi-shot"),
]


class TestApprovedLabels:
    @pytest.mark.parametrize("new,old", HELPER_LABEL_TABLE)
    def test_helper_label_and_alias_resolve_to_one_entry(self, new, old):
        assert resolve_helper_mode(new) is resolve_helper_mode(old)

    @pytest.mark.parametrize("new,old", COMPOSER_LABEL_TABLE)
    def test_composer_label_and_alias_resolve_to_one_entry(self, new, old):
        assert resolve_composer_style(new) is resolve_composer_style(old)

    def test_custom_system_prompt_label_is_unchanged(self):
        """The catch-all mode was not renamed, so it needs no alias."""
        entry = resolve_helper_mode("Custom System Prompt")
        assert entry.id == "custom_system_prompt"
        assert entry.aliases == ()

    def test_vision_labels_are_unchanged(self):
        """Vision modes were not renamed — none of them needs an alias."""
        for entry in load_catalog().vision_modes:
            assert entry.aliases == ()

    def test_dropdown_order_is_catalog_order(self):
        """Saved workflows store the label, but users navigate by position."""
        assert catalog_mod.helper_mode_labels() == (
            [new for new, _ in HELPER_LABEL_TABLE] + ["Custom System Prompt"]
        )
        assert catalog_mod.composer_style_labels() == [
            new for new, _ in COMPOSER_LABEL_TABLE
        ]

    def test_unknown_label_is_rejected(self):
        for resolve in (resolve_helper_mode, resolve_composer_style, resolve_vision_mode):
            with pytest.raises(KeyError):
                resolve("No Such Thing")


# ---------------------------------------------------------------------------
# Validation — one test per failure mode
# ---------------------------------------------------------------------------
class TestValidation:
    def test_valid_minimal_catalog_parses(self):
        """Guards the fixture itself: the mutations below start from valid."""
        cat = _parse(_minimal())
        assert cat.version == 1

    def test_missing_version_is_rejected(self):
        data = _minimal()
        del data["version"]
        with pytest.raises(CatalogError, match="missing top-level 'version'"):
            _parse(data)

    def test_unsupported_version_is_rejected(self):
        data = _minimal()
        data["version"] = 99
        with pytest.raises(CatalogError, match="unsupported catalog version"):
            _parse(data)

    def test_missing_section_is_rejected(self):
        data = _minimal()
        del data["vision_modes"]
        with pytest.raises(CatalogError, match="missing section 'vision_modes'"):
            _parse(data)

    def test_section_that_is_not_a_list_is_rejected(self):
        data = _minimal()
        data["composer_styles"] = {"id": "x"}
        with pytest.raises(CatalogError, match="must be a list"):
            _parse(data)

    def test_entry_that_is_not_a_mapping_is_rejected(self):
        data = _minimal()
        data["target_models"].append("just a string")
        with pytest.raises(CatalogError, match=r"target_models\[3\] must be a mapping"):
            _parse(data)

    def test_entry_without_id_is_rejected(self):
        data = _minimal()
        data["target_models"].append({"label": "No Id"})
        with pytest.raises(CatalogError, match="has no usable 'id'"):
            _parse(data)

    def test_duplicate_id_in_a_section_is_rejected(self):
        data = _minimal()
        data["target_models"].append({"id": "sdxl", "label": "SDXL Again"})
        with pytest.raises(CatalogError, match="duplicate id"):
            _parse(data)

    def test_entry_without_label_is_rejected(self):
        data = _minimal()
        del data["prompt_helper_modes"][0]["label"]
        with pytest.raises(CatalogError, match="'label' must be a non-empty string"):
            _parse(data)

    def test_duplicate_label_in_a_list_is_rejected(self):
        data = _minimal()
        data["composer_styles"].append(
            {
                "id": "composer_flux2",
                "label": "SDXL – Tags",  # already taken
                "template": "composer_flux2",
                "target_model": "sdxl",
            }
        )
        with pytest.raises(CatalogError, match="is used by both"):
            _parse(data)

    def test_alias_colliding_with_another_label_is_rejected(self):
        """An alias is resolved like a label, so a collision is ambiguous."""
        data = _minimal()
        data["composer_styles"].append(
            {
                "id": "composer_flux2",
                "label": "FLUX.2 – Natural Language",
                "aliases": ["SDXL – Tags"],  # already a label in this list
                "template": "composer_flux2",
                "target_model": "sdxl",
            }
        )
        with pytest.raises(CatalogError, match="is used by both"):
            _parse(data)

    def test_non_string_alias_is_rejected(self):
        data = _minimal()
        data["prompt_helper_modes"][0]["aliases"] = [42]
        with pytest.raises(CatalogError, match="every alias must be a non-empty string"):
            _parse(data)

    def test_missing_template_field_is_rejected(self):
        data = _minimal()
        del data["vision_modes"][0]["template"]
        with pytest.raises(CatalogError, match="'template' must be a non-empty string"):
            _parse(data)

    def test_template_file_that_does_not_exist_is_rejected(self):
        data = _minimal()
        data["vision_modes"][0]["template"] = "no_such_template_anywhere"
        with pytest.raises(CatalogError, match="not found"):
            _parse(data)

    def test_missing_target_model_reference_is_rejected(self):
        data = _minimal()
        del data["composer_styles"][0]["target_model"]
        with pytest.raises(CatalogError, match="'target_model' must be a non-empty string"):
            _parse(data)

    def test_unknown_target_model_reference_is_rejected(self):
        data = _minimal()
        data["composer_styles"][0]["target_model"] = "nope"
        with pytest.raises(CatalogError, match="is not an id in 'target_models'"):
            _parse(data)

    def test_image_ref_without_the_number_placeholder_is_rejected(self):
        data = _minimal()
        data["target_models"][1]["image_ref"] = "the reference image"
        with pytest.raises(CatalogError, match=r"must contain '\{n\}'"):
            _parse(data)

    def test_unknown_vision_kind_is_rejected(self):
        data = _minimal()
        data["vision_modes"][0]["kind"] = "transform"
        with pytest.raises(CatalogError, match="must be one of edit, describe"):
            _parse(data)

    def test_missing_vision_kind_is_rejected(self):
        data = _minimal()
        del data["vision_modes"][0]["kind"]
        with pytest.raises(CatalogError, match="'kind' must be a non-empty string"):
            _parse(data)

    @pytest.mark.parametrize("images", [0, 3, None, "1"])
    def test_bad_vision_image_count_is_rejected(self, images):
        data = _minimal()
        data["vision_modes"][0]["images"] = images
        with pytest.raises(CatalogError, match="'images' must be 1 or 2"):
            _parse(data)

    def test_missing_generic_target_model_is_rejected(self):
        """Without ``generic`` there is no neutral default to render with."""
        data = _minimal()
        data["target_models"] = [m for m in data["target_models"] if m["id"] != "generic"]
        data["prompt_helper_modes"][0]["target_model"] = "sdxl"
        with pytest.raises(CatalogError, match="must contain an entry with id 'generic'"):
            _parse(data)

    def test_generic_without_image_ref_is_rejected(self):
        data = _minimal()
        data["target_models"][0].pop("image_ref")
        with pytest.raises(CatalogError, match="must define an 'image_ref'"):
            _parse(data)

    def test_missing_catalog_file_is_rejected(self, isolated_catalog, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog_mod, "_EXAMPLE_FILE", tmp_path / "gone.yaml")
        with pytest.raises(CatalogError, match="Prompt catalog not found"):
            load_catalog(refresh=True)

    def test_malformed_yaml_is_rejected(self, isolated_catalog):
        with pytest.raises(CatalogError, match="Failed to read"):
            isolated_catalog("version: 1\n  bad indentation: [\n")

    def test_top_level_that_is_not_a_mapping_is_rejected(self, isolated_catalog):
        with pytest.raises(CatalogError, match="top level must be a mapping"):
            isolated_catalog("- just\n- a\n- list\n")


# ---------------------------------------------------------------------------
# Local overlay
# ---------------------------------------------------------------------------
BASE_YAML = """
version: 1
target_models:
  - id: generic
    label: "Generic"
    image_ref: "image {n}"
  - id: sdxl
    label: "SDXL"
prompt_helper_modes:
  - id: sdxl_photorealistic
    label: "SDXL – Photorealistic"
    aliases: ["SDXL Photorealistic"]
    template: sdxl_photorealistic
    target_model: sdxl
composer_styles:
  - id: composer_sdxl
    label: "SDXL – Tags"
    template: composer_sdxl
    target_model: sdxl
  - id: composer_flux2
    label: "FLUX.2 – Natural Language"
    template: composer_flux2
    target_model: generic
vision_modes:
  - id: vision_describe_face
    label: "Describe Face"
    template: vision_describe_face
    kind: describe
    images: 1
"""


class TestLocalOverlay:
    def test_no_user_file_means_shipped_values(self, isolated_catalog):
        cat = isolated_catalog(BASE_YAML)
        assert [e.label for e in cat.composer_styles] == [
            "SDXL – Tags",
            "FLUX.2 – Natural Language",
        ]

    def test_same_id_replaces_only_the_listed_fields(self, isolated_catalog):
        """A user relabels one style; its template and aliases stay put."""
        cat = isolated_catalog(
            BASE_YAML,
            'version: 1\ncomposer_styles:\n  - id: composer_sdxl\n    label: "My SDXL"\n',
        )
        entry = cat.composer_style("My SDXL")
        assert entry.id == "composer_sdxl"
        assert entry.template == "composer_sdxl"
        assert entry.target_model == "sdxl"
        with pytest.raises(KeyError):
            cat.composer_style("SDXL – Tags")

    def test_new_id_is_appended_at_the_end_of_its_section(self, isolated_catalog):
        cat = isolated_catalog(
            BASE_YAML,
            "version: 1\n"
            "composer_styles:\n"
            '  - id: my_style\n'
            '    label: "My Style"\n'
            "    template: composer_zimage\n"
            "    target_model: generic\n",
        )
        assert [e.label for e in cat.composer_styles][-1] == "My Style"
        assert cat.composer_style("My Style").template == "composer_zimage"

    def test_enabled_false_hides_a_shipped_entry(self, isolated_catalog):
        cat = isolated_catalog(
            BASE_YAML,
            "version: 1\ncomposer_styles:\n  - id: composer_flux2\n    enabled: false\n",
        )
        assert [e.label for e in cat.composer_styles] == ["SDXL – Tags"]
        with pytest.raises(KeyError):
            cat.composer_style("FLUX.2 – Natural Language")

    def test_new_target_model_becomes_selectable(self, isolated_catalog):
        """The whole point of the overlay: a new model without a code change."""
        cat = isolated_catalog(
            BASE_YAML,
            "version: 1\n"
            "target_models:\n"
            "  - id: my_model\n"
            '    label: "My Model"\n'
            '    image_ref: "REF_{n}"\n',
        )
        model = cat.image_target_model("My Model")
        assert render_image_refs("{img1}/{img2}", model) == "REF_1/REF_2"

    def test_overlay_is_validated_too(self, isolated_catalog):
        with pytest.raises(CatalogError, match=r"must contain '\{n\}'"):
            isolated_catalog(
                BASE_YAML,
                "version: 1\n"
                "target_models:\n"
                "  - id: broken\n"
                '    label: "Broken"\n'
                '    image_ref: "no placeholder"\n',
            )

    def test_overlay_without_version_is_rejected(self, isolated_catalog):
        with pytest.raises(CatalogError, match="missing top-level 'version'"):
            isolated_catalog(BASE_YAML, "composer_styles: []\n")


# ---------------------------------------------------------------------------
# Image-reference rendering
# ---------------------------------------------------------------------------
class TestImageRefRendering:
    #: The wording each image-capable target model expects, per the briefing.
    EXPECTED = {
        "Generic": ("image 1", "image 2"),
        "FLUX.2": ("image 1", "image 2"),
        "FLUX Kontext": ("image 1", "image 2"),
        "Krea 2": ("Picture 1", "Picture 2"),
        "Qwen-Image-Edit 2511": ("Picture 1", "Picture 2"),
        "Qwen-Image 2.1": ("<image1>", "<image2>"),
    }

    def test_every_image_capable_model_is_covered_by_this_test(self):
        assert set(catalog_mod.image_target_model_labels()) == set(self.EXPECTED)

    @pytest.mark.parametrize("label", sorted(EXPECTED))
    def test_placeholders_render_to_the_model_wording(self, label):
        first, second = self.EXPECTED[label]
        model = resolve_image_target_model(label)
        assert render_image_refs("{img1} then {img2}", model) == f"{first} then {second}"

    def test_third_image_renders_too(self):
        """The Qwen couple-scene mode addresses a third reference image."""
        model = resolve_image_target_model("Qwen-Image 2.1")
        assert render_image_refs("{img3}", model) == "<image3>"

    def test_text_without_placeholders_is_untouched(self):
        model = resolve_target_model(None)
        assert render_image_refs("no placeholders here", model) == "no placeholders here"

    def test_none_target_model_is_the_generic_default(self):
        assert resolve_target_model(None).id == "generic"
        assert resolve_image_target_model(None).id == "generic"

    def test_text_only_model_is_not_accepted_as_an_image_target(self):
        """``SDXL`` has no ``image_ref``, so it cannot render a reference."""
        with pytest.raises(KeyError, match="takes no reference images"):
            resolve_image_target_model("SDXL")

    def test_render_without_image_ref_raises(self):
        model = resolve_target_model("SDXL")
        with pytest.raises(ValueError, match="has no image_ref"):
            render_image_refs("{img1}", model)

    @pytest.mark.parametrize("label", sorted(EXPECTED))
    def test_edit_mode_prompt_uses_the_selected_wording(self, label):
        """End to end: the rendered vision prompt carries the model's wording."""
        first, _ = self.EXPECTED[label]
        rendered = get_vision_system_prompt("Hair Change", target_model=label)
        assert first in rendered
        assert "{img1}" not in rendered

    def test_unknown_target_model_is_rejected(self):
        with pytest.raises(KeyError, match="Unknown target model"):
            get_vision_system_prompt("Hair Change", target_model="Nope 9000")


# ---------------------------------------------------------------------------
# Describe modes carry no image reference
# ---------------------------------------------------------------------------
class TestDescribeModesHaveNoImageReference:
    @pytest.mark.parametrize(
        "label",
        [e.label for e in load_catalog().vision_modes if e.kind == "describe"],
    )
    def test_rendered_describe_prompt_names_no_image(self, label):
        """A describe snippet is pasted into someone else's prompt, so an
        ``image 1`` in it would address the wrong picture downstream."""
        rendered = get_vision_system_prompt(label)
        found = IMAGE_REF_RE.findall(rendered)
        assert not found, f"{label} still references {found}"

    @pytest.mark.parametrize(
        "label",
        [e.label for e in load_catalog().vision_modes if e.kind == "describe"],
    )
    def test_describe_templates_hold_no_image_placeholder(self, label):
        """Not even a placeholder — there is nothing to render it for."""
        for target in catalog_mod.image_target_model_labels():
            rendered = get_vision_system_prompt(label, target_model=target)
            assert not IMAGE_REF_RE.findall(rendered)

    @pytest.mark.parametrize(
        "label",
        [e.label for e in load_catalog().vision_modes if e.kind == "edit"],
    )
    def test_edit_modes_do_reference_an_image(self, label):
        """Counterpart: edit modes must keep addressing their input."""
        rendered = get_vision_system_prompt(label)
        assert IMAGE_REF_RE.findall(rendered)


# ---------------------------------------------------------------------------
# VALIDATE_INPUTS on the nodes
# ---------------------------------------------------------------------------
class TestNodeValidateInputs:
    """ComfyUI skips its own combo-list check for an input that appears as a
    named parameter of ``VALIDATE_INPUTS`` (verified against the
    ``execution.py`` of the ComfyUI build this repo deploys to). That is what
    lets a workflow saved with a pre-catalog label still queue.
    """

    def test_validate_signatures_name_the_aliased_inputs(self):
        """Without the parameter name, ComfyUI rejects the value before the
        node ever runs — and no ``**kwargs``, which would disable the
        built-in min/max checks on every other input too."""
        import inspect

        for node, expected in (
            (PromptHelper, {"mode"}),
            (PromptComposer, {"output_style"}),
            (VisionPromptHelper, {"mode", "target_model"}),
        ):
            spec = inspect.getfullargspec(node.VALIDATE_INPUTS)
            assert set(spec.args) - {"cls"} == expected
            assert spec.varkw is None

    @pytest.mark.parametrize("new,old", HELPER_LABEL_TABLE)
    def test_prompt_helper_accepts_both_labels(self, new, old):
        assert PromptHelper.VALIDATE_INPUTS(new) is True
        assert PromptHelper.VALIDATE_INPUTS(old) is True

    @pytest.mark.parametrize("new,old", COMPOSER_LABEL_TABLE)
    def test_composer_accepts_both_labels(self, new, old):
        assert PromptComposer.VALIDATE_INPUTS(new) is True
        assert PromptComposer.VALIDATE_INPUTS(old) is True

    def test_vision_helper_accepts_every_shipped_combination(self):
        for mode in catalog_mod.vision_mode_labels():
            for target in catalog_mod.image_target_model_labels():
                assert VisionPromptHelper.VALIDATE_INPUTS(mode, target) is True

    def test_vision_helper_accepts_a_missing_target_model(self):
        """An optional input is absent from workflows saved before the update.

        ComfyUI builds the ``VALIDATE_INPUTS`` call from the inputs the
        workflow actually stores (``get_input_data`` iterates ``inputs``), so
        for such a workflow the classmethod is called with ``mode`` only. It
        must therefore be callable that way and treat the absent value as
        ``Generic``.
        """
        assert VisionPromptHelper.VALIDATE_INPUTS("Hair Change") is True
        assert VisionPromptHelper.VALIDATE_INPUTS("Hair Change", None) is True
        assert VisionPromptHelper.VALIDATE_INPUTS(mode="Hair Change") is True

    def test_unknown_mode_is_rejected_with_a_readable_message(self):
        result = PromptHelper.VALIDATE_INPUTS("Totally Made Up")
        assert result is not True
        assert "Totally Made Up" in result
        assert "Known values" in result

    def test_unknown_style_is_rejected(self):
        result = PromptComposer.VALIDATE_INPUTS("Totally Made Up")
        assert result is not True
        assert "Totally Made Up" in result

    def test_unknown_vision_mode_is_rejected(self):
        result = VisionPromptHelper.VALIDATE_INPUTS("Totally Made Up", "Generic")
        assert result is not True
        assert "Totally Made Up" in result

    def test_text_only_target_model_is_rejected(self):
        result = VisionPromptHelper.VALIDATE_INPUTS("Hair Change", "SDXL")
        assert result is not True
        assert "takes no reference images" in result


# ---------------------------------------------------------------------------
# Node dropdowns come from the catalog
# ---------------------------------------------------------------------------
class TestNodeDropdowns:
    def test_prompt_helper_mode_dropdown(self):
        spec = PromptHelper.INPUT_TYPES()["required"]["mode"]
        assert spec[0] == catalog_mod.helper_mode_labels()
        assert spec[1]["default"] == catalog_mod.helper_mode_labels()[0]

    def test_composer_style_dropdown(self):
        spec = PromptComposer.INPUT_TYPES()["required"]["output_style"]
        assert spec[0] == catalog_mod.composer_style_labels()
        assert spec[1]["default"] == catalog_mod.composer_style_labels()[0]

    def test_vision_mode_dropdown(self):
        spec = VisionPromptHelper.INPUT_TYPES()["required"]["mode"]
        assert spec[0] == catalog_mod.vision_mode_labels()
        assert spec[1]["default"] == "Outfit Transfer"

    def test_vision_target_model_is_optional_and_defaults_to_generic(self):
        spec = VisionPromptHelper.INPUT_TYPES()["optional"]["target_model"]
        assert spec[0] == catalog_mod.image_target_model_labels()
        assert spec[1]["default"] == "Generic"
