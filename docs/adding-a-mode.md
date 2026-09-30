# Adding a Mode, a Style or a Target Model

Every selectable value in the three LLM nodes comes from one file:
`config/catalog.yaml.example`. Adding a PromptHelper mode, a
PromptComposer output style, a VisionPromptHelper mode or a target
model means **one entry in that file** plus, for the entries that have
a `template`, **one template file** under
`comfyui_prompt_tools/system_prompts/`. There is no code to touch — the
loader builds the dropdowns from the catalog when ComfyUI starts.

See [`architecture/prompt-catalog.md`](architecture/prompt-catalog.md)
for how the pieces fit together.

## Where to put your entry

| You want to… | Section in the catalog |
|---|---|
| add a PromptHelper mode | `prompt_helper_modes` |
| add a PromptComposer output style | `composer_styles` |
| add a VisionPromptHelper mode | `vision_modes` |
| add an image / video model the prompts are written for | `target_models` |

`config/catalog.yaml.example` is the shipped truth and is overwritten by
`git pull`. Put **your** entries in a sibling `config/catalog.yaml`
(gitignored) — it is overlaid on the shipped file, matched by `id`:

* same `id` → the fields you list replace the shipped ones
* new `id` → appended at the end of its section
* `enabled: false` → the entry disappears from the dropdown

You only repeat the fields you want to change. Contributing a mode
upstream means adding it to `config/catalog.yaml.example` instead.

## Step 1 — pick an id, a label and a basename

The **`id`** is the stable internal key. Lower-case, underscores. Never
rename it once it has shipped: local overlays and the `target_model`
references point at it.

The **`label`** is what shows in the UI (e.g.
`"Z-Image – Text-to-Image"`). Two conventions worth keeping:

- Target model first, task second, separated by an en dash — the
  dropdown then groups by model when read top to bottom.
- A label is **frozen once released.** Saved ComfyUI workflows store it
  as a plain string, so renaming one breaks every workflow that used it.
  If you must rename, move the old label into `aliases` (see step 4).

The **basename** is the filename-safe name of the template on disk
(e.g. `zimage_text_to_image`). The per-model override cascade builds on
it (see [`system-prompt-overrides.md`](system-prompt-overrides.md)).

## Step 2 — write the template

Create `comfyui_prompt_tools/system_prompts/<basename>.txt.example`.

The committed templates ship as `.txt.example` so a `git pull` never
overwrites user-edited `<basename>.txt` copies (the loader prefers the
`.txt` if present and falls back to the `.example`). When contributing a
new mode upstream, commit the `.txt.example`; users can then copy it to
`<basename>.txt` to customise locally.

The template is a system prompt the LLM receives ahead of the user's
input. It should:

- Open with one line of role framing
  ("You are a prompt engineer for X…").
- Spell out the output style the downstream model expects —
  tag list, natural language, dense vs verbose, etc.
- List concrete inclusions (technical markers, anatomy safety, quality
  anchors the model needs).
- End with `{shared_rules}` on its own line. The loader substitutes
  that placeholder with the contents of `_shared_rules.txt`.

Length budget: 100–300 words for the template body. Existing files
(`flux_text_to_image.txt.example`, `zimage_text_to_image.txt.example`,
`random_character_pony.txt.example`) are good references.

### Referring to reference images

Never write `image 1` or `Picture 1` into a template. Target models
address their inputs differently and the catalog is what decides the
wording. Two different jobs, two different notations:

| What you are writing | Notation |
|---|---|
| an instruction **to the vision LLM** about which input is which | fixed prose: "the first image", "the second image" |
| text the LLM must **put into the prompt it produces** | `{img1}`, `{img2}`, `{img3}` |

`{imgN}` is replaced with the `image_ref` pattern of the selected target
model — `image 1` for FLUX Kontext, `Picture 1` for Qwen-Image-Edit 2511
and Krea 2, `<image1>` for Qwen-Image 2.1. `Generic` keeps the neutral
`image {n}`.

Describe-mode templates get **no image reference at all**: their output
is a snippet that a composer pastes into someone else's prompt, where an
`image 1` would address the wrong picture.

## Step 3 — register the entry

Add one entry to the right section. The order in the file is the
dropdown order, so place it next to similar entries.

A PromptHelper mode:

```yaml
prompt_helper_modes:
  - id: zimage_text_to_image
    label: "Z-Image – Text-to-Image"
    aliases: ["Z-Image Text-to-Image"]
    template: zimage_text_to_image
    target_model: zimage
```

A composer style takes the same fields. A vision mode replaces
`target_model` (the user picks that on the node) with `kind` and
`images`:

```yaml
vision_modes:
  - id: vision_describe_hair
    label: "Describe Hair"
    template: vision_describe_hair
    kind: describe        # edit | describe
    images: 1             # 1 | 2 — 2 requires image_2 to be wired up
```

A target model needs `image_ref` only if it addresses reference images
inside the prompt. Models without it are text-to-image / text-to-video
and are not offered in the VisionPromptHelper `target_model` dropdown:

```yaml
target_models:
  - id: qwen_image_21
    label: "Qwen-Image 2.1"
    image_ref: "<image{n}>"     # {n} is the image number
```

`AVAILABLE_MODES`, `OUTPUT_STYLES` and `AVAILABLE_VISION_MODES` are
derived from the catalog — no separate update needed. The dropdowns
include the new label the next time ComfyUI reloads.

The catalog is validated on load and **fails loudly**: a duplicate `id`,
a label that collides with another entry's alias, a missing template
file, a dangling `target_model` or an `image_ref` without `{n}` raises a
`CatalogError` naming the offending entry and field.

## Step 4 — renaming a released label

Move the old label into `aliases` and give the entry the new `label`:

```yaml
  - id: random_character_pony
    label: "SDXL Pony – Random Character"
    aliases: ["Random Character (Pony)"]
```

Both strings then resolve to the same entry, so a workflow saved with
the old label keeps running. This works because the three nodes declare
a `VALIDATE_INPUTS` classmethod naming `mode` / `output_style` /
`target_model`, which makes ComfyUI skip its own "Value not in list"
check for those inputs and delegate to the catalog. **Never remove an
alias once it has shipped.**

The label in a *loaded* workflow is a separate question: whether the
ComfyUI frontend keeps a widget value that is no longer in the list can
only be checked in the UI.

## Step 5 — (optional) ship model-family overrides

If the mode benefits from a different tone for a specific LLM family
(e.g. a narrative variant for a chat-tuned model, a tag-only variant
for another), add
`system_prompts/<basename>.<family>.txt.example` (or `.txt` for a
purely local override that should not be committed). The cascade in
`prompts.py:render_template` picks it up automatically when the
user selects a matching model. See
[`system-prompt-overrides.md`](system-prompt-overrides.md) for the
family list and style guide.

## Step 6 — add a unit test

In `tests/unit/test_prompts_loader.py` (or
`test_vision_prompts.py` / `test_prompt_composer.py`), add a short test
that:

- asserts the label appears in the derived list,
- maps to the expected basename,
- loads a non-empty rendered prompt,
- has `{shared_rules}` substituted (placeholder absent from output),
- contains one or two stable fingerprint strings from the template
  body (so a future accidental file truncation surfaces).

Do **not** assert a mode count — adding a mode should not require
editing an unrelated test.

The parametrised tests already cover every entry the moment it lands in
the catalog: the cascade fallback test in `test_prompts_loader.py`, the
template-exists and label-uniqueness checks in `test_catalog.py`, and —
for vision modes — the "describe modes carry no image reference" /
"edit modes do reference an image" pair in `test_catalog.py`.

Run the suite from the repo root:

```bash
python3 -m pytest tests/unit -q
```
