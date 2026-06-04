# Adding a Mode to PromptHelper

Every PromptHelper "mode" (visible in the node's mode dropdown) is
backed by exactly two things: one row in `prompts.py:MODE_TO_FILE`
and one template file in `system_prompts/`. Adding a mode means
adding both. No code changes are needed inside the node itself —
the loader picks it up automatically.

The walkthrough below shows the FLUX Text-to-Image / Z-Image
Text-to-Image pattern. The same recipe applies to all PromptHelper
modes (Random-Character modes additionally use the slot-pool
machinery in `random_pools.py`, but the loader contract is identical).

## Step 1 — pick a name and a basename

The **mode label** is what shows in the UI (e.g.
`"Z-Image Text-to-Image"`). Keep it concise and grouped with similar
modes — text-to-image modes live next to each other, edit modes
next to each other.

The **basename** is the filename-safe version used on disk
(e.g. `zimage_text_to_image`). Lower-case, underscores, no
parentheses or slashes. The override cascade builds on this
basename (see `docs/system-prompt-overrides.md`).

## Step 2 — write the default template

Create `comfyui_prompt_tools/system_prompts/<basename>.txt.example`.

The committed templates ship as `.txt.example` so a `git pull` never
overwrites user-edited `<basename>.txt` copies (the loader prefers
the `.txt` if present and falls back to the `.example`). When
contributing a new mode upstream, commit the `.txt.example`; users
can then copy it to `<basename>.txt` to customise locally.

The template is a system prompt the LLM receives ahead of the
user's input. It should:

- Open with one line of role framing
  ("You are a prompt engineer for X…").
- Spell out the output style the downstream model expects —
  tag list, natural language, dense vs verbose, etc.
- List concrete inclusions (technical markers, anatomy safety,
  token anchors like `image 1` / `image 2` if the model needs them).
- End with `{shared_rules}` on its own line. The loader substitutes
  that placeholder with the contents of `_shared_rules.txt`.

Length budget: 100–300 words for the template body. Existing files
(`flux_text_to_image.txt.example`, `zimage_text_to_image.txt.example`,
`random_character_pony.txt.example`) are good references.

## Step 3 — register the mode

In `comfyui_prompt_tools/prompts.py`, add the entry to `MODE_TO_FILE`
in the location that groups it with similar modes:

```python
MODE_TO_FILE: Dict[str, str] = {
    ...
    "FLUX Text-to-Image":             "flux_text_to_image",
    "Z-Image Text-to-Image":          "zimage_text_to_image",
    "SDXL Photorealistic":            "sdxl_photorealistic",
    ...
}
```

`AVAILABLE_MODES` is derived from `MODE_TO_FILE.keys()` — no
separate update needed. The PromptHelper node's dropdown will
include the new label the next time ComfyUI reloads.

## Step 4 — (optional) ship model-family overrides

If the mode benefits from a different tone for a specific model
family (e.g. a narrative variant for a chat-tuned model, a tag-only
variant for another), add
`system_prompts/<basename>.<family>.txt.example` (or `.txt` for a
purely local override that should not be committed). The cascade in
`prompts.py:render_template` picks it up automatically when the
user selects a matching model. See `docs/system-prompt-overrides.md`
for the family list and style guide.

## Step 5 — add a unit test

In `tests/unit/test_prompts_loader.py`, add a short test that:

- asserts the mode label appears in `AVAILABLE_MODES`,
- maps to the expected basename in `MODE_TO_FILE`,
- loads a non-empty rendered prompt via `get_system_prompt(label)`,
- has `{shared_rules}` substituted (placeholder absent from output),
- contains one or two stable fingerprint strings from the template
  body (so a future accidental file truncation surfaces).

The parametrised cascade test
(`test_override_tag_falls_back_to_default_in_shipped_library`) runs
across every mode in `AVAILABLE_MODES`, so a new mode is covered the
moment it lands in `MODE_TO_FILE` — no separate registration needed.

Run the suite from the repo root:

```bash
python -m pytest tests/unit/ -v
```

No nodes need editing — `PromptHelper`, `VisionPromptHelper`, and
`PromptComposer` read `AVAILABLE_MODES` and call the loader by
label, so the new mode is wired up the moment the registry and
template land.
