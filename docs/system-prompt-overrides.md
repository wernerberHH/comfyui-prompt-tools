# Per-Model System-Prompt Overrides

Different LLM families respond best to different prompt styles. A
hard-rule-heavy template written for one model ("MUST start with…",
"DO NOT use…") can produce stubborn or off-style output from another
that prefers narrative framing and ignores tag syntax. The override
cascade lets each `(mode, model-family)` pair ship its own prompt
variant — without forking the loader or duplicating the default.

## The cascade

`get_system_prompt(mode, model_name=None)` resolves in this order:

1. `system_prompts/<mode>.<family>.txt` — model-family override
2. `system_prompts/<mode>.txt` — default
3. `KeyError` — only when the mode is not registered

If `model_name` is `None` or the family is unknown, step 1 is skipped
and the loader goes straight to the default. The shared
`render_template()` helper applies the cascade and substitutes
`{shared_rules}` in one place, so `vision_prompts.get_vision_system_prompt`
and `prompt_composer._load_composer_system_prompt` honour the same rules.

## Model-family detection

Families are detected from the model tag by an ordered list. The
built-in defaults live in `prompts.py`:

```python
MODEL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"^qwen3-vl",              "qwen3vl"),
    (r"^qwen3(\.|$|-prompt)",   "qwen3"),
    (r"^gemma",                 "gemma"),
    (r"^llama",                 "llama"),
]
```

Matching is case-sensitive — Ollama and vLLM model tags are
case-sensitive too. First match wins.

### Adding your own families

You do not need to edit code to route your own model tags. Drop a
`config/model_families.yaml` (gitignored — your model names stay
private and survive `git pull`) next to `endpoints.yaml`:

```yaml
families:
  - { pattern: "^MyOrg/MyModel", family: mymodel }
```

These custom patterns are **prepended** to the built-in defaults, so a
custom entry wins over (or extends) the defaults. See
`config/model_families.yaml.example` for the full schema. A missing
file, missing `pyyaml`, or malformed YAML degrades gracefully — the
built-in defaults still apply.

## Naming convention

Override files live next to their defaults in `system_prompts/` and
follow the pattern `<existing_default_basename>.<family>.txt`
(or `.txt.example` if contributed upstream).

| Mode label | Default file | Example override filename |
|---|---|---|
| FLUX Kontext – Couple Scene | `flux_kontext_couple_scene.txt.example` | `flux_kontext_couple_scene.qwen3vl.txt` |
| Qwen-Image-Edit – Couple Scene | `qwen_image_edit_couple_scene.txt.example` | `qwen_image_edit_couple_scene.qwen3vl.txt` |
| SDXL Pony – Random Character | `random_character_pony.txt.example` | `random_character_pony.qwen3vl.txt` |
| Z-Image – Random Character | `random_character_zimage.txt.example` | `random_character_zimage.qwen3vl.txt` |
| Z-Image – Text-to-Image | `zimage_text_to_image.txt.example` | `zimage_text_to_image.qwen3vl.txt` |

The basename comes from the mode's `template` field in
`config/catalog.yaml.example` — see
[`adding-a-mode.md`](adding-a-mode.md).

**The public release ships no override files.** The table above shows
the naming convention for overrides users can author themselves. Modes
with strict tag syntax (SDXL – Photorealistic, SDXL Pony – Illustrious),
structural edits (FLUX Kontext – Scene Edit), or user-supplied prompts
(Custom System Prompt) generally benefit less from a model-specific
override than free-form narrative modes do.

## Writing a good override

The model-specific override is a *flavour* of the default, not a
rewrite. Style guidance:

- **Match the tone to the model.** A narrative-tuned model responds to
  framing ("You are writing the scene…"), not imperative walls. Avoid
  `MUST`, `DO NOT`, `ONLY` in caps if your model gets stubborn under
  them.
- **Keep the format anchors.** If the downstream model requires image
  reference tokens, Pony quality tags, or identity-preservation
  phrasing, those still appear — they're contracts with the
  generator, not stylistic choices. Write the reference tokens as
  `{img1}` / `{img2}`, never as a literal `image 1`: the catalog renders
  them into the wording of the selected target model (see
  [`architecture/prompt-catalog.md`](architecture/prompt-catalog.md)).
- **Length budget: 150–300 words** of system prompt, similar to
  the defaults.
- **Keep `{shared_rules}`** wherever the default uses it — the
  substitution runs over override files too.
- **Test it.** The parametrised fallback test in
  `tests/unit/test_prompts_loader.py` already covers every mode in
  `AVAILABLE_MODES`. For a more targeted assertion (override-vs-default
  divergence for a specific mode), follow the pattern in
  `tests/unit/test_prompts_example_fallback.py`.

## Worked example — author your own override

Suppose the default `flux_kontext_couple_scene.txt.example` opens with
bullet-style rules: *"The prompt MUST reference both images"*,
*"Always include 'Preserve the exact facial features…'"*, *"Do NOT use
markdown"*. A narrative-tuned model may treat imperative walls as stage
directions rather than constraints.

First register the model's family in `config/model_families.yaml` (e.g.
map its tag to `qwen3vl`, or define your own family name). Then create
`system_prompts/flux_kontext_couple_scene.qwen3vl.txt` and reframe the
task in the register your model responds to. The opening line might be
a neutral framing such as *"You are writing the scene where two people
share a frame, drawing on {img1} and {img2} as references."* The hard
contracts the downstream generator needs — the opening anchor phrase,
both image tokens, identity preservation, anatomy safety, length
budget — should survive as quiet sentences inside the prose, not as a
numbered rule list.

The file is gitignored locally; commit it as
`flux_kontext_couple_scene.qwen3vl.txt.example` if you want to
contribute it upstream.

## A local copy hides the shipped template — and stays hidden

The same precedence that protects your edits from `git pull` also keeps
updates away from them. `<name>.txt` always wins over
`<name>.txt.example`, and nothing removes your `.txt` on an update: the
ComfyUI-Manager unpacks a release over the node folder and only deletes
files that belonged to the previous package. A copy you made two releases
ago therefore still drives the prompt today, even though the shipped
template has improved and may have gained placeholders your copy does not
have.

The nodes report this instead of leaving it silent. Once per process, at
startup, every local `.txt` in `system_prompts/` is compared against
`system_prompts/shipped_hashes.json` — the digest of every version of
every template this package has ever shipped — and the result goes into
the ComfyUI log as one block. Nothing is changed, renamed or deleted; see
[the README](../README.md#local-copies-can-hide-the-updated-templates)
for the message, the labels and how to clean up.

### How a file is classified

| Classification | Condition |
|---|---|
| `identical` | Same content as the current shipped template. Counted only. |
| `outdated-copy` | Same content as an *older* shipped version — taken over unchanged, now hiding the current one. |
| `customized` | Matches no shipped version: a real local edit. |
| `stale-shipped-override` | A `<name>.<family>.txt` with no `.txt.example` of its own whose content matches a version once shipped under that name. Left behind by an update. |
| `local override` | No shipped counterpart at all — an override you wrote. Counted only. |
| `unverified` | `shipped_hashes.json` missing or unreadable, or the file is not valid UTF-8. |

Two classifications deserve a word on *why* they are what they are:

- **`_shared_rules.txt` is treated like any other template.** It ships as
  `_shared_rules.txt.example` and a local copy shadows it the same way —
  and because every template that writes `{shared_rules}` substitutes it,
  a stale copy of this one file reaches nearly every mode. Exempting it
  would hide the widest-reaching case. Only the `{imgN}` note below is
  withheld: it is not a mode template and addresses no reference images.
- **A family override with no `.txt.example` is not reported as out of
  date by default.** There is no current shipped version it could be
  behind, so it counts as yours — unless its content matches a version
  this package itself once shipped under that name, which means an update
  left it behind rather than you writing it. That case is
  `stale-shipped-override`.

A `customized` copy backing an image-**edit** mode gets one extra note
when it contains no `{imgN}` placeholder:

```
[no {imgN} placeholders — the target_model setting has no effect on this mode]
```

Such a copy predates the `target_model` input: the image-reference wording
is baked into its text, so switching `target_model` changes nothing for
that mode. Re-apply your edits on top of the current `.txt.example`, which
writes `{img1}` / `{img2}`, to get the selector working again.

### Comparison is on normalised text

Digests cover the text as the loader uses it — decoded as UTF-8, line
endings converted to `\n`, trailing newlines stripped. An editor that
rewrites line endings or appends a final newline therefore does not turn
an untouched copy into a `customized` one.

### After changing a shipped template (maintainers)

`shipped_hashes.json` is generated from the git history, so it has to be
regenerated whenever a `*.txt.example` changes:

```bash
python3 scripts/update_shipped_hashes.py        # rewrite the file
python3 scripts/update_shipped_hashes.py --check  # CI-style: report drift only
```

Commit the result together with the template change.
`tests/unit/test_shipped_hashes.py` fails until you do — otherwise the new
template content would be missing from the history and an untouched copy
of the *previous* version could no longer be recognised as an
`outdated-copy`.

Retired versions — in practice model-family overrides that were shipped
and later externalised — are kept in a second section,
`retired_override_versions`, keyed by **base template name only**. The
family part of the filename is deliberately dropped: this is a public
repository and the model tags this project routes on stay out of it (they
live in the gitignored `config/model_families.yaml`). Knowing "did this
exact content ever ship?" for a file whose name the check already has is
all the classification needs.
