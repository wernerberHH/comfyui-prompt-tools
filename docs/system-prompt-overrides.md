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
| FLUX Kontext (Couple Scene) | `flux_kontext_couple_scene.txt.example` | `flux_kontext_couple_scene.qwen3vl.txt` |
| Qwen Image Edit (Couple Scene) | `qwen_image_edit_couple_scene.txt.example` | `qwen_image_edit_couple_scene.qwen3vl.txt` |
| Random Character (Pony) | `random_character_pony.txt.example` | `random_character_pony.qwen3vl.txt` |
| Random Character (Z-Image) | `random_character_zimage.txt.example` | `random_character_zimage.qwen3vl.txt` |
| Z-Image Text-to-Image | `zimage_text_to_image.txt.example` | `zimage_text_to_image.qwen3vl.txt` |

**The public release ships no override files.** The table above shows
the naming convention for overrides users can author themselves. Modes
with strict tag syntax (SDXL, SDXL Pony/Illustrious), structural edits
(FLUX Kontext Scene Edit), or user-supplied prompts (Custom System
Prompt) generally benefit less from a model-specific override than
free-form narrative modes do.

## Writing a good override

The model-specific override is a *flavour* of the default, not a
rewrite. Style guidance:

- **Match the tone to the model.** A narrative-tuned model responds to
  framing ("You are writing the scene…"), not imperative walls. Avoid
  `MUST`, `DO NOT`, `ONLY` in caps if your model gets stubborn under
  them.
- **Keep the format anchors.** If the downstream model requires
  `image 1` tokens, Pony quality tags, or identity-preservation
  phrasing, those still appear — they're contracts with the
  generator, not stylistic choices.
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
share a frame, drawing on image 1 and image 2 as references."* The hard
contracts the downstream generator needs — the opening anchor phrase,
both image tokens, identity preservation, anatomy safety, length
budget — should survive as quiet sentences inside the prose, not as a
numbered rule list.

The file is gitignored locally; commit it as
`flux_kontext_couple_scene.qwen3vl.txt.example` if you want to
contribute it upstream.
