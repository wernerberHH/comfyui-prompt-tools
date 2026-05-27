# Per-Model System-Prompt Overrides

Different LLM families respond best to different prompt styles. A
hard-rule-heavy template written for Qwen3-VL ("MUST start with…",
"DO NOT use…") produces stubborn or off-style output from Cydonia,
which is RP-tuned and ignores tag syntax. v0.5 introduces a small
override cascade so each `(mode, model-family)` pair can ship its own
prompt variant — without forking the loader or duplicating the
default.

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

Families are detected by a single ordered list in `prompts.py`:

```python
MODEL_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"^Fermi/Cydonia",         "cydonia"),
    (r"^qwen3-vl",              "qwen3vl"),
    (r"^qwen3(\.|$|-prompt)",   "qwen3"),
    (r"^gemma",                 "gemma"),
    (r"^huihui_ai/",            "abliterated"),
    (r"^llama",                 "llama"),
]
```

Matching is case-sensitive — Ollama and vLLM model tags are
case-sensitive too. First match wins. To add a new family, insert a
line in the desired priority position; the loader picks it up
without further wiring.

## Naming convention

Override files live next to their defaults in `system_prompts/` and
follow the pattern `<existing_default_basename>.<family>.txt`
(or `.txt.example` if contributed upstream).

| Mode label | Default file | Example override filename |
|---|---|---|
| FLUX Kontext (Couple Scene) | `flux_kontext_couple_scene.txt.example` | `flux_kontext_couple_scene.cydonia.txt` |
| Qwen Image Edit (Couple Scene) | `qwen_image_edit_couple_scene.txt.example` | `qwen_image_edit_couple_scene.cydonia.txt` |
| Random Character (Pony) | `random_character_pony.txt.example` | `random_character_pony.cydonia.txt` |
| Random Character (Z-Image) | `random_character_zimage.txt.example` | `random_character_zimage.cydonia.txt` |
| Z-Image Text-to-Image | `zimage_text_to_image.txt.example` | `zimage_text_to_image.cydonia.txt` |

**The public release ships no override files.** The table above
shows the naming convention for overrides users can author
themselves. Modes well-suited to a Cydonia narrative tone are
listed; modes with strict tag syntax (SDXL, SDXL Pony/Illustrious),
structural edits (FLUX Kontext Scene Edit), or user-supplied
prompts (Custom System Prompt) generally do not benefit from a
narrative-tone override.

## Writing a good override

The model-specific override is a *flavour* of the default, not a
rewrite. Style guidance:

- **Match the tone to the model.** Cydonia responds to narrative
  framing ("You are writing the scene…"), not imperative walls.
  Avoid `MUST`, `DO NOT`, `ONLY` in caps — Cydonia gets stubborn
  under them.
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

## Worked example — author your own Cydonia override

Suppose the default `flux_kontext_couple_scene.txt.example` opens
with bullet-style rules: *"The prompt MUST reference both images"*,
*"Always include 'Preserve the exact facial features…'"*, *"Do NOT
use markdown"*. Cydonia tends to ignore those — it is RP-tuned and
treats imperative walls as stage directions rather than constraints.

To author an override, create
`system_prompts/flux_kontext_couple_scene.cydonia.txt` and reframe
the task as character-voiced narration. Opening line might be a
neutral framing such as *"You are writing the scene where two people
share a frame, drawing on image 1 and image 2 as references."* The
hard contracts that the downstream generator needs — the opening
anchor phrase, both image tokens, identity preservation, anatomy
safety, length budget — should survive as quiet sentences inside the
prose, not as a numbered rule list. The result reads like a
director's brief, which is the register Cydonia speaks in.

The file is gitignored locally; commit it as
`flux_kontext_couple_scene.cydonia.txt.example` if you want to
contribute it upstream.
