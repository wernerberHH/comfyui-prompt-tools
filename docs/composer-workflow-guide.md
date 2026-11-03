# Composer Workflow Guide

**Audience:** ComfyUI users building multi-subject scenes who want stable
identity, reusable scene snippets, and debuggable prompts.

**Prerequisites:** `comfyui-prompt-tools` installed (v0.4 or later). See
[`prompt-pipeline-pattern.md`](prompt-pipeline-pattern.md) for the design
rationale behind the pipeline approach.

---

## The pattern in one picture

```
[Setting reference image] ──> VisionPromptHelper           ──> "candlelit bistro,
                                (Describe Background)           dark wooden tables..."
                                                                       │
[Static text: outfit_A]   ────────────────────────────────────────────│
                                                                       │
[Static text: outfit_B]   ────────────────────────────────────────────│
                                                                       │
                                                                       ▼
                                                          PromptComposer
                                                          user_instruction:
                                                          "two people sharing
                                                           dessert, warm tones"
                                                          output_style:
                                                       "FLUX.2 – Natural Language"
                                                                       │
                                                                       ▼
                                                            composed_prompt
                                                                       │
                                                                       ▼
                                                          CLIPTextEncode → KSampler
```

The composer fuses an LLM-extracted scene description with two static outfit
descriptions and the user's scene intent into one coherent FLUX.2 prompt.
Each upstream node is debuggable in isolation (wire a `ShowText` to its
output to see what it produced).

---

## Choosing where to wire each helper

The composer has **one `user_instruction` field** and **five generic
input slots**. They serve different purposes — the choice is not
cosmetic.

| Field | Role |
|-------|------|
| `user_instruction` | **Dominant intent.** Decides scene framing, mood, emphasis (image styles), or motion + camera (Wan video style). On conflict between two inputs, the composer prefers the one consistent with the instruction. |
| `input_1` … `input_5` | **Raw material.** Per-aspect snippets the composer fuses into the dominant intent. Irrelevant snippets are dropped rather than forced in. |

**Rule of thumb**

- If a helper answers *"what is shown?"* → its output goes into
  `user_instruction`.
- If a helper answers *"what does one aspect of it look like?"* → its
  output goes into one of the five input slots.

**For image composers** (`zimage`, `flux2`, `sdxl`): the
`user_instruction` carries the scene, subject framing and mood; the
inputs carry per-aspect detail (face, hair, body, outfit, background,
lighting, composition).

**For the Wan 2.2 video composer**: the `user_instruction` carries
the motion and camera intent. Inputs carry subject and atmosphere only;
image-description snippets in the inputs are explicitly *not* treated
as motion sources. A common cause of weak Wan motion is putting the
movement description into an input slot instead of the instruction.

Two common configurations:

1. **One helper produces a complete scene description** — wire it into
   `user_instruction`. Inputs stay empty (or carry one or two extra
   detail snippets).
2. **Several aspect-specific helpers** (classic pipeline-of-specialists)
   — each goes into an input slot. You write the `user_instruction`
   yourself as the connective tissue ("two people at a candlelit
   bistro, warm tones").

---

## Step-by-step build

### 1. Source the scene

Drop a `LoadImage` node and load a reference photo of the setting you want
(restaurant, beach, studio, wherever). Wire it into a `VisionPromptHelper`:

- `mode`: `Describe Background`
- `engine`: `ollama` (or `vllm` if you have a vision-capable vLLM)
- `base_url` + `model`: pick from the dropdown if you populated
  `config/endpoints.yaml`; otherwise type them. The dropdowns show URLs
  and models from **all** engines combined — you must pick entries that
  actually belong to your chosen `engine`, otherwise the call will fail
- `image_1`: ← from `LoadImage`
- `intent`: optional, e.g. `"warm intimate lighting"`

Output `vision_prompt` is a 30–60-word snippet ready to feed downstream.
Wire a `ShowText` here to verify it captured what you wanted.

### 2. Inputs for the outfits (or any other text)

Drop two `String (Multiline)` primitive nodes (or use your favourite text
node). One per outfit. Type the outfit description manually:

- Outfit A: `"navy linen suit, white open-collar shirt, brown leather belt"`
- Outfit B: `"emerald silk wrap dress, knee length, gold drop earrings"`

These are the **static** inputs — they don't need to come from an image, and
PromptComposer doesn't care where the text originated.

### 3. The composer

Drop a `PromptComposer` node:

- `engine`: `vllm` (text-only, faster than the vision engine you used in step 1)
- `base_url` + `model`: a text LLM, e.g. `Qwen2.5-7B-Instruct`
- `user_instruction`: `"two people sharing dessert at a candlelit bistro, warm tones"`
- `output_style`: `FLUX.2 – Natural Language`
- `input_1`: ← from the `VisionPromptHelper` (background snippet)
- `input_2`: ← from the outfit A text primitive
- `input_3`: ← from the outfit B text primitive
- `input_4` / `input_5`: leave unwired

Empty / unwired inputs are silently dropped before the LLM call, so the
composer sees a clean 3-item list.

Output `composed_prompt` is the final fused prompt. Wire another `ShowText`
to debug — and wire it to `CLIPTextEncode` to feed your sampler.

### 4. Generator side

Use the composed prompt as the positive conditioning for whatever you're
running — FLUX.2 dev / Kontext, Qwen Image Edit, SDXL, etc. Pick the
`output_style` in step 3 that matches your generator.

---

## Identity-stable multi-subject scenes

The pattern shines when you have two reference-photo subjects:

```
Person A image  ──> VisionPromptHelper (Describe Face) ──> face_a snippet
Person A image  ──> VisionPromptHelper (Describe Hair) ──> hair_a snippet
Person B image  ──> VisionPromptHelper (Describe Face) ──> face_b snippet
Person B image  ──> VisionPromptHelper (Describe Hair) ──> hair_b snippet
Setting image   ──> VisionPromptHelper (Describe Background)
                                                             │
                                                             ▼
                                                      PromptComposer
                                                      input_1: face_a
                                                      input_2: hair_a
                                                      input_3: face_b
                                                      input_4: hair_b
                                                      input_5: background
```

Pair the composed prompt with an image-conditioning adapter (IPAdapter,
ReActor, PuLID, InfiniteYou) for the actual identity transfer at sample
time. The prompt already carries the identity anchors; the adapter then
refines a starting point that's correctly differentiated, instead of
guessing everything from the conditioning image alone.

This is the recommended replacement for the older
`couple_scene_v3_qwen_reactor.json` workflow.

---

## Starting from an existing prompt (`base_prompt`)

Sometimes the scene is already written — a prompt you found, saved, or tuned
earlier — and you only want to swap in a different face, hairstyle or outfit.
Paste that prompt into the composer's `base_prompt` field and wire the
describe-mode snippets into the inputs as usual. The composer then edits the
base prompt instead of writing a new one: the parts that describe a feature
an input covers are replaced by that input, features the base prompt does not
mention are added, and everything else — scene, light, pose, camera — stays.

`base_prompt_rest` decides what happens to that rest:

| Value | What the LLM does | The rest of the base prompt |
|---|---|---|
| `keep verbatim` (default) | Returns a list of targeted edits as JSON; the node applies them | Unchanged, character for character — guaranteed by the node, not just requested from the LLM |
| `adapt to output style` | Rewrites the whole prompt with the selected `output_style` template | Content kept, wording adapted to the output style |

**Example** (`keep verbatim`):

- `base_prompt`: `a woman with long red curly hair in a yellow summer dress
  walking through a foggy forest at dawn, 35mm film`
- `input_1`: ← `Describe Hair` → `short platinum bob`

Result: `a woman with short platinum bob in a yellow summer dress walking
through a foggy forest at dawn, 35mm film`. `debug_info` reports
`Base prompt: verbatim | Replaced: 1 | Added: 0 | Skipped: 0`.

The same works for tag prompts: in `masterpiece, 1girl, long red hair,
curly hair, yellow sundress, sunflower field` a hair input replaces
`long red hair` and removes `curly hair` together with its comma.

Things to know about `keep verbatim`:

- **`output_style` is not used.** The replacements follow the language and
  form of the base prompt (tags stay tags, prose stays prose). Use
  `adapt to output style` when you want the result in a particular style.
- **`user_instruction` refines the edits**, e.g. `keep the background` stops a
  background input from replacing the setting.
- **Edits that cannot be placed are skipped, never guessed.** The LLM has to
  quote the text it replaces; if the quote is not in the base prompt (the
  match tolerates only case and extra spaces) or overlaps another edit, it is
  dropped and counted under `Skipped` in `debug_info`. A feature whose
  position cannot be found is appended at the end.
- **An unreadable LLM reply is an error** (`ERROR: unreadable edit list …`),
  not a silent fallback to a free rewrite. Small models occasionally produce
  one; run it again or pick a stronger text model.
- **With no inputs and no `user_instruction`** the base prompt is returned as
  it is, without an LLM call.
- `lora_keywords` work in both modes: the tokens are requested from the LLM
  and any token still missing is appended.

The template behind `keep verbatim` is
`system_prompts/composer_base_prompt_replace.txt.example` — override it like
any other template (see [system-prompt-overrides.md](system-prompt-overrides.md)).
It is not an output style and does not appear in the style list.

An empty `base_prompt` changes nothing: the composer behaves exactly as
before the field existed.

---

## Tips

- **Use a fast text LLM in the composer**, even if your describe-modes use a
  vision LLM. The composer doesn't need vision capability — it only fuses
  text snippets.
- **Wire `ShowText` between every stage** while building. Once you trust
  the pipeline, you can remove them; they don't affect performance.
- **Cache aggressively**. Describe-mode outputs are deterministic-ish for a
  given image + temperature + seed. If you're varying only the user
  instruction across a batch, the upstream snippets stay constant and only
  the composer re-runs.
- **Pick the right output style**:
  - `FLUX.2 – Natural Language` for FLUX.2 dev / Kontext, Qwen-Image-Edit
  - `SDXL – Tags` for SDXL / Pony / Illustrious
  - `Z-Image – Compact` for Z-Image (denser is better there)
- **Drop irrelevant snippets** instead of forcing them in. The composer is
  instructed to drop inputs that conflict with or are irrelevant to the
  user instruction — but if you know one is irrelevant, just leave the
  slot unwired.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Composed prompt ignores one snippet | Snippet conflicts with `user_instruction` (composer prefers the instruction by design) | Reword the snippet, or drop the conflicting bit |
| Output is too long for SDXL | Wrong `output_style` | Switch to `SDXL – Tags` |
| `ERROR: nothing to compose` | Both `user_instruction` and all 5 inputs are empty / unwired | Wire at least one input or type an instruction |
| `ERROR: connection refused` | Engine URL unreachable | Check `engine`, `base_url`, and that the LLM server is up |
| Dropdown for `model` is empty | No `models:` list configured for that URL in `endpoints.yaml` | Add the model name to the list, or type it into the field if it's a text input |
