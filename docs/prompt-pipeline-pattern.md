# Prompt Pipeline Pattern

**Purpose:** explain the design philosophy of the package — why we have separate nodes for description, edit, and composition, instead of one mega-node.

---

## The Problem with Monolithic Prompt Helpers

A single mega-prompt-helper that takes "two reference images and a user instruction, output one prompt" is conceptually simple but in practice fragile:

- The LLM has to decide what's important about each image (often guesses wrong)
- Hard to debug — "why did it lose Person B's hair color?" is invisible inside one black-box call
- Can't reuse intermediate results (e.g., re-generate the scene with a new background but same people)
- Hard to vary one aspect (different lighting, same outfit) without re-doing everything

## The Pipeline Approach

Split the work into specialists:

```
[Image: Person A]──┐
                   ├──> VisionPromptHelper(describe_face)──> "asian woman, almond eyes, ..."
[Image: Person A]──┘
                                                                   │
[Image: Person B]──┐                                               │
                   ├──> VisionPromptHelper(describe_face)──> ".."  │
[Image: Person B]──┘                                               │
                                                                   ├──> PromptComposer
[Image: Setting] ───> VisionPromptHelper(describe_background)──> ".."│   user_instruction:
                                                                   │   "warm romantic dinner"
[Manual: Outfit A] ─────────────────────────────────────────────── │
[Manual: Outfit B] ─────────────────────────────────────────────── │
                                                                   ▼
                                                            final prompt string
                                                                   │
                                                                   ▼
                                                            CLIPTextEncode → KSampler
```

**Each node has one job. Each output is debuggable. Each can be cached or replaced.**

---

## When to Use Which Node

| Scenario | Use |
|---|---|
| Single-subject image, no reference | `PromptHelper` (text mode) |
| Edit an image (change outfit, hair, etc.) | `VisionPromptHelper` edit-mode |
| Need the "blueprint" of one aspect of an image | `VisionPromptHelper` describe-mode |
| Combine multiple descriptions + user intent | `PromptComposer` |
| Switch between AI-generated and manual prompt | `TextMux` |

---

## Why describe-modes matter for identity stability

When generating a couple scene with two reference subjects, image-conditioning adapters (IPAdapter, ReActor, PuLID, InfiniteYou) work best when the prompt **already has hooks** for what they should reinforce.

**Vague prompt** ("two people at a cafe") forces the adapter to do all the identity work alone — it must guess everything from the conditioning image, often resulting in feature-bleed between subjects (especially with similar facial structures).

**Specific prompt** ("asian woman in her 30s with almond eyes... european man with blue eyes...") gives the adapter explicit anchors. The adapter then refines a starting point that's already correctly differentiated, instead of constructing identity from scratch.

→ **describe-modes give you that specificity automatically**, sourced from the actual reference image rather than your memory of what the person looks like.

---

## Trade-offs

**Wins:**
- Better identity stability for multi-subject scenes
- Each output is visible in `ShowText` for debugging
- Reusability — describe Person A once, use the snippet across many generations
- New modes are pure data (system_prompt files), not new code

**Costs:**
- More nodes on the canvas
- 3-5 extra LLM calls per workflow (~3-8 seconds total with vision models)
- Steeper learning curve for users new to ComfyUI

---

## Example workflows

Ready-to-load demo workflows live in [`examples/`](../examples/) — see
[`examples/README.md`](../examples/README.md) for the per-workflow
description, expected models, and default LLM provider. The
`video_v1_wan22_i2v_demo.json` workflow exercises all four nodes
(VisionPromptHelper → PromptHelper → PromptComposer → TextMux) in a
single pipeline.

---

## Future Extensions (post-v0.4)

- **Caching** — `PromptComposer` could memoize `(input_hash, user_instruction, output_style) -> result` to avoid recomputing on parameter sweeps
- **Conditional pipelines** — a router node that picks describe-modes based on what aspects the user wants to vary
- **Prompt-quality scoring** — a separate node that rates a generated prompt for clarity/specificity before sending to the generator

These are ideas, not commitments.
