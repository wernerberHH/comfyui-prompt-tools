# Example workflows

This directory contains ready-to-load ComfyUI workflows that demonstrate
the typical use patterns for **comfyui-prompt-tools**.

## Files

### `z_image_turbo_txt2img_demo.json` — Text-to-image with PromptHelper

A minimal text-to-image pipeline that shows the core node pattern of
this package:

```
PromptHelper → ShowText → TextMux (AI/Manual Switch) → CLIPTextEncode
```

The PromptHelper takes a short scene description from the user (e.g.
"a young woman in a cafe"), enhances it into a long, photo-realistic
description tailored to Z-Image Turbo, and routes the result through a
ShowText preview and a TextMux switch (so you can override the LLM
output with manual text if you want) before it hits the CLIP encoder.

**Models expected:**

- `z_image_turbo_bf16.safetensors` (UNET)
- `qwen_3_4b.safetensors` (CLIP / text encoder)
- `ae.safetensors` (FLUX VAE)
- Optional: `z-image/Z-Detail-Slider.safetensors` LoRA (node ships
  bypassed — press `Ctrl+B` on the LoRA node to enable if installed).

**LLM provider:** OpenAI `gpt-4o-mini` by default; switch the
`engine` / `model` dropdowns on the PromptHelper to use any other
configured provider (`ollama`, `vllm`, `gemini`, …).

---

### `flux2_fashion_v4_demo.json` — Outfit transfer with VisionPromptHelper

A two-reference-image workflow for FLUX.2 that transfers an outfit
from one image to a person in another:

```
LoadImage (Person)   ──┐
                       ├──→ VisionPromptHelper (Outfit Transfer mode)
LoadImage (Outfit)   ──┘            │
                                    ▼
                       ShowText  →  TextMux  →  CLIPTextEncode
                                                       │
                              + ReferenceLatent ×2 ────┴──→ FLUX.2 sampler
```

The VisionPromptHelper looks at both images and writes a single
identity-preserving prompt that describes the person from image 1
wearing the outfit from image 2. Both images are also fed into the
sampler as latent references via `ReferenceLatent` so FLUX.2 can keep
identity and outfit visually consistent.

**Models expected:**

- `flux2/flux2_dev_fp8mixed.safetensors` (UNET)
- `flux2/mistral_3_small_flux2_fp8.safetensors` (CLIP / text encoder)
- `flux2/flux2-vae.safetensors` (VAE)

**LLM provider:** OpenAI `gpt-5-mini` by default. Replace the two
`example.png` LoadImage references with your own person + outfit
images before running.

---

### `video_v1_wan22_i2v_demo.json` — Image-to-video with the full pipeline

The most complete demo: a Wan 2.2 image-to-video workflow that uses
**all four** custom nodes — VisionPromptHelper for scene description,
PromptHelper for motion writing, PromptComposer for fusing them into a
Wan-shaped prompt, and TextMux for AI-vs-manual switching.

```
LoadImage  ──→  VisionPromptHelper (Describe Picture)  ──→  ShowText
                                                                  │
user motion idea  ──→  PromptHelper (motion writer)  ──→  ShowText│
                                                                  ├─→ PromptComposer (Wan 2.2 motion style)
                                                                  │              │
                                                                  ▼              ▼
                                                              TextMux  ──→  CLIPTextEncode  ──→  Wan 2.2 i2v sampler
```

The two-stage sampling matches the Wan 2.2 reference workflow
(high-noise UNET for steps 0–10, low-noise UNET for steps 10–20). A
Lightning LoRA pair is wired in but bypassed by default — press
`Ctrl+B` on the LoRA nodes to enable a 4-step fast mode (see the
workflow's note for the sampler settings that go with it).

**Models expected:**

- `wan22/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- `wan22/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (CLIP, Wan)
- `wan_2.1_vae.safetensors` (VAE)
- Optional: `wan22/wan2.2_i2v_lightx2v_4steps_lora_v1_*.safetensors`
  for Lightning mode (bypassed by default).

**LLM providers:** the VisionPromptHelper is preconfigured for a local
vLLM endpoint at `http://localhost:8001/v1`; the PromptHelper and
PromptComposer are preconfigured for OpenAI `gpt-4.1-mini`. Adjust the
`engine` / `api_url` / `model` dropdowns to point at whatever you have
configured.

The PromptHelper system prompt also demonstrates the multilingual
input pattern: the included few-shot examples cover German user
inputs to show the "always-translate-to-English" behaviour, but the
output is always English regardless of input language.

**Defaults:** 480×704, 97 frames @ 16fps (~6 seconds).
