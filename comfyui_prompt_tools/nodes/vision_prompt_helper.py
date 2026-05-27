"""VisionPromptHelper node: extract or edit prompts from reference images.

Supports 14 modes split into edit-modes (transform an image) and describe-modes
(extract one aspect as a snippet). Engine selection (Ollama vs vLLM) is
shared with PromptHelper via :class:`BasePromptNode`.
"""

from __future__ import annotations

import time

from ..engines import OllamaError, OpenAIError
from ..image_io import tensor_to_b64
from ..vision_prompts import (
    AVAILABLE_VISION_MODES,
    get_vision_system_prompt,
    mode_uses_two_images,
)
from .base_prompt_node import BasePromptNode


class VisionPromptHelper(BasePromptNode):
    """Generate edit instructions or aspect snippets from reference image(s).

    Single-image modes use ``image_1``. ``Outfit Transfer`` is the only mode
    in v0.4 that also consumes ``image_2`` (the outfit source). Other modes
    silently ignore ``image_2`` even if wired up.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI API contract
        engine_inputs = cls._build_engine_inputs()
        return {
            "required": {
                "intent":        ("STRING", {"multiline": True, "default": ""}),
                "mode":          (AVAILABLE_VISION_MODES, {"default": "Outfit Transfer"}),
                **engine_inputs,
                "jpeg_quality":  ("INT",    {"default": 85,  "min": 60,  "max": 95,  "step": 5}),
                "max_edge":      ("INT",    {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "image_1":       ("IMAGE",),
            },
            "optional": {
                "image_2":              ("IMAGE",),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("vision_prompt", "debug_info")
    FUNCTION      = "generate"
    CATEGORY      = "prompt"
    OUTPUT_NODE   = False

    def generate(
        self,
        intent: str,
        mode: str,
        engine: str,
        base_url: str,
        model: str,
        temperature: float,
        jpeg_quality: int,
        max_edge: int,
        image_1,
        image_2=None,
        custom_system_prompt: str = "",
    ):
        # ---- 1. Encode images ------------------------------------------
        images_b64 = [
            tensor_to_b64(image_1, max_edge=max_edge, jpeg_quality=jpeg_quality)
        ]
        if mode_uses_two_images(mode):
            if image_2 is None:
                msg = (
                    f"Mode '{mode}' requires a second reference image (image_2) "
                    "but none was provided."
                )
                print(f"[VisionPromptHelper] {msg}")
                return (f"ERROR: {msg}", f"Mode: {mode} | Missing image_2")
            images_b64.append(
                tensor_to_b64(image_2, max_edge=max_edge, jpeg_quality=jpeg_quality)
            )

        # ---- 2. Build system prompt ------------------------------------
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            system_prompt = get_vision_system_prompt(mode, model_name=model)

        user_message = (intent.strip() or "no specific intent") + " /no_think"

        # ---- 3. Resolve engine -----------------------------------------
        try:
            engine_obj = self._resolve_engine(
                engine=engine,
                base_url=base_url,
                model=model,
            )
        except ValueError as exc:
            print(f"[VisionPromptHelper] {exc}")
            return (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")

        # ---- 4. Call LLM -----------------------------------------------
        t0 = time.perf_counter()
        try:
            result = self._call_engine(
                engine_obj,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                images_b64=images_b64,
            )
        except (OpenAIError, OllamaError) as exc:
            print(f"[VisionPromptHelper] {exc}")
            return (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface to user via ShowText
            print(f"[VisionPromptHelper] Unexpected error: {exc}")
            return (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")

        latency = time.perf_counter() - t0

        # ---- 5. Defensive: empty output --------------------------------
        if not result:
            msg = "ERROR: empty LLM output - increase num_predict or check model"
            return (msg, f"Engine: {engine} | Model: {model} | Empty output")

        # ---- 6. Estimate token count (rough: chars / 4) ----------------
        in_chars = len(system_prompt) + len(user_message)
        in_tokens_est = in_chars // 4

        debug_info = (
            f"Engine: {engine} | Model: {model} | Mode: {mode} | "
            f"Images: {len(images_b64)} | InTokens: ~{in_tokens_est} | "
            f"Latency: {latency:.1f}s"
        )

        print(f"[VisionPromptHelper] {debug_info}")
        print(f"[VisionPromptHelper] Output: {result[:200]}...")

        return (result, debug_info)
