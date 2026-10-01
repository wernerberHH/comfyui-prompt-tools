"""VisionPromptHelper node: extract or edit prompts from reference images.

Modes come from the prompt catalog and split into edit-modes (transform an
image) and describe-modes (extract one aspect as a snippet). Engine selection
(Ollama vs vLLM) is shared with PromptHelper via :class:`BasePromptNode`.

The ``target_model`` input decides how the generated prompt addresses the
reference images — ``image 1`` for FLUX Kontext, ``Picture 1`` for
Qwen-Image-Edit 2511 and Krea 2, ``<image1>`` for Qwen-Image 2.1. The
dropdown only lists models that reference images inside the prompt at all;
``Generic`` is the default and keeps the wording that predates the catalog.

The order of the images on this node must match their order on the image
model: ``image_1`` is the first image there too.
"""

from __future__ import annotations

import time
from typing import Optional

from ..catalog import (
    default_target_model_label,
    image_target_model_labels,
    resolve_image_target_model,
    resolve_vision_mode,
    vision_mode_labels,
)
from ..engines import OllamaError, OpenAIError
from ..image_io import tensor_to_b64
from ..template_status import describe_resolved_source
from ..vision_prompts import get_vision_system_prompt, mode_uses_two_images
from .base_prompt_node import BasePromptNode
from .catalog_validation import validation_message


class VisionPromptHelper(BasePromptNode):
    """Generate edit instructions or aspect snippets from reference image(s).

    Single-image modes use ``image_1``. Modes declared with ``images: 2`` in
    the catalog also consume ``image_2`` (for ``Outfit Transfer`` that is the
    outfit source). Other modes silently ignore ``image_2`` even if wired up.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI API contract
        engine_inputs = cls._build_engine_inputs()
        modes = vision_mode_labels()
        return {
            "required": {
                "intent":        ("STRING", {"multiline": True, "default": ""}),
                "mode":          (modes, {"default": modes[0]}),
                **engine_inputs,
                "jpeg_quality":  ("INT",    {"default": 85,  "min": 60,  "max": 95,  "step": 5}),
                "max_edge":      ("INT",    {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "image_1":       ("IMAGE",),
            },
            "optional": {
                "image_2":              ("IMAGE",),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
                # NEW INPUTS GO LAST — never between existing ones. The
                # ComfyUI frontend maps a saved workflow's widgets_values by
                # POSITION, not by name, so inserting an input here would
                # shift every value after it. target_model sat before
                # custom_system_prompt in the first draft of this package and
                # swallowed the saved custom prompt of every existing node.
                # Appended, a pre-existing 9-value node keeps its mapping and
                # simply has no stored value for target_model -> Generic.
                "target_model":         (image_target_model_labels(),
                                        {"default": default_target_model_label()}),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("vision_prompt", "debug_info")
    FUNCTION      = "generate"
    CATEGORY      = "prompt"
    OUTPUT_NODE   = False

    @classmethod
    def VALIDATE_INPUTS(cls, mode, target_model=None):  # noqa: N802 — ComfyUI API contract
        """Accept current labels and the former labels kept as aliases.

        Naming ``mode`` and ``target_model`` here makes ComfyUI skip its own
        "Value not in list" check for both inputs (see the guard in
        ``execution.py``), which is what lets a workflow saved with an old
        label still queue. Unknown values are rejected here instead.

        ``target_model`` needs a default: ComfyUI only passes the inputs a
        workflow actually stores, and a workflow saved before this input
        existed has none — without the default that call would raise
        ``TypeError`` during validation. Unset means ``Generic``.

        Each message names the input it belongs to. ComfyUI attaches a
        single non-``True`` return to *every* input in this signature (it
        loops ``for x in input_filtered`` over one result), so without the
        prefix a bad ``target_model`` read as if ``mode`` were the problem.
        """
        return validation_message(
            ("mode", resolve_vision_mode, mode),
            ("target_model", resolve_image_target_model, target_model),
        )

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
        target_model: Optional[str] = None,
        custom_system_prompt: str = "",
    ):
        # ---- 0. Resolve the image-reference wording --------------------
        # Done before any encoding so a bad value fails cheaply. ``None``
        # (node not yet re-saved after the update) yields ``Generic``.
        try:
            target = resolve_image_target_model(target_model)
        except KeyError as exc:
            print(f"[VisionPromptHelper] {exc}")
            return (f"ERROR: {exc}", f"Mode: {mode} | Error: {exc}")

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
        # ``template_source`` names the file the prompt actually came from,
        # so debug_info shows when a local copy shadows the shipped template.
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
            template_source = "(custom_system_prompt input)"
        else:
            system_prompt = get_vision_system_prompt(
                mode, model_name=model, target_model=target.label
            )
            template_source = describe_resolved_source(
                resolve_vision_mode(mode).template, model_name=model
            )

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
            f"Target: {target.label} | "
            f"Images: {len(images_b64)} | InTokens: ~{in_tokens_est} | "
            f"Latency: {latency:.1f}s | "
            f"Template: {template_source}"
        )

        print(f"[VisionPromptHelper] {debug_info}")
        print(f"[VisionPromptHelper] Output: {result[:200]}...")

        return (result, debug_info)
