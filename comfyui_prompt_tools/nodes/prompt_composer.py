"""PromptComposer node: fuse N description snippets + a user instruction.

This is the third node in the v0.4 prompt-pipeline pattern (see
``docs/prompt-pipeline-pattern.md``). Upstream describe-mode VisionPromptHelpers
produce snippets about face, hair, outfit, background, lighting, etc. The
composer fuses them with the user's scene intent into one coherent prompt in
the chosen output style.

Empty inputs are silently dropped so unused slots in the UI never end up in
the LLM call. The composer always runs through ``BasePromptNode`` so it
shares the same engine selector (Ollama / vLLM) as the other nodes.
"""

from __future__ import annotations

import time
from typing import Optional

from ..engines import OllamaError, OpenAIError
from ..post_processing import strip_llm_noise
from ..prompts import render_template
from .base_prompt_node import BasePromptNode

OUTPUT_STYLES = [
    "FLUX.2 natural language",
    "SDXL tag-based",
    "Z-Image compact",
    "Wan 2.2 motion",
]

_STYLE_TO_FILE = {
    "FLUX.2 natural language": "composer_flux2",
    "SDXL tag-based":          "composer_sdxl",
    "Z-Image compact":         "composer_zimage",
    "Wan 2.2 motion":          "composer_wan22",
}


def _load_composer_system_prompt(
    style: str, model_name: Optional[str] = None
) -> str:
    """Return the rendered composer system prompt for the given output style.

    Honours the per-model override cascade (see :mod:`prompts` module
    docstring): if ``model_name`` resolves to a known family, an
    ``composer_<style>.<family>.txt`` override takes precedence over the
    default. ``{shared_rules}`` substitution applies to both.

    Raises ``KeyError`` for unknown styles.
    """
    if style not in _STYLE_TO_FILE:
        raise KeyError(f"Unknown output style: {style!r}")
    return render_template(_STYLE_TO_FILE[style], model_name)


def _build_user_message(user_instruction: str, inputs: list[str]) -> str:
    """Assemble the user-facing payload sent alongside the system prompt.

    Empty input strings are skipped; remaining entries are enumerated 1..N
    so the LLM sees a clean numbered list without gaps.
    """
    parts: list[str] = []
    instruction = (user_instruction or "").strip()
    parts.append("USER INSTRUCTION:")
    parts.append(instruction if instruction else "(no specific instruction)")
    parts.append("")

    non_empty = [s.strip() for s in inputs if s and s.strip()]
    if non_empty:
        parts.append("INPUT DESCRIPTIONS:")
        for i, snippet in enumerate(non_empty, start=1):
            parts.append(f"[{i}] {snippet}")
        parts.append("")

    parts.append("Compose ONE prompt that fuses these inputs with the user instruction, in the required output style.")
    parts.append("/no_think")
    return "\n".join(parts)


def _parse_lora_keywords(raw: str) -> list[str]:
    """Parse a comma-separated LoRA-trigger string into a clean token list.

    Splits on commas, strips per-token whitespace, drops empty entries.
    Preserves the original token order so position-sensitive triggers
    (e.g. LoRAs trained on a specific leading token) remain stable.
    """
    if not raw or not raw.strip():
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _build_lora_keywords_directive(keywords: list[str]) -> str:
    """Render the mandatory-tokens directive appended to the system prompt.

    Empty list yields an empty string so the caller can concatenate
    unconditionally without branching.
    """
    if not keywords:
        return ""
    formatted = ", ".join(keywords)
    return (
        "\n\nMANDATORY TOKENS — these must appear verbatim in your output. "
        "Do not paraphrase, translate, or rewrite them. Weave them naturally "
        "into the prompt where contextually fitting; otherwise append at the "
        f"end. Tokens: {formatted}"
    )


def _ensure_verbatim_tokens(
    output: str, required_tokens: list[str]
) -> tuple[str, list[str]]:
    """Append any required tokens missing from output as a safety net.

    The LLM is instructed to include the tokens verbatim, but compliance
    is not guaranteed. This post-check appends any missing token as a
    comma-separated suffix so the downstream CLIPTextEncode never loses
    a LoRA trigger silently.

    Returns ``(final_output, missing_tokens)`` so callers can surface the
    count in debug information.
    """
    if not required_tokens:
        return output, []
    missing = [t for t in required_tokens if t not in output]
    if not missing:
        return output, []
    base = output.rstrip().rstrip(",")
    suffix = ", ".join(missing)
    return f"{base}, {suffix}", missing


class PromptComposer(BasePromptNode):
    """Fuse 1-5 description snippets and a user instruction into one prompt.

    Inputs ``input_1`` .. ``input_5`` are optional ``STRING`` slots with
    ``forceInput=True`` so they only accept wired connections from upstream
    nodes (typically describe-mode VisionPromptHelpers). Empty / unwired
    slots are dropped before the LLM call.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI API contract
        engine_inputs = cls._build_engine_inputs()
        return {
            "required": {
                **engine_inputs,
                "user_instruction": ("STRING", {"multiline": True, "default": ""}),
                "output_style":     (OUTPUT_STYLES, {"default": "FLUX.2 natural language"}),
            },
            "optional": {
                "input_1": ("STRING", {"forceInput": True}),
                "input_2": ("STRING", {"forceInput": True}),
                "input_3": ("STRING", {"forceInput": True}),
                "input_4": ("STRING", {"forceInput": True}),
                "input_5": ("STRING", {"forceInput": True}),
                "lora_keywords": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("composed_prompt", "debug_info")
    FUNCTION     = "compose"
    CATEGORY     = "prompt"
    OUTPUT_NODE  = False

    def compose(
        self,
        engine: str,
        base_url: str,
        model: str,
        temperature: float,
        user_instruction: str,
        output_style: str,
        input_1: Optional[str] = None,
        input_2: Optional[str] = None,
        input_3: Optional[str] = None,
        input_4: Optional[str] = None,
        input_5: Optional[str] = None,
        lora_keywords: str = "",
    ):
        # ---- 1. Collect non-empty inputs -------------------------------
        raw_inputs = [input_1, input_2, input_3, input_4, input_5]
        inputs = [s for s in raw_inputs if s and s.strip()]

        if not user_instruction.strip() and not inputs:
            msg = "ERROR: no user_instruction and no inputs — nothing to compose."
            print(f"[PromptComposer] {msg}")
            return (msg, f"Style: {output_style} | Inputs: 0")

        # ---- 2. Build system prompt and user message -------------------
        try:
            system_prompt = _load_composer_system_prompt(output_style, model_name=model)
        except KeyError as exc:
            print(f"[PromptComposer] {exc}")
            return (f"ERROR: {exc}", f"Style: {output_style} | Error: {exc}")

        # Parse LoRA-trigger keywords (comma-separated) and append the
        # mandatory-tokens directive to the system prompt if any are given.
        keywords = _parse_lora_keywords(lora_keywords)
        system_prompt += _build_lora_keywords_directive(keywords)

        user_message = _build_user_message(user_instruction, raw_inputs)

        # ---- 3. Resolve engine -----------------------------------------
        try:
            engine_obj = self._resolve_engine(
                engine=engine,
                base_url=base_url,
                model=model,
            )
        except ValueError as exc:
            print(f"[PromptComposer] {exc}")
            return (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")

        # ---- 4. Call LLM -----------------------------------------------
        t0 = time.perf_counter()
        try:
            result = self._call_engine(
                engine_obj,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
            )
        except (OllamaError, OpenAIError) as exc:
            print(f"[PromptComposer] {exc}")
            return (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface to user via ShowText
            print(f"[PromptComposer] Unexpected error: {exc}")
            return (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")

        latency = time.perf_counter() - t0

        # ---- 5. Defensive: empty output --------------------------------
        if not result:
            msg = "ERROR: empty LLM output - increase num_predict or check model"
            return (msg, f"Engine: {engine} | Model: {model} | Empty output")

        # ---- 6. Cleanup + verbatim-token guarantee + debug summary ------
        result = strip_llm_noise(result)
        result, missing_tokens = _ensure_verbatim_tokens(result, keywords)

        debug_info = (
            f"Engine: {engine} | Model: {model} | Style: {output_style} | "
            f"Inputs: {len(inputs)} | Latency: {latency:.1f}s"
        )
        if keywords:
            debug_info += (
                f" | LoRA tokens: {len(keywords)} "
                f"({len(missing_tokens)} appended)"
            )

        print(f"[PromptComposer] {debug_info}")
        print(f"[PromptComposer] Output: {result[:200]}...")

        return (result, debug_info)
