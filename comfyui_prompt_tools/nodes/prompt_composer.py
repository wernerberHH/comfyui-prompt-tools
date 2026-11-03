"""PromptComposer node: fuse N description snippets + a user instruction.

This is the third node in the v0.4 prompt-pipeline pattern (see
``docs/prompt-pipeline-pattern.md``). Upstream describe-mode VisionPromptHelpers
produce snippets about face, hair, outfit, background, lighting, etc. The
composer fuses them with the user's scene intent into one coherent prompt in
the chosen output style.

Empty inputs are silently dropped so unused slots in the UI never end up in
the LLM call. The composer always runs through ``BasePromptNode`` so it
shares the same engine selector (Ollama / vLLM) as the other nodes.

Base prompt
-----------
An optional ``base_prompt`` turns the composer from "write a new prompt" into
"edit this prompt": the parts of the base prompt that describe a feature an
input covers are replaced by that input, missing features are added, the rest
stays. ``base_prompt_rest`` picks what happens to the rest:

``keep verbatim``
    One LLM call with ``composer_base_prompt_replace`` returns a JSON edit
    list; :mod:`base_prompt_edits` applies it in Python, so untouched text is
    guaranteed to survive character for character. ``output_style`` is not
    used — replacements follow the form of the base prompt.
``adapt to output style``
    One LLM call with the normal style template plus the "BASE PROMPT mode"
    directive; the LLM writes the whole prompt in the output style.

An empty ``base_prompt`` sends exactly the same system prompt and user
message as before the feature existed.
"""

from __future__ import annotations

import time
from typing import Optional

from ..base_prompt_edits import EditListError, apply_edits, parse_edit_list
from ..catalog import composer_style_labels, load_catalog, resolve_composer_style
from ..engines import OllamaError, OpenAIError
from ..post_processing import strip_llm_noise
from ..prompts import render_template
from ..template_status import describe_resolved_source
from .base_prompt_node import BasePromptNode
from .catalog_validation import validation_message

# Derived views on the catalog, kept for callers that used to read the
# hard-coded lists. The catalog is the source of truth; these are snapshots
# taken at import time, while ``INPUT_TYPES`` reads the catalog live.
OUTPUT_STYLES = [entry.label for entry in load_catalog().composer_styles]

_STYLE_TO_FILE = {
    entry.label: entry.template for entry in load_catalog().composer_styles
}


#: ``base_prompt_rest`` choices. Stored in saved workflows — never rename.
BASE_PROMPT_KEEP = "keep verbatim"
BASE_PROMPT_ADAPT = "adapt to output style"
BASE_PROMPT_REST_CHOICES = [BASE_PROMPT_KEEP, BASE_PROMPT_ADAPT]

#: Template basename for ``keep verbatim``. Not a composer style: it returns
#: a JSON edit list, not a prompt, so it is not listed in the catalog.
BASE_PROMPT_REPLACE_TEMPLATE = "composer_base_prompt_replace"


def _load_composer_system_prompt(
    style: str, model_name: Optional[str] = None
) -> str:
    """Return the rendered composer system prompt for the given output style.

    ``style`` may be the current label or any former label kept as an alias
    in the catalog, so saved workflows keep resolving after a rename.

    Honours the per-model override cascade (see :mod:`prompts` module
    docstring): if ``model_name`` resolves to a known family, an
    ``composer_<style>.<family>.txt`` override takes precedence over the
    default. ``{shared_rules}`` substitution applies to both. Image
    references render with the neutral ``Generic`` wording — the composer
    has no target-model selector of its own.

    Raises ``KeyError`` for unknown styles.
    """
    return render_template(resolve_composer_style(style).template, model_name)


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


def _build_base_prompt_directive() -> str:
    """Render the BASE PROMPT mode directive appended in ``adapt`` mode."""
    return (
        "\n\nBASE PROMPT MODE — the user message contains a BASE PROMPT. Use "
        "it as the foundation of your output instead of composing from "
        "scratch. Where an input description covers the same feature as part "
        "of the base prompt (face, hair, outfit, background, ...), the input "
        "replaces that part. Features from the inputs that the base prompt "
        "does not mention are added. Keep every other element of the base "
        "prompt — scene, setting, lighting, pose, camera, composition — with "
        "its content intact, but phrase the whole prompt in the required "
        "output style. The user instruction, if any, still takes precedence."
    )


def _build_base_prompt_user_message(
    user_instruction: str, base_prompt: str, inputs: list[str]
) -> str:
    """User message for ``adapt`` mode: the normal one plus a BASE PROMPT block."""
    parts: list[str] = []
    instruction = (user_instruction or "").strip()
    parts.append("USER INSTRUCTION:")
    parts.append(instruction if instruction else "(no specific instruction)")
    parts.append("")
    parts.append("BASE PROMPT:")
    parts.append(base_prompt.strip())
    parts.append("")

    non_empty = [s.strip() for s in inputs if s and s.strip()]
    if non_empty:
        parts.append("INPUT DESCRIPTIONS:")
        for i, snippet in enumerate(non_empty, start=1):
            parts.append(f"[{i}] {snippet}")
        parts.append("")

    parts.append(
        "Compose ONE prompt from the base prompt: replace what the inputs "
        "cover, add what is missing, keep the rest, in the required output style."
    )
    parts.append("/no_think")
    return "\n".join(parts)


def _build_replace_user_message(
    user_instruction: str, base_prompt: str, inputs: list[str]
) -> str:
    """User message for ``keep verbatim`` mode (JSON edit list expected)."""
    parts: list[str] = ["BASE PROMPT:", base_prompt.strip(), ""]

    non_empty = [s.strip() for s in inputs if s and s.strip()]
    parts.append("INPUT DESCRIPTIONS:")
    if non_empty:
        for i, snippet in enumerate(non_empty, start=1):
            parts.append(f"[{i}] {snippet}")
    else:
        parts.append("(none)")
    parts.append("")

    instruction = (user_instruction or "").strip()
    parts.append("USER INSTRUCTION:")
    parts.append(instruction if instruction else "(no specific instruction)")
    parts.append("")

    parts.append("Return the JSON edit list only.")
    parts.append("/no_think")
    return "\n".join(parts)


def _build_replace_lora_directive(keywords: list[str]) -> str:
    """Mandatory-tokens directive for ``keep verbatim`` mode.

    The normal directive asks for the tokens "in your output" — here the
    output is a JSON edit list, so the tokens are requested as additions.
    :func:`_ensure_verbatim_tokens` remains the safety net either way.
    """
    if not keywords:
        return ""
    formatted = ", ".join(keywords)
    return (
        "\n\nMANDATORY TOKENS — every token below must end up in the edited "
        "prompt verbatim. For each token the base prompt does not already "
        "contain, add an addition whose insert is exactly that token (with "
        "joining punctuation), placed where it fits or appended at the end. "
        f"Tokens: {formatted}"
    )


def _format_edit_counts(result) -> str:
    """``Replaced: 2 | Added: 1 | Skipped: 0`` with a breakdown when skipped."""
    text = (
        f"Replaced: {result.replaced} | Added: {result.added} | "
        f"Skipped: {result.skipped}"
    )
    if result.skipped:
        text += (
            f" (not found: {result.not_found}, overlap: {result.overlap}, "
            f"invalid: {result.invalid})"
        )
    return text


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
        styles = composer_style_labels()
        return {
            "required": {
                **engine_inputs,
                "user_instruction": ("STRING", {"multiline": True, "default": ""}),
                "output_style":     (styles, {"default": styles[0]}),
            },
            "optional": {
                "input_1": ("STRING", {"forceInput": True}),
                "input_2": ("STRING", {"forceInput": True}),
                "input_3": ("STRING", {"forceInput": True}),
                "input_4": ("STRING", {"forceInput": True}),
                "input_5": ("STRING", {"forceInput": True}),
                "lora_keywords": ("STRING", {"multiline": False, "default": ""}),
                # New inputs go AFTER lora_keywords: saved workflows map
                # widget values by position (tests/unit/test_widget_positions.py).
                "base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_prompt_rest": (
                    BASE_PROMPT_REST_CHOICES,
                    {"default": BASE_PROMPT_KEEP},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("composed_prompt", "debug_info")
    FUNCTION     = "compose"
    CATEGORY     = "prompt"
    OUTPUT_NODE  = False

    @classmethod
    def VALIDATE_INPUTS(cls, output_style):  # noqa: N802 — ComfyUI API contract
        """Accept current labels and the former labels kept as aliases.

        Naming ``output_style`` here makes ComfyUI skip its own "Value not in
        list" check for that input (see the guard in ``execution.py``), which
        is what lets a workflow saved with an old label still queue. Unknown
        values are rejected here instead, with the known ones listed.
        """
        return validation_message(
            ("output_style", resolve_composer_style, output_style)
        )

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
        base_prompt: Optional[str] = "",
        base_prompt_rest: Optional[str] = BASE_PROMPT_KEEP,
    ):
        # ---- 1. Collect non-empty inputs -------------------------------
        raw_inputs = [input_1, input_2, input_3, input_4, input_5]
        inputs = [s for s in raw_inputs if s and s.strip()]

        if base_prompt and base_prompt.strip():
            return self._compose_with_base_prompt(
                engine=engine,
                base_url=base_url,
                model=model,
                temperature=temperature,
                user_instruction=user_instruction,
                output_style=output_style,
                raw_inputs=raw_inputs,
                lora_keywords=lora_keywords,
                base_prompt=base_prompt,
                base_prompt_rest=base_prompt_rest,
            )

        if not user_instruction.strip() and not inputs:
            msg = "ERROR: no user_instruction and no inputs — nothing to compose."
            print(f"[PromptComposer] {msg}")
            return (msg, f"Style: {output_style} | Inputs: 0")

        # ---- 2. Build system prompt and user message -------------------
        try:
            system_prompt = _load_composer_system_prompt(output_style, model_name=model)
            # Names the file the prompt actually came from, so debug_info
            # shows when a local copy shadows the shipped template.
            template_source = describe_resolved_source(
                resolve_composer_style(output_style).template, model_name=model
            )
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
            f"Inputs: {len(inputs)} | Latency: {latency:.1f}s | "
            f"Template: {template_source}"
        )
        if keywords:
            debug_info += (
                f" | LoRA tokens: {len(keywords)} "
                f"({len(missing_tokens)} appended)"
            )

        print(f"[PromptComposer] {debug_info}")
        print(f"[PromptComposer] Output: {result[:200]}...")

        return (result, debug_info)

    # ------------------------------------------------------------------
    # Base prompt
    # ------------------------------------------------------------------
    def _run_llm(
        self,
        engine: str,
        base_url: str,
        model: str,
        temperature: float,
        system_prompt: str,
        user_message: str,
    ) -> tuple[Optional[str], float, Optional[tuple[str, str]]]:
        """Resolve the engine and call it once.

        Returns ``(reply, latency, None)`` on success and
        ``(None, 0.0, (output, debug_info))`` with the node's usual
        ``ERROR: …`` pair on failure.
        """
        try:
            engine_obj = self._resolve_engine(
                engine=engine, base_url=base_url, model=model
            )
        except ValueError as exc:
            print(f"[PromptComposer] {exc}")
            return None, 0.0, (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")

        t0 = time.perf_counter()
        try:
            reply = self._call_engine(
                engine_obj,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
            )
        except (OllamaError, OpenAIError) as exc:
            print(f"[PromptComposer] {exc}")
            return None, 0.0, (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface to user via ShowText
            print(f"[PromptComposer] Unexpected error: {exc}")
            return None, 0.0, (f"ERROR: {exc}", f"Engine: {engine} | Error: {exc}")
        latency = time.perf_counter() - t0

        if not reply:
            msg = "ERROR: empty LLM output - increase num_predict or check model"
            return None, latency, (msg, f"Engine: {engine} | Model: {model} | Empty output")
        return reply, latency, None

    def _compose_with_base_prompt(
        self,
        engine: str,
        base_url: str,
        model: str,
        temperature: float,
        user_instruction: str,
        output_style: str,
        raw_inputs: list,
        lora_keywords: str,
        base_prompt: str,
        base_prompt_rest: Optional[str],
    ):
        """``compose`` for a non-empty ``base_prompt``. See the module docstring."""
        rest = base_prompt_rest or BASE_PROMPT_KEEP
        if rest not in BASE_PROMPT_REST_CHOICES:
            msg = (
                f"ERROR: unknown base_prompt_rest {rest!r} — "
                f"expected one of {BASE_PROMPT_REST_CHOICES}"
            )
            print(f"[PromptComposer] {msg}")
            return (msg, f"Style: {output_style} | Error: {msg}")

        base = base_prompt.strip()
        inputs = [s for s in raw_inputs if s and s.strip()]
        keywords = _parse_lora_keywords(lora_keywords)
        instruction = user_instruction or ""

        if rest == BASE_PROMPT_KEEP:
            return self._compose_keep_verbatim(
                engine, base_url, model, temperature, instruction,
                output_style, raw_inputs, inputs, keywords, base,
            )

        # ---- adapt to output style -------------------------------------
        try:
            system_prompt = _load_composer_system_prompt(output_style, model_name=model)
            template_source = describe_resolved_source(
                resolve_composer_style(output_style).template, model_name=model
            )
        except KeyError as exc:
            print(f"[PromptComposer] {exc}")
            return (f"ERROR: {exc}", f"Style: {output_style} | Error: {exc}")

        system_prompt += _build_base_prompt_directive()
        system_prompt += _build_lora_keywords_directive(keywords)
        user_message = _build_base_prompt_user_message(instruction, base, raw_inputs)

        reply, latency, error = self._run_llm(
            engine, base_url, model, temperature, system_prompt, user_message
        )
        if error is not None:
            return error

        result = strip_llm_noise(reply)
        result, missing_tokens = _ensure_verbatim_tokens(result, keywords)

        debug_info = (
            f"Engine: {engine} | Model: {model} | Style: {output_style} | "
            f"Inputs: {len(inputs)} | Latency: {latency:.1f}s | "
            f"Template: {template_source} | Base prompt: adapted"
        )
        if keywords:
            debug_info += (
                f" | LoRA tokens: {len(keywords)} "
                f"({len(missing_tokens)} appended)"
            )
        print(f"[PromptComposer] {debug_info}")
        print(f"[PromptComposer] Output: {result[:200]}...")
        return (result, debug_info)

    def _compose_keep_verbatim(
        self,
        engine: str,
        base_url: str,
        model: str,
        temperature: float,
        instruction: str,
        output_style: str,
        raw_inputs: list,
        inputs: list,
        keywords: list,
        base: str,
    ):
        """Ask for a JSON edit list and apply it to ``base`` in Python."""
        prefix = (
            f"Engine: {engine} | Model: {model} | Style: {output_style} (not used) | "
            f"Inputs: {len(inputs)}"
        )

        # Nothing to merge in: the base prompt is the answer, no LLM call.
        if not inputs and not instruction.strip():
            result, missing_tokens = _ensure_verbatim_tokens(base, keywords)
            debug_info = (
                f"{prefix} | Base prompt: verbatim | Replaced: 0 | Added: 0 | "
                "Skipped: 0 | No inputs, LLM not called"
            )
            if keywords:
                debug_info += (
                    f" | LoRA tokens: {len(keywords)} "
                    f"({len(missing_tokens)} appended)"
                )
            print(f"[PromptComposer] {debug_info}")
            return (result, debug_info)

        try:
            system_prompt = render_template(BASE_PROMPT_REPLACE_TEMPLATE, model)
        except FileNotFoundError as exc:
            print(f"[PromptComposer] {exc}")
            return (f"ERROR: {exc}", f"{prefix} | Error: {exc}")
        template_source = describe_resolved_source(
            BASE_PROMPT_REPLACE_TEMPLATE, model_name=model
        )
        system_prompt += _build_replace_lora_directive(keywords)
        user_message = _build_replace_user_message(instruction, base, raw_inputs)

        reply, latency, error = self._run_llm(
            engine, base_url, model, temperature, system_prompt, user_message
        )
        if error is not None:
            return error

        try:
            replacements, additions = parse_edit_list(reply)
        except EditListError as exc:
            msg = f"ERROR: unreadable edit list from the LLM ({exc})"
            print(f"[PromptComposer] {msg}. Reply: {reply[:300]}")
            return (
                msg,
                f"{prefix} | Latency: {latency:.1f}s | "
                f"Template: {template_source} | Base prompt: verbatim | Error: {exc}",
            )

        edits = apply_edits(base, replacements, additions)
        result, missing_tokens = _ensure_verbatim_tokens(edits.text, keywords)

        debug_info = (
            f"{prefix} | Latency: {latency:.1f}s | Template: {template_source} | "
            f"Base prompt: verbatim | {_format_edit_counts(edits)}"
        )
        if keywords:
            debug_info += (
                f" | LoRA tokens: {len(keywords)} "
                f"({len(missing_tokens)} appended)"
            )
        print(f"[PromptComposer] {debug_info}")
        print(f"[PromptComposer] Output: {result[:200]}...")
        return (result, debug_info)
