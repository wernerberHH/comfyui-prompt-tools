"""PromptHelper node: enhances prompts using a local LLM (Ollama or vLLM)."""

from __future__ import annotations

from ..catalog import helper_mode_labels, resolve_helper_mode
from ..engines import OllamaError, OpenAIError
from ..post_processing import strip_llm_noise
from ..prompts import get_system_prompt
from ..random_pools import (
    DEFAULT_AGE_POOL,
    DEFAULT_ETHNICITY_POOL,
    DEFAULT_HAIR_POOL,
    DEFAULT_MOOD_POOL,
    pick_age,
    pick_from_pool,
)
from .base_prompt_node import BasePromptNode
from .catalog_validation import validation_message

#: Catalog id of the catch-all mode that takes its system prompt from the
#: node's own text field instead of a template.
_CUSTOM_SYSTEM_PROMPT_ID = "custom_system_prompt"


def _mode_entry(mode):
    """Resolve a mode label or alias, or ``None`` if it is not in the catalog.

    Unknown values are reported as ``None`` rather than raised so the
    dispatch below can fall through to the regular path, where
    ``get_system_prompt`` produces the error message.
    """
    if not isinstance(mode, str):
        return None
    try:
        return resolve_helper_mode(mode)
    except KeyError:
        return None


def _uses_random_pools(mode) -> bool:
    """True if the mode feeds the random slot pools into the user message."""
    entry = _mode_entry(mode)
    return entry is not None and entry.uses_random_pools


def _is_custom_system_prompt(mode) -> bool:
    """True for the catch-all mode driven by the node's own text field."""
    entry = _mode_entry(mode)
    return entry is not None and entry.id == _CUSTOM_SYSTEM_PROMPT_ID


class PromptHelper(BasePromptNode):
    """Enhances prompts via a local LLM. Backend selectable (Ollama / vLLM)."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI API contract
        engine_inputs = cls._build_engine_inputs()
        modes = helper_mode_labels()
        return {
            "required": {
                "prompt":      ("STRING", {"multiline": True, "default": ""}),
                "mode":        (modes, {"default": modes[0]}),
                **engine_inputs,
                "keep_alive":  ("STRING", {"default": "30s"}),
            },
            "optional": {
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
                # Random Character pools (defaults filled, used in Random modes)
                "ethnicity_pool":    ("STRING", {"multiline": True,  "default": DEFAULT_ETHNICITY_POOL}),
                "age_pool":          ("STRING", {"multiline": False, "default": DEFAULT_AGE_POOL}),
                "mood_pool":         ("STRING", {"multiline": True,  "default": DEFAULT_MOOD_POOL}),
                "hair_pool":         ("STRING", {"multiline": True,  "default": DEFAULT_HAIR_POOL}),
                # Optional pools (no defaults — empty means slot is skipped)
                "lighting_pool":     ("STRING", {"multiline": True, "default": ""}),
                "setting_pool":      ("STRING", {"multiline": True, "default": ""}),
                "outfit_style_pool": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt")
    FUNCTION = "enhance"
    CATEGORY = "prompt"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, mode):  # noqa: N802 — ComfyUI API contract
        """Accept current labels and the former labels kept as aliases.

        Naming ``mode`` here makes ComfyUI skip its own "Value not in list"
        check for that input (see the guard in ``execution.py``), which is
        what lets a workflow saved with an old label still queue. Unknown
        values are rejected here instead, with the known ones listed.
        """
        return validation_message(("mode", resolve_helper_mode, mode))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):  # noqa: N802 — ComfyUI API contract
        # Random-pool modes force re-execution every queue (bypass cache).
        # The flag lives in the catalog, so a renamed label keeps working
        # and a user-defined random mode gets the same treatment.
        mode = kwargs.get("mode", "")
        if not mode and len(args) >= 2:
            mode = args[1]
        if _uses_random_pools(mode):
            return float("NaN")
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_random_character_message(
        prompt: str,
        ethnicity_pool: str,
        age_pool: str,
        mood_pool: str,
        hair_pool: str,
        lighting_pool: str,
        setting_pool: str,
        outfit_style_pool: str,
    ) -> tuple[str, dict]:
        """Build the user message for Random Character modes and return it
        together with the actually chosen slot values (for logging)."""
        slots: list[str] = []
        chosen: dict = {}

        ethnicity = pick_from_pool(ethnicity_pool)
        if ethnicity:
            slots.append(f"Ethnicity: {ethnicity}")
            chosen["ethnicity"] = ethnicity

        age = pick_age(age_pool)
        if age is not None:
            slots.append(f"Age: {age} years old")
            chosen["age"] = age

        mood = pick_from_pool(mood_pool)
        if mood:
            slots.append(f"Mood/expression: {mood}")
            chosen["mood"] = mood

        hair = pick_from_pool(hair_pool)
        if hair:
            slots.append(f"Hair: {hair}")
            chosen["hair"] = hair

        lighting = pick_from_pool(lighting_pool)
        if lighting:
            slots.append(f"Lighting: {lighting}")
            chosen["lighting"] = lighting

        setting = pick_from_pool(setting_pool)
        if setting:
            slots.append(f"Setting: {setting}")
            chosen["setting"] = setting

        outfit = pick_from_pool(outfit_style_pool)
        if outfit:
            slots.append(f"Outfit style: {outfit}")
            chosen["outfit"] = outfit

        constraints = prompt.strip() or "(no specific constraints provided)"

        user_message = (
            f"CONSTRAINTS (must be included in the output):\n{constraints}\n\n"
            f"RANDOM VARIATION SLOTS for this character:\n"
            + "\n".join(f"- {s}" for s in slots)
            + "\n\nGenerate ONE detailed character prompt incorporating ALL slots and constraints."
            + " /no_think"
        )

        return user_message, chosen

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def enhance(
        self,
        prompt: str,
        mode: str,
        engine: str,
        base_url: str,
        model: str,
        temperature: float,
        keep_alive: str = "30s",
        custom_system_prompt: str = "",
        ethnicity_pool: str = "",
        age_pool: str = "",
        mood_pool: str = "",
        hair_pool: str = "",
        lighting_pool: str = "",
        setting_pool: str = "",
        outfit_style_pool: str = "",
    ):
        # ---- 1. Decide system prompt and user message ------------------
        if _uses_random_pools(mode):
            user_message, chosen = self._build_random_character_message(
                prompt, ethnicity_pool, age_pool, mood_pool, hair_pool,
                lighting_pool, setting_pool, outfit_style_pool,
            )
            system_prompt = get_system_prompt(mode, model_name=model)
            print(f"\n[PromptHelper] === {mode} ===")
            print(f"[PromptHelper] Slots: {chosen}")
            constraints_preview = (prompt.strip() or "(none)")[:120]
            print(f"[PromptHelper] Constraints: {constraints_preview}")

        elif _is_custom_system_prompt(mode) and custom_system_prompt.strip():
            if not prompt.strip():
                return ("", "")
            system_prompt = custom_system_prompt.strip()
            user_message = prompt + " /no_think"

        else:
            if not prompt.strip():
                return ("", "")
            system_prompt = get_system_prompt(mode, model_name=model)
            user_message = prompt + " /no_think"

        # ---- 2. Resolve engine and call --------------------------------
        try:
            engine_obj = self._resolve_engine(
                engine=engine,
                base_url=base_url,
                model=model,
                keep_alive=keep_alive,
            )
        except ValueError as exc:
            print(f"[PromptHelper] {exc}")
            return (f"ERROR: {exc}", prompt)

        try:
            enhanced = self._call_engine(
                engine_obj,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
            )
        except (OllamaError, OpenAIError) as exc:
            print(f"[PromptHelper] {exc}")
            return (f"ERROR: {exc}", prompt)
        except Exception as exc:  # noqa: BLE001 — surface to user via ShowText
            print(f"[PromptHelper] Unexpected error: {exc}")
            return (f"ERROR: {exc}", prompt)

        # ---- 3. Defensive: empty output --------------------------------
        if not enhanced:
            print("[PromptHelper] ERROR: empty output (token budget exhausted?)")
            return (
                "ERROR: empty LLM output - increase num_predict or check thinking mode",
                prompt,
            )

        # ---- 4. Cleanup + return ---------------------------------------
        enhanced = strip_llm_noise(enhanced)

        print(f"[PromptHelper] Mode: {mode} | Engine: {engine} | Model: {model}")
        print(f"[PromptHelper] Enhanced: {enhanced[:200]}...")

        return (enhanced, prompt)
