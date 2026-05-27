"""TextMux node: switches between AI-enhanced and manual text input."""


class TextMux:
    """Pick between an upstream AI-generated text and a manual override."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 — ComfyUI API contract
        return {
            "required": {
                "manual_text": ("STRING", {"multiline": True, "default": ""}),
                "source": (["AI (PromptHelper)", "Manual"], {"default": "AI (PromptHelper)"}),
            },
            "optional": {
                "ai_text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "switch"
    CATEGORY = "prompt"

    def switch(self, manual_text: str, source: str, ai_text: str | None = None):
        if source == "Manual":
            result = manual_text.strip()
            print(f"[TextMux] Source: Manual | Text: {result[:100]}...")
            return (result,)

        # source == "AI (PromptHelper)"
        if ai_text and ai_text.strip():
            print(f"[TextMux] Source: AI | Text: {ai_text[:100]}...")
            return (ai_text,)

        # Empty AI output -> fall back to manual so workflow doesn't break
        print(f"[TextMux] Source: AI (empty, fallback to manual) | Text: {manual_text[:100]}...")
        return (manual_text.strip(),)
