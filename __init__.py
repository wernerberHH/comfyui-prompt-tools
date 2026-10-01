"""ComfyUI custom-node loader entry point.

ComfyUI's loader looks at ``NODE_CLASS_MAPPINGS`` and
``NODE_DISPLAY_NAME_MAPPINGS`` in the top-level ``__init__.py`` of the
custom-node directory. We re-export them from the inner package here so the
implementation stays cleanly modular but ComfyUI still finds everything.
"""

from .comfyui_prompt_tools import (
    PromptComposer,
    PromptHelper,
    TextMux,
    VisionPromptHelper,
)

NODE_CLASS_MAPPINGS = {
    "PromptHelper":       PromptHelper,
    "TextMux":            TextMux,
    "VisionPromptHelper": VisionPromptHelper,
    "PromptComposer":     PromptComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptHelper":       "Prompt Helper",
    "TextMux":            "Text Mux (AI / Manual)",
    "VisionPromptHelper": "Vision Prompt Helper",
    "PromptComposer":     "Prompt Composer",
}

# Tell ComfyUI to serve our JS extension from the ``web/`` subdir.
WEB_DIRECTORY = "./web"

# Register HTTP routes for the Settings UI. Defensive: returns False
# and logs (rather than raising) if PromptServer is not importable.
from .comfyui_prompt_tools.web_api import register_routes as _register_routes
_register_routes()

# Report local system_prompts/*.txt copies that hide a shipped template —
# one summary block, once per process. Read-only: nothing is changed,
# renamed or deleted, and any problem inside is logged, not raised.
from .comfyui_prompt_tools.template_status import (
    report_local_copies as _report_local_copies,
)
_report_local_copies()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
