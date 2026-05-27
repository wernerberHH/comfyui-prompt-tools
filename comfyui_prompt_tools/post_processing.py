"""Post-processing helpers for LLM output cleanup."""

from __future__ import annotations

# Common preambles that small LLMs prepend even after we tell them not to
_LEADING_NOISE_PREFIXES = (
    '"',
    "'",
    "Here is",
    "Here's",
    "Sure,",
    "Enhanced prompt:",
)


def strip_llm_noise(text: str) -> str:
    """Remove leading/trailing quotes and common preamble phrases."""
    cleaned = text.strip()
    lowered = cleaned.lower()
    for prefix in _LEADING_NOISE_PREFIXES:
        if lowered.startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            lowered = cleaned.lower()
    return cleaned.strip("\"'").strip()
