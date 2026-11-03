"""Targeted edits to a user-supplied base prompt (PromptComposer, keep verbatim).

In ``keep verbatim`` mode the LLM never rewrites the base prompt. It returns
a small JSON edit list instead, and the functions here apply it in Python::

    {"replacements": [{"find": "<exact excerpt>", "replace": "<new text>"}],
     "additions":    [{"after": "<exact excerpt or empty>", "insert": "<new text>"}]}

Everything the edit list does not touch is copied from the base prompt
character for character — that is the guarantee the mode makes, and it
holds even when the LLM ignores its instructions: an edit that cannot be
located is skipped and counted, never guessed.

Both functions are pure (no I/O, no LLM) so they can be tested in isolation.

Applying the edits
------------------
* ``find`` is searched exactly first. Fallback: case-insensitive with any run
  of whitespace matching any other run — nothing looser. Not found ->
  skipped (``not_found``). Only the first occurrence is used.
* All positions refer to the ORIGINAL base prompt, so one edit never shifts
  another. A replacement whose span overlaps an earlier accepted one is
  dropped (``overlap``) — the earlier entry in the list wins.
* An empty ``replace`` deletes the span. One adjacent list separator
  (``,`` / ``;``) or one surplus space goes with it, so deleting a tag from
  ``a, b, c`` gives ``a, c`` and not ``a, , c``.
* ``additions`` are inserted after their ``after`` anchor. An empty or
  unlocatable anchor appends at the end — the feature is still wanted, only
  its position is unknown. An anchor ending inside a replaced span moves to
  the end of that span.
* An ``insert`` that starts with punctuation (``, text``) is glued to the
  anchor; otherwise it is joined with a space, so two word characters never
  touch. At the very end of the prompt an unpunctuated insert is joined with
  ``", "`` so it does not run into the last word.
* Closing side, language-independent: if an ``insert`` does not end with a
  separator (``, ; : . ! ?``) and the next original character after any
  whitespace is a word or number character, a comma is placed between them
  (``with <insert>, short bob``) — otherwise a mid-phrase insert runs into the
  following description. The original whitespace is kept; ``", "`` is only
  written in full when the original had none. Nothing beyond these two rules
  touches grammar.
* Entries of the wrong shape (not an object, ``find`` / ``insert`` missing or
  empty, a non-string value) are skipped (``invalid``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

#: Punctuation that may open an ``insert`` and is then glued to the left side.
_JOINING_PUNCTUATION = ",;:.!?)"

#: Characters that already close an ``insert`` off from the following text.
_SEPARATORS = ",;:.!?"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", re.DOTALL)


class EditListError(ValueError):
    """The LLM reply contains no usable edit list."""


@dataclass(frozen=True)
class EditResult:
    """Outcome of :func:`apply_edits`."""

    text: str
    replaced: int = 0
    added: int = 0
    not_found: int = 0
    overlap: int = 0
    invalid: int = 0

    @property
    def skipped(self) -> int:
        """Every edit that was not applied, whatever the reason."""
        return self.not_found + self.overlap + self.invalid


# ---------------------------------------------------------------------------
# Reading the LLM reply
# ---------------------------------------------------------------------------
def _strip_reasoning(text: str) -> str:
    """Drop ``<think>`` blocks, including the unbalanced forms.

    A reasoning parser may have removed the opening tag only (everything up to
    ``</think>`` is then reasoning), and a cut-off reply may lack the closing
    tag (everything after ``<think>`` is then reasoning).
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    lowered = cleaned.lower()
    if "</think>" in lowered:
        cleaned = cleaned[lowered.rindex("</think>") + len("</think>"):]
        lowered = cleaned.lower()
    if "<think>" in lowered:
        cleaned = cleaned[: lowered.index("<think>")]
    return cleaned


def _candidates(text: str) -> list[str]:
    """Text regions worth scanning for the JSON object, most specific first."""
    fenced = [m.group(1) for m in _CODE_FENCE_RE.finditer(text)]
    return fenced + [text]


def _first_edit_object(text: str) -> Optional[dict]:
    """Return the first JSON object in ``text`` that looks like an edit list."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (
            "replacements" in value or "additions" in value
        ):
            return value
    return None


def parse_edit_list(reply: str) -> tuple[list, list]:
    """Extract ``(replacements, additions)`` from the raw LLM reply.

    Tolerates ``<think>`` blocks, Markdown code fences and prose around the
    object. A missing key counts as an empty list. Raises
    :class:`EditListError` when no object with ``replacements`` or
    ``additions`` can be found, or when either key is not a list — the caller
    must not fall back to a free rewrite.
    """
    if not reply or not reply.strip():
        raise EditListError("empty reply")
    text = _strip_reasoning(reply)
    obj = None
    for region in _candidates(text):
        obj = _first_edit_object(region)
        if obj is not None:
            break
    if obj is None:
        raise EditListError(
            "no JSON object with 'replacements' or 'additions' in the reply"
        )
    replacements = obj.get("replacements", [])
    additions = obj.get("additions", [])
    if replacements is None:
        replacements = []
    if additions is None:
        additions = []
    for key, value in (("replacements", replacements), ("additions", additions)):
        if not isinstance(value, list):
            raise EditListError(
                f"'{key}' must be a list, got {type(value).__name__}"
            )
    return replacements, additions


# ---------------------------------------------------------------------------
# Locating excerpts
# ---------------------------------------------------------------------------
def locate(haystack: str, needle: str) -> Optional[tuple[int, int]]:
    """Return the ``(start, end)`` span of ``needle`` in ``haystack``, or ``None``.

    Exact match first; fallback ignores case and treats every whitespace run
    as equivalent. Leading / trailing whitespace of ``needle`` is irrelevant
    for the fallback.
    """
    if not needle:
        return None
    index = haystack.find(needle)
    if index != -1:
        return index, index + len(needle)
    words = needle.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, haystack, re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()


# ---------------------------------------------------------------------------
# Applying the edits
# ---------------------------------------------------------------------------
@dataclass
class _Span:
    start: int
    end: int
    text: str  # replacement text, "" = delete


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _extend_deletion(base: str, span: _Span, lower: int, upper: int) -> None:
    """Widen a deletion by one separator, staying within ``[lower, upper]``."""
    after = re.match(r"\s*[,;]\s*", base[span.end:upper])
    if after:
        span.end += after.end()
        return
    before = re.search(r"\s*[,;]\s*$", base[lower:span.start])
    if before:
        span.start = lower + before.start()
        return
    # Plain words: drop one of the two spaces that would otherwise meet.
    if span.end < upper and base[span.end].isspace() and (
        span.start == lower or base[span.start - 1].isspace()
    ):
        span.end += 1
    elif span.start > lower and base[span.start - 1].isspace() and (
        span.end == len(base)
    ):
        span.start -= 1


def _replacement_text(base: str, start: int, end: int, replace: str) -> str:
    """Keep the whitespace that framed the matched excerpt."""
    matched = base[start:end]
    lead = matched[: len(matched) - len(matched.lstrip())]
    trail = matched[len(matched.rstrip()):]
    return f"{lead}{replace.strip()}{trail}"


def _glue(left: str, insert: str, right: str, at_end: bool) -> str:
    """Join ``insert`` between ``left`` and ``right`` with sensible spacing."""
    if insert[0] in _JOINING_PUNCTUATION:
        piece = insert
        left = left.rstrip(" ")
    elif not left:
        piece = insert
    elif at_end and left.rstrip()[-1:] not in _JOINING_PUNCTUATION:
        left = left.rstrip()
        piece = ", " + insert
    elif left[-1].isspace():
        piece = insert
    else:
        piece = " " + insert
    following = right.lstrip()
    if (
        following
        and _is_word_char(following[0])
        and insert[-1] not in _SEPARATORS
    ):
        # Mid-phrase insert followed by more description: keep the two apart.
        piece += "," if right[0].isspace() else ", "
    elif right and not right[0].isspace() and right[0] not in _JOINING_PUNCTUATION:
        piece += " "
    return left + piece


def _is_word_char(char: str) -> bool:
    """Letter or digit in any script (``str.isalnum``), or an underscore."""
    return char.isalnum() or char == "_"


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def apply_edits(base: str, replacements: list, additions: list) -> EditResult:
    """Apply the LLM edit list to ``base``. See the module docstring."""
    not_found = overlap = invalid = 0

    # ---- replacements: locate in the original, reject overlaps ----------
    spans: list[_Span] = []
    for entry in replacements or []:
        if not isinstance(entry, dict):
            invalid += 1
            continue
        find, replace = entry.get("find"), entry.get("replace", "")
        if not _is_text(find) or not find.strip() or not _is_text(replace):
            invalid += 1
            continue
        found = locate(base, find)
        if found is None:
            not_found += 1
            continue
        start, end = found
        if any(_overlaps(start, end, s.start, s.end) for s in spans):
            overlap += 1
            continue
        spans.append(_Span(start, end, replace))

    spans.sort(key=lambda s: s.start)
    # Right to left, so a deletion sees its right neighbour's final extent:
    # in "a, b, c" deleting b and c, c takes ", " before it and b then takes
    # the ", " before it as well.
    for index in reversed(range(len(spans))):
        span = spans[index]
        if span.text.strip():
            span.text = _replacement_text(base, span.start, span.end, span.text)
            continue
        span.text = ""
        lower = spans[index - 1].end if index > 0 else 0
        upper = spans[index + 1].start if index + 1 < len(spans) else len(base)
        _extend_deletion(base, span, lower, upper)

    # ---- additions: resolve insertion points in the original ------------
    inserts: list[tuple[int, int, str]] = []  # (position, order, text)
    appended: list[str] = []
    for order, entry in enumerate(additions or []):
        if not isinstance(entry, dict):
            invalid += 1
            continue
        insert, after = entry.get("insert"), entry.get("after", "")
        if not _is_text(insert) or not insert.strip():
            invalid += 1
            continue
        if after is None:
            after = ""
        if not _is_text(after):
            invalid += 1
            continue
        found = locate(base, after) if after.strip() else None
        if found is None:
            appended.append(insert.strip())
            continue
        position = found[1]
        for span in spans:
            if span.start < position < span.end:
                position = span.end
        inserts.append((position, order, insert.strip()))

    # ---- assemble ---------------------------------------------------------
    events: list[tuple[int, int, str, Any]] = []
    for span in spans:
        events.append((span.start, 1, "span", span))
    for position, order, text in inserts:
        events.append((position, 0, "insert", (order, text)))
    events.sort(key=lambda e: (e[0], e[1], e[3][0] if e[2] == "insert" else 0))

    out = ""
    cursor = 0
    for position, _, kind, payload in events:
        if kind == "span":
            out += base[cursor:payload.start] + payload.text
            cursor = payload.end
        else:
            out += base[cursor:position]
            cursor = position
            out = _glue(out, payload[1], base[cursor:], at_end=False)
    out += base[cursor:]

    for text in appended:
        out = _glue(out, text, "", at_end=True)

    return EditResult(
        text=out,
        replaced=len(spans),
        added=len(inserts) + len(appended),
        not_found=not_found,
        overlap=overlap,
        invalid=invalid,
    )
