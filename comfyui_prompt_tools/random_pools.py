"""Helpers for the Random Character modes.

A pool is a string with items separated by either commas or newlines (both
treated equally). Items are stripped of surrounding whitespace; empty items
are dropped.

For age, additionally the format ``min-max`` is recognised and resolved to a
random integer in that inclusive range.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Union


_SPLIT_RE = re.compile(r"[,\n]")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


# Default pools used when the user leaves a slot blank
DEFAULT_ETHNICITY_POOL = (
    "European\nEast Asian\nSouth Asian\nSoutheast Asian\n"
    "Latin American\nAfrican\nMiddle Eastern\nMixed heritage"
)
DEFAULT_AGE_POOL = "22-45"
DEFAULT_MOOD_POOL = (
    "confident\nplayful\nserene\nintense\n"
    "thoughtful\nwarm\nmysterious\nrelaxed"
)
DEFAULT_HAIR_POOL = (
    "long straight\nshoulder-length wavy\nshort pixie cut\n"
    "long curly\nbraided\nbob cut\nlong wavy\nshort textured"
)


def pick_from_pool(pool_string: str) -> Optional[str]:
    """Pick a random item from a comma- or newline-separated pool.

    Returns ``None`` if the pool is empty or whitespace-only.
    """
    if not pool_string or not pool_string.strip():
        return None
    items = [x.strip() for x in _SPLIT_RE.split(pool_string) if x.strip()]
    if not items:
        return None
    return random.choice(items)


def pick_age(age_string: str) -> Optional[Union[int, str]]:
    """Pick an age from a pool.

    Recognises:
      - ``"min-max"`` -> random int in [min, max]
      - ``"22, 28, 35"`` or ``"22\\n28\\n35"`` -> random pick of any token
    """
    if not age_string or not age_string.strip():
        return None
    s = age_string.strip()
    m = _RANGE_RE.match(s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= hi:
            return random.randint(lo, hi)
    return pick_from_pool(s)
