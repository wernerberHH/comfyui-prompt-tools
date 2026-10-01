"""Shared ``VALIDATE_INPUTS`` helper for the catalog-backed combo inputs.

Why the nodes need a ``VALIDATE_INPUTS`` at all
----------------------------------------------
ComfyUI rejects a combo value that is not in the current list *before* any
node code runs. Mode and style labels are frozen in saved workflows, so a
rename would break every workflow using the old label. Naming an input as a
parameter of ``VALIDATE_INPUTS`` makes ComfyUI skip its own list check for
that input and call the classmethod instead, which resolves label **or**
alias through the catalog.

Why the message names the input
-------------------------------
ComfyUI attaches a single non-``True`` return to *every* input in the
signature::

    for x in input_filtered:
        for i, r in enumerate(ret):
            if r is not True ...:
                details = f"{x}" + (f" - {r}" if r is not False else "")

There is no way for one return value to say "this input is fine, that one
is not". Observed consequence before this helper existed: an empty
``target_model`` produced *"mode - Unknown target model: ''"*, which reads
as if the mode were wrong. So every message carries its own input name, and
only the inputs that actually failed are mentioned.

``str()`` on a ``KeyError`` is ``repr()`` of its argument, which wrapped the
whole message in quotes and escaped the inner ones (the smoke-test log read
``Unknown target model: \\x27\\x27``). :func:`_reason` unwraps ``args[0]``
instead.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

#: One input to check: its name, a catalog resolver, and the received value.
InputCheck = Tuple[str, Callable[[Any], Any], Any]


def _reason(exc: KeyError) -> str:
    """Return a ``KeyError``'s message without ``repr`` quoting."""
    if exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def validation_message(*checks: InputCheck) -> Optional[object]:
    """Run the resolvers and build a ``VALIDATE_INPUTS`` return value.

    Returns ``True`` when every value resolves, otherwise one string naming
    each offending input and why it was rejected.
    """
    problems = []
    for name, resolve, value in checks:
        try:
            resolve(value)
        except KeyError as exc:
            problems.append(f"{name}: {_reason(exc)}")
    if problems:
        return " | ".join(problems)
    return True
