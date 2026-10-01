"""The payment gate.

Every worker command runs through `bash`, which is set to always_ask, so the
session pauses on requires_action and waits indefinitely. This decides what to
do about that.

Shape matters here. An earlier version ALLOWLISTED safe commands and the agent
thrashed -- denied on `ls /mnt/memory`, `browse --help`, `date -u` -- burning
turns probing around the gate. So: deny the small set of things that spend money
or are hard to undo, allow the rest. Same security property, no thrash.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("shot.approvals")

# Anything that could complete a purchase or is otherwise irreversible.
# Matched against the whole command, case-folded.
_SPEND = re.compile(
    r"\b(checkout|place[\s\-_]*order|submit[\s\-_]*order|confirm[\s\-_]*(order|purchase|payment)"
    r"|pay[\s\-_]*now|buy[\s\-_]*now|complete[\s\-_]*purchase|add[\s\-_]*card"
    r"|billing|payment[\s\-_]*method|cvv|card[\s\-_]*number)\b")
# URLs that mean we are standing on a checkout page
_SPEND_URL = re.compile(r"/(checkout|payment|billing|place-order|purchase|cart/checkout)")
# Destructive shell, regardless of context
_DESTRUCTIVE = re.compile(
    r"(\brm\s+-[a-z]*[rf]|\bmkfs\b|\bdd\s+if=|>\s*/dev/sd|\bshutdown\b|\breboot\b"
    r"|\bcurl\b[^|;]*\|\s*(ba)?sh)")


class Decision:
    __slots__ = ("allow", "reason")

    def __init__(self, allow: bool, reason: str = "") -> None:
        self.allow, self.reason = allow, reason

    def __repr__(self) -> str:  # pragma: no cover
        return f"Decision({self.allow}, {self.reason!r})"


def review(command: str) -> Decision:
    """Auto-approve, or escalate to a human.

    Escalation is not a denial: the session parks at $0.00/hr and waits
    indefinitely, so 'ask the owner on the next call' is a real option.
    """
    low = " ".join((command or "").lower().split())
    if not low:
        return Decision(True)
    if _DESTRUCTIVE.search(low):
        return Decision(False, "destructive shell command")
    if _SPEND.search(low) or _SPEND_URL.search(low):
        return Decision(False, "would spend money or complete an order; "
                               "needs spoken confirmation from the owner plus his PIN")
    return Decision(True)
