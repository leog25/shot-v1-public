"""The callback identity check: is this actually the owner?

Caller ID proves nothing about *who answered* an outbound call, so this is the
only real identity signal on that path. It is deliberately separate from the
AMD gate: AMD answers "is a person there", this answers "is it the owner".

Pure functions only. Collecting the digits -- from the keypad or from speech,
across the whole call -- lives in `identity.py`. That split exists because the
collector used to be livekit's `GetDtmfTask`, whose lifecycle produced two
separate live failures and whose listening window opened far too late.
"""

from __future__ import annotations

import re
import secrets
from enum import StrEnum

_DIGITS = re.compile(r"\D")


class PinResult(StrEnum):
    """StrEnum, not (str, Enum): `str()` on the latter yields "PinResult.WRONG",
    which is how AMDCategory once made the disclosure gate never match "human"
    and withhold every result from the real user."""

    VERIFIED = "verified"      # it is the owner
    WRONG = "wrong"            # digits arrived, none of them were the code
    NO_INPUT = "no_input"      # nobody typed or said anything at all
    ABANDONED = "abandoned"    # the line went away before we could tell


def normalize(entry: str | None) -> str:
    """Keypad and speech both land here. '1-2-3-4', 'one two three four' as
    digits, and '1234#' must all compare equal."""
    return _DIGITS.sub("", entry or "")


def verify(entry: str | None, *, expected: str | None = None) -> bool:
    """Constant-time compare against the configured PIN.

    An empty configured PIN is treated as *deny*, not *allow* -- a missing
    secret must never become an open door. `expected` is injectable so that
    branch is directly testable without patching frozen settings.
    """
    if expected is None:
        from shot_core.settings import get_settings

        expected = get_settings().callback_pin.get_secret_value()
    expected = normalize(expected)
    got = normalize(entry)
    if not expected or not got:
        return False
    return secrets.compare_digest(expected, got)


def required(pin_policy: str, *, direction: str) -> bool:
    """Kept as a thin alias; the rule itself lives on Settings so the supervisor
    can apply it without importing the voice tier."""
    from shot_core.settings import Settings

    return Settings.pin_required(
        Settings.model_construct(pin_policy=pin_policy), direction=direction)
