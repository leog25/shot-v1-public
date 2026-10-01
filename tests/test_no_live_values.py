"""L0 — the live PIN, phone numbers and owner's name live in .env and nowhere else.

This repo is meant to be publishable. With PIN_POLICY=callbacks_only an inbound
call is admitted on caller ID alone, so the owner's number beside the agent's is most
of what a spoofed call needs -- and both, with the PIN, used to sit in the
docs and half the tests, because the callback tests could only pass by keying
the real PIN.

Failures name the file, never the value, so the check cannot leak what it guards.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from shot_core.settings import Settings

from support import TEST_NAME, TEST_PIN

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _live() -> Settings:
    # Not get_settings(): conftest swaps TEST_PIN in there, and this needs the real one.
    return Settings()  # type: ignore[call-arg]


def _tracked() -> list[tuple[str, str]]:
    try:
        # --others: a new file is checked before it is ever `git add`ed.
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    files = []
    for name in out.decode().split("\0"):
        p = ROOT / name
        if name and name != "uv.lock" and p.is_file():
            files.append((name, p.read_text(errors="ignore")))
    return files


@pytest.mark.parametrize("field", ["owner_phone_number", "twilio_phone_number"])
def test_no_tracked_file_carries_a_live_number(field):
    """In any spelling: +1XXXXXXXXXX, XXX XXX XXXX, (XXX) XXX-XXXX."""
    national = getattr(_live(), field)[-10:]
    pattern = re.compile(r"[\s().-]*".join(national))
    hits = [name for name, text in _tracked() if pattern.search(text)]
    assert not hits, f"{field.upper()} is written into {hits}; it belongs in .env only"


def test_no_tracked_file_states_the_live_pin():
    """A bare four-digit search would hit dates and ports, so look for it beside
    the word: 'PIN NNNN', 'Your PIN is **NNNN**', 'CALLBACK_PIN=NNNN',
    'live_pin("NNNN")'."""
    pin = _live().callback_pin.get_secret_value()
    if not pin:
        pytest.skip("CALLBACK_PIN is unset")
    pattern = re.compile(rf"(?i)pin[^\d\n]{{0,24}}(?<!\d){re.escape(pin)}(?!\d)")
    hits = [name for name, text in _tracked() if pattern.search(text)]
    assert not hits, f"CALLBACK_PIN is written into {hits}; it belongs in .env only"


def test_the_published_test_pin_is_not_the_live_one():
    assert _live().callback_pin.get_secret_value() != TEST_PIN, (
        "TEST_PIN is in this repo for anyone to read; rotate CALLBACK_PIN")


def test_no_tracked_file_names_the_owner():
    """Whole word, case-sensitive: it is a name. prompts.OWNER and owner_named()
    carry it to the model, so the source never needs to spell it."""
    name = _live().owner_name
    if name == Settings.model_fields["owner_name"].default:
        pytest.skip("OWNER_NAME is unset")
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    hits = [n for n, text in _tracked() if pattern.search(text)]
    assert not hits, f"OWNER_NAME is written into {hits}; it belongs in .env only"


def test_the_published_test_name_is_not_the_live_one():
    assert _live().owner_name != TEST_NAME, (
        "TEST_NAME is in this repo for anyone to read; it cannot be the real one")
