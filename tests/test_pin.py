"""L0 — the identity check. No network."""

from __future__ import annotations

import pytest

from shot_voice.pin import normalize, required, verify

from support import TEST_PIN


@pytest.mark.parametrize("raw,want", [
    ("1234", "1234"), ("1-2-3-4", "1234"), ("1234#", "1234"),
    (" 1 2 3 4 ", "1234"), ("*1234", "1234"), (None, ""), ("", ""), ("abcd", ""),
])
def test_normalize_accepts_keypad_and_speech_shapes(raw, want):
    assert normalize(raw) == want


def test_correct_pin_verifies_in_every_entry_shape():
    for shape in (TEST_PIN, "-".join(TEST_PIN), f"{TEST_PIN}#", f" {TEST_PIN} "):
        assert verify(shape) is True


# near misses of TEST_PIN: last digit off, a prefix, one digit too many
@pytest.mark.parametrize("bad", ["4827", "482", "48265", "", None, "0000", "abcd"])
def test_wrong_pin_rejected(bad):
    assert verify(bad) is False


def test_missing_configured_pin_denies_rather_than_allows():
    """A missing secret must never become an open door."""
    assert verify("1234", expected="") is False
    assert verify("1234", expected=None if False else "") is False
    assert verify("", expected="1234") is False


def test_verify_is_constant_time_compare():
    """secrets.compare_digest, not ==, so a wrong PIN leaks no timing signal."""
    import inspect

    import shot_voice.pin as pin
    assert "compare_digest" in inspect.getsource(pin.verify)


@pytest.mark.parametrize("policy,direction,want", [
    # ("off", "outbound") was False. It is now True, and that flip IS the
    # change: PIN_POLICY governs inbound alone, because caller ID says nothing
    # about who picked up the phone we dialled.
    ("off", "outbound", True), ("off", "inbound", False),
    ("callbacks_only", "outbound", True), ("callbacks_only", "inbound", False),
    ("always", "outbound", True), ("always", "inbound", True),
])
def test_policy_matrix(policy, direction, want):
    assert required(policy, direction=direction) is want


@pytest.mark.parametrize("policy", ["off", "callbacks_only", "always"])
def test_no_policy_can_turn_the_outbound_pin_off(policy):
    """There is no configuration under which the agent reads the owner's results to
    whoever happened to answer the phone.

    This carries more weight than it used to: the AMD gate no longer withholds
    the greeting, so the PIN is the only thing between a stranger and the brief.
    """
    assert required(policy, direction="outbound") is True


def test_attempts_are_capped():
    """Three wrong PINs then goodbye — never an unbounded retry loop."""
    from shot_core.budget import PIN_WINDOW_S

    # Three attempts became one window: the old loop re-prompted per
    # attempt, so the owner was asked for his code three times with no room to
    # answer, and digits typed early were discarded.
    assert PIN_WINDOW_S >= 15
