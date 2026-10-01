"""L0 — who is on the line, from anything they type or say, all call long.

Replaces the tests for `pin.challenge`/`GetDtmfTask`, which is deleted. That
collector caused two of the four callback failures the owner hit in one evening --
it refused to be awaited outside `on_enter`, and it raised "the activity that
awaited the inline task is closing" when the caller hung up, which the retry
loop counted as three wrong codes in the same millisecond.

Its real flaw was the window: it only listened once it asked, so a code typed
during the greeting was discarded, and it re-prompted once per attempt.
"""

from __future__ import annotations

import asyncio

import pytest

from shot_voice.identity import Identity, digits_in
from shot_voice.pin import PinResult

from support import TEST_PIN, TEST_PIN_SPOKEN

PIN = TEST_PIN


def make() -> Identity:
    return Identity(expected=PIN)


# -------------------------------------------------------------- transcripts

@pytest.mark.parametrize("said,want", [
    ("1234", "1234"),
    ("one two three four", "1234"),
    ("One, two, three, four.", "1234"),
    ("my code is 12 34", "1234"),
    ("it's one 2 three 4", "1234"),
    ("oh one two three", "0123"),          # "oh" is how people say zero aloud
    ("hello can you hear me", ""),
])
def test_digits_are_read_out_of_speech(said, want):
    assert digits_in(said) == want


# ------------------------------------------------------------- the window

async def test_a_code_typed_before_we_ask_counts():
    """The owner's own request: if he keys it during the ring or over the greeting,
    the agent should already know and not ask."""
    ident = make()
    for d in PIN:
        ident.feed(d)

    assert ident.already_verified is True
    assert await ident.verified(within=0.01) is PinResult.VERIFIED


async def test_a_code_split_across_the_greeting_and_the_prompt_counts():
    ident = make()
    ident.feed(PIN[:2])
    assert ident.already_verified is False
    ident.feed(PIN[2:])
    assert await ident.verified(within=0.01) is PinResult.VERIFIED


async def test_it_returns_the_moment_the_code_arrives():
    """One long window, not three prompts -- so it must not wait it out."""
    ident = make()

    async def press_later():
        await asyncio.sleep(0.05)
        ident.feed(PIN)

    asyncio.create_task(press_later())
    result = await asyncio.wait_for(ident.verified(within=10), timeout=2)
    assert result is PinResult.VERIFIED


async def test_a_stray_keypress_first_does_not_poison_the_code():
    """He hits a menu digit, or the carrier injects one. Matching is on a
    rolling window precisely so the rest of the call still works."""
    ident = make()
    ident.feed("5")
    ident.feed(PIN)
    assert await ident.verified(within=0.01) is PinResult.VERIFIED


async def test_spoken_digits_verify():
    ident = make()
    ident.feed(digits_in(TEST_PIN_SPOKEN))
    assert await ident.verified(within=0.01) is PinResult.VERIFIED


# ------------------------------------------------------------- rejections

async def test_a_wrong_code_is_wrong_not_silence():
    ident = make()
    ident.feed("0000")
    assert await ident.verified(within=0.05) is PinResult.WRONG


async def test_saying_nothing_is_distinguishable_from_getting_it_wrong():
    """shot.calls has to tell "he wasn't there" from "he failed the check" --
    they mean different things when you look back at why a call failed."""
    ident = make()
    assert await ident.verified(within=0.05) is PinResult.NO_INPUT


async def test_a_prefix_of_the_code_is_not_the_code():
    ident = make()
    ident.feed(PIN[:3])
    assert await ident.verified(within=0.05) is PinResult.WRONG


def test_an_empty_configured_pin_denies():
    """A missing secret must never become an open door."""
    ident = Identity(expected="")
    ident.feed(PIN)
    assert ident.already_verified is False


# ----------------------------------------------------------------- wiring

async def test_it_listens_to_the_room_and_the_session():
    """The buffer is opened before the dial, so nothing keyed during the ring
    or over the greeting is lost."""
    from livekit import rtc

    room, session = rtc.Room(), _FakeSession()
    ident = make()
    ident.listen(room, session)
    try:
        for d in PIN:
            room.emit("sip_dtmf_received", rtc.SipDTMF(code=int(d), digit=d,
                                                       participant=None))
        assert ident.already_verified is True
    finally:
        ident.stop()
    assert "sip_dtmf_received" not in room._events or not room._events["sip_dtmf_received"]


async def test_spoken_digits_arrive_through_the_session():
    room, session = _FakeRoom(), _FakeSession()
    ident = make()
    ident.listen(room, session)
    session.emit("user_input_transcribed",
                 _Transcript(f"my code is {TEST_PIN_SPOKEN}", True))
    assert ident.already_verified is True


async def test_a_non_final_transcript_is_ignored():
    """Interim transcripts churn; acting on them would match on noise."""
    room, session = _FakeRoom(), _FakeSession()
    ident = make()
    ident.listen(room, session)
    session.emit("user_input_transcribed", _Transcript(TEST_PIN_SPOKEN, False))
    assert ident.already_verified is False


class _FakeRoom:
    def __init__(self):
        self.handlers = {}

    def on(self, name, cb):
        self.handlers.setdefault(name, []).append(cb)

    def off(self, name, cb):
        self.handlers.get(name, []).remove(cb)


class _FakeSession(_FakeRoom):
    def emit(self, name, ev):
        for cb in self.handlers.get(name, []):
            cb(ev)


class _Transcript:
    def __init__(self, transcript, is_final):
        self.transcript, self.is_final = transcript, is_final
