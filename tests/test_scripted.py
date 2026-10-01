"""L0 — making the agent say a prepared line, and knowing whether it did.

Two live failures sit behind this file.

The agent was asked to read the owner his finished results and instead said "No, I
don't need it. Your identity is already verified. What would you like to do
next?" -- because the OpenAI plugin concatenates a per-response instruction onto
the session prompt, so a 300-character script arrived as 11% of a 2,500-char
prompt telling it to "just talk with the owner". Replaying that call, the diluted form
lost 3 times out of 3; emptying the session prompt won 3 out of 3.

And that call was recorded as `delivered=True`, because `wait_for_playout()`
returns normally after an interruption. A record that lies is worse than none.
"""

from __future__ import annotations

import pytest

from shot_voice.scripted import scripted, spoken_from

SCRIPT = "Say exactly this."
REAL = "the real instructions, two thousand characters of them"


class _Msg:
    type = "message"
    role = "assistant"

    def __init__(self, text, interrupted=False):
        self.text_content, self.interrupted = text, interrupted


class _Handle:
    def __init__(self, items=(), *, interrupted=False, exc=None):
        self.chat_items = list(items)
        self.interrupted = interrupted
        self._exc = exc

    def done(self):
        return True

    def exception(self):
        return self._exc

    def __await__(self):
        async def _done():
            return self
        return _done().__await__()


class _Agent:
    def __init__(self):
        self.instructions = REAL
        self.seen: list[str] = []

    async def update_instructions(self, text):
        self.instructions = text
        self.seen.append(text)


class _Session:
    def __init__(self, handle, agent):
        self._handle, self._agent = handle, agent
        self.calls: list[dict] = []

    def generate_reply(self, *, instructions, allow_interruptions=True):
        # what the session prompt was AT THE MOMENT of the call is the whole point
        self.calls.append({"instructions": instructions,
                           "session_prompt": self._agent.instructions,
                           "interruptible": allow_interruptions})
        return self._handle


# ------------------------------------------------- the dilution, defeated

async def test_the_session_prompt_is_empty_while_the_script_runs():
    """`if is_given(instructions) and self._instructions:` -- an empty session
    prompt is falsy, so the plugin skips the concatenation and the script is the
    entire instruction."""
    agent = _Agent()
    session = _Session(_Handle([_Msg("said it")]), agent)

    await scripted(agent, session, SCRIPT, limit=5)

    assert session.calls[0]["session_prompt"] == "", (
        "the real prompt was still in place; the script gets diluted")
    assert session.calls[0]["instructions"] == SCRIPT


async def test_the_real_instructions_come_back():
    agent = _Agent()
    await scripted(agent, _Session(_Handle([_Msg("ok")]), agent), SCRIPT, limit=5)
    assert agent.instructions == REAL


async def test_they_come_back_even_when_the_turn_blows_up():
    """A session left on an empty prompt stays empty for every later turn and is
    re-pushed on any agent handoff."""
    agent = _Agent()

    class Boom(_Session):
        def generate_reply(self, **kw):
            raise RuntimeError("kaboom")

    await scripted(agent, Boom(None, agent), SCRIPT, limit=5)
    assert agent.instructions == REAL


async def test_restoring_happens_after_the_turn_not_before():
    """generate_reply returns before anything is sent -- response.create is
    issued later -- so restoring early puts the long prompt back on the wire
    first and the dilution returns."""
    agent = _Agent()
    session = _Session(_Handle([_Msg("ok")]), agent)
    await scripted(agent, session, SCRIPT, limit=5)
    assert agent.seen == ["", REAL], f"restore ordering wrong: {agent.seen}"


@pytest.mark.parametrize("interruptible", [True, False])
async def test_interruptibility_is_passed_through(interruptible):
    """The "enter your code" prompt was cut off mid-word -- "please enter
    your" -- when he spoke over it."""
    agent = _Agent()
    session = _Session(_Handle([_Msg("ok")]), agent)
    await scripted(agent, session, SCRIPT, limit=5, interruptible=interruptible)
    assert session.calls[0]["interruptible"] is interruptible


# --------------------------------------------------- delivery is a fact

def test_a_full_playout_is_delivered():
    spoken = spoken_from(_Handle([_Msg("the whole brief")]))
    assert spoken.delivered is True
    assert spoken.text == "the whole brief"
    assert bool(spoken) is True


def test_an_interrupted_handle_is_not_delivered():
    assert spoken_from(_Handle([_Msg("half a br")], interrupted=True)).delivered is False


def test_a_partially_played_message_is_not_delivered():
    """chat_items carries the transcript truncated at the PLAYBACK position, so
    a partial message is what he actually heard -- not delivery."""
    assert spoken_from(_Handle([_Msg("half a br", interrupted=True)])).delivered is False


def test_speech_that_never_reached_the_speakers_is_not_delivered():
    """Skipped speech produces no chat item at all."""
    spoken = spoken_from(_Handle([]))
    assert spoken.delivered is False and spoken.text == ""


def test_a_failed_turn_is_not_delivered():
    assert spoken_from(_Handle([_Msg("x")], exc=RuntimeError("timed out"))).delivered is False


def test_no_handle_is_not_delivered():
    assert spoken_from(None).delivered is False


def test_a_handle_that_is_still_playing_does_not_blow_up():
    """SpeechHandle.exception() RAISES InvalidStateError when the speech is not
    finished, and asyncio.wait_for timing out leaves exactly that. Reading it
    unconditionally turned "the brief is still playing" into a crash that took a
    live call down and recorded outcome=error."""
    class NotDone(_Handle):
        def done(self):
            return False

        def exception(self):
            raise AssertionError("must not be called on an unfinished handle")

    spoken = spoken_from(NotDone([_Msg("half the brief")]))
    assert spoken.delivered is False


# --------------------------------- three outcomes, not two (the 22:06 call)

def test_nothing_played_and_cut_off_are_different_facts():
    """Collapsing these into one `delivered` flag is what hung up on the owner six
    seconds into an eighty-second brief: "he interrupted me" and "nothing came
    out of the speaker" were the same value, and the caller answered both by
    deleting the room."""
    nothing = spoken_from(_Handle([]))
    cut = spoken_from(_Handle([_Msg("So, Harbor Lights is the nightc", interrupted=True)]))
    whole = spoken_from(_Handle([_Msg("the whole brief")]))

    assert (nothing.started, nothing.interrupted) == (False, False)
    assert (cut.started, cut.interrupted) == (True, True)
    assert (whole.started, whole.interrupted) == (True, False)
    assert not nothing.delivered and not cut.delivered and whole.delivered


def test_what_he_heard_is_kept_even_when_it_was_cut_off():
    """chat_items is truncated at the PLAYBACK position, so this is the only
    record of how much of his results actually reached him."""
    cut = spoken_from(_Handle([_Msg("So, Harbor Lights is the nightc", interrupted=True)]))
    assert cut.text == "So, Harbor Lights is the nightc"


def test_an_interruption_on_the_handle_counts_even_if_the_message_looks_clean():
    """handle.interrupted and message.interrupted are set on different paths;
    trusting either alone has under-reported an interruption."""
    assert spoken_from(_Handle([_Msg("clean")], interrupted=True)).interrupted is True
