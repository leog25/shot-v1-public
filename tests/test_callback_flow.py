"""L0 — the callback decision tree, every branch, no network.

The outbound path failed on the owner's real phone five times in one evening for four
different root causes, and every one lived in code no test ever entered: the
outbound branch of the job entrypoint had zero coverage, contract or offline.

The tree now lives in `callback_flow.run_callback` behind a six-method protocol,
so all of it is reachable here in milliseconds with no session, no room and no
paid API.
"""

from __future__ import annotations

import pytest

from shot_voice.callback_flow import run_callback
from shot_voice.outbound import MACHINE, UNANSWERED, DialOutcome
from shot_voice.pin import PinResult
from shot_voice.scripted import Spoken

BRIEF = "Jazz Bistro has three shows tonight."


class FakeIO:
    """A scripted CallbackIO that records what the flow did."""

    def __init__(self, outcome: DialOutcome, *, pin: PinResult = PinResult.VERIFIED,
                 speech_plays: bool = True,
                 code_early: bool = False, brief_cut_after: int | None = None) -> None:
        self._outcome = outcome
        self._pin = pin
        self._speech_plays = speech_plays
        self._code_early = code_early
        # the owner talks over the brief. NOT the same as the brief failing to play,
        # and telling them apart is the whole point of Spoken.
        self._brief_cut_after = brief_cut_after
        self.said: list[str] = []
        self.uninterruptible: list[bool] = []
        self.pin_asked = 0
        self.hung_up = 0
        self.handed_back = 0
        # Which variant the wind-down prompt must be: he heard it all, or he cut
        # in and has not heard the rest. Collapsing the two tells him he is done
        # when he is not.
        self.handed_back_delivered: list[bool] = []
        # How many turns had been spoken when the floor was released. The
        # closing line must come AFTER it.
        self.handed_back_at: int | None = None
        # How many turns had been spoken by the time the code was asked for.
        # Everything before this index is pre-verification and must name nothing.
        self.pin_at = 0

    async def dial(self) -> DialOutcome:
        return self._outcome

    async def say(self, script: str, *, limit: float,
                  interruptible: bool = True) -> Spoken:
        self.said.append(script)
        self.uninterruptible.append(not interruptible)
        if not self._speech_plays:
            return Spoken(False, False, "")
        if self._brief_cut_after is not None and BRIEF in script:
            return Spoken(True, True, script[:self._brief_cut_after])
        return Spoken(True, False, script)

    def code_already_entered(self) -> bool:
        return self._code_early

    async def wait_for_code(self, limit: float) -> PinResult:
        self.pin_asked += 1
        self.pin_at = len(self.said)
        return self._pin

    async def hang_up(self) -> None:
        self.hung_up += 1

    async def hand_back(self, *, delivered: bool) -> None:
        self.handed_back += 1
        self.handed_back_at = len(self.said)
        self.handed_back_delivered.append(delivered)

    # helpers the assertions read
    @property
    def leaked(self) -> bool:
        return any(BRIEF in s for s in self.said)

    @property
    def spoke_at_all(self) -> bool:
        return bool(self.said)


async def _run(io: FakeIO, *, require_pin: bool = True, task_ids=None, titles=None):
    return await run_callback(io, brief=BRIEF, require_pin=require_pin,
                              task_ids=task_ids, task_titles=titles)


# --------------------------------------------------------- the happy path

async def test_human_answers_and_gets_the_brief():
    io = FakeIO(DialOutcome(kind="human", amd_speech_s=0.34, amd_reason="short_greeting"))
    rec = await _run(io, task_ids=["t1"])

    assert rec.outcome == "human"
    assert rec.delivered is True
    assert rec.pin_verified is True
    assert io.pin_asked == 1
    assert io.leaked, "a verified human must actually hear the brief"
    assert io.hung_up == 0, "it must not hang up on him after delivering"
    assert io.handed_back == 1, "it should stay on the line, ready for more work"
    assert rec.stayed_on_line is True
    assert rec.task_ids == ["t1"]
    # instrumentation must survive into the durable record
    assert (rec.amd_category, rec.amd_reason, rec.amd_speech_s) == (
        "human", "short_greeting", 0.34)


async def test_pin_can_be_switched_off_for_a_human():
    io = FakeIO(DialOutcome(kind="human"))
    rec = await _run(io, require_pin=False)
    assert rec.outcome == "human" and rec.delivered and rec.stayed_on_line
    assert io.pin_asked == 0


# ------------------------------------------------------- withholding

@pytest.mark.parametrize("kind", sorted(MACHINE))
async def test_a_machine_never_hears_the_results(kind):
    """The highest-consequence path: the agent must not recite the owner's results
    into a voicemail box.

    It DOES hear the greeting now, which names nothing. That is deliberate: AMD
    is not infallible, and a person misclassified as a machine getting dead air
    and a dropped line is a worse failure than a box recording an apology.
    """
    io = FakeIO(DialOutcome(kind=kind, amd_speech_s=3.7), pin=PinResult.VERIFIED)
    rec = await _run(io, titles=["Jazz Bistro schedule"])

    assert rec.outcome == kind
    assert rec.delivered is False
    assert not io.leaked, "the brief reached a machine"
    assert not any("Jazz Bistro schedule" in t for t in io.said), \
        "a machine must not even learn the subject"
    assert len(io.said) == 2, "greeting then sign-off, and nothing else"
    assert io.pin_asked == 0, "a machine must never even be prompted for a PIN"
    assert io.hung_up == 1


@pytest.mark.parametrize("kind", sorted(MACHINE))
async def test_a_machine_is_not_made_to_wait_out_the_pin_window(kind):
    """Twenty-five seconds of an agent talking to itself, recorded onto a
    voicemail box, waiting for a keypress a machine cannot make."""
    io = FakeIO(DialOutcome(kind=kind, amd_speech_s=3.7))
    await _run(io)
    assert io.pin_asked == 0


@pytest.mark.parametrize("kind", sorted(UNANSWERED))
async def test_nobody_home_says_nothing(kind):
    io = FakeIO(DialOutcome(kind=kind, sip_status_code=486))
    rec = await _run(io)

    assert rec.outcome == kind
    assert not io.spoke_at_all, "there is no audio path to speak into"
    assert io.pin_asked == 0
    assert rec.sip_status_code == 486


# ------------------------------------------------- the silent answerer

async def test_a_silent_answer_is_greeted_and_taken_to_the_pin():
    """The owner answers and waits for the agent to speak -- exactly what he asked it
    to do. AMD waits for him. Both waiting is the deadlock that killed every
    polite answer, and it is gone: `uncertain` now goes to the PIN like anything
    else, because the keypad answers "is anyone there" better than a voice does.
    """
    io = FakeIO(DialOutcome(kind="uncertain", amd_speech_s=0.0,
                            amd_reason="no_speech_timeout"))
    rec = await _run(io)

    assert rec.outcome == "human"
    assert rec.delivered and io.leaked
    assert io.pin_asked == 1


async def test_a_line_that_never_keys_a_code_hears_nothing():
    """The safety property the old greet-and-listen branch was carrying, now
    enforced by the PIN instead -- and enforced harder, because a voicemail box
    cannot press a key however long it stays on the line."""
    io = FakeIO(DialOutcome(kind="uncertain", amd_speech_s=0.0),
                pin=PinResult.NO_INPUT)
    rec = await _run(io)

    assert rec.outcome == "pin_failed"
    assert rec.pin_result == "no_input"
    assert not io.leaked
    assert io.spoke_at_all, "never sit in dead air on a live handset"
    assert io.hung_up == 1


@pytest.mark.parametrize("speech", [0.0, None, 2.5])
async def test_uncertain_goes_to_the_pin_whatever_amd_heard(speech):
    """Whether AMD heard nothing, could not measure, or heard speech it could
    not classify, the answer is the same now: greet, and ask for the code."""
    io = FakeIO(DialOutcome(kind="uncertain", amd_speech_s=speech))
    rec = await _run(io)
    assert rec.outcome == "human" and io.pin_asked == 1


async def test_it_greets_exactly_once():
    """This used to greet from two places behind a flag."""
    for io in (FakeIO(DialOutcome(kind="human")),
               FakeIO(DialOutcome(kind="uncertain", amd_speech_s=0.0)),
               FakeIO(DialOutcome(kind="human"), code_early=True)):
        await _run(io)
        assert sum("Open the call." in t for t in io.said) == 1


# ------------------------------------------------------------- the PIN

async def test_wrong_pin_withholds_and_is_terminal():
    io = FakeIO(DialOutcome(kind="human"), pin=PinResult.WRONG)
    rec = await _run(io)

    assert rec.outcome == "pin_failed"
    assert rec.pin_verified is False
    assert not io.leaked
    assert io.spoke_at_all, "say why you are going, do not just drop the line"
    assert io.hung_up == 1


async def test_a_closing_session_is_abandoned_not_a_failed_pin():
    """The 19:08 call: AMD said human, then all three PIN attempts failed in the
    same millisecond with ToolError "the activity that awaited the inline task
    is closing". Recording that as `pin_failed` is wrong twice over -- nobody
    failed an identity check, and pin_failed is terminal, so the result would
    never be delivered."""
    io = FakeIO(DialOutcome(kind="human"), pin=PinResult.ABANDONED)
    rec = await _run(io)

    assert rec.outcome == "abandoned"
    assert rec.outcome != "pin_failed"
    assert rec.delivered is False
    assert not io.leaked


# ------------------------------------------------------------ contract

async def test_every_path_produces_a_record():
    """place_callback blocks on this outcome for four minutes; a path that
    returns nothing leaves the line open and the workflow stuck."""
    for kind in ["human", "uncertain", *MACHINE, *UNANSWERED]:
        rec = await _run(FakeIO(DialOutcome(kind=kind, amd_speech_s=1.0)))
        assert rec.outcome, f"{kind} produced no outcome"
        assert rec.outcome != "no_report", f"{kind} left the default outcome"


async def test_a_brief_that_never_plays_is_not_recorded_as_delivered():
    """The first successful real callback recorded `delivered=True` for a call
    where the agent never read the brief: it was still answering as the
    digit-collecting sub-agent, said "I do need your four digit code", and the
    line dropped. say() returned None regardless, so the flow assumed success
    and the durable record lied."""
    io = FakeIO(DialOutcome(kind="human"), speech_plays=False)
    rec = await _run(io, task_ids=["t1"])

    assert rec.outcome == "human", "a verified human is still a verified human"
    assert rec.delivered is False, "nothing played, so nothing was delivered"


# ------------------------------------------- the greeting the owner asked for

async def test_the_greeting_names_nothing_and_asks_for_the_code():
    """It used to lead with the task title. That is the owner's business and nobody
    else's, and at this point in the call we do not know who is holding the
    phone -- so the greeting says who is calling and asks for the code, and the
    subject waits for the PIN."""
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io, titles=["Jazz Bistro schedule", "jazz dinner times"])

    opening = io.said[0]
    assert "Jazz Bistro schedule" not in opening
    assert "jazz dinner times" not in opening
    assert "four digit code" in opening, "it must ask for the code in the same breath"
    # a goal, not a script -- the wording is the model's, so assert the brief it
    # is given names the points rather than dictating a sentence
    assert "in your own words" in opening.lower()
    assert io.uninterruptible[0] is True, (
        'the greeting was cut off mid-word once -- "please enter your"')


def test_titles_read_as_english():
    """It is spoken aloud, so a list has to sound like a sentence."""
    from shot_voice.callback_flow import _phrase

    assert _phrase(["Jazz Bistro schedule"]) == "Jazz Bistro schedule"
    assert _phrase(["a", "b"]) == "a and b"
    assert _phrase(["a", "b", "c"]) == "a, b and c"
    assert _phrase([]) == "what you asked me to look into"
    assert _phrase([None, "", "a"]) == "a"


async def test_with_no_titles_it_still_says_something_sensible():
    """The fallback phrase lives in the DELIVERY turn now, not the greeting."""
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io, titles=[])
    delivery = next(t for t in io.said if BRIEF in t)
    assert "what you asked me to look into" in delivery
    assert "what you asked me to look into" not in io.said[0]


async def test_a_code_typed_early_skips_the_ask_entirely():
    """The owner's request: if he keys it during the ring or the greeting, do not ask
    for something we already have."""
    io = FakeIO(DialOutcome(kind="human"), code_early=True)
    rec = await _run(io, titles=["Jazz Bistro schedule"])

    assert io.pin_asked == 0, "it asked for a code it already had"
    assert "four digit code" not in io.said[0]
    assert rec.outcome == "human" and rec.delivered and rec.stayed_on_line


async def test_the_brief_stays_interruptible():
    """A user turn during uninterruptible speech is dropped outright, so a long
    brief must not be locked -- eating a real question is worse than being cut
    off, and delivery is verified separately."""
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io)
    # Locate the brief. This used to assert on uninterruptible[-1], which is the
    # OFFER_MORE turn -- so it was testing the wrong index and passing anyway.
    brief_at = next(i for i, t in enumerate(io.said) if BRIEF in t)
    assert io.uninterruptible[brief_at] is False


async def test_no_input_is_recorded_distinctly_from_a_wrong_code():
    io = FakeIO(DialOutcome(kind="human"), pin=PinResult.NO_INPUT)
    rec = await _run(io)
    assert rec.outcome == "pin_failed"
    assert rec.pin_result == "no_input", "shot.calls must tell silence from a bad code"
    assert not io.leaked


async def test_no_prompt_placeholder_ever_reaches_the_model():
    """A literal {what} went out on a live call once: the owner heard "This is Shot
    calling the owner back, and what came back."

    Neither opening takes a format argument any more, so that class is dead by
    construction -- but {what} moved into the delivery turn, so the guard moves
    with it and now covers every path.
    """
    for io in (FakeIO(DialOutcome(kind="human")),
               FakeIO(DialOutcome(kind="human"), code_early=True),
               FakeIO(DialOutcome(kind="uncertain", amd_speech_s=0.0)),
               FakeIO(DialOutcome(kind="machine-vm", amd_speech_s=3.0))):
        await _run(io, titles=["Jazz Bistro schedule"])
        for said in io.said:
            assert "{what}" not in said, "the placeholder reached the model verbatim"
            assert "{brief}" not in said


async def test_the_work_is_never_named_before_the_code_checks_out():
    """THE point of the reorder, as an ordering assertion rather than a wording
    one. `io.pin_at` is how many turns had been spoken when the code was asked
    for; everything before it went out to a line nobody had identified yet."""
    for kind in ("human", "uncertain"):
        io = FakeIO(DialOutcome(kind=kind, amd_speech_s=0.0))
        await _run(io, titles=["Jazz Bistro schedule"])
        for said in io.said[:io.pin_at]:
            assert "Jazz Bistro schedule" not in said
            assert BRIEF not in said
        assert any("Jazz Bistro schedule" in t for t in io.said[io.pin_at:]), \
            "and once it checks out, he is told what it was about"


async def test_a_code_keyed_during_the_ring_still_names_nothing_early():
    """The early-PIN opening is a different prompt, and it used to be the other
    site that interpolated the title."""
    io = FakeIO(DialOutcome(kind="human"), code_early=True)
    await _run(io, titles=["Jazz Bistro schedule"])
    assert "Jazz Bistro schedule" not in io.said[0]
    assert any("Jazz Bistro schedule" in t for t in io.said[1:])


async def test_it_offers_to_keep_working_instead_of_hanging_up():
    """The owner: don't hang up right after the brief -- ask if there's anything else
    and stay ready. He often has the next thing in mind while it's on the line."""
    io = FakeIO(DialOutcome(kind="human"))
    rec = await _run(io, task_ids=["t1"])

    assert rec.stayed_on_line is True
    assert io.hung_up == 0
    assert "anything else" in io.said[-1].lower()
    assert "do not end the call" in io.said[-2].lower(), "the brief must not sign off"


async def test_the_offer_of_more_work_is_its_own_turn():
    """It used to be the last bullet of the brief, which invited the owner to answer
    over the tail -- and an interruption there marked a brief he had heard in
    full as undelivered, leaving the task to be announced again next call."""
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io)

    brief, offer = io.said[-2], io.said[-1]
    assert BRIEF in brief and BRIEF not in offer
    assert "anything else" not in brief.lower(), (
        "the brief must not ask him a question; that is the next turn's job")
    assert "anything else" in offer.lower()


async def test_the_brief_turn_is_one_flowing_turn_not_three_announcements():
    """He heard the first show twice: the model answered his spoken code with a
    summary of its own, then the prepared brief ran. One turn now -- acknowledge
    and deliver -- and the agent holds the floor until it is done."""
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io)

    turn = io.said[-2]
    assert "ONE natural turn" in turn
    # it must not draw attention to the code it just checked
    assert "do not mention codes" in turn.lower()


async def test_a_withheld_call_still_hangs_up():
    """Staying on the line is for the owner, not for a voicemail box.

    `uncertain` is no longer one of these -- it goes to the PIN now -- so the
    second case is a live line that never produced the code.
    """
    io = FakeIO(DialOutcome(kind="machine-vm", amd_speech_s=3.0))
    rec = await _run(io)
    assert rec.stayed_on_line is False
    assert io.hung_up == 1, "a machine must not hold the line open"

    io = FakeIO(DialOutcome(kind="uncertain", amd_speech_s=3.0),
                pin=PinResult.NO_INPUT)
    rec = await _run(io)
    assert rec.stayed_on_line is False
    assert io.hung_up == 1, "a line that never identified itself must not linger"


async def test_an_undelivered_brief_hangs_up_rather_than_lingering():
    io = FakeIO(DialOutcome(kind="human"), speech_plays=False)
    rec = await _run(io)

    assert rec.delivered is False
    assert rec.stayed_on_line is False
    assert io.hung_up == 1


async def test_a_brief_that_never_played_still_says_something_first():
    """Deleting the room without a word is how a callback becomes a mystery on
    his end -- he answered, gave his code, and the line died."""
    io = FakeIO(DialOutcome(kind="human"), speech_plays=False)
    await _run(io)
    assert "could not read out what you called about" in io.said[-1]
    assert BRIEF not in io.said[-1], "the failure line must not leak the findings"


# ---------------------------------------- he talks over it (the 22:06 call)

async def test_being_interrupted_mid_brief_does_not_hang_up_on_him():
    """THE bug. 22:06:42: `interrupted=True, items=1` six seconds into an
    eighty-second brief, then `the brief did not play` and the room was deleted
    on the same millisecond. The owner spoke, got silence -- `_holding` was still set
    -- and then the line dropped. He called back twice.

    An interruption is proof he is ON the line and listening. It is the moment
    to stop talking and answer him, not to hang up."""
    io = FakeIO(DialOutcome(kind="human"), brief_cut_after=12)
    rec = await _run(io, task_ids=["t1"])

    assert io.hung_up == 0, "it hung up on a man who was mid-sentence"
    assert io.handed_back == 1, "the floor must be released so he gets a reply"
    assert rec.stayed_on_line is True
    assert rec.outcome == "human"


async def test_an_interrupted_brief_is_not_recorded_as_heard():
    """`reported_at` means he heard it. We cannot prove he heard the part he
    spoke over, and being told twice is far cheaper than burying a finished
    task forever."""
    io = FakeIO(DialOutcome(kind="human"), brief_cut_after=12)
    rec = await _run(io, task_ids=["t1"])

    assert rec.delivered is False
    assert rec.interrupted is True
    assert rec.heard_chars == 12


async def test_nothing_played_is_distinguishable_from_cut_off():
    """Both were `delivered=False`, and the code answered both by hanging up.
    The durable record has to tell them apart or the next investigation is the
    same guesswork."""
    silent = FakeIO(DialOutcome(kind="human"), speech_plays=False)
    cut = FakeIO(DialOutcome(kind="human"), brief_cut_after=40)

    a, b = await _run(silent), await _run(cut)

    assert (a.delivered, a.interrupted, a.heard_chars) == (False, False, 0)
    assert (b.delivered, b.interrupted, b.heard_chars) == (False, True, 40)
    assert silent.hung_up == 1 and cut.hung_up == 0


async def test_an_interrupted_brief_skips_the_offer_of_more_work():
    """He is already talking. Asking "anything else?" over him is exactly the
    tone-deafness that made the agent feel like a recording."""
    io = FakeIO(DialOutcome(kind="human"), brief_cut_after=12)
    await _run(io)
    assert "anything else" not in io.said[-1].lower()


async def test_hand_back_is_told_whether_the_brief_landed():
    """The wind-down prompt installed afterwards must not tell him he heard it
    all when he cut in six seconds into an eighty-second brief."""
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io)
    assert io.handed_back_delivered == [True]

    io = FakeIO(DialOutcome(kind="human"), brief_cut_after=20)
    rec = await _run(io)
    assert rec.delivered is False
    assert io.handed_back_delivered == [False], (
        "an interrupted brief must hand back with the cut-off prompt")


async def test_the_floor_is_released_before_the_last_line():
    """He talked over "anything else?" and got thirteen seconds of silence.

    The floor is held so the model cannot improvise over a prepared line, and
    the brief IS that line. Holding on through the closing courtesy left a seam:
    the barge-in resolved the handle, the flow handed back, his turn completed,
    and nothing was generated. Releasing first puts a barge-in there back on the
    ordinary path, where it gets an ordinary answer.
    """
    io = FakeIO(DialOutcome(kind="human"))
    await _run(io)

    offer_at = next(i for i, t in enumerate(io.said) if "anything else" in t.lower())
    assert io.handed_back == 1
    assert io.pin_at <= offer_at, "sanity: the offer comes after the code"
    assert offer_at == len(io.said) - 1, "the offer is still the last thing said"
    # and the handback happened BEFORE it was said
    assert io.handed_back_at is not None and io.handed_back_at <= offer_at, (
        "the floor must be released before the closing line, not after")
