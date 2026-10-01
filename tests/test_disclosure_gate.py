"""L0 — the gate that decides whether the owner's results are read onto a line.

This is the highest-consequence branch in the system: get it wrong the wrong way
and the agent recites his task results into a stranger's voicemail box.

These tests previously asserted a `disclosure_allowed` that production never
called -- the live gate was a bare `if outcome.kind != "human"` in the job
entrypoint, which nothing tested. The function is now the single call site in
`callback_flow.run_callback`, and `test_callback_flow.py` proves the flow
actually consults it.
"""

from __future__ import annotations

import pytest

from shot_supervisor.callbacks import RETRYABLE, TERMINAL
from shot_voice.outbound import (
    MACHINE,
    UNANSWERED,
    DialOutcome,
    DialRequest,
    classify_sip_error,
    disclosure_allowed,
)

ANSWERED_NOT_HUMAN = sorted(MACHINE) + ["uncertain"]


# ----------------------------------------------------------- the hard rule

@pytest.mark.parametrize("kind", sorted(MACHINE))
def test_a_detected_machine_never_discloses(kind):
    """The invariant the gate exists for, and the reason it stays written out
    even though DISCLOSABLE already excludes these.

    Note what this no longer covers: the GREETING. It names nothing, so a
    detected machine hears it -- deliberately, because AMD is not infallible and
    a person misclassified as a machine getting dead air and a dropped line is
    worse than a box recording an apology. What a machine never gets is the
    results, and it is never even asked for a code.
    """
    assert disclosure_allowed(DialOutcome(kind=kind)) is False


@pytest.mark.parametrize("kind", sorted(UNANSWERED))
def test_nothing_discloses_before_anyone_answers(kind):
    assert disclosure_allowed(DialOutcome(kind=kind)) is False


def test_human_discloses():
    assert disclosure_allowed(DialOutcome(kind="human")) is True


@pytest.mark.parametrize("speech", [0.0, None, 2.5])
def test_uncertain_reaches_the_pin_whatever_amd_heard(speech):
    """`uncertain` used to need a live voice answering our greeting.

    That rule is gone with the thing that made it necessary. Nothing about the
    work is said before the PIN now, so AMD no longer has to be certain before
    the agent may speak -- and the PIN is strictly stronger evidence than "a
    voice answered", because a voicemail box cannot press a key. The old rule
    deadlocked: the owner answers and waits for the agent, AMD waits for the owner, six
    seconds later every polite answer died as `uncertain`.
    """
    assert disclosure_allowed(DialOutcome(kind="uncertain", amd_speech_s=speech)) is True


def test_unknown_categories_fail_closed():
    """Why the gate is an allowlist and not "anything that is not a machine".

    A new AMDCategory shipped in a future SDK release must withhold by default.
    A denylist would silently open for it.
    """
    assert disclosure_allowed(DialOutcome(kind="something-new")) is False


def test_the_gate_is_a_table_of_categories():
    """The whole rule, in one place, so a change to it is visible in one diff."""
    allowed = {"human", "uncertain"}
    for kind in sorted(MACHINE | UNANSWERED | {"human", "uncertain", "something-new"}):
        assert disclosure_allowed(DialOutcome(kind=kind)) is (kind in allowed), kind


# ------------------------------------------------------------- plumbing

def test_amd_category_is_unwrapped_to_its_value():
    """AMDCategory is a (str, Enum): str() yields "AMDCategory.HUMAN", so the
    gate would never match "human" and every callback would withhold from the
    real user."""
    from livekit.agents import AMDCategory

    assert AMDCategory.HUMAN.value == "human"
    assert str(AMDCategory.HUMAN) != "human"
    assert disclosure_allowed(DialOutcome(kind=AMDCategory.HUMAN.value)) is True


@pytest.mark.parametrize("code,kind", [
    (486, "busy"), (600, "busy"), (603, "rejected"),
    (480, "no_answer"), (408, "no_answer"), (404, "trunk_error"), (999, "trunk_error"),
])
def test_sip_errors_classify(code, kind):
    class FakeSipError(Exception):
        sip_status_code = code

    out = classify_sip_error(FakeSipError())
    assert out.kind == kind
    assert out.answered is False, "a call that errored was never answered"


def test_dial_request_parse():
    assert DialRequest.parse(None) is None
    assert DialRequest.parse("") is None
    assert DialRequest.parse("not json") is None
    assert DialRequest.parse('{"brief":"x"}') is None          # no to_number
    d = DialRequest.parse('{"to_number":"+15550000000","brief":"b",'
                          '"callback_id":"c1","task_id":"t1"}')
    assert (d.to_number, d.brief, d.callback_id, d.task_id) == (
        "+15550000000", "b", "c1", "t1")
    assert d.require_pin is True
    assert DialRequest.parse(
        '{"to_number":"+15550000000","require_pin":false}').require_pin is True, \
        "dispatch metadata must not be able to turn the identity check off"


# --------------------------------------------------------- retry policy

def test_nothing_ever_redials():
    """Automatic retries rang the owner three times in one evening while a systematic
    bug went unfixed, each attempt failing exactly like the last. A redial is
    the one failure mode with a real-world cost, so it is now manual only."""
    assert RETRYABLE == set()


def test_delivered_and_failed_identity_stay_terminal():
    assert "human" in TERMINAL
    assert "pin_failed" in TERMINAL
