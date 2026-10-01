"""Every timeout on the callback path, in one place.

These used to be seven hand-written constants across four files that had to nest
correctly by coincidence. They did not: the worst-case dial was 59s against a
60s gate, so a slow answer silently skipped the PIN entirely and the caller sat
on an open line while the job waited out a 150s poll.

Plain ints and one function on purpose. The voice worker forks a subprocess per
job and pays import cost against a 10s `initialize_process_timeout`, so nothing
heavy may be imported here.

`test_budget.py` asserts these nest. Change a number, and the test tells you
whether the chain still fits.
"""

from __future__ import annotations

# --- dialling -------------------------------------------------------------
# A US mobile rolls to voicemail at ~20-25s, and the LiveKit SDK silently pins
# 30s if you leave it unset -- so we would reliably reach voicemail.
RING_S = 20

# AMD's own detection window, and the outer belt for livekit/agents#6996, where
# AMD stalls 17-20s on ~7% of answered calls and it lands as dead air.
#
# MIRRORED as env-overridable settings fields (AMD_DETECTION_TIMEOUT_SECONDS,
# AMD_NO_SPEECH_SECONDS, AMD_WAIT_SECONDS), and production reads the SETTINGS
# field, never these -- same shape as RING_S. outbound.py hardcoded 8.0/6.0/9.0
# for months while this table claimed to own them: three numbers that agreed
# only by coincidence, which is the exact failure this file exists to prevent.
# test_budget.py asserts they agree, so they cannot drift apart again.
AMD_DETECT_S = 8.0

# How long a SILENT line is given before AMD gives up and says `uncertain`.
# It is a give-up-if-silent timer only: once any speech starts, AMD_DETECT_S
# governs instead, so this is not a cap on classifying a talkative voicemail.
#
# It is also dead air. AMD holds speech-playout authorization for the whole of
# its context manager, so nothing can be said until it settles -- and a person
# who answers and politely waits for the agent (which is what the callback asks
# of the owner) produces exactly no speech. Lowering it shortens that silence and
# costs detection only for a box that stays quiet through its own opening.
# Measured against real answers: machine-vm classified on 5.01s and 5.66s of
# speech, machine-ivr on 3.70s and 3.75s -- all of them talking well before
# this fires. Was 6.0; 3.0 halves the dead air and keeps a second of margin.
AMD_NO_SPEECH_S = 3.0

# Outer belt for #6996. Summed into worst_case_call_s(), so it is NOT free:
# raising it eats the DBOS rendezvous headroom, and past ~12.0 a slow-but-
# healthy call starts being recorded as `no_report`. test_budget.py pins that.
AMD_WAIT_S = 9.0

# --- speaking -------------------------------------------------------------
GREETING_PLAYOUT_S = 20.0
SIGNOFF_PLAYOUT_S = 15.0
# A ~400-character brief read aloud runs 35-45s, so 40 was a coin flip --
# and overrunning it did not just truncate, it crashed the call.
BRIEF_PLAYOUT_S = 100.0

# "Anything else you want me to pick up?" -- one short sentence, and its own
# turn. It used to be the last bullet of the brief, which invited the owner to answer
# over the tail; an interruption there marked a fully-heard brief as undelivered.
OFFER_MORE_S = 10.0

# The inbound greeting that leads with finished work. Not on the callback path,
# so neither this nor NEWS_OFFER_S is part of worst_case_call_s().
NEWS_GREETING_S = 25.0

# How long the opening line has to actually START before we conclude it never
# will. Not a playout budget -- a "did any audio come out at all" watchdog.
# Measured against the live worker: audio begins ~0.2s after the pipeline is
# ready, three runs of three. Three seconds is an order of magnitude of slack,
# and it has to stay short: the whole point is to notice and say it again while
# the caller is still on the line, not to discover it afterwards.
GREETING_START_S = 3.0

# Turn TWO of that greeting -- the offer to run through the rest of the day.
# Its own turn, and its delivery is deliberately NOT measured: appended to the
# news it would invite the owner to answer over the tail, and an interruption there
# marks work he heard in full as undelivered, which leaves the task unreported
# and re-announced on his next call. Same shape and same number as OFFER_MORE_S
# on the callback path, for exactly the same reason.
NEWS_OFFER_S = 10.0

# --- PIN ------------------------------------------------------------------
# ONE window, not three attempts. The old loop re-prompted per attempt, so the owner
# was asked for his code three times with no room to answer. Digits are buffered
# from the moment the call connects, so this only bounds how long we wait after
# asking.
PIN_WINDOW_S = 25.0

# After the brief the agent stays on the line, ready for more work. The owner usually
# just hangs up, and the room closing ends the job -- this only covers the case
# where he does not, so an open realtime session cannot run indefinitely.
IDLE_HANGUP_S = 180.0

# --- inline tools ---------------------------------------------------------
# One Brave call. Measured ~0.4s warm, 0.9s cold; a generous ceiling that still
# leaves room for the internal retry inside a single conversational turn.
WEB_SEARCH_ATTEMPT_S = 2.5
# Both attempts plus the reformulation, end to end. The bucket table at the top
# of tools.py says anything over 5s should have been delegated instead -- this
# is the ceiling that keeps web_search an inline tool rather than a worker task.
WEB_SEARCH_TOTAL_S = 5.0

# --- Linear ---------------------------------------------------------------
# ONE GraphQL POST returning both "assigned to the owner" and "everything still open"
# via aliased root fields. Measured 0.28-0.42s against the real API. Budgeted
# like a single Brave call rather than like web_search's retrying pair: there is
# no retry, because a second identical query answers nothing.
LINEAR_QUERY_S = 2.5

# check_my_day end to end. The Linear POST and GET /internal/tasks run
# CONCURRENTLY, so this bounds the SLOWER LEG, not the sum. Under the 5s bucket
# ceiling at the top of tools.py: anything that cannot answer inside that
# belongs in a background task rather than inside a conversational turn.
CHECK_MY_DAY_S = 3.0

# Writing to Linear, end to end -- BOTH round trips. `teamId` is the only
# required field on IssueCreateInput, so the first write in a process resolves
# the team and then mutates: two POSTs at the measured 0.28-0.42s each. Later
# writes are one, because the team id is cached for the life of the process.
# Sits in the same bucket as check_my_day and under the same 5s ceiling.
LINEAR_WRITE_S = 3.0

# --- supervisor -----------------------------------------------------------
# The rendezvous the voice job reports into. Must exceed the worst case below
# or a slow-but-healthy call is recorded as `no_report`.
DBOS_RECV_S = 240

SUPERVISOR_HTTP_S = 5.0


def worst_case_call_s() -> float:
    """The longest a healthy callback can legitimately take.

    Ring, then classify, then greet and ask for the code in one turn, then wait
    out the code window, then read the brief, then offer to pick up more work.
    Anything above this is a hang, not a slow call.

    Two terms came out when the disclosure moved behind the PIN: the reply
    window (we no longer greet and then listen to find out who is there -- the
    keypad answers that) and a second SIGNOFF_PLAYOUT_S for the separate "enter
    your code" ask, which the one-turn greeting made unnecessary.
    """
    return (
        RING_S
        + AMD_WAIT_S
        + GREETING_PLAYOUT_S
        + PIN_WINDOW_S
        + BRIEF_PLAYOUT_S
        + OFFER_MORE_S
    )
