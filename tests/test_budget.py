"""L0 — the timeouts have to nest.

They used to be seven constants across four files that nested by coincidence,
and they did not: the worst-case dial was 59s against a 60s gate. One slow
answer and the PIN was never asked at all -- the caller sat on an open line
while the job waited out a separate 150s poll and then reported nothing.
"""

from __future__ import annotations

from shot_core import budget


def test_a_healthy_call_fits_inside_the_supervisor_rendezvous():
    """place_callback blocks on DBOS.recv for this long. A legitimate slow call
    that outlives it is recorded as `no_report` while it is still in progress."""
    worst = budget.worst_case_call_s()
    assert worst < budget.DBOS_RECV_S, (
        f"worst case {worst:.0f}s exceeds the {budget.DBOS_RECV_S}s rendezvous")
    assert budget.DBOS_RECV_S - worst >= 30, "leave real headroom, not one second"


def test_the_ring_ends_before_voicemail_picks_up():
    """A US mobile rolls to voicemail at ~20-25s, and the LiveKit SDK silently
    pins 30s when this is left unset -- so we would reliably reach voicemail."""
    assert 5 <= budget.RING_S <= 25


def test_amd_has_an_outer_belt():
    """livekit/agents#6996: AMD stalls 17-20s on ~7% of answered calls, and it
    lands as dead air on the first thing a callback does."""
    assert budget.AMD_WAIT_S > budget.AMD_DETECT_S
    assert budget.AMD_NO_SPEECH_S < budget.AMD_DETECT_S


def test_the_greeting_is_not_gated_on_a_reply_window():
    """SPEECH_REPLY_S is gone and must stay gone.

    Greeting and then listening for eight seconds to find out whether anyone
    was there existed only to break the AMD deadlock. Nothing about the work is
    said before the PIN now, and a keypress is a stronger answer than a voice --
    a voicemail box cannot press one. Re-adding this would re-add eight seconds
    of dead air in exchange for nothing.
    """
    assert not hasattr(budget, "SPEECH_REPLY_S")


def test_the_pin_window_is_one_generous_wait():
    """Three attempts became one window. The old loop re-prompted per attempt,
    so the owner was asked for his code three times with no room to answer -- and a
    correct entry was still eaten, because the deadline ran from when the
    attempt started rather than from when the prompt finished."""
    assert budget.PIN_WINDOW_S >= 15, "leave time to find the keypad"
    assert budget.PIN_WINDOW_S < budget.worst_case_call_s()


def test_settings_ring_timeout_agrees_with_the_budget():
    from shot_core.settings import get_settings

    assert get_settings().ringing_timeout_seconds == budget.RING_S, (
        "RINGING_TIMEOUT_SECONDS in .env has drifted from the budget table")


def test_settings_amd_windows_agree_with_the_budget():
    """outbound.py hardcoded 8.0/6.0/9.0 while this table claimed to own them.

    Two sets of numbers agreeing only by coincidence is the exact failure this
    file exists to prevent -- the same lesson as the 59s dial against a 60s
    gate. Production reads the settings fields; these are the source of truth.
    """
    from shot_core.settings import get_settings

    s = get_settings()
    assert s.amd_detection_timeout_seconds == budget.AMD_DETECT_S
    assert s.amd_no_speech_seconds == budget.AMD_NO_SPEECH_S
    assert s.amd_wait_seconds == budget.AMD_WAIT_S


def test_the_amd_belt_cannot_eat_the_rendezvous_headroom():
    """AMD_WAIT_S is summed into worst_case_call_s().

    Raising the #6996 belt is not free: past this ceiling a slow-but-healthy
    call gets recorded as `no_report`, which reads exactly like a hang.
    """
    assert budget.AMD_WAIT_S <= 12.0


def test_a_search_and_its_retry_still_fit_inside_one_turn():
    """web_search retries itself once, so two Brave calls happen inline.

    Over WEB_SEARCH_TOTAL_S the bucket table at the top of tools.py says it
    should have been a background task instead -- and a background task rings
    the owner's phone, which is the thing the retry exists to avoid.
    """
    assert budget.WEB_SEARCH_ATTEMPT_S * 2 <= budget.WEB_SEARCH_TOTAL_S
    assert budget.WEB_SEARCH_TOTAL_S <= 5.0


def test_the_offer_turn_is_shorter_than_the_news_it_follows():
    """One short sentence versus two or three with substance in them. An offer
    budgeted longer than the news it follows is the sign that someone put
    content back into it -- which is what "anything else?" as the last line of
    the brief already cost once."""
    assert budget.NEWS_OFFER_S < budget.NEWS_GREETING_S


def test_the_two_turn_greeting_is_not_summed_into_the_callback_budget():
    """NEWS_* bound an INBOUND turn, on a path with no ring, no AMD and no PIN.

    Folding them into worst_case_call_s() would eat the DBOS rendezvous headroom
    for a path they are not on, and a slow-but-healthy callback would start
    being recorded as `no_report`. Pinning the COMPOSITION, not just the total,
    is what makes that a red test rather than a silently shifted budget.
    """
    assert budget.worst_case_call_s() == (
        budget.RING_S + budget.AMD_WAIT_S + budget.GREETING_PLAYOUT_S
        + budget.PIN_WINDOW_S + budget.BRIEF_PLAYOUT_S + budget.OFFER_MORE_S)
    assert budget.NEWS_GREETING_S + budget.NEWS_OFFER_S < budget.IDLE_HANGUP_S


def test_the_linear_write_stays_an_inline_tool():
    """Two round trips on the first write of a process -- teamId is the only
    required field on IssueCreateInput, so the team is resolved and then the
    mutation runs -- and it still has to fit the bucket table at the top of
    tools.py."""
    assert budget.LINEAR_QUERY_S <= budget.LINEAR_WRITE_S
    assert budget.LINEAR_WRITE_S <= budget.WEB_SEARCH_TOTAL_S


def test_the_mcp_connect_budget_is_gone():
    """It timed the Linear MCP toolset's per-job connect, which landed on the
    front of EVERY call. The toolset came off this tier because 78KB of schema
    re-charged against a 40,000 TPM bucket on every response left room for two
    responses a minute, so this has no reader left. Pinned absent for the same
    reason as SPEECH_REPLY_S: a constant nothing reads is a claim about the
    system that is no longer true."""
    assert not hasattr(budget, "LINEAR_MCP_CONNECT_S")


def test_the_day_check_stays_an_inline_tool():
    """Both legs run concurrently, so this bounds the slower one -- but it still
    has to fit the bucket table at the top of tools.py, which says anything that
    cannot answer inside ~5s belongs in a background task. And a background task
    rings the owner's phone, which is never the right answer to "what's on today?"."""
    assert budget.LINEAR_QUERY_S <= budget.CHECK_MY_DAY_S
    assert budget.CHECK_MY_DAY_S <= budget.WEB_SEARCH_TOTAL_S
