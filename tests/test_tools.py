"""L0 — the tool descriptions, where the decisions actually get made.

Guidance that must beat the session prompt belongs on the tool: as one line of a
2,500-character prompt it loses, which is the lesson EXTRA_END_CALL and
scripted.py were both written from. These assert the load-bearing sentences are
still there, because deleting one is a silent behaviour change.
"""

from __future__ import annotations

from livekit.agents.testing import fake_job_context

from shot_voice.agent import ShotAgent
from shot_voice.tools import (
    cancel_task,
    check_my_day,
    comment_on_linear_issue,
    find_linear_issue,
    get_task_status,
    start_background_task,
    web_search,
    write_linear_issue,
)

from support import TEST_NAME


def _desc(tool) -> str:
    """Lowercased and whitespace-normalised: these are wrapped docstrings, so a
    phrase that matters can straddle a line break."""
    return " ".join(tool.info.description.split()).lower()


def test_the_delegation_tool_says_what_it_costs():
    """It opens a paid session and RINGS HIS PHONE. The old docstring said
    "Returns immediately", which reads as free."""
    d = _desc(start_background_task)
    assert f"rings {TEST_NAME.lower()}'s phone" in d
    assert "opened" in d, "the bar is a page that has to be OPENED"


def test_the_delegation_tool_forbids_escalating_a_thin_search():
    """The incident: a weather question with no number in the snippets became a
    worker session and an outbound call ninety seconds later."""
    d = _desc(start_background_task)
    assert "thin" in d
    assert "weather" in d, "name the actual shape that went wrong"


def test_the_cancel_tool_refuses_to_be_used_for_it_taking_a_while():
    """Cancelling work he still wants loses it silently: there is no callback
    left to tell him, and he finds out by asking for something gone."""
    d = _desc(cancel_task)
    assert "taking a while" in d
    assert f"only when {TEST_NAME.lower()} has said so" in d
    assert "interrupted you" in d


def test_the_cancel_tool_says_it_cannot_un_ring_a_call():
    assert "cannot be recalled" in _desc(cancel_task)


def test_the_status_tool_closes_the_topic():
    """Its no-detail return invited the third re-poll that filled forty seconds
    of one call while the owner waited."""
    assert "never guess" in _desc(get_task_status)


def test_the_day_tool_claims_the_whats_on_today_phrasings():
    d = _desc(check_my_day)
    assert "what's on today" in d
    assert "what's on my plate" in d


def test_the_day_tool_steers_off_going_ticket_by_ticket():
    """One round trip for the whole picture. It used to steer off the Linear
    MCP toolset, which is gone from this tier -- 65 tools and 78KB of schema
    re-charged against a 40,000 TPM bucket on every response left room for two
    responses a minute. find_linear_issue is what remains to steer off, and
    only because looping it is several round trips inside one turn."""
    d = _desc(check_my_day)
    assert "do not go ticket by ticket with find_linear_issue" in d
    assert "do not also call list_my_tasks" in d


def test_the_search_tool_names_the_cases_that_are_not_its_job():
    """The prohibition belongs on the tool being wrongly CHOSEN. As one bullet
    of a 2,500-character session prompt it lost, and the owner got five web searches
    after asking for the browser."""
    d = _desc(web_search)
    assert "use the browser" in d and "wrong tool" in d
    assert "already did for him" in d, "re-finding reported work is not a search"


def test_the_delegation_tool_refuses_to_re_find_reported_work():
    """It made him wait twice for the same answer, and rang his phone to do it."""
    assert "not for re-finding something you already reported" in _desc(
        start_background_task)


# --- the three Linear tools ------------------------------------------------


def test_the_ticket_lookup_steers_back_to_the_day_tool():
    """"What's on today" through find_linear_issue is several round trips
    inside one turn, and check_my_day answers it in one -- tickets AND
    background work together."""
    d = _desc(find_linear_issue)
    assert "check_my_day" in d
    assert "read-only" in d


def test_the_ticket_lookup_promises_no_identifiers():
    """The session prompt says never read out an identifier, a link or a
    reference number. The query selects neither id nor url, so the tool cannot
    return one -- and the description says so, where the model reads it."""
    d = _desc(find_linear_issue)
    assert "identifier" in d and "link" in d


def test_both_writers_say_they_change_something():
    """The owner asked for no approval gate, so the description is the only thing
    between "a remark" and a ticket appearing in his workspace."""
    for tool in (write_linear_issue, comment_on_linear_issue):
        d = _desc(tool)
        assert "changes" in d, f"{tool.info.name} does not say it writes"
        assert "asked for" in d


def test_the_writer_does_not_read_as_expensive():
    """start_background_task is the one that costs money and rings his phone.
    Writing a ticket does neither, and a model that confuses them either
    refuses to write or spends three dollars to do it."""
    d = _desc(write_linear_issue)
    assert "costs nothing" in d and "rings nobody" in d


def test_the_two_writers_point_at_each_other():
    """"Add that to the demo video ticket" must not create a second ticket
    with the same title."""
    assert "comment_on_linear_issue" in _desc(write_linear_issue)
    assert "write_linear_issue" in _desc(comment_on_linear_issue)


async def test_every_shot_agent_tool_is_mocked_in_the_behaviour_lane():
    """The invariant _MOCKS states in a comment, enforced.

    mock_tools patches only the names it is given, so a tool left out of the
    dict reaches the REAL supervisor and the REAL Linear API from the paid
    lane -- and two of them now WRITE. `end_call` is deliberately absent:
    three tests assert it IS called.

    Lived in test_linear_toolset_wiring.py until the MCP toolset came off this
    tier and the rest of that file went with it.
    """
    from test_agent_behavior import _MOCKS

    with fake_job_context():
        agent = ShotAgent()
    named = {t.info.name for t in agent.tools if hasattr(t, "info")}
    assert named - {"end_call"} <= set(_MOCKS), (
        f"unmocked tool(s) reach production from the paid lane: "
        f"{sorted(named - {'end_call'} - set(_MOCKS))}")
