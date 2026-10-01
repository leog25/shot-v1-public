"""L0 — check_my_day, offline, against a mocked Linear and a mocked supervisor.

Why this tool exists at all rather than letting the model use the Linear MCP
toolset it already has: that toolset is 65 tools and ~83KB of schema whose
list_issues returns 5,159 characters of JSON -- UUIDs and URLs included -- for
five issues, against a session prompt that says never to read out IDs. And
"tickets" plus "background tasks" chained by the model is two round trips inside
a conversational turn.

So these are the guarantees: ONE Linear call, both legs concurrent, titles only,
and an answer no matter which half falls over.
"""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from shot_core import budget
from shot_voice import tools
from shot_voice.tools import _due, _speak_day, _title, check_my_day


def _linear_route():
    return respx.post(host="api.linear.app", path="/graphql")


def _tasks_route():
    # _client carries a base_url (127.0.0.1), so match on the path.
    return respx.get(path="/internal/tasks")


def _gql(mine=(), every=()) -> httpx.Response:
    return httpx.Response(200, json={"data": {
        "viewer": {"assignedIssues": {"nodes": list(mine)}},
        "issues": {"nodes": list(every)}}})


def _tasks(running=(), finished=()) -> httpx.Response:
    return httpx.Response(200, json={"tasks": list(running),
                                     "finished": list(finished)})


ISSUE = {"identifier": "GAO-7", "title": "Ship the callback fix",
         "dueDate": None, "state": {"name": "In Progress"}}
RUNNING = {"ref": "jazz-bistro-schedule", "title": "Jazz Bistro schedule",
           "state": "running", "age_s": 30}
DONE = {"ref": "harbor-lights-friday", "title": "Harbor Lights Friday", "state": "succeeded",
        "ago": "20 minutes ago", "told": False, "gist": "nothing confirmed yet"}


# ------------------------------------------------------------- both halves

@respx.mock
async def test_both_halves_come_back_in_one_string():
    _linear_route().mock(return_value=_gql(mine=[ISSUE]))
    _tasks_route().mock(return_value=_tasks(running=[RUNNING]))
    out = await check_my_day(None)
    assert "Ship the callback fix" in out
    assert "Jazz Bistro schedule" in out


@respx.mock
async def test_it_costs_exactly_one_linear_call():
    """The whole justification for not using the MCP toolset."""
    route = _linear_route().mock(return_value=_gql(mine=[ISSUE]))
    _tasks_route().mock(return_value=_tasks())
    await check_my_day(None)
    assert route.call_count == 1


@respx.mock
async def test_the_two_legs_actually_overlap():
    """Asserting overlap directly, not total wall time, which is flaky.

    Serialised this is two round trips inside a conversational turn -- which is
    the thing the bucket table at the top of tools.py exists to prevent.
    """
    seen: list[float] = []

    def _stamp(response):
        def _h(request):
            seen.append(time.monotonic())
            return response
        return _h

    _linear_route().mock(side_effect=_stamp(_gql(mine=[ISSUE])))
    _tasks_route().mock(side_effect=_stamp(_tasks(running=[RUNNING])))
    await check_my_day(None)
    assert len(seen) == 2
    assert abs(seen[0] - seen[1]) < 0.05, "the legs ran one after the other"


# ------------------------------------------------------------- degradation

@respx.mock
async def test_an_empty_workspace_says_nothing_is_on_rather_than_erroring():
    """Today's live state: seven issues, all completed, none assigned. It has to
    read as an ANSWER, not as a failure."""
    _linear_route().mock(return_value=_gql())
    _tasks_route().mock(return_value=_tasks())
    out = await check_my_day(None)
    assert "Nothing on your plate" in out
    assert "could not reach" not in out


@respx.mock
async def test_linear_down_still_returns_the_background_tasks():
    _linear_route().mock(side_effect=httpx.ConnectError("nope"))
    _tasks_route().mock(return_value=_tasks(running=[RUNNING]))
    out = await check_my_day(None)
    assert "Jazz Bistro schedule" in out
    assert "could not reach it" in out


@respx.mock
async def test_the_supervisor_being_down_still_returns_linear():
    _linear_route().mock(return_value=_gql(mine=[ISSUE]))
    _tasks_route().mock(side_effect=httpx.ConnectError("nope"))
    out = await check_my_day(None)
    assert "Ship the callback fix" in out


@respx.mock
async def test_an_unset_key_says_nothing_about_linear(monkeypatch):
    """"I could not reach Linear" when it was never configured is a lie about a
    structural absence -- the same distinction as "I could not ask" not being
    "he got it wrong"."""
    monkeypatch.setattr(tools, "_LINEAR_KEY", "")
    route = _linear_route().mock(return_value=_gql(mine=[ISSUE]))
    _tasks_route().mock(return_value=_tasks(running=[RUNNING]))
    out = await check_my_day(None)
    assert route.call_count == 0
    assert "Linear" not in out
    assert "Jazz Bistro schedule" in out


@respx.mock
async def test_a_graphql_error_payload_is_not_treated_as_data():
    """A GraphQL failure is an HTTP 200 with a top-level `errors` array, so
    raise_for_status() sees nothing at all."""
    _linear_route().mock(return_value=httpx.Response(
        200, json={"errors": [{"message": "Authentication failed"}]}))
    _tasks_route().mock(return_value=_tasks(running=[RUNNING]))
    out = await check_my_day(None)
    assert "Authentication failed" not in out, "the error text was read as work"
    assert "could not reach it" in out
    assert "Jazz Bistro schedule" in out


# --------------------------------------------------------------- the wire

@respx.mock
async def test_no_uuids_and_no_urls_ever_reach_the_model():
    """Fed nodes carrying id and url, as they would be if the query were later
    widened. Asserting the invariant, not the current field list."""
    import re

    fat = dict(ISSUE, id="eb53a99e-2c57-467c-8e69-266dc39796bd",
               url="https://linear.app/example/issue/GAO-7")
    _linear_route().mock(return_value=_gql(mine=[fat]))
    _tasks_route().mock(return_value=_tasks())
    out = await check_my_day(None)
    assert "http" not in out
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", out)
    assert "GAO-7" not in out


@respx.mock
async def test_the_key_is_sent_raw_not_as_a_bearer():
    """Linear personal API keys are not bearer tokens: "Bearer <key>" is a 401
    that reads exactly like a bad key. worker.py used to send Bearer for this
    same secret against the MCP endpoint -- two conventions, one key, one repo
    -- and that toolset is off this tier now, so raw is the only convention
    left here. Bearer survives in ops.bootstrap.anthropic_res, for tier 3."""
    route = _linear_route().mock(return_value=_gql())
    _tasks_route().mock(return_value=_tasks())
    await check_my_day(None)
    sent = route.calls[0].request.headers["authorization"]
    assert sent == tools._LINEAR_KEY
    assert not sent.startswith("Bearer ")


def test_the_client_timeout_comes_from_the_budget():
    assert tools._linear.timeout.read == budget.LINEAR_QUERY_S
    assert tools._client.timeout.read == budget.SUPERVISOR_HTTP_S


# ------------------------------------------------ the pure bits, no network

def _day(n):
    from datetime import date
    return date(2026, 9, n)


@pytest.mark.parametrize("due,expected", [
    (None, ""), ("", ""), ("not-a-date", ""),
    ("2026-09-08", "overdue"),
    ("2026-09-09", "due today"),
    ("2026-09-11", "due Friday"),
    ("2026-10-03", "due 3 October"),
])
def test_a_due_date_is_spoken_as_when_not_as_a_date(due, expected):
    assert _due({"dueDate": due}, _day(9)) == expected


def test_the_most_urgent_ticket_is_named_first():
    said = _speak_day(
        [{"identifier": "A", "title": "Later thing", "dueDate": "2026-09-30"},
         {"identifier": "B", "title": "Overdue thing", "dueDate": "2026-09-01"},
         {"identifier": "C", "title": "Today thing", "dueDate": "2026-09-09"}],
        [], [], [], linear_ok=True, linear_on=True, today=_day(9))
    assert said.index("Overdue thing") < said.index("Today thing") < said.index("Later thing")


def test_an_unassigned_board_falls_back_to_what_is_open():
    """viewer.assignedIssues is empty until the owner starts assigning, so the tool
    would answer "nothing" every time on a board that has plenty on it."""
    said = _speak_day([], [{"identifier": "A", "title": "Complete application"}],
                      [], [], linear_ok=True, linear_on=True, today=_day(9))
    assert "Nothing is assigned to you" in said
    assert "Complete application" in said


def test_work_he_has_already_heard_is_not_news_today():
    """`told` is reported_at. Re-announcing it is what reported_at prevents."""
    said = _speak_day([], [], [], [], linear_ok=True, linear_on=True, today=_day(9))
    assert "Nothing on your plate" in said


def test_a_title_is_trimmed_never_left_as_markdown_soup():
    assert _title({"title": "  Ship   the\nfix  "}) == "Ship the fix"
    assert len(_title({"title": "x" * 200})) <= 80
