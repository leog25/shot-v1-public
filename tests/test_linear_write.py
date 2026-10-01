"""L0 -- the three Linear tools, offline, against a mocked Linear.

These replaced the Linear MCP toolset on the voice tier. That toolset was 65
tools and 78KB of schema, and the realtime API re-charges the WHOLE tool schema
against the account's 40,000 TPM ceiling on every response -- so carrying it
cost 14,923 per response and left room for two a minute. A tool-using turn
needs two, one to call and one to speak the result, so the second reliably came
back `response failed: [tokens] rate_limit_exceeded`, which the model cannot
see and cannot recover from: it just says nothing.

Two of these WRITE, with no approval gate, at the owner's explicit request. So the
thing under test is mostly the honesty of the return strings: "Linear has
nothing matching that", "I could not reach Linear", and "nothing was saved"
have to be three different sentences. Believing a ticket landed when it did not
is the failure that costs him work.
"""

from __future__ import annotations

import re

import httpx
import pytest
import respx

from shot_core import budget
from shot_voice import tools
from shot_voice.tools import (
    comment_on_linear_issue,
    find_linear_issue,
    write_linear_issue,
)


def _linear_route():
    # _linear carries a base_url, so match host+path rather than a bare URL.
    return respx.post(host="api.linear.app", path="/graphql")


def _ok(data: dict) -> httpx.Response:
    return httpx.Response(200, json={"data": data})


def _issues(*nodes) -> dict:
    return {"issues": {"nodes": list(nodes)}}


ISSUE = {"title": "Complete the founder profile", "state": {"name": "Done"}}


@pytest.fixture(autouse=True)
def _forget_the_team():
    """The team id is cached for the life of the PROCESS, which is right in a
    forked job and wrong across tests."""
    tools._team_id = None
    yield
    tools._team_id = None


# --- find ------------------------------------------------------------------


@respx.mock
async def test_a_hit_comes_back_as_title_and_status():
    _linear_route().mock(return_value=_ok(_issues(ISSUE)))
    out = await find_linear_issue(None, "founder")
    assert "Complete the founder profile" in out
    assert "Done" in out


@respx.mock
async def test_several_hits_name_the_first_three_and_count_the_rest():
    nodes = [{"title": f"Ticket {n}", "state": {"name": "Todo"}} for n in range(5)]
    _linear_route().mock(return_value=_ok(_issues(*nodes)))
    out = await find_linear_issue(None, "ticket")
    assert "Ticket 0" in out and "Ticket 2" in out
    assert "Ticket 3" not in out
    assert "And 2 more" in out


@respx.mock
async def test_nothing_matching_is_not_the_same_as_could_not_look():
    """The distinction _linear_day exists to protect, one tool over: "Linear
    says you have nothing" and "I could not reach Linear" are different
    answers and must be said differently."""
    _linear_route().mock(return_value=_ok(_issues()))
    empty = await find_linear_issue(None, "nothing like this")
    assert "could not" not in empty.lower() and "couldn't" not in empty.lower()

    _linear_route().mock(side_effect=httpx.ConnectError("nope"))
    down = await find_linear_issue(None, "founder")
    assert "couldn't reach linear" in down.lower()


@respx.mock
async def test_a_graphql_error_payload_is_not_read_as_an_empty_workspace():
    """A Linear GraphQL failure is an HTTP 200 with a top-level `errors`
    array, so raise_for_status() sees nothing at all. Read as data it becomes
    "you have no tickets", which is a lie he would act on."""
    _linear_route().mock(return_value=httpx.Response(
        200, json={"errors": [{"message": "Access denied"}]}))
    out = await find_linear_issue(None, "founder")
    assert "couldn't reach linear" in out.lower()
    assert "nothing in linear" not in out.lower()


@respx.mock
async def test_no_identifier_uuid_or_url_can_reach_the_model():
    """The session prompt says never read out an identifier, a link or a
    reference number. The query selects neither id nor url, so even a server
    that volunteers them cannot get them spoken."""
    _linear_route().mock(return_value=_ok(_issues({
        "title": "Complete the founder profile", "state": {"name": "Done"},
        "id": "eb53a99e-2c57-467c-8e69-266dc39796bd",
        "identifier": "GAO-8",
        "url": "https://linear.app/example/issue/GAO-8/x"})))
    out = await find_linear_issue(None, "founder")
    assert "http" not in out
    assert "GAO-8" not in out
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", out)


@respx.mock
async def test_an_unset_key_says_so_rather_than_claiming_a_failure():
    """Reporting a structural absence as a failure is the same mistake as
    recording ABANDONED as pin_failed. _speak_day already refuses to make it."""
    route = _linear_route()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tools, "_LINEAR_KEY", "")
        out = await find_linear_issue(None, "founder")
    assert route.call_count == 0
    assert "not set up" in out.lower()
    assert "couldn't reach" not in out.lower()


# --- write -----------------------------------------------------------------


@respx.mock
async def test_a_new_ticket_resolves_the_team_then_creates():
    route = _linear_route().mock(side_effect=[
        _ok({"teams": {"nodes": [{"id": "team-1"}]}}),
        _ok({"issueCreate": {"success": True}})])
    out = await write_linear_issue(None, "Try the new callback flow")
    assert route.call_count == 2
    assert "Try the new callback flow" in out
    body = route.calls[1].request.content.decode()
    assert "team-1" in body


@respx.mock
async def test_the_team_is_resolved_once_per_process():
    """One extra round trip on the first write, none after. teamId is the only
    required field on IssueCreateInput, so it cannot be skipped -- only
    cached."""
    _linear_route().mock(side_effect=[
        _ok({"teams": {"nodes": [{"id": "team-1"}]}}),
        _ok({"issueCreate": {"success": True}}),
        _ok({"issueCreate": {"success": True}})])
    await write_linear_issue(None, "One")
    await write_linear_issue(None, "Two")
    assert tools._team_id == "team-1"


@respx.mock
async def test_a_failed_write_says_nothing_was_saved():
    """"I couldn't reach Linear" alone leaves him believing it might have
    landed, which is worse than either fact on its own."""
    _linear_route().mock(side_effect=httpx.ConnectError("nope"))
    out = await write_linear_issue(None, "Try the new callback flow")
    assert "nothing was saved" in out.lower()


@respx.mock
async def test_linear_declining_the_create_is_not_reported_as_success():
    _linear_route().mock(side_effect=[
        _ok({"teams": {"nodes": [{"id": "team-1"}]}}),
        _ok({"issueCreate": {"success": False}})])
    out = await write_linear_issue(None, "Try the new callback flow")
    assert "nothing was saved" in out.lower()


@respx.mock
async def test_a_workspace_with_no_team_cannot_silently_succeed():
    _linear_route().mock(return_value=_ok({"teams": {"nodes": []}}))
    out = await write_linear_issue(None, "Try the new callback flow")
    assert "nothing was saved" in out.lower()


async def test_an_empty_title_never_reaches_linear():
    out = await write_linear_issue(None, "   ")
    assert "need a line" in out.lower()


# --- comment ---------------------------------------------------------------


@respx.mock
async def test_a_comment_finds_the_issue_then_writes_to_it():
    route = _linear_route().mock(side_effect=[
        _ok(_issues({"id": "issue-1", "title": "Make demo video"})),
        _ok({"commentCreate": {"success": True}})])
    out = await comment_on_linear_issue(None, "demo video", "already sent it")
    assert route.call_count == 2
    assert "Make demo video" in out
    assert "issue-1" in route.calls[1].request.content.decode()
    assert "issue-1" not in out


@respx.mock
async def test_an_ambiguous_match_refuses_to_guess():
    """Writing to the wrong ticket is not undoable from a phone call."""
    _linear_route().mock(return_value=_ok(_issues(
        {"id": "a", "title": "Complete the founder profile"},
        {"id": "b", "title": "Complete the cofounder profile"})))
    out = await comment_on_linear_issue(None, "founder", "done")
    assert "more than one" in out.lower()
    assert "do not guess" in out.lower()


@respx.mock
async def test_no_such_ticket_is_not_reported_as_a_failure_to_reach_linear():
    _linear_route().mock(return_value=_ok(_issues()))
    out = await comment_on_linear_issue(None, "nothing like this", "hi")
    assert "couldn't find" in out.lower()
    assert "nothing was saved" not in out.lower()


async def test_an_empty_comment_never_reaches_linear():
    out = await comment_on_linear_issue(None, "demo video", "  ")
    assert "need something to write" in out.lower()


# --- budget ----------------------------------------------------------------


def test_the_timeouts_come_from_the_budget():
    assert tools._linear.timeout.read == budget.LINEAR_QUERY_S
    assert budget.LINEAR_WRITE_S <= budget.WEB_SEARCH_TOTAL_S
