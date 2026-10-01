"""L0 — web_search, offline, against a mocked Brave.

The incident: "what's the weather in Portland tomorrow" came back as four
sentences about weather websites and not one number, so the model did what the
prompt told it to and opened a background task -- which cost money and rang
the owner's phone ninety seconds later to tell him it would be hot. The SAME Brave
response carried an infobox with the forecast in it; the code read `web.results`
and nothing else, then kept three of the four hits it had paid for.

So these are the guarantees, not the wording: mine every block, use every hit,
retry once with a sharper question, and never leave delegation as the only move.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from shot_core import budget
from shot_voice import tools
from shot_voice.tools import _mine, _reformulate, _thin, web_search


def _route():
    # _brave carries a base_url, so match host+path rather than a bare URL.
    return respx.get(host="api.search.brave.com", path="/res/v1/web/search")


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


# Prose, no digits: exactly the shape that produced the incident.
THIN = {"web": {"results": [
    {"title": "AccuWeather Portland",
     "description": "Get the latest forecast for your area with maps, radar and "
                    "everything you need to plan your day and your week ahead."},
    {"title": "The Weather Channel",
     "description": "Local weather forecasts, conditions and severe alerts for "
                    "wherever you happen to be, updated through the day."},
]}}

RICH = {
    "infobox": {"results": [
        {"title": "Portland weather",
         "attributes": [["High", "88 F"], ["Low", "67 F"]]}]},
    "faq": {"results": [
        {"question": "How hot tomorrow?", "answer": "Around <strong>88</strong>."}]},
    "web": {"results": [
        {"title": f"M{i}", "description": f"result number {i} with detail"}
        for i in range(1, 5)]},
}

EMPTY: dict = {}


# ------------------------------------------------------- mining the payload

@respx.mock
async def test_the_infobox_answers_when_the_snippets_do_not():
    """The incident, in one line. The number was in the response all along."""
    _route().mock(return_value=_ok(RICH))
    out = await web_search(None, "weather in portland tomorrow")
    assert "88" in out


@respx.mock
async def test_all_four_paid_results_are_used():
    """It asked Brave for four and rendered three, discarding a hit it had
    already paid for."""
    _route().mock(return_value=_ok(RICH))
    out = await web_search(None, "portland things")
    for marker in ("M1", "M2", "M3", "M4"):
        assert marker in out, f"{marker} was dropped"


@respx.mock
async def test_extra_snippets_are_requested():
    """Extra sentences from the same page are often where the number is when the
    description is boilerplate."""
    route = _route().mock(return_value=_ok(RICH))
    await web_search(None, "anything")
    assert route.calls[0].request.url.params["extra_snippets"]


@respx.mock
async def test_html_never_reaches_the_model():
    _route().mock(return_value=_ok(RICH))
    out = await web_search(None, "weather in portland tomorrow")
    assert "<" not in out and ">" not in out


@respx.mock
async def test_a_payload_full_of_nulls_does_not_crash():
    """Brave omits blocks freely, and `.get("web")` can be an explicit null."""
    _route().mock(return_value=_ok({"web": None, "infobox": None, "faq": None}))
    out = await web_search(None, "xyzzy plugh")
    assert "background task" in out


# -------------------------------------------------------------- the retry

@respx.mock
async def test_a_thin_first_pass_retries_itself_once():
    """The whole point: the model never sees the thin pass, so it cannot be
    talked out of the retry -- and cannot escalate on the strength of it."""
    route = _route().mock(side_effect=[_ok(THIN), _ok(RICH)])
    out = await web_search(None, "weather in portland tomorrow")
    assert route.call_count == 2
    assert "88" in out


@respx.mock
async def test_the_retry_asks_a_different_question():
    """Repeating a similar query never works, which is why the old rule said to
    delegate instead. The retry has to actually sharpen it."""
    route = _route().mock(side_effect=[_ok(THIN), _ok(RICH)])
    await web_search(None, "what's the weather in portland tomorrow")
    first = route.calls[0].request.url.params["q"]
    second = route.calls[1].request.url.params["q"]
    assert first != second


@respx.mock
async def test_it_never_searches_a_third_time():
    route = _route().mock(side_effect=[_ok(THIN), _ok(THIN)])
    await web_search(None, "weather in portland tomorrow")
    assert route.call_count == 2


@respx.mock
async def test_a_good_first_pass_costs_exactly_one_call():
    """Brave is metered, and this runs inside a conversational turn."""
    route = _route().mock(return_value=_ok(RICH))
    await web_search(None, "weather in portland tomorrow")
    assert route.call_count == 1


@respx.mock
async def test_a_query_with_nothing_to_sharpen_is_not_asked_twice():
    route = _route().mock(return_value=_ok(EMPTY))
    await web_search(None, "xyzzy plugh")
    assert route.call_count == 1, "asking the identical question twice is useless"


# ------------------------------------------------------------ dead ends

@respx.mock
async def test_a_dead_end_search_does_not_push_a_background_task():
    """The failure that rang the owner's phone. A search that found nothing must close
    the topic, not leave delegation as the only move left."""
    _route().mock(side_effect=[_ok(THIN), _ok(EMPTY)])
    out = await web_search(None, "weather in portland tomorrow")
    assert "could not find it" in out
    assert "do NOT start a background task" in out


@respx.mock
async def test_a_thin_result_still_closes_the_topic():
    """Two passes, both prose, no number. Handing the model the snippets and
    nothing else is exactly where the incident started: with nothing to state,
    the only move the prompt left was a paid worker session and a phone call.
    The snippets still go along as context -- they are just not an answer."""
    _route().mock(side_effect=[_ok(THIN), _ok(THIN)])
    out = await web_search(None, "weather in portland tomorrow")
    assert "do NOT start a background task" in out
    assert "AccuWeather" in out, "the context it did find is still worth having"


@respx.mock
async def test_a_search_outage_never_raises_into_the_call():
    """And is not retried: a broken network is not a thin result, and doubling
    the dead air helps nobody."""
    route = _route().mock(side_effect=httpx.ConnectError("boom"))
    out = await web_search(None, "weather in portland tomorrow")
    assert out == "I couldn't reach search just then."
    assert route.call_count == 1


# ------------------------------------------------------------ pure helpers

@pytest.mark.parametrize("query", [
    "what's the weather in portland tomorrow",
    "how much does a pixel 9 cost",
    "what was the final score in the lakers game",
    "what time does the library open",
])
def test_the_reformulation_is_never_the_same_question_again(query):
    assert _reformulate(query).lower() != query.lower()


def test_a_numbery_answer_with_no_number_is_thin():
    assert _thin("weather tomorrow", _mine(THIN))
    assert not _thin("weather tomorrow", _mine(RICH))


def test_nothing_at_all_is_thin():
    assert _thin("anything", [])


def test_the_client_timeout_comes_from_the_budget():
    """Every timeout comes from budget.py -- they used to be seven constants
    across four files that nested by coincidence, and did not."""
    assert tools._brave.timeout.read == budget.WEB_SEARCH_ATTEMPT_S
