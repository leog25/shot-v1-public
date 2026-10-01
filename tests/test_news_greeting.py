"""L0 — the inbound greeting that leads with finished work, and when it counts.

`tasks.reported_at` means "the owner actually heard it". The news greeting was the one
place still deciding that from `wait_for_playout()`, which returns normally
after an interruption -- the exact trap `scripted.py` exists to avoid, left in
place on the other path.

It cost him the Harbor Lights findings. The callback was cut off six seconds in and hung
up on him; he called straight back; the greeting started, `wait_for_playout()`
returned five seconds later, the task was stamped `reported_at`; the model then
ended the call. He called back a third time and got `0 unreported task(s)` and
"what do you need?" -- the finished work filed as already told to him.
"""

from __future__ import annotations

import pytest

from shot_voice.agent import ShotAgent


class _Msg:
    type = "message"
    role = "assistant"

    def __init__(self, text, interrupted=False):
        self.text_content, self.interrupted = text, interrupted


class _Handle:
    """A SpeechHandle far enough along to be read back."""

    def __init__(self, items=(), *, interrupted=False):
        self.chat_items = list(items)
        self.interrupted = interrupted

    def done(self):
        return True

    def exception(self):
        return None

    def __await__(self):
        async def _done():
            return self
        return _done().__await__()

    def wait_for_playout(self):
        raise AssertionError(
            "wait_for_playout() returns normally after an interruption; it is "
            "not proof of delivery and must not decide reported_at")


class _StateEv:
    def __init__(self, new_state: str) -> None:
        self.new_state = new_state


class _Item:
    def __init__(self, role: str) -> None:
        self.role = role


class _History:
    def __init__(self) -> None:
        self.items: list[_Item] = []


class _Session:
    """Records every instruction handed to the model, in order.

    Takes one handle or two. Reusing the last one when only a single handle is
    supplied is what lets every pre-existing single-turn test here keep working
    untouched now that the news greeting is two turns.

    `speaks=False` models the failure that cost a real call: the generation
    produces NO audio and NO chat item, silently. The only signal the agent has
    is the session never entering the `speaking` state, so that is what the fake
    withholds.
    """

    def __init__(self, *handles, speaks: bool = True):
        self._handles = list(handles)
        self.replies: list[str] = []
        self.speaks = speaks
        self.interrupts = 0
        self.history = _History()
        self._listeners: list = []

    def on(self, name, fn):
        if name == "agent_state_changed":
            self._listeners.append(fn)

    def off(self, name, fn):
        if fn in self._listeners:
            self._listeners.remove(fn)

    def interrupt(self, **kw):
        self.interrupts += 1

    def generate_reply(self, *, instructions, **kw):
        self.replies.append(instructions)
        if self.speaks:
            # The listener is registered before generate_reply is called, so
            # firing synchronously is faithful enough.
            for fn in list(self._listeners):
                fn(_StateEv("speaking"))
        return self._handles[min(len(self.replies) - 1, len(self._handles) - 1)]


class _Agent(ShotAgent):
    # A plain class attribute shadows Agent.session, which is a read-only
    # property, so the greeting can run with no room and no realtime model.
    session = None


NEWS = [{"id": "t1", "ref": "harbor-lights-friday", "title": "Harbor Lights Friday",
         "ok": True, "gist": "nothing confirmed for Friday yet"}]


def _kill_the_supervisor(monkeypatch):
    """seeded_chat_ctx builds its own client, and /internal/context WRITES
    (save_summary) -- so an offline test must never let it reach a supervisor
    that happens to be running on this machine."""
    import httpx

    class _Dead:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **kw):
            raise httpx.ConnectError("supervisor is gone")

    monkeypatch.setattr(httpx, "AsyncClient", _Dead)


@pytest.fixture
def reported(monkeypatch):
    """Capture what would have been stamped reported_at."""
    seen: list[list[str]] = []

    async def _fake(task_ids):
        seen.append(list(task_ids))

    monkeypatch.setattr("shot_voice.agent._mark_reported", _fake)
    return seen


async def _greet(handle, reported):
    agent = _Agent(news=NEWS)
    agent.session = _Session(handle)
    await agent.on_enter()
    return agent.session, reported


async def test_a_greeting_he_heard_in_full_is_reported(reported):
    session, seen = await _greet(_Handle([_Msg("Hey, the Harbor Lights thing is done.")]),
                                 reported)
    assert "Harbor Lights Friday" in session.replies[0]
    assert seen == [["t1"]]


async def test_a_greeting_he_talked_over_is_NOT_reported(reported):
    """He is on the line and can ask, and he hears it again next call. Filing it
    as heard is what left him with nothing on the third call."""
    _, seen = await _greet(_Handle([_Msg("Hey, the Harbor Lights", interrupted=True)]),
                           reported)
    assert seen == [], "an interrupted greeting was filed as heard"


async def test_a_greeting_that_never_played_is_NOT_reported(reported):
    """Skipped speech produces no chat item at all."""
    _, seen = await _greet(_Handle([]), reported)
    assert seen == []


async def test_an_interruption_on_the_handle_alone_is_enough(reported):
    _, seen = await _greet(_Handle([_Msg("Hey")], interrupted=True), reported)
    assert seen == []


async def test_with_no_news_it_just_greets_and_reports_nothing(reported):
    agent = _Agent(news=[])
    agent.session = _Session(_Handle([_Msg("Hi, what do you need?")]))
    await agent.on_enter()

    assert agent.session.replies, "it must still speak first"
    assert reported == []


# ------------------------------------ the greeting must carry the findings

def test_the_news_block_carries_the_gist_not_just_the_title():
    """`/internal/context` has always returned a gist; on_enter threw it away
    and passed only titles, so the prompt asked for a headline the model did not
    have. It opened "It's done -- if you want, I can read the highlights next",
    and the task was marked reported on that."""
    from shot_voice.agent import _news_block

    block = _news_block([{"title": "Harbor Lights Friday", "ok": True,
                          "gist": "nothing confirmed for Friday yet"}])
    assert "Harbor Lights Friday" in block
    assert "nothing confirmed for Friday yet" in block


def test_a_failed_task_says_so_rather_than_inventing_a_result():
    from shot_voice.agent import _news_block

    assert "did not finish" in _news_block(
        [{"title": "Portland food", "ok": False, "gist": ""}])


def test_a_missing_gist_does_not_produce_a_dangling_headline():
    from shot_voice.agent import _news_block

    block = _news_block([{"title": "Portland food", "ok": True, "gist": ""}])
    assert "Portland food" in block and "no summary" in block


async def test_the_greeting_states_the_result_and_asks_nothing(reported):
    session, _ = await _greet(_Handle([_Msg("done")]), reported)
    prompt = session.replies[0]

    assert "nothing confirmed for Friday yet" in prompt, (
        "the findings never reached the model, so it can only announce that "
        "work exists -- which is what got stamped reported_at")
    assert "Harbor Lights Friday" in prompt
    assert "Do NOT ask whether he would like to hear it" in prompt
    assert "{news}" not in prompt and "{titles}" not in prompt, (
        "a placeholder reached the model verbatim once already")


# ------------------------------------------------ the offer is its own turn

async def test_the_offer_is_its_own_turn(reported):
    """Turn one is measured and gates reported_at; turn two does the asking.

    Appended to the news instead, the offer would invite the owner to answer over the
    tail -- and an interruption there marks a brief he heard IN FULL as
    undelivered, leaving the task unreported and re-announced next call. That is
    the DELIVER_CALLBACK / OFFER_MORE incident, one path over.
    """
    agent = _Agent(news=NEWS)
    agent.session = _Session(_Handle([_Msg("The Harbor Lights thing came back.")]),
                             _Handle([_Msg("Want the rest of the day?")]))
    await agent.on_enter()

    assert len(agent.session.replies) == 2, "the offer must not ride in turn one"
    news, offer = agent.session.replies
    assert "Harbor Lights Friday" in news
    assert "Harbor Lights Friday" not in offer, "the offer must not recap"
    assert "on today" in offer
    assert reported == [["t1"]]


async def test_an_interrupted_news_turn_gets_no_offer_over_the_top(reported):
    """He is talking. Reading a prepared line over him is exactly what the
    interruption rule forbids, and the callback flow already returns before
    OFFER_MORE for the same reason."""
    agent = _Agent(news=NEWS)
    agent.session = _Session(_Handle([_Msg("The Harbor Lights", interrupted=True)]))
    await agent.on_enter()

    assert len(agent.session.replies) == 1
    assert reported == []


async def test_a_news_turn_that_never_played_gets_no_offer(reported):
    """With nothing in front of it, "what ELSE is on today?" is the first thing
    he hears and has nothing to be else than."""
    agent = _Agent(news=NEWS)
    agent.session = _Session(_Handle([]))
    await agent.on_enter()

    assert len(agent.session.replies) == 1
    assert reported == []


async def test_with_no_news_the_offer_rides_in_the_single_turn(reported):
    """Nothing to protect there: no reported_at to gate, so no interruption can
    mis-file anything, so no reason to spend a second turn on it."""
    agent = _Agent(news=[])
    agent.session = _Session(_Handle([_Msg("Morning.")]))
    await agent.on_enter()

    assert len(agent.session.replies) == 1
    assert "on today" in agent.session.replies[0]
    assert reported == []


# ------------------------------------------------------- the time of day

async def test_the_greeting_carries_the_time_of_day(reported, monkeypatch):
    """OPENING_INBOUND_NEWS asserts the model was told the time. When a prompt
    asserts the model knows something, check what is actually interpolated --
    that gap is what produced "It's done, I can read the highlights next"."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setattr("shot_voice.agent.now_local",
                        lambda: datetime(2026, 9, 9, 8, 14,
                                         tzinfo=ZoneInfo("America/New_York")))
    agent = _Agent(news=NEWS)
    agent.session = _Session(_Handle([_Msg("done")]))
    await agent.on_enter()

    prompt = agent.session.replies[0]
    assert "morning" in prompt
    assert "8:14 AM in New York" in prompt


async def test_no_placeholder_ever_reaches_the_model(reported, monkeypatch):
    """A literal {what} went out on a live call once. OPENING_INBOUND gained
    placeholders in this change, so the no-news branch needs the guard too."""
    for news in ([], NEWS):
        agent = _Agent(news=news)
        agent.session = _Session(_Handle([_Msg("done")]))
        await agent.on_enter()
        for prompt in agent.session.replies:
            for placeholder in ("{part}", "{when}", "{news}", "{titles}"):
                assert placeholder not in prompt, (placeholder, prompt[:80])


# ---------------------------------------- the time survives a flaky supervisor

async def test_the_time_is_seeded_even_when_the_supervisor_is_down(monkeypatch):
    """seeded_chat_ctx swallows every exception and returns an EMPTY context.

    A time line added inside that try would vanish exactly when the supervisor
    is flaky -- leaving INSTRUCTIONS asserting "you were told the day and the
    time" with nothing interpolated. An instruction the model cannot follow is
    worse than none: that is the gap that produced "It's done -- if you want, I
    can read the highlights next" on a greeting with no findings in it.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setattr("shot_voice.agent.now_local",
                        lambda: datetime(2026, 9, 9, 20, 5,
                                         tzinfo=ZoneInfo("America/New_York")))
    _kill_the_supervisor(monkeypatch)
    ctx, news = await ShotAgent.seeded_chat_ctx()

    assert news == []
    said = " ".join(i.text_content or "" for i in ctx.items)
    assert "8:05 PM in New York" in said
    assert "evening" in said


async def test_the_seeded_time_does_not_claim_to_be_the_time_now(monkeypatch):
    """A twenty-minute call ends with a twenty-minute-old timestamp. echo_time
    stays the authority for the time NOW, and says so on the tool."""
    _kill_the_supervisor(monkeypatch)
    ctx, _ = await ShotAgent.seeded_chat_ctx()
    said = " ".join(i.text_content or "" for i in ctx.items)
    assert "at the start of this call" in said


# --------------------------------------------- an opening that made no sound

@pytest.fixture
def _fast_watchdog(monkeypatch):
    """GREETING_START_S is 3s in production; the suite must not wait it out."""
    monkeypatch.setattr("shot_voice.agent.GREETING_START_S", 0.01)


async def test_a_greeting_that_speaks_is_not_repeated(reported, _fast_watchdog):
    agent = _Agent(news=[])
    agent.session = _Session(_Handle([_Msg("Evening.")]), speaks=True)
    await agent.on_enter()

    assert len(agent.session.replies) == 1
    assert agent.session.interrupts == 0


async def test_a_greeting_that_makes_no_sound_is_said_again(reported, _fast_watchdog):
    """The failure that cost a real call: no audio, no chat item, no error.

    The owner heard nine seconds of dead air, said "hello?", and hung up -- and the
    stored transcript held two turns, neither of them a greeting. Nothing
    anywhere said the agent had failed to speak.
    """
    agent = _Agent(news=[])
    agent.session = _Session(_Handle([]), speaks=False)
    await agent.on_enter()

    assert len(agent.session.replies) == 2, "a silent opening must be retried"
    assert agent.session.interrupts == 1, (
        "the silent generation must be dropped first, or a late one lands on "
        "top of the retry and he hears it twice")


async def test_it_does_not_repeat_the_greeting_over_him(reported, _fast_watchdog):
    """Once he has spoken the silence is his turn to fill. Talking over him is
    the worse failure -- an interruption is never a reason to keep going."""
    agent = _Agent(news=[])
    session = _Session(_Handle([]), speaks=False)
    session.history.items.append(_Item("user"))
    agent.session = session
    await agent.on_enter()

    assert len(session.replies) == 1
    assert session.interrupts == 0
