"""L0 — the call gets written down, and a failure to write it never breaks a call.

Three calls in ninety seconds went wrong and the words existed nowhere: the room
closes and the realtime session goes with it, `enable_recording` left no egress,
journald logs no transcript at INFO, and the dashboard needs a browser login. The
investigation ended with "the model called end_call six seconds in" and no way to
say what the owner had just said to it.

This is evidence, not control flow. Nothing here may raise into the call path.
"""

from __future__ import annotations

import time

from shot_voice.transcript import MAX_TURN_CHARS, MAX_TURNS, Transcript


class _Item:
    def __init__(self, role, text, *, interrupted=False, type="message",
                 created_at=None):
        self.role, self.interrupted, self.type = role, interrupted, type
        self.text_content = text
        # Real ChatMessages always carry this; the realtime path overwrites it
        # with when the user STARTED SPEAKING, which is not when we hear about it.
        if created_at is not None:
            self.created_at = created_at


class _Ev:
    def __init__(self, item):
        self.item = item


class _Session:
    def __init__(self):
        self.handlers: dict[str, list] = {}

    def on(self, name, fn):
        self.handlers.setdefault(name, []).append(fn)

    def off(self, name, fn):
        self.handlers.get(name, []).remove(fn)

    def emit(self, name, ev):
        for fn in list(self.handlers.get(name, [])):
            fn(ev)


def _tx(session=None):
    tx = Transcript(room_name="cb-1", direction="outbound", callback_id="c1")
    tx.listen(session or _Session())
    return tx


def test_both_sides_are_recorded_in_order():
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added", _Ev(_Item("assistant", "Hi, it's Shot.")))
    s.emit("conversation_item_added", _Ev(_Item("user", "yeah go ahead")))

    assert [t["role"] for t in tx.turns] == ["assistant", "user"]
    assert [t["text"] for t in tx.turns] == ["Hi, it's Shot.", "yeah go ahead"]


def test_an_interrupted_turn_is_marked_as_such():
    """The text is truncated at the PLAYBACK position -- what he heard, not what
    was generated -- so "he heard half of this" has to be a separate fact."""
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added",
           _Ev(_Item("assistant", "So, Harbor Lights is the nightc", interrupted=True)))

    assert tx.turns[0]["interrupted"] is True
    assert tx.turns[0]["text"] == "So, Harbor Lights is the nightc"


def test_notes_interleave_with_the_words():
    """A transcript alone cannot explain a hang-up; what the code decided has to
    sit in line with what was said."""
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added", _Ev(_Item("assistant", "enter your code")))
    tx.note("pin verified")

    assert [t["role"] for t in tx.turns] == ["assistant", "note"]
    assert tx.turns[1]["text"] == "pin verified"


def test_non_speech_items_are_ignored():
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added", _Ev(_Item("assistant", "x", type="agent_handoff")))
    s.emit("conversation_item_added", _Ev(_Item("assistant", "   ")))
    assert tx.turns == []


def test_a_malformed_event_cannot_break_the_call():
    """Recording is evidence. An exception here would drop a live call, which is
    a far worse bug than the blind spot this closes."""
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added", _Ev(None))
    s.emit("conversation_item_added", "not an event at all")
    s.emit("conversation_item_added", _Ev(_Item("user", "still working")))

    assert [t["text"] for t in tx.turns] == ["still working"]


def test_a_session_that_will_not_be_listened_to_does_not_raise():
    class Hostile:
        def on(self, *a):
            raise RuntimeError("no")

    Transcript(room_name="r", direction="inbound").listen(Hostile())


def test_runaway_output_cannot_write_an_unbounded_row():
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added", _Ev(_Item("assistant", "x" * (MAX_TURN_CHARS + 500))))
    for _ in range(MAX_TURNS + 20):
        s.emit("conversation_item_added", _Ev(_Item("user", "again")))

    assert len(tx.turns) == MAX_TURNS
    assert len(tx.turns[0]["text"]) == MAX_TURN_CHARS


def test_stop_detaches_the_listener():
    s = _Session()
    tx = _tx(s)
    tx.stop()
    s.emit("conversation_item_added", _Ev(_Item("user", "after the end")))
    assert tx.turns == []


async def test_flushing_an_empty_call_writes_nothing_and_does_not_raise():
    await Transcript(room_name="r", direction="inbound").flush()


async def test_a_supervisor_that_is_down_does_not_take_the_call_with_it(monkeypatch):
    s = _Session()
    tx = _tx(s)
    s.emit("conversation_item_added", _Ev(_Item("user", "hello")))

    import httpx

    class Boom:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            raise httpx.ConnectError("supervisor is down")

    monkeypatch.setattr(httpx, "AsyncClient", Boom)
    await tx.flush()          # must not raise


# ----------------------------------------------------------- tool calls

class _Call:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Out:
    def __init__(self, output, *, is_error=False):
        self.output, self.is_error = output, is_error


class _ToolsEv:
    def __init__(self, calls, outs):
        self.function_calls, self.function_call_outputs = calls, outs


def _tools_ev(name="web_search", args='{"query": "weather"}',
              out="AccuWeather: 88 F", is_error=False):
    return _ToolsEv([_Call(name, args)], [_Out(out, is_error=is_error)])


def test_a_tool_call_and_what_it_returned_are_both_recorded():
    """Nobody could prove what Brave returned on the call that got escalated to
    a paid worker session: the only record of that search was the model's own
    paraphrase of it. conversation_item_added never carries tool items at all,
    so no filter change here could ever have fixed it."""
    tx = Transcript(room_name="r", direction="inbound")
    tx.listen(_Session())
    tx._on_tools(_tools_ev())

    assert len(tx.turns) == 1
    turn = tx.turns[0]
    assert turn["role"] == "tool"
    assert "web_search" in turn["text"]
    assert "weather" in turn["text"], "what was asked"
    assert "88 F" in turn["text"], "and what came back"


def test_the_stored_turn_shape_never_grows_a_fifth_key():
    """ops/transcript.py reads exactly these four, and schema.sql documents
    them."""
    tx = Transcript(room_name="r", direction="inbound")
    tx._on_tools(_tools_ev())
    assert set(tx.turns[0]) == {"at", "role", "text", "interrupted"}


def test_the_tool_role_fits_the_transcript_column():
    """ops/transcript.py right-aligns the role in a 9-character field, so
    "tool_result" would misalign every line in the file."""
    assert len("tool") <= len("assistant")


def test_a_huge_tool_result_is_truncated_harder_than_speech():
    from shot_voice.transcript import MAX_TOOL_CHARS

    tx = Transcript(room_name="r", direction="inbound")
    tx._on_tools(_tools_ev(out="x" * 10_000))
    assert len(tx.turns[0]["text"]) < MAX_TURN_CHARS
    assert tx.turns[0]["text"].count("x") <= MAX_TOOL_CHARS


def test_a_failed_tool_is_marked_as_such():
    tx = Transcript(room_name="r", direction="inbound")
    tx._on_tools(_tools_ev(out="boom", is_error=True))
    assert "!>" in tx.turns[0]["text"]


def test_a_malformed_tool_event_cannot_break_the_call():
    """Same reasoning as the message recorder: evidence-gathering must never be
    able to take down the thing it is observing."""
    tx = Transcript(room_name="r", direction="inbound")
    tx._on_tools(None)
    tx._on_tools(_ToolsEv(None, None))
    tx._on_tools(_ToolsEv([_Call("a", "{}")], []))      # mismatched lengths
    assert tx.turns == []
    tx._on_tools(_tools_ev())
    assert len(tx.turns) == 1, "and a good event after a bad one still lands"


def test_stop_detaches_both_listeners():
    session = _Session()
    tx = Transcript(room_name="r", direction="inbound")
    tx.listen(session)
    assert sorted(session.handlers) == ["conversation_item_added",
                                        "error",
                                        "function_tools_executed",
                                        "metrics_collected"]
    tx.stop()
    assert all(not fns for fns in session.handlers.values()), (
        "both recorders must come off, or a second call double-records")


# --- the model being STOPPED, as opposed to choosing not to speak ----------


class _Err:
    """The shape livekit hands over: ErrorEvent.error is a RealtimeModelError
    whose own .error is the underlying exception."""

    def __init__(self, detail, *, recoverable=True):
        self.error = RuntimeError(detail)
        self.recoverable = recoverable


class _ErrEv:
    def __init__(self, err):
        self.error = err


class _Metrics:
    def __init__(self, input_tokens, cached=0):
        self.input_tokens = input_tokens
        self.input_token_details = type("D", (), {"cached_tokens": cached})()


class _MetricsEv:
    def __init__(self, metrics):
        self.metrics = metrics


def test_a_rejected_response_is_recorded_instead_of_vanishing():
    """Five responses in one call came back
    `response failed: [tokens] rate_limit_exceeded` and NOTHING wrote it down.
    The plugin sets the exception on a future nobody awaits, so it surfaced
    only as asyncio's "Future exception was never retrieved" -- no shot.* line,
    no shot.calls row, no turn here. From the transcript the agent had simply
    chosen not to answer, which reads exactly like a routing bug."""
    session = _Session()
    tx = _tx(session)
    session.emit("error", _ErrEv(_Err(
        "response failed: [tokens] rate_limit_exceeded")))
    assert len(tx.turns) == 1
    assert tx.turns[0]["role"] == "note"
    assert "rate_limit_exceeded" in tx.turns[0]["text"]


def test_a_model_error_lands_in_line_with_the_words():
    """A note between the turns is what makes the silence legible: he asked,
    nothing was said, he asked again."""
    session = _Session()
    tx = _tx(session)
    session.emit("conversation_item_added", _Ev(_Item("user", "check my day")))
    session.emit("error", _ErrEv(_Err("response failed: [tokens] rate_limit_exceeded")))
    session.emit("conversation_item_added", _Ev(_Item("user", "hello?")))
    assert [t["role"] for t in tx.turns] == ["user", "note", "user"]


def test_a_malformed_error_event_cannot_break_the_call():
    """Transcript is deliberately passive: a bookkeeping failure that drops a
    call is a far worse bug than the blind spot it closes."""
    session = _Session()
    tx = _tx(session)
    session.emit("error", object())
    session.emit("error", _ErrEv(None))
    assert isinstance(tx.turns, list)


def test_token_headroom_is_accumulated_and_never_written_as_turns():
    """metrics_collected fires once per RESPONSE. A row each would bury the
    words, and the peak is the number that matters -- the ceiling is per
    minute and per response, not per call."""
    session = _Session()
    tx = _tx(session)
    session.emit("metrics_collected", _MetricsEv(_Metrics(14455, cached=0)))
    session.emit("metrics_collected", _MetricsEv(_Metrics(1426, cached=1400)))
    assert tx.turns == []
    assert tx.responses == 2
    assert tx.peak_input_tokens == 14455


def test_a_malformed_metrics_event_cannot_break_the_call():
    session = _Session()
    tx = _tx(session)
    session.emit("metrics_collected", object())
    assert tx.responses == 0


# --- the order things HAPPENED, not the order we heard about them ---------


async def test_a_late_transcript_is_filed_when_it_was_SPOKEN():
    """The realtime model answers AUDIO. The text transcription of that audio
    comes back on a slower channel and can land after the reply it caused.

    On 2026-09-10 this produced:

        118.5s  note  the model called end_call
        118.6s  user  Okay, this one.

    -- which reads as the agent hanging up and the owner speaking afterwards. He had
    in fact dismissed it a couple of seconds earlier and the model obeyed. Read
    in the recorded order it is a report of an unprompted hang-up, and that is
    how it was read: as a regression, in a written report, wrongly.

    The SDK hands us the true time on ChatMessage.created_at for exactly this
    reason. Using it is what keeps the transcript a record of the call rather
    than a record of our own event loop.
    """
    session = _Session()
    tx = Transcript(room_name="r", direction="inbound")
    tx.listen(session)
    # Ten seconds into the call. He spoke at eight; both events reach us now.
    tx._started = time.time() - 10.0
    spoke_at = tx._started + 8.0

    tx.note("the model called end_call")        # observed now, at ~10s
    session.emit("conversation_item_added",     # transcribed now, spoken at 8s
                 _Ev(_Item("user", "you're dismissed", created_at=spoke_at)))
    await tx.flush()

    roles = [t["role"] for t in tx.turns]
    assert roles == ["user", "note"], (
        "the dismissal was spoken BEFORE the hang-up and must be filed there")
    assert tx.turns[0]["at"] < tx.turns[1]["at"]


def test_a_turn_with_no_created_at_still_lands():
    """Notes and tool rows have no created_at, and neither does a stub item.
    They fall back to now, which for those IS when they happened."""
    session = _Session()
    tx = _tx(session)
    session.emit("conversation_item_added", _Ev(_Item("user", "hi")))
    assert tx.turns[0]["at"] >= 0


def test_a_bogus_timestamp_cannot_reorder_the_call():
    """A provider timestamp from 1970 would sort every real turn after it and
    produce a negative offset in the stored row."""
    session = _Session()
    tx = _tx(session)
    session.emit("conversation_item_added", _Ev(_Item("user", "hi", created_at=0.0)))
    session.emit("conversation_item_added",
                 _Ev(_Item("user", "later", created_at=time.time() + 9999)))
    assert all(t["at"] >= 0 for t in tx.turns)
