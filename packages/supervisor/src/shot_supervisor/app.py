"""Supervisor: task registry, durable callbacks, and the localhost API the
voice worker calls.

Every /internal endpoint sits INSIDE a conversational turn, so each must stay
well under ~200ms. That is why POST /internal/tasks writes the row and returns
without awaiting sessions.create — delegate() stays in the "inline, silent"
latency bucket rather than becoming an async tool.

Binds to 127.0.0.1; no auth beyond that.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import sqlalchemy as sa
from dbos import DBOS, DBOSConfig
from fastapi import FastAPI
from pydantic import BaseModel

from shot_core.db import make_engine
from shot_core.settings import get_settings
from shot_supervisor import context as ctxmod
from shot_supervisor import registry
from shot_supervisor.callbacks import (  # noqa: F401  (schedule_callback registers the queue)
    schedule_callback,
    speakable_failure,
)

log = logging.getLogger("shot.supervisor")
_s = get_settings()

app = FastAPI(title="shot-supervisor")
dbos = DBOS(config=DBOSConfig(
    name="shot-supervisor",
    system_database_url=_s.database_url,
    application_database_url=_s.database_url,
    run_migrations=True,
    enable_otlp=False,
), fastapi=app)

_engine = make_engine(_s.database_url)
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="session-start")


class NewTask(BaseModel):
    goal: str
    title: str


class NewCallback(BaseModel):
    brief: str
    delay_s: int = 0
    reason: str = "user_requested"
    # Defaults to the owner. Set explicitly to dial elsewhere -- e.g. a Twilio
    # test number -- so a probe can never ring a real phone by omission.
    to_number: str | None = None


class Outcome(BaseModel):
    outcome: str


def _start_session_bg(task_id: str) -> None:
    try:
        with _engine.begin() as cx:
            registry.start_session(cx, task_id)
    except Exception as e:
        # Do NOT leave it at `queued`: list_my_tasks would report it as still
        # running forever and no callback would ever come. Surface the reason
        # so the agent can say "that one couldn't start" out loud.
        log.exception("failed to start session for task %s", task_id)
        reason = str(e)
        import sqlalchemy as sa
        with _engine.begin() as cx:
            cx.execute(sa.text("""
                UPDATE shot.tasks
                   SET state='failed', finished_at=now(), updated_at=now(),
                       error = :err,
                       result_summary = :sum
                 WHERE id=:i AND session_id IS NULL"""),
                {"i": task_id, "err": json.dumps({"stage": "start_session",
                                                  "message": reason[:400]}),
                 "sum": f"I couldn't start that one: {speakable_failure(reason)}."})


@app.post("/internal/tasks")
def create_task(body: NewTask) -> dict:
    with _engine.begin() as cx:
        t = registry.create_task(cx, owner=_s.owner_phone_number,
                                 goal=body.goal, title=body.title)
    _pool.submit(_start_session_bg, t.id)   # do NOT await: keeps this <200ms
    return {"ref": t.ref, "spoken": f"Started. I'll call it {t.ref.replace('-', ' ')}."}


@app.get("/internal/tasks")
def list_tasks() -> dict:
    """Open work AND work that has finished.

    `tasks` stays open-only because ops.status keys the "can this ring the owner?"
    check off it. `finished` is the half the agent was missing: asked on a call
    what had come back, it could only say "I can only see what's running right
    now, and that list is empty."
    """
    with _engine.begin() as cx:
        return {
            "tasks": [{"ref": t.ref, "title": t.title, "state": t.state,
                       "age_s": t.age_s}
                      for t in registry.open_tasks(cx, _s.owner_phone_number)],
            "finished": registry.finished_tasks(cx, _s.owner_phone_number),
        }


@app.get("/internal/tasks/{ref}")
def task_status(ref: str) -> dict:
    with _engine.begin() as cx:
        t = registry.get_task(cx, _s.owner_phone_number, ref)
        if t is None:
            return {"found": False}
        if t.session_id and t.state in ("running", "idle", "starting"):
            registry.refresh_from_anthropic(cx, t.id)
            t = registry.get_task(cx, _s.owner_phone_number, ref)
    return {"found": True, "ref": t.ref, "title": t.title, "state": t.state,
            "summary": t.result_summary, "age_s": t.age_s,
            # The full report, cleaned for speech. Without this a follow-up
            # like "what are the three shows?" had nowhere to go but a new
            # browsing task for work already done.
            "detail": registry.speakable(t.result_raw or "", limit=900)}


@app.post("/internal/callbacks")
def new_callback(body: NewCallback) -> dict:
    cid = schedule_callback(owner=_s.owner_phone_number, brief=body.brief,
                            reason=body.reason, delay_s=body.delay_s,
                            to_number=body.to_number)
    when = "right now" if body.delay_s < 60 else f"in about {body.delay_s // 60} minutes"
    target = body.to_number or _s.owner_phone_number
    return {"callback_id": cid, "to_number": target,
            "spoken": f"I'll call you back {when}."}


@app.post("/internal/calls/{callback_id}/outcome")
def call_outcome(callback_id: str, body: Outcome) -> dict:
    """The voice job reports how the callback went. This is the durable
    rendezvous that unblocks place_callback's DBOS.recv."""
    DBOS.send(f"cb-{callback_id}", body.outcome, topic="call_outcome")
    return {"ok": True}


@app.post("/internal/tasks/{ref}/cancel")
def cancel_task(ref: str) -> dict:
    """The owner asked for this to stop.

    USER-INITIATED ONLY -- see registry.cancel_task for why that matters. The DB
    work is one transaction and stays well under 200ms; capping Anthropic spend
    is a network call and goes to the pool unawaited, exactly like
    sessions.create does on the way in. The spoken answer is computed from the
    DB result alone, so it is true whether or not the Anthropic call lands.

    No request body on purpose: a `reason` field would be the first step toward
    a policy input on a decision that is his alone.
    """
    with _engine.begin() as cx:
        c = registry.cancel_task(cx, _s.owner_phone_number, ref)
    if c.session_id:
        _pool.submit(_stop_spend_bg, c.session_id)
    return {"found": c.found, "ref": c.ref, "cancelled": c.was_open,
            "callbacks_cancelled": c.callbacks_cancelled,
            "callback_already_fired": c.callback_already_fired,
            "spoken": registry.spoken_cancel(c)}


def _stop_spend_bg(session_id: str) -> None:
    try:
        registry.stop_session_spend(session_id)
    except Exception:
        # Never fatal. The task is already cancelled in Postgres and the owner has
        # already been told; a session that keeps spending is still bounded by
        # SESSION_BUDGET_CENTS and shows up in ops.status.
        log.exception("could not stop spend for session %s", session_id)


@app.get("/internal/context")
def call_context() -> dict:
    """What the agent should already know when a call starts."""
    with _engine.begin() as cx:
        summary = ctxmod.build_summary(cx, _s.owner_phone_number)
        if summary:
            ctxmod.save_summary(cx, _s.owner_phone_number, summary)
        open_n = len(registry.open_tasks(cx, _s.owner_phone_number))
        news = ctxmod.unreported(cx, _s.owner_phone_number)
    return {"summary": summary, "open_tasks": open_n, "news": news}


class CallReport(BaseModel):
    callback_id: str | None = None
    direction: str = "outbound"
    room_name: str
    livekit_job_id: str | None = None
    outcome: str
    amd_category: str | None = None
    amd_reason: str | None = None
    amd_delay_ms: int | None = None
    amd_speech_s: float | None = None
    sip_status_code: int | None = None
    pin_verified: bool | None = None
    # verified|wrong|no_input|abandoned -- WHY, where pin_verified says only
    # whether. Nullable, unlike pin_verified, so it can say "never asked".
    pin_result: str | None = None
    delivered: bool = False
    # "he heard nothing" vs "he cut me off part-way" are different failures with
    # different fixes, and `delivered=False` alone cannot tell them apart.
    heard_chars: int | None = None
    interrupted: bool | None = None


@app.post("/internal/calls")
def record_call(body: CallReport) -> dict:
    """Durable record of one call attempt, plus the DBOS rendezvous.

    shot.calls existed from day one with exactly the columns every failure
    needed -- amd_category, amd_reason, amd_speech_s, pin_verified -- and was
    never written to once. Diagnosing a bad callback therefore meant grepping a
    log file that gets deleted on every restart, and twice it was simply gone.
    """
    # Release the rendezvous FIRST. place_callback blocks on it for four
    # minutes, so a bookkeeping failure must never be able to hold up the
    # workflow -- which is exactly what happened when a NOT NULL violation on
    # room_name took the whole endpoint down with it.
    if body.callback_id:
        DBOS.send(f"cb-{body.callback_id}", body.outcome, topic="call_outcome")

    recorded = False
    try:
        with _engine.begin() as cx:
            cx.execute(sa.text("""
                INSERT INTO shot.calls (direction, owner_phone, room_name, livekit_job_id,
                                        callback_id, amd_category, amd_reason, amd_delay_ms,
                                        amd_speech_s, pin_verified, pin_result,
                                        sip_status_code,
                                        failure_kind, heard_chars, interrupted,
                                        started_at, ended_at, end_reason)
                VALUES (:d, :o, :rn, :jid, CAST(:c AS uuid), :ac, :ar, :ad, :as_, :pv,
                        :prr, :sc,
                        :fk, :hc, :itr, now(), now(), :er)"""),
                {"d": body.direction, "o": _s.owner_phone_number, "rn": body.room_name,
                 "jid": body.livekit_job_id, "c": body.callback_id,
                 "ac": body.amd_category, "ar": body.amd_reason, "ad": body.amd_delay_ms,
                 # NOT NULL with a default of false -- passing an explicit NULL
                 # overrides the default and fails the constraint. "was the PIN
                 # verified" is false whether it failed or was never asked; the
                 # difference lives in end_reason.
                 # LiveKit's AMD has reported a very slightly negative speech
                 # duration (-0.005s) when the classifier never saw a boundary.
                 # Floor it: 0.0 is the documented tell for the answer-and-wait
                 # deadlock, and a negative reads as a different bug entirely.
                 "as_": (max(0.0, body.amd_speech_s)
                         if body.amd_speech_s is not None else None),
                 "pv": bool(body.pin_verified),
                 "prr": body.pin_result,
                 "sc": body.sip_status_code,
                 "fk": None if body.delivered else body.outcome,
                 "hc": body.heard_chars, "itr": body.interrupted,
                 "er": body.outcome})
        recorded = True
    except Exception:
        log.exception("could not record call %s", body.room_name)
    return {"ok": True, "recorded": recorded}


class TranscriptReport(BaseModel):
    room_name: str
    direction: str
    turns: list[dict]
    callback_id: str | None = None


@app.post("/internal/transcripts")
def record_transcript(body: TranscriptReport) -> dict:
    """What was said on one call.

    Written at job shutdown, once. Upserted on room_name so a retry cannot
    duplicate a call. Like /internal/calls this must never be able to take a
    call down -- it is evidence, not control flow.
    """
    recorded = False
    try:
        with _engine.begin() as cx:
            cx.execute(sa.text("""
                INSERT INTO shot.transcripts (room_name, direction, callback_id, turns)
                VALUES (:rn, :d, CAST(:c AS uuid), CAST(:t AS jsonb))
                ON CONFLICT (room_name) DO UPDATE
                   SET turns = EXCLUDED.turns, created_at = now()"""),
                {"rn": body.room_name, "d": body.direction,
                 "c": body.callback_id, "t": json.dumps(body.turns)})
        recorded = True
    except Exception:
        log.exception("could not record transcript for %s", body.room_name)
    return {"ok": True, "recorded": recorded}


class Reported(BaseModel):
    task_ids: list[str]


@app.post("/internal/tasks/reported")
def tasks_reported(body: Reported) -> dict:
    """The agent has now SAID this out loud to the verified owner."""
    with _engine.begin() as cx:
        n = ctxmod.mark_reported(cx, body.task_ids)
    return {"marked": n}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "missing_resource_ids": _s.missing_resource_ids()}


DBOS.launch()
