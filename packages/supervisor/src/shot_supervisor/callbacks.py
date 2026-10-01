"""Durable callbacks: the timer that survives a restart, and the dialer.

Three independent paths can fire the same callback — the delayed enqueue, the
due-callback sweep, and a manual retrigger. They all converge on claim_callback,
and exactly one wins. A double-fire means the phone rings twice and the owner is told
the same thing twice, so this is stacked four deep:

  1. claim_callback()                  compare-and-set in Postgres
  2. SetWorkflowID("cb-<id>")          DBOS workflow-id idempotency
  3. callbacks_one_pending_per_task    partial unique index
  4. queue global_concurrency=1        never two live outbound calls

Any one of them is sufficient. Together they cost nothing at runtime.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from dbos import DBOS, Queue, SetEnqueueOptions, SetWorkflowID

from shot_core.budget import DBOS_RECV_S
from shot_core.db import claim_callback, make_engine
from shot_core.settings import get_settings

log = logging.getLogger("shot.callbacks")

# never two outbound calls at once — the user only has one phone
callbacks_q = Queue("callbacks", global_concurrency=1)

_engine = make_engine(get_settings().database_url)

# Retry backoff, in seconds. Ring-out waits longest: they are probably busy,
# not absent. Busy is short: they declined one specific call.
# Nothing redials. A callback that does not reach the owner leaves its task
# unreported, and the result is delivered when he next calls in -- `reported_at`
# is what makes it resurface.
#
# Automatic retries rang him three times in one evening while a systematic bug
# went unfixed, each attempt failing exactly as the last, and a redial is the
# one failure mode with a real-world cost attached. Re-trigger by hand with
# POST /internal/callbacks when you actually want one.
RETRYABLE: set[str] = set()

# Deliberately NOT retryable. A failed identity check means either it was not
# the owner who answered -- in which case calling back is exactly the wrong move --
# or it was the owner and he fumbled it, where an unrequested redial half an hour
# later is still presumptuous. Terminal; he can always call in.
TERMINAL = {"human", "pin_failed"}


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
def dispatch_voice_agent(room: str, metadata: dict) -> str:
    """CreateDispatch ONLY. The dispatched job creates the SIP participant
    itself, because AMD needs a live AgentSession to attach to."""
    import asyncio

    from livekit import api

    s = get_settings()

    async def _go() -> str:
        async with api.LiveKitAPI(s.livekit_url, s.livekit_api_key,
                                  s.livekit_api_secret.get_secret_value()) as lk:
            d = await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
                agent_name="shot-voice", room=room, metadata=json.dumps(metadata)))
            return d.id

    return asyncio.run(_go())


@DBOS.workflow(name="place_callback", max_recovery_attempts=3)
def place_callback(callback_id: str) -> str:
    with _engine.begin() as cx:
        if not claim_callback(cx, callback_id):
            log.info("callback %s already claimed", callback_id)
            return "already_fired"
        row = cx.execute(sa.text("""
            SELECT c.owner_phone, COALESCE(c.to_number, c.owner_phone) AS to_number,
                   c.brief, c.attempt, c.max_attempts, t.ref, t.title, c.task_id
              FROM shot.callbacks c LEFT JOIN shot.tasks t ON t.id = c.task_id
             WHERE c.id = :c"""), {"c": callback_id}).mappings().one()

    room = f"cb-{callback_id}"
    with _engine.begin() as cx:
        cx.execute(sa.text("UPDATE shot.callbacks SET room_name=:r WHERE id=:c"),
                   {"r": room, "c": callback_id})

    dispatch_voice_agent(room, {
        "to_number": row["to_number"],
        "brief": row["brief"],
        "callback_id": callback_id,
        "require_pin": get_settings().pin_required(direction="outbound"),
        # So the job can record that the owner actually HEARD this, once the brief
        # has finished playing to a PIN-verified human.
        "task_id": str(row["task_id"]) if row["task_id"] else None,
        # So the greeting can name what came back rather than saying "what you
        # asked me to look into", which made the owner do the remembering.
        "task_titles": [row["title"]] if row["title"] else [],
    })

    # Durable rendezvous: the voice job posts its outcome to the supervisor,
    # which DBOS.send()s it here. Blocking a pooled thread is acceptable
    # precisely because this queue is global_concurrency=1.
    outcome = DBOS.recv(topic="call_outcome", timeout_seconds=DBOS_RECV_S) or "no_report"

    with _engine.begin() as cx:
        cx.execute(sa.text("UPDATE shot.callbacks SET outcome=:o WHERE id=:c"),
                   {"o": outcome, "c": callback_id})
    # No redial, ever -- see RETRYABLE. The task stays unreported and the owner gets
    # it on his next inbound call.
    log.info("callback %s finished as %s; not redialling", callback_id, outcome)
    return outcome


def enqueue_callback(callback_id: str, due_at: datetime) -> None:
    """Schedule the dial. DBOS persists the delay in Postgres, so it survives a
    supervisor restart — which is why there is no second Cloud Tasks path."""
    delay = max(0.0, (due_at - datetime.now(UTC)).total_seconds())
    with SetWorkflowID(f"cb-{callback_id}"), SetEnqueueOptions(
            deduplication_id=f"cb-{callback_id}",
            duplication_policy="return-existing",
            delay_seconds=delay):
        callbacks_q.enqueue(place_callback, callback_id)


def schedule_callback(*, owner: str, brief: str, reason: str = "task_done",
                      task_id: str | None = None, delay_s: int = 0,
                      to_number: str | None = None) -> str:
    """Queue a callback. `to_number` defaults to the owner; pass it explicitly
    to dial somewhere else (tests point it at a Twilio test number so a probe
    can never ring a real phone)."""
    due = datetime.now(UTC) + timedelta(seconds=delay_s)
    with _engine.begin() as cx:
        cid = str(cx.execute(sa.text("""
            INSERT INTO shot.callbacks
                   (owner_phone, task_id, reason, brief, due_at, to_number)
            VALUES (:o,:t,CAST(:r AS shot.callback_reason),:b,:d,:n) RETURNING id"""),
            {"o": owner, "t": task_id, "r": reason, "b": brief, "d": due,
             "n": to_number}).scalar_one())
    enqueue_callback(cid, due)
    return cid


@DBOS.scheduled("* * * * *")
@DBOS.workflow(name="sweep_due_callbacks")
def sweep_due_callbacks(scheduled: datetime, actual: datetime) -> None:
    """Safety net for a delayed enqueue that was never written or was lost.

    Safe to run alongside the primary path *because* of SetWorkflowID plus the
    CAS guard: a duplicate is a no-op, not a second phone call.
    """
    with _engine.begin() as cx:
        ids = [str(r[0]) for r in cx.execute(sa.text("""
            SELECT id FROM shot.callbacks
             WHERE fired_at IS NULL AND cancelled_at IS NULL AND due_at <= now()
             LIMIT 20"""))]
    for cid in ids:
        enqueue_callback(cid, datetime.now(UTC))


@DBOS.scheduled("*/2 * * * *")
@DBOS.workflow(name="reconcile_sessions")
def reconcile_sessions(scheduled: datetime, actual: datetime) -> None:
    """The webhook backstop.

    Anthropic webhooks are explicitly not a durable log: 3 attempts, then the
    event is dropped with no signal. This sweep is what makes a dropped webhook
    cost latency instead of correctness, and it is sufficient on its own.
    """
    from shot_supervisor import registry

    with _engine.begin() as cx:
        open_ids = [(str(r[0]), r[1]) for r in cx.execute(sa.text("""
            SELECT id, ref FROM shot.tasks
             WHERE session_id IS NOT NULL
               AND state IN ('starting','running','idle','waiting_on_user')"""))]
    for tid, _ref in open_ids:
        with _engine.begin() as cx:
            registry.refresh_from_anthropic(cx, tid)

    # Notify separately from refreshing. A task can reach a terminal state via
    # ANY path -- most often because the owner asked "how's it going" mid-call and the
    # status endpoint refreshed it -- and the notification must not depend on
    # this sweep being the one that moved it. Backed by tasks_owner_unreported_idx.
    notify_finished_tasks()


def speakable_failure(reason: str) -> str:
    """One clause a callback can READ ALOUD. Never the raw exception.

    A task that never started has no `result_raw`, so brief_for() falls back
    to `result_summary` -- which means whatever lands there is spoken down the
    phone verbatim, under a prompt that says to read out every fact. This used
    to be `str(e)[:200]`, with only the out-of-credit case translated, and on
    2026-09-10 the owner was read: "Error code four hundred. Type error. The error
    type is invalid request error. The message says MCP server hosts blocked
    by environment network policy..." before it stopped mid-word at "Add these
    hosts to the e".

    The raw text is NOT lost -- it goes to `error` jsonb, which is where an
    investigation looks and where the request_id survives. Only the half that
    gets spoken is translated. Same split as `shot.calls` versus the log: the
    record keeps everything, the phone gets a sentence.

    The fallback is deliberately vague rather than raw. An unrecognised
    failure is a bug to read in `error`, not JSON to recite at someone.
    """
    if "credit balance is too low" in reason:
        return "the Anthropic account is out of credit"
    if "blocked by environment network policy" in reason:
        return ("the worker environment is not allowed to reach one of its "
                "tools, so the session could not be set up")
    if "rate_limit" in reason or "429" in reason:
        return "the account is being rate limited"
    return "the worker session could not be started"


def brief_for(row) -> str:
    """What the callback actually reads out.

    result_raw FIRST, not result_summary. `summary` is speakable() trimmed to a
    sentence boundary, and a worker's opening sentence is usually what it DID
    rather than what it FOUND -- "I pulled the National Weather Service forecast
    for Portland, plus the hourly breakdown and the severe weather outlook":
    140 characters with no forecast in them.

    DELIVER_CALLBACK then tells the model to read out every fact, and there are
    none, so it goes looking. It called list_my_tasks and read the owner a MENU of his
    finished tasks instead -- and because a turn did play, `delivered=True` was
    recorded and `reported_at` stamped, filing a forecast he had never heard as
    told to him. He rang back ninety seconds later to ask for it.

    Same lesson as OPENING_INBOUND_NEWS: an instruction the model cannot follow
    is worse than none, and when a prompt asserts the model knows something,
    check what is actually interpolated.

    700 rather than the 900 the inbound `detail` uses: a ~400-character brief
    reads aloud in 35-45s, so 700 lands near 70s against BRIEF_PLAYOUT_S of 100
    and still leaves margin.
    """
    # Function-local, matching reconcile_sessions: registry pulls in the
    # Anthropic SDK and this module is imported at queue-registration time.
    from shot_supervisor.registry import speakable

    return (speakable(row["result_raw"] or "", limit=700)
            or row["result_summary"]
            or f"Your task {row['ref'].replace('-', ' ')} finished as {row['state']}.")


@DBOS.workflow(name="notify_finished_tasks")
def notify_finished_tasks() -> int:
    """Schedule a callback for every finished-but-unreported task. Idempotent:
    notified_at is claimed in the same statement that selects the row."""
    with _engine.begin() as cx:
        rows = cx.execute(sa.text("""
            UPDATE shot.tasks SET notified_at = now()
             WHERE id IN (
                 SELECT id FROM shot.tasks
                  WHERE finished_at IS NOT NULL
                    AND notified_at IS NULL
                    AND state IN ('succeeded','failed','budget_reached')
              -- He told us to stop. Without this, cancelling a task that had
              -- ALREADY finished still let the sweep schedule a fresh callback
              -- about it -- which is the exact call he asked us not to make.
              AND cancelled_at IS NULL
                  ORDER BY finished_at LIMIT 10
                  FOR UPDATE SKIP LOCKED)
         RETURNING id, ref, owner_phone, result_summary, result_raw, state""")).mappings().all()

    for r in rows:
        schedule_callback(owner=r["owner_phone"], brief=brief_for(r),
                          task_id=str(r["id"]), delay_s=0)
        log.info("task %s finished; callback scheduled", r["ref"])
    return len(rows)
