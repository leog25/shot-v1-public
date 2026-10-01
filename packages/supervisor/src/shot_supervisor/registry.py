"""Task registry + the Managed Agents seam.

This module is the ONLY place allowed to touch client.beta.sessions.

The rule is NOT "no events.send()". This file has always contained one --
resolve_pending_approvals answers tool confirmations on a session parked idle on
`requires_action` -- and stop_session_spend now sends a second. The header said
otherwise for months, and claiming an invariant the code already breaks is worse
than having no invariant: the next reader either believes the comment or stops
believing all of them.

The real rule is: NEVER send a `user.message` to a RUNNING session. A
user.message is QUEUED rather than processed, so asking a running session for
its status hangs until the task finishes. Status is `retrieve` + `events.list`,
plain reads that do not touch the running turn. `user.tool_confirmation` and
`user.interrupt` are CONTROL events, not messages, and are not queued.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic
import sqlalchemy as sa

from shot_core.settings import get_settings
from shot_supervisor.approvals import review

_OPEN = ("queued", "starting", "running", "waiting_on_user", "idle")
_TERMINAL = {"succeeded", "failed", "cancelled", "budget_reached"}


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())


_MD_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_MD_NOISE = re.compile(r"[*_`#>]+")
_PATHY = re.compile(r"\S*/\S+\.\w+|\bhttps?://\S+")

# Sentences about the worker's own bookkeeping. The owner asked for showtimes, not a
# report on where the report was filed or which memory note got corrected --
# read aloud, this is the agent talking about itself. Matched anywhere in the
# sentence: the earlier version only caught it at the very end, so "Full
# write-up saved to outputs, and I logged the venue details, ..." sailed
# through into the greeting context.
_HOUSEKEEPING = re.compile(
    r"\b(saved to|written to|wrote (it |them )?to|full (write-?up|report|summary)"
    r"|logged (the|a|an)|memory (note|store)|prior session|outputs?\b.{0,12}$"
    r"|scraping recipe|cross-?checked against|to memory\b"
    r"|re-?scraped|verification to|month grid|list view)\b", re.I)

# Section headers from a written report. Spoken, "What I found dash Jazz Bistro
# has three shows" is a document being read at you rather than someone telling
# you something.
_LABEL = re.compile(
    r"(?:^|(?<=[.!?:;]\s))\s*(what i found|here'?s what i found|summary|tl;?dr"
    r"|result|results|findings|notes? worth flagging|notes?|bottom line"
    r"|the short version)\s*[-—:]+\s*", re.I)


def speakable(text: str, limit: int = 400) -> str | None:
    """Turn a worker's markdown report into something a phone can read out.

    Workers write for a file: tables, bold, and `/mnt/session/outputs/...`
    paths. Read aloud that becomes "asterisk asterisk What I found". Strip the
    markup, drop paths and URLs entirely, and keep the first few sentences.
    """
    if not text:
        return None
    rows = [r.strip(" |") for r in _MD_TABLE.findall(text)]
    body = _MD_TABLE.sub(" ", text)
    body = _LABEL.sub("", _PATHY.sub("", _MD_NOISE.sub("", body)))
    body = " ".join(body.split())
    # drop clauses that only existed to name a file we just removed
    keep = [c for c in re.split(r"(?<=[.!?])\s+", body)
            if len(re.sub(r"[^A-Za-z0-9]", "", c)) > 12
            and not _HOUSEKEEPING.search(c)]
    body = " ".join(keep) if keep else body
    # A table usually holds the actual answer -- three showtimes, say -- and
    # stripping it for speech is what left "has three shows today:" with
    # nothing after the colon. Fold the data rows back in, and reserve room for
    # them BEFORE trimming: appended after the prose they sat right on the
    # limit and were cut off, so the agent re-delegated a task to re-find what
    # it had already been told.
    data = [" ".join(c.strip() for c in r.split("|") if c.strip())
            for r in rows if not set(r) <= set("-| :")]
    rows_data = data[1:6] if len(data) > 1 else []
    # Any pipe that survived the table regex (ragged rows, nested markup) would
    # be read aloud as nothing at all, leaving a confusing gap. Drop them.
    body = re.sub(r"\s*\|\s*", ". ", body)
    body = re.sub(r"(\.\s*){2,}", ". ", body)
    body = " ".join(body.split()).lstrip(". ")
    if rows_data:
        # The rows ARE the answer, so they get the budget -- but never at the
        # cost of the lead-in. Trimming prose to fit the rows produced
        # "...has three shows today: the Señor.", and dropping it entirely made
        # the agent open with a bare timestamp. Keep the first sentence, then
        # fit whole rows, then whole extra sentences with whatever is left.
        sentences = [x for x in re.split(r"(?<=[.!?:])\s+", body) if x]
        intro = _trim(sentences[0], 140) if sentences else ""
        out = [intro] if intro else []
        used = len(intro)
        for row in rows_data:
            if used + len(row) + 2 > limit:
                break
            out.append(row)
            used += len(row) + 2
        for extra in sentences[1:]:
            if used + len(extra) + 2 > limit:   # ". " join, not " "
                break
            out.append(extra)
            used += len(extra) + 2
        body = ". ".join(x.rstrip(". ") if x is not intro else x for x in out)
        # the intro usually ends in ":" and the join adds "." -> "2026:."
        body = re.sub(r"([:;,])\s*\.\s*", r"\1 ", body)
    else:
        body = _trim(body, limit)
    return body.strip() or None


def _trim(body: str, limit: int) -> str:
    """Cut to a sentence boundary, or failing that a word boundary.

    Never a hard slice: that produced "(Night 2 is Sat", which the agent then
    read out loud exactly as written.
    """
    if len(body) <= limit:
        return body
    cut = body[:limit]
    if ". " in cut:
        return cut[:cut.rfind(". ") + 1]
    cut = cut[:cut.rfind(" ")] if " " in cut else cut
    return cut.rstrip(" ,;:(-—") + "."


def slugify(title: str, existing: set[str]) -> str:
    """A short ref the model can actually say out loud."""
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:24] or "task"
    ref, n = base, 2
    while ref in existing:
        ref, n = f"{base}-{n}", n + 1
    return ref


@dataclass(frozen=True)
class Task:
    id: str
    ref: str
    title: str
    state: str
    session_id: str | None
    result_summary: str | None
    result_raw: str | None
    age_s: int


def create_task(cx: sa.Connection, *, owner: str, goal: str, title: str) -> Task:
    refs = {r for (r,) in cx.execute(
        sa.text("SELECT ref FROM shot.tasks WHERE owner_phone=:o"), {"o": owner})}
    ref = slugify(title, refs)
    row = cx.execute(sa.text("""
        INSERT INTO shot.tasks (owner_phone, ref, title, goal, state)
        VALUES (:o,:r,:t,:g,'queued')
        RETURNING id, ref, title, state, session_id, result_summary,
                  result_raw, 0 AS age_s
    """), {"o": owner, "r": ref, "t": title, "g": goal}).mappings().one()
    return Task(**{**row, "id": str(row["id"])})


def start_session(cx: sa.Connection, task_id: str) -> str:
    """Create the worker session. Returns immediately: POST /v1/sessions comes
    back with the session already `running`, which is the real background-task
    primitive (the multiagent roster is not — a coordinator stays `running`
    until its children finish and queues anything you send it)."""
    s = get_settings()
    t = cx.execute(sa.text("SELECT goal, title, session_id FROM shot.tasks WHERE id=:i"),
                   {"i": task_id}).mappings().one()
    if t["session_id"]:
        return t["session_id"]

    sess = _client().beta.sessions.create(
        agent=s.anthropic_agent_id,
        environment_id=s.anthropic_environment_id,
        vault_ids=[s.anthropic_vault_id],
        resources=[{"type": "memory_store",
                    "memory_store_id": s.anthropic_memory_store_id,
                    "access": "read_write"},
                   # The signed-in browser. Mounted rather than described in the
                   # prompt: `browse` cannot attach a Browserbase Context itself,
                   # so reaching the owner's logged-in profile takes a real script, and
                   # a model improvising a multi-step curl is how that breaks.
                   *([{"type": "file",
                       "file_id": s.anthropic_browser_helper_file_id,
                       "mount_path": "/workspace/bb_connect.sh"}]
                     if s.anthropic_browser_helper_file_id else [])],
        title=t["title"][:120],
        metadata={"task_id": task_id, "owner": s.owner_phone_number},
        budget={"type": "limit",
                "max_list_cost": {"amount": str(s.session_budget_cents), "currency": "USD"}},
        initial_events=[{"type": "user.message",
                         "content": [{"type": "text", "text": t["goal"]}]}],
    )
    # session_id is UNIQUE, so a retried create can never fork a task in two.
    #
    # The session id is recorded UNCONDITIONALLY, even for a task cancelled
    # while this create was in flight -- losing the handle would mean a paid
    # session nobody can cap. Only the STATE is guarded, or a cancel that landed
    # a moment ago would be silently undone and the task set back to running.
    row = cx.execute(sa.text("""
        UPDATE shot.tasks
           SET session_id=:s,
               state = CASE WHEN cancelled_at IS NULL THEN 'running' ELSE state END,
               updated_at=now()
         WHERE id=:i AND session_id IS NULL
     RETURNING cancelled_at"""), {"s": sess.id, "i": task_id}).first()
    if row is not None and row[0] is not None:
        # Born already cancelled. Cap it now; nothing else will.
        try:
            stop_session_spend(sess.id)
        except Exception:
            _log().warning("could not cap a session that was cancelled while "
                           "it was starting: %s", sess.id, exc_info=True)
    return sess.id


def open_tasks(cx: sa.Connection, owner: str) -> list[Task]:
    rows = cx.execute(sa.text(f"""
        SELECT id, ref, title, state, session_id, result_summary, result_raw,
               EXTRACT(EPOCH FROM now()-created_at)::int AS age_s
          FROM shot.tasks
         WHERE owner_phone=:o AND state IN {_OPEN}
         ORDER BY updated_at DESC"""), {"o": owner}).mappings().all()
    return [Task(**{**r, "id": str(r["id"])}) for r in rows]


def finished_tasks(cx: sa.Connection, owner: str, limit: int = 6) -> list[dict]:
    """Work that is DONE, most recent first.

    `open_tasks` filters on the open states, so finished work was invisible to
    the agent entirely: asked what had come back, it could only answer "I can
    only see what's running right now, and that list is empty." A task could
    only be reached by guessing its exact reference.
    """
    rows = cx.execute(sa.text("""
        SELECT ref, title, state, result_summary, reported_at,
               EXTRACT(EPOCH FROM now()-finished_at)::int AS ago
          FROM shot.tasks
         WHERE owner_phone=:o AND finished_at IS NOT NULL
           AND state IN ('succeeded','failed','budget_reached')
           AND cancelled_at IS NULL
         ORDER BY finished_at DESC LIMIT :n"""),
        {"o": owner, "n": limit}).mappings().all()
    return [{"ref": r["ref"], "title": r["title"], "state": r["state"],
             "ago": _ago_s(r["ago"]), "told": r["reported_at"] is not None,
             "gist": speakable(r["result_summary"] or "", limit=160) or ""}
            for r in rows]


def _ago_s(seconds: int | None) -> str:
    if seconds is None:
        return "recently"
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes ago"
    if seconds < 172800:
        return f"{round(seconds / 3600)} hours ago"
    return f"{round(seconds / 86400)} days ago"


def get_task(cx: sa.Connection, owner: str, ref: str) -> Task | None:
    """Look up by ref, forgiving how it was spoken.

    The agent says refs aloud, so it hands back "jazz bistro schedule" for
    `jazz-bistro-schedule`. Matching strictly meant every status check during a
    call returned "I have no task called that" while the work was in fact
    running. Be liberal here: compare on letters and digits only.
    """
    want = re.sub(r"[^a-z0-9]", "", (ref or "").lower())
    if not want:
        return None
    rows = cx.execute(sa.text("""
        SELECT id, ref, title, state, session_id, result_summary, result_raw,
               EXTRACT(EPOCH FROM now()-created_at)::int AS age_s
          FROM shot.tasks WHERE owner_phone=:o
         ORDER BY created_at DESC"""), {"o": owner}).mappings().all()
    for r in rows:
        if re.sub(r"[^a-z0-9]", "", r["ref"].lower()) == want:
            return Task(**{**r, "id": str(r["id"])})
    # last resort: the model may quote the title instead of the ref
    for r in rows:
        if re.sub(r"[^a-z0-9]", "", (r["title"] or "").lower()) == want:
            return Task(**{**r, "id": str(r["id"])})
    return None


def resolve_pending_approvals(session_id: str) -> int:
    """Answer any tool confirmations the worker is parked on.

    Without this a delegated task stalls at `waiting_on_user` forever and no
    callback ever comes -- every command goes through bash, and bash is
    always_ask. Returns how many were resolved.
    """
    c = _client()
    idles = list(c.beta.sessions.events.list(
        session_id, types=["session.status_idle"], order="desc", limit=1))
    if not idles:
        return 0
    stop = getattr(idles[0], "stop_reason", None)
    if getattr(stop, "type", None) != "requires_action":
        return 0

    wanted = set(getattr(stop, "event_ids", []) or [])
    if not wanted:
        return 0
    cmds: dict[str, str] = {}
    for ev in c.beta.sessions.events.list(
            session_id, types=["agent.tool_use"], order="desc", limit=40):
        if ev.id in wanted:
            inp = getattr(ev, "input", None) or {}
            cmds[ev.id] = str(inp.get("command") or inp.get("file_path") or "") \
                if isinstance(inp, dict) else ""

    events = []
    for eid in wanted:
        d = review(cmds.get(eid, ""))
        events.append({"type": "user.tool_confirmation", "tool_use_id": eid,
                       "result": "allow" if d.allow else "deny",
                       **({} if d.allow else {"deny_message": d.reason})})
    if events:
        c.beta.sessions.events.send(session_id=session_id, events=events)
    return len(events)


def refresh_from_anthropic(cx: sa.Connection, task_id: str) -> str:
    """Plain reads only: retrieve + events.list. Neither touches the running turn,
    which is what makes 'what's the status of my order' safe to ask mid-call."""
    sid = cx.execute(sa.text("SELECT session_id FROM shot.tasks WHERE id=:i"),
                     {"i": task_id}).scalar_one_or_none()
    if not sid:
        return "queued"
    c = _client()
    sess = c.beta.sessions.retrieve(sid)
    stop = None
    if sess.status == "idle":
        evs = list(c.beta.sessions.events.list(
            sid, types=["session.status_idle"], order="desc", limit=1))
        if evs:
            stop = getattr(getattr(evs[0], "stop_reason", None), "type", None)

    state = {"running": "running", "rescheduling": "running",
             "terminated": "succeeded"}.get(sess.status, "idle")
    if sess.status == "idle":
        state = {"end_turn": "succeeded", "requires_action": "waiting_on_user",
                 "budget_reached": "budget_reached",
                 "retries_exhausted": "failed"}.get(stop or "", "idle")

    summary = raw = None
    if state in _TERMINAL:
        msgs = list(c.beta.sessions.events.list(
            sid, types=["agent.message"], order="desc", limit=1))
        if msgs:
            parts = [getattr(b, "text", "") for b in (getattr(msgs[0], "content", None) or [])]
            raw = " ".join(p for p in parts if p) or None
            summary = speakable(raw or "")

    # Kept for when the payment gate goes back on: with bash=always_allow the
    # worker never parks, so this is a no-op today.
    if state == "waiting_on_user":
        try:
            n = resolve_pending_approvals(sid)
            if n:
                state = "running"
        except Exception:  # never let the sweep die on one stuck task
            import logging
            logging.getLogger("shot.registry").warning(
                "could not resolve approvals for %s", sid, exc_info=True)

    cx.execute(sa.text("""
        UPDATE shot.tasks
           SET state=CAST(:st AS shot.task_state), stop_reason=:sr,
               result_summary=COALESCE(:sum, result_summary),
               result_raw=COALESCE(:raw, result_raw),
               updated_at=now(),
               finished_at=CASE WHEN :fin THEN COALESCE(finished_at, now()) ELSE finished_at END
         -- A cancel can land between the retrieve above and this write. Without
         -- this guard the sweep would set a task the owner just killed back to
         -- 'running', or to 'succeeded' -- which then schedules the very
         -- callback he asked us not to make.
         WHERE id=:i AND cancelled_at IS NULL"""),
        {"st": state, "sr": stop, "sum": summary, "raw": raw,
         "fin": state in _TERMINAL, "i": task_id})
    return state


def _log():
    import logging
    return logging.getLogger("shot.registry")


def _consumed_cents(sess) -> int:
    """What this session has already spent, in minor units."""
    usage = getattr(sess, "usage", None)
    cost = getattr(usage, "list_cost", None)
    try:
        return int(getattr(cost, "amount", 0) or 0)
    except (TypeError, ValueError):
        return 0


def stop_session_spend(session_id: str) -> None:
    """Stop a worker session spending any more of the owner's money.

    THE INTERRUPT FIRST, and independently. `user.interrupt` is what actually
    stops the turn that is running right now, and it must not be skipped because
    the ceiling call failed -- which is exactly what happened the first time this
    ran for real: the ceiling 400'd, the exception propagated, the interrupt
    never went out, and a cancelled task carried on to $2.35. It is a CONTROL
    event, not a `user.message`, so it is not queued behind the running turn.
    That is the second events.send() in this module and it is legal for the same
    reason as the first.

    Then the ceiling, which is the durable half: it survives even if the
    interrupt was lost. `max_list_cost` must be strictly GREATER than what the
    session has already consumed -- the API rejects anything lower with a 400,
    so a flat "1" is only ever valid on a session that has spent nothing.

    The margin matters. `consumed + 1` is a race: the session keeps spending
    between the read and the write, and a rejected ceiling means NO ceiling,
    which lets it run all the way to SESSION_BUDGET_CENTS. Losing that race
    costs far more than the margin does, so the first attempt allows a few cents
    of slack and the retry allows a dollar. A cap a dollar above where it stands
    is still much better than the $3 it would otherwise reach. Measured: a live
    browsing session moves about six cents a minute, and two API round trips
    take a second or two.

    Non-destructive on purpose. NOT sessions.delete -- that destroys exactly the
    evidence of what the worker did, and every piece of this system exists
    because evidence went missing once. NOT sessions.archive either: ops.status
    counts `sessions.list(statuses=["running"])` under the heading "these are
    spending money", so archiving a still-running session hides the one thing
    that line exists to show. There is no sessions.stop.
    """
    c = _client()
    try:
        c.beta.sessions.events.send(
            session_id=session_id, events=[{"type": "user.interrupt"}])
    except Exception:
        _log().warning("could not interrupt session %s; still setting a ceiling",
                       session_id, exc_info=True)

    margins = (5, 100)
    for i, margin in enumerate(margins):
        spent = _consumed_cents(c.beta.sessions.retrieve(session_id))
        ceiling = spent + margin
        try:
            c.beta.sessions.update(session_id, budget={
                "type": "limit",
                # Minor units, and strictly above consumed or the API 400s.
                "max_list_cost": {"amount": str(ceiling), "currency": "USD"}})
            _log().info("capped session %s at %d cents (consumed %d)",
                        session_id, ceiling, spent)
            return
        except Exception:
            if i == len(margins) - 1:
                raise
            _log().info("ceiling %d for %s was overtaken while reading; retrying "
                        "with more room", ceiling, session_id)


@dataclass(frozen=True)
class Cancellation:
    """What actually happened, in the facts the phone needs.

    Four things the caller cannot reconstruct afterwards: whether we stopped
    live work, what state it had already reached if we did not, whether we
    stopped a callback before it dialled, and whether one was already out of the
    gate. "Stopped, I won't call you back" and "that already finished" are
    different sentences, and the agent must not have to guess which is true.
    """
    found: bool
    ref: str = ""
    was_open: bool = False              # we stopped work that was still running
    already: str = ""                   # the terminal state it had already reached
    callbacks_cancelled: int = 0        # pending callbacks killed before dialling
    callback_already_fired: bool = False
    session_id: str | None = None       # so the caller can cap spend off the request path


def cancel_task(cx: sa.Connection, owner: str, ref: str) -> Cancellation:
    """Stop a task, and the callback it would have triggered.

    USER-INITIATED ONLY. Nothing in this process may call this on its own
    judgement -- not on an interruption, not on a timeout, not because a task is
    slow, not because the agent has decided the work was a mistake. An automatic
    cancel is indistinguishable from a bug that silently drops his work, and by
    construction there is no callback left to tell him it happened: he finds out
    by asking for something that no longer exists. The single legitimate caller
    is POST /internal/tasks/{ref}/cancel, whose single legitimate caller is the
    cancel_task voice tool, reached only because the owner said so out loud.
    tests/test_registry.py asserts that call graph.
    """
    t = get_task(cx, owner, ref)
    if t is None:
        return Cancellation(found=False, ref=ref)

    # 1. Stamp cancelled_at whatever the state. It means "the owner said stop", which
    #    is true of a task that had already finished -- he still does not want
    #    to hear about it, and every query that decides what to tell him reads
    #    this column. The STATE only moves to 'cancelled' if there was live work
    #    to stop: you cannot un-finish work, and claiming otherwise in the record
    #    would be a lie. Race-safe against refresh_from_anthropic either way.
    row = cx.execute(sa.text("""
        UPDATE shot.tasks
           SET cancelled_at=now(),
               state = CASE
                   WHEN state IN ('queued','starting','running','waiting_on_user','idle')
                   THEN 'cancelled' ELSE state END,
               -- finished_at means "stopped running", which is now true. Leaving
               -- it NULL would make cancelled the only terminal state with no
               -- stop time and break every _ago() that reads it. A DIFFERENT
               -- fact from cancelled_at, which is why both columns exist.
               finished_at=COALESCE(finished_at, now()),
               updated_at=now()
         WHERE id=CAST(:i AS uuid) AND cancelled_at IS NULL
     RETURNING state"""), {"i": t.id}).first()
    was_open = row is not None and row[0] == "cancelled"

    # 2. Kill any callback that has not dialled yet -- the same three predicates
    #    as claim_callback, so fire-vs-cancel has exactly one winner.
    killed = cx.execute(sa.text("""
        UPDATE shot.callbacks SET cancelled_at=now()
         WHERE task_id=CAST(:i AS uuid) AND fired_at IS NULL AND cancelled_at IS NULL
     RETURNING id"""), {"i": t.id}).fetchall()

    # 3. Is one already out of the gate? place_callback stamps fired_at at the
    #    top and writes outcome only after the rendezvous, so this pair means
    #    "ringing right now", which nothing can recall.
    ringing = cx.execute(sa.text("""
        SELECT count(*) FROM shot.callbacks
         WHERE task_id=CAST(:i AS uuid) AND fired_at IS NOT NULL AND outcome IS NULL"""),
        {"i": t.id}).scalar_one()

    return Cancellation(
        found=True, ref=t.ref, was_open=was_open,
        # "" when we stopped live work; otherwise what it had already become --
        # and 'cancelled' when this is the second time he has asked.
        already="" if was_open else ("cancelled" if row is None else t.state),
        callbacks_cancelled=len(killed),
        callback_already_fired=bool(ringing),
        # Even when the task was already terminal, a session may still be live
        # and worth capping -- budget_reached in particular.
        session_id=t.session_id,
    )


def spoken_cancel(c: Cancellation) -> str:
    """What to say. It must never claim more than actually happened.

    Three different truths, and the agent cannot tell them apart on its own:
    work was stopped; work had already finished and only the callback was
    stopped; a callback was already dialling and cannot be recalled. Saying
    "stopped, I won't call you back" about a call that is already ringing is
    exactly the kind of confident wrong answer he would act on.
    """
    if not c.found:
        return f"I have no task called {c.ref or 'that'}."
    if c.was_open:
        if c.callback_already_fired:
            return ("Stopped. A call about it was already going out, though, so "
                    "I can't pull that one back.")
        return "Stopped — I won't call you back about it."
    if c.already == "cancelled":
        return "I'd already stopped that one."
    if c.callback_already_fired:
        return ("That one already finished, and the call about it is already on "
                "its way — I can't pull that back.")
    if c.callbacks_cancelled:
        return ("That one already finished, but I've cancelled the callback, so "
                "I won't ring you about it.")
    return "That one already finished, so there's nothing left to stop."
