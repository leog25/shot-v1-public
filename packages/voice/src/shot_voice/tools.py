"""Voice tools, bucketed by measured p95 latency.

  <200ms   inline, silent
  200-800  inline + one spoken preamble
  0.8-5s   async tool with a filler
  >5s      delegate to a worker session

The supervisor-backed tools are in the first bucket: it is on localhost and each
endpoint is one transaction. `web_search` is NOT -- it calls Brave, and with its
internal retry it is budgeted at WEB_SEARCH_TOTAL_S, the top of the third
bucket. That is deliberate, and it is the ceiling: anything that cannot answer
inside it belongs in a background task rather than here.

`check_my_day` is in the SECOND bucket: one Linear GraphQL POST (0.28-0.42s,
measured against the real API) run CONCURRENTLY with the supervisor's task list,
so a turn costs the slower leg rather than the sum. Serialised -- or worse, left
to the model to chain as two separate tool calls -- it is two round trips inside
a conversational turn.

This docstring used to claim nothing here touched the network. That had been
false since the day web_search was added, and it was read as a reason not to
look at how long a search actually took.
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx
from livekit.agents import RunContext, function_tool

from shot_core.budget import (
    CHECK_MY_DAY_S,
    LINEAR_QUERY_S,
    LINEAR_WRITE_S,
    SUPERVISOR_HTTP_S,
    WEB_SEARCH_ATTEMPT_S,
    WEB_SEARCH_TOTAL_S,
)
from shot_core.clock import now_local
from shot_core.settings import get_settings
from shot_voice.prompts import OWNER, owner_named

log = logging.getLogger("shot.tools")

# Was a literal 5.0, which is what SUPERVISOR_HTTP_S already says. Every timeout
# comes from budget.py -- two numbers agreeing by coincidence is the failure
# that table exists to prevent.
_client = httpx.AsyncClient(base_url=get_settings().supervisor_base_url,
                            timeout=SUPERVISOR_HTTP_S)
# Brave is the only search API fast enough to stay inside a conversational turn
# (~0.4s warm, 0.9s cold, measured). Anything slower gets delegated instead.
_brave = httpx.AsyncClient(
    base_url="https://api.search.brave.com/res/v1",
    headers={"X-Subscription-Token": get_settings().brave_api_key.get_secret_value(),
             "Accept": "application/json"},
    timeout=WEB_SEARCH_ATTEMPT_S)


@function_tool(on_duplicate="reject")
@owner_named
async def start_background_task(ctx: RunContext, goal: str, short_title: str) -> str:
    """Hand work to a background worker that keeps running after the call ends.

    This is expensive and it is LOUD. It opens a paid worker session -- up to
    three dollars -- and when that session finishes the supervisor RINGS THE OWNER'S
    PHONE to read him the result, whatever time it is and whether or not he
    still cares. Once started, the only thing that stops it is him asking you to
    cancel it.

    Use it when the answer needs a page to be OPENED: a site, a schedule, a
    menu, a listing, availability, a logged-in account, a purchase, a booking,
    a form filled in.

    NEVER use it for something a search could answer -- the weather, a score, a
    price, who won, what time a place opens -- and not even when the search came
    back thin. A thin search is a bad query, not a job for a worker: web_search
    already retried itself once with a sharper one before it answered you. If
    two passes found nothing, say so and let the owner decide. Ringing him back in two
    minutes to tell him it will be hot tomorrow is worse than saying you could
    not find it.

    NOT for re-finding something you already reported. If he is asking about
    work you already did -- the details, the times, which one, read me the rest
    -- that is get_task_status. Starting a fresh task makes him wait twice for
    the same answer, and rings his phone to do it.

    goal: the full instruction for the worker, in one or two sentences.
    short_title: three or four words naming the task.
    """
    r = await _client.post("/internal/tasks", json={"goal": goal, "title": short_title})
    r.raise_for_status()
    return r.json()["spoken"]


@function_tool
@owner_named
async def list_my_tasks(ctx: RunContext) -> str:
    """Every background task — both still running AND already finished.

    Use this for "what's running", "what came back", "did anything finish",
    "what have you got for me", or any question about work you were given
    earlier. It covers completed work too, so never tell the owner you can only see
    what is running.

    For the full findings of one of them, call get_task_status with its
    reference. Read-only.

    NOT the tool for "what's on today" or "what's on my plate" -- that is
    check_my_day, which covers his Linear tickets as well as this.
    """
    r = await _client.get("/internal/tasks")
    d = r.json()
    parts: list[str] = []
    for t in d.get("tasks") or []:
        parts.append(f"{t['title']} is still {t['state']}, reference {t['ref']}")
    for f in d.get("finished") or []:
        if f["state"] == "succeeded":
            gist = f" — {f['gist']}" if f["gist"] else ""
            parts.append(f"{f['title']} finished {f['ago']}, "
                         f"reference {f['ref']}{gist}")
        else:
            parts.append(f"{f['title']} did not complete ({f['state']}), {f['ago']}")
    if not parts:
        return "Nothing running, and nothing has finished recently."
    return ". ".join(parts)


@function_tool
async def get_task_status(ctx: RunContext, task_ref: str) -> str:
    """Status AND full findings of one background task, by reference name.

    Read-only and fast. Use this for any follow-up about work already done —
    "what were the times", "which one was first", "read me the rest" — before
    even considering a search or a new task. The full report is here; the
    headline you already said out loud was only its first line.

    Never guess at a task's status — always call this.
    """
    from urllib.parse import quote

    r = await _client.get(f"/internal/tasks/{quote(task_ref, safe='')}")
    d = r.json()
    if not d.get("found"):
        return f"I have no task called {task_ref}."
    state = d["state"]
    body = d.get("detail") or d.get("summary")
    if body:
        return f"{d['ref']} is {state}. {body}"
    if state in ("succeeded", "failed", "cancelled", "budget_reached"):
        # Terminal with nothing to read out. The old line claimed it was
        # "running for about N seconds", which was simply false.
        return (f"{d['ref']} is {state}, and there is nothing more to read out. "
                f"That is the whole answer -- do not check it again.")
    # A non-answer that invites another poll is how one call spent three turns
    # saying "it's still running" while the owner waited.
    return (f"{d['ref']} is still {state} and nothing has come back yet. Tell "
            f"{OWNER} that in one line and move on: he will be called the moment it "
            f"finishes, so there is nothing to gain by checking again on this "
            f"call, and checking twice makes it sound like something is wrong.")


@function_tool
@owner_named
async def cancel_task(ctx: RunContext, task_ref: str) -> str:
    """Stop a background task, and cancel the callback it would have triggered.

    Call this ONLY when the owner has said so -- "cancel that", "stop it", "forget
    it", "don't call me about that", "never mind". It is his decision, and the
    only signal that it is his decision is that he said it.

    Do NOT call this because a task is slow, because he sounds impatient,
    because you now think a search would have been quicker, because he
    interrupted you, or because the call is ending. "It's taking a while" is not
    a cancellation -- say how it is going and leave it running. Cancelling work
    he still wants loses it silently: there is no callback left to tell him it
    is gone, and he finds out by asking for something that no longer exists.

    Read the answer back honestly rather than promising more than it says. A
    task that has already finished cannot be un-finished, and a call already
    dialling cannot be recalled -- the answer says which of those happened.

    task_ref: the reference or title of the task, as it was said out loud.
    """
    from urllib.parse import quote

    r = await _client.post(f"/internal/tasks/{quote(task_ref, safe='')}/cancel")
    r.raise_for_status()
    return r.json()["spoken"]


# --- what's on today ------------------------------------------------------
#
# One tool, one turn, two concurrent legs. Linear's MCP toolset could answer the
# same question and deliberately does not: it is 65 tools and 78KB of schema
# whose list_issues returns 4,625 characters of JSON -- full of UUIDs and URLs
# -- for five issues, against a session prompt that says never to read out IDs.
# And asking the model to chain "tickets" and "tasks" itself is two round trips
# inside a conversational turn.
#
# That was written when the toolset sat beside this tool. It no longer does --
# see the ticket section below for what it cost and what replaced it.

# The RAW key, with no "Bearer " prefix: Linear personal API keys are not bearer
# tokens, and "Bearer <key>" is a 401 that reads exactly like a bad key. This
# used to be one of TWO conventions for one secret in one repo -- worker.py sent
# Bearer to the MCP endpoint for the same key -- which is a trap, and it is gone
# from this tier. Bearer is now tier 3's business alone (ops.bootstrap
# .anthropic_res registers it in the Anthropic vault). Pinned by tests.
_LINEAR_KEY = get_settings().linear_api_key_voice.get_secret_value()
_linear = httpx.AsyncClient(
    base_url="https://api.linear.app",
    headers={"Authorization": _LINEAR_KEY, "Content-Type": "application/json"},
    timeout=LINEAR_QUERY_S)

# Fetch what we intend to speak, not a page of it.
_DAY_N = 5

# Both halves in ONE round trip, via two aliased root fields. Verified live at
# 0.28-0.42s. `issues` is deliberately NOT assignee-filtered: it is the fallback
# for a board where nothing is assigned yet, and it is deduped against the first
# list below. `identifier` is selected only to dedupe -- it never reaches the
# model. No `id` and no `url` are selected at all, so nothing UUID- or URL-
# shaped can leak even if the projection below is later loosened.
_DAY_QUERY = """
query Day($n: Int!) {
  viewer {
    assignedIssues(first: $n, orderBy: updatedAt,
                   filter: {state: {type: {nin: ["completed", "canceled"]}}}) {
      nodes { identifier title dueDate state { name } }
    }
  }
  issues(first: $n, orderBy: updatedAt,
         filter: {state: {type: {nin: ["completed", "canceled"]}}}) {
    nodes { identifier title dueDate state { name } }
  }
}
"""


class LinearRejected(RuntimeError):
    """A GraphQL error. NOT an HTTP error -- see _linear_day."""


async def _gql(query: str, variables: dict | None = None) -> dict:
    """POST one GraphQL document, return `data`. RAISES on failure.

    A Linear GraphQL failure is an HTTP **200** with a top-level `errors`
    array, so `raise_for_status()` sees nothing at all. That check lived inline
    in `_linear_day` while it was the only caller. It is not any more, and a
    caller that forgot it would read a rejection as an empty workspace -- the
    exact "nothing there" versus "could not look" confusion the tools below
    exist to keep apart. One copy, so nobody has to remember.
    """
    r = await _linear.post("/graphql",
                           json={"query": query, "variables": variables or {}})
    r.raise_for_status()
    body = r.json() or {}
    if body.get("errors"):
        first = (body["errors"] or [{}])[0]
        raise LinearRejected(str(first.get("message") or first)[:200])
    return body.get("data") or {}


async def _linear_day(n: int) -> tuple[list[dict], list[dict]]:
    """(assigned to the owner, everything else still open). RAISES on failure.

    Raising rather than returning empties is the point: "Linear says you have
    nothing" and "I could not reach Linear" are different answers and must be
    said differently. Same distinction as `NO_INPUT` versus `ABANDONED` on the
    PIN -- "I could not ask" is not "he got it wrong".
    """
    data = await _gql(_DAY_QUERY, {"n": n})
    mine = ((data.get("viewer") or {}).get("assignedIssues") or {}).get("nodes") or []
    every = (data.get("issues") or {}).get("nodes") or []
    seen = {i.get("identifier") for i in mine}
    return mine, [i for i in every if i.get("identifier") not in seen]


async def _supervisor_day() -> tuple[list[dict], list[dict]]:
    """(still running, finished that he has NOT been told about). Raises."""
    r = await _client.get("/internal/tasks")
    r.raise_for_status()
    d = r.json() or {}
    # `told` is reported_at: work he has already heard is not news today, and
    # re-announcing it is what reported_at exists to prevent.
    return (list(d.get("tasks") or []),
            [f for f in (d.get("finished") or []) if not f.get("told")])


def _leg(task, label: str, default):
    """(value, ok). `ok=False` means the leg timed out or blew up, which is a
    different thing from it answering "nothing" -- and they are said
    differently."""
    if task is None:
        return default, False
    if task.cancelled() or not task.done():
        log.warning("check_my_day: the %s leg did not answer in time", label)
        return default, False
    if (exc := task.exception()) is not None:
        log.warning("check_my_day: the %s leg failed: %s: %s",
                    label, type(exc).__name__, exc)
        return default, False
    return task.result(), True


def _due(issue: dict, today) -> str:
    """"overdue" | "due today" | "due Friday" | "due 3 October" | "". Pure."""
    from datetime import date

    raw = (issue.get("dueDate") or "")[:10]
    if not raw:
        return ""
    try:
        d = date.fromisoformat(raw)
    except ValueError:
        return ""
    if d < today:
        return "overdue"
    if d == today:
        return "due today"
    if (d - today).days <= 6:
        return f"due {d:%A}"
    return f"due {d:%-d %B}"


_RANK = {"overdue": 0, "due today": 1, "": 3}


def _title(issue: dict) -> str:
    """Titles only, never identifiers. `GAO-7` would put this return string in
    direct conflict with the session prompt's "never read out IDs" -- and the
    ref.replace('-', ' ') convention exists for word-shaped task refs
    (`harbor-lights-friday`), not for alphanumeric tokens that carry nothing the owner can
    act on over a phone."""
    return " ".join((issue.get("title") or "").split())[:80]


def _issues(items: list[dict], today) -> list[str]:
    """Most urgent first, spoken."""
    ranked = sorted(items, key=lambda i: (_RANK.get(_due(i, today), 2),
                                          (i.get("dueDate") or "")))
    out = []
    for i in ranked:
        title = _title(i)
        if not title:
            continue
        when = _due(i, today)
        out.append(f"{title}, {when}" if when else title)
    return out


def _speak_day(mine, others, running, finished, *, linear_ok: bool,
               linear_on: bool, today) -> str:
    """The whole answer as one speakable string. Pure, so it tests without a
    clock and without a network."""
    parts: list[str] = []

    if linear_on and linear_ok:
        if said := _issues(mine, today):
            parts.append(f"On your plate in Linear: {'; '.join(said[:3])}."
                         + (f" And {len(said) - 3} more." if len(said) > 3 else ""))
        elif said := _issues(others, today):
            parts.append(f"Nothing is assigned to you in Linear, but "
                         f"{len(said)} still open on the board: "
                         f"{'; '.join(said[:3])}.")
        else:
            parts.append("No open Linear tickets.")
    elif linear_on:
        # Configured but unreachable. Say so once, in passing.
        parts.append("(You have Linear, but I could not reach it just then -- "
                     "mention that once in passing and do not offer to retry.)")
    # Not configured at all says NOTHING about Linear. "I could not reach it"
    # when it was never set up is a lie about a structural absence.

    work = []
    if running:
        names = [" ".join((t.get("title") or "").split()) for t in running]
        work.append(f"still running: {', '.join(n for n in names[:3] if n)}")
    if finished:
        names = [" ".join((t.get("title") or "").split()) for t in finished]
        work.append(f"finished and you have not heard it yet: "
                    f"{', '.join(n for n in names[:3] if n)}")
    if work:
        parts.append("Background work -- " + "; ".join(work) + ".")
    elif running is not None and not running and not finished:
        parts.append("Nothing running in the background.")

    if not parts:
        return "Nothing on your plate that I can see."
    # An all-clear has to read as an ANSWER, not as a shrug.
    if (linear_on and linear_ok and not mine and not others
            and not running and not finished):
        return ("Nothing on your plate: no open Linear tickets, nothing running "
                "in the background, and nothing new that finished.")
    return " ".join(parts)


@function_tool
@owner_named
async def check_my_day(ctx: RunContext) -> str:
    """What is on the owner's plate today -- his Linear tickets AND his background work.

    Use this for "what's on today", "what's on my plate", "what am I meant to be
    doing", "run me through the day", "anything I should know about" -- and
    whenever you have just offered to go through what else is on.

    ONE call covers both halves. Do NOT also call list_my_tasks, and do NOT go
    ticket by ticket with find_linear_issue: this is the whole picture in one
    round trip, and assembling it yourself is several inside a single turn.

    Say one short line first, like "let me look", then answer -- it takes about
    half a second. Say it as prose and never as a list: give the shape of it,
    name the first two or three, and let the owner ask for the rest. Read-only; it
    changes nothing.
    """
    legs: set[asyncio.Task] = set()
    linear_t = None
    if _LINEAR_KEY:
        linear_t = asyncio.create_task(_linear_day(_DAY_N))
        legs.add(linear_t)
    tasks_t = asyncio.create_task(_supervisor_day())
    legs.add(tasks_t)

    # asyncio.wait, NOT wait_for() around a gather: a wait_for timeout cancels
    # BOTH legs, so a hanging Linear would take the background-task half down
    # with it -- the exact opposite of what this tool exists to do.
    _, slow = await asyncio.wait(legs, timeout=CHECK_MY_DAY_S)
    for t in slow:
        t.cancel()

    (mine, others), linear_ok = _leg(linear_t, "linear", ([], []))
    (running, finished), _ = _leg(tasks_t, "supervisor", ([], []))
    return _speak_day(mine, others, running, finished,
                      linear_ok=linear_ok, linear_on=bool(_LINEAR_KEY),
                      today=now_local().date())


# --- one ticket -----------------------------------------------------------
#
# Three tools, ~1,000 bytes of schema between them, against the 78KB and 65
# tools of Linear's MCP server -- which the voice tier used to carry and cannot
# afford. The realtime API re-charges the WHOLE tool schema against a 40,000
# TPM bucket on EVERY response, so the toolset alone took 14,923 of it and left
# room for two responses a minute. A tool-using turn needs two, one to call and
# one to speak the result, so the second reliably came back
# `response failed: [tokens] rate_limit_exceeded` -- which is not an error the
# model can see or recover from. It simply says nothing. The owner asked it to check
# his day, got thirty-one seconds of silence, asked again, and got another
# eighteen.
#
# Caching does not rescue that: measured at a 99.7% cache hit, the bucket was
# still charged the full ~15k. Schema size is the whole variable. These three
# measure 2,157 charged, which is eighteen responses a minute -- the same
# headroom as carrying no Linear tools at all.
#
# They are small because they take two plain strings where Linear's own
# issueCreate takes a 36-field input object. Anything past them -- closing,
# reassigning, projects, documents -- is a background task, which runs on
# tier 3 and still has the entire MCP surface.

_FIND_N = 5

# Titles and status, nothing else. Same projection rule as _DAY_QUERY: no `id`
# and no `url` are selected at all, so nothing UUID- or URL-shaped can reach
# the model even if this is later loosened.
_FIND_QUERY = """
query Find($q: String!, $n: Int!) {
  issues(first: $n, orderBy: updatedAt,
         filter: {title: {containsIgnoreCase: $q}}) {
    nodes { title state { name } }
  }
}
"""

# The writers' version. `id` is selected only to ADDRESS the issue in the
# mutation that follows -- it is never returned and never spoken.
_FIND_ID_QUERY = """
query FindId($q: String!, $n: Int!) {
  issues(first: $n, orderBy: updatedAt,
         filter: {title: {containsIgnoreCase: $q}}) {
    nodes { id title }
  }
}
"""

_CREATE_QUERY = """
mutation Create($teamId: String!, $title: String!, $description: String) {
  issueCreate(input: {teamId: $teamId, title: $title,
                      description: $description}) {
    success
  }
}
"""

_COMMENT_QUERY = """
mutation Comment($issueId: String!, $body: String!) {
  commentCreate(input: {issueId: $issueId, body: $body}) { success }
}
"""

# `teamId` is the ONLY required field on IssueCreateInput -- title and
# description are both optional -- so this one lookup is the whole difference
# between "can create a ticket" and "cannot".
#
# Resolved lazily and cached for the life of the process, NEVER at import: the
# voice worker forks a subprocess per job and pays import cost against a 10s
# initialize_process_timeout, so a network call at module scope is one that
# takes the phone number down. The lock stops two concurrent writes both
# paying for it.
_team_id: str | None = None
_team_lock = asyncio.Lock()


async def _team() -> str:
    """The team new tickets land in. One query per process. Raises."""
    global _team_id
    async with _team_lock:
        if _team_id is None:
            data = await _gql("query { teams(first: 1) { nodes { id } } }")
            nodes = (data.get("teams") or {}).get("nodes") or []
            if not nodes:
                raise LinearRejected("this Linear workspace has no team")
            _team_id = nodes[0]["id"]
    return _team_id


# Said when the key is unset. NOT "I could not reach Linear": reporting a
# structural absence as a failure is the same mistake as recording ABANDONED
# as pin_failed, and _speak_day already refuses to make it.
_NO_LINEAR = "Linear is not set up on my end, so I cannot look at his tickets."


@function_tool
@owner_named
async def find_linear_issue(ctx: RunContext, query: str) -> str:
    """Look up one of the owner's Linear tickets by words from its title.

    Use it when he names a specific piece of work: "what's the status of the
    founder profile ticket", "did I have something about the demo video",
    "look up the YC application", "is that one still open".

    NOT for "what's on today" or "what's on my plate" -- check_my_day answers
    that, covering his tickets AND your background work in one round trip.
    Going ticket by ticket through here instead is several round trips inside
    one conversational turn.

    Titles and status only. It cannot return an identifier, a link or a
    reference number, so there is nothing here you could read out by mistake.
    Read-only; it changes nothing.

    query: words from the ticket title, as he said them.
    """
    if not _LINEAR_KEY:
        return _NO_LINEAR
    try:
        data = await _gql(_FIND_QUERY, {"q": query, "n": _FIND_N})
    except Exception as e:
        # The MESSAGE, never type(e).__name__: a bare class name once turned a
        # wiring bug into a bland "no input" and hid it for two calls.
        log.warning("find_linear_issue failed for %r: %s: %s",
                    query, type(e).__name__, e)
        return "I couldn't reach Linear just then."

    said = []
    for node in (data.get("issues") or {}).get("nodes") or []:
        title = _title(node)
        if not title:
            continue
        state = " ".join(((node.get("state") or {}).get("name") or "").split())
        said.append(f"{title}, {state}" if state else title)

    # "Linear has nothing matching that" and "I could not ask Linear" are
    # different answers and are said differently. See _linear_day.
    if not said:
        return f"Nothing in Linear with {query} in the title."
    if len(said) == 1:
        return said[0] + "."
    more = f" And {len(said) - 3} more." if len(said) > 3 else ""
    return "; ".join(said[:3]) + "." + more


@function_tool
async def write_linear_issue(ctx: RunContext, title: str, notes: str = "") -> str:
    """Make a new Linear ticket. This CHANGES his workspace.

    Use it when he asks you to write something down, note it, make a ticket, or
    add it to his list: "put that in Linear", "make a ticket for that", "write
    that down", "add that to my list".

    Only when creating a ticket is what he actually asked for. A remark is not
    a request, and thinking out loud is not a request -- let him land on it
    himself. To add to a ticket that already exists, use
    comment_on_linear_issue instead of making a second one.

    Costs nothing and rings nobody, so it is never a reason to start a
    background task.

    title: one short line, as he would say it out loud.
    notes: any detail he gave. Leave it empty if he gave none.
    """
    if not _LINEAR_KEY:
        return _NO_LINEAR
    title = " ".join((title or "").split())[:250]
    if not title:
        return "I need a line to put on the ticket before I can make it."

    async def _do() -> dict:
        # Two round trips on the first write of a process, one after that.
        return await _gql(_CREATE_QUERY, {
            "teamId": await _team(), "title": title,
            "description": " ".join((notes or "").split()) or None})

    try:
        data = await asyncio.wait_for(_do(), timeout=LINEAR_WRITE_S)
    except Exception as e:
        log.warning("write_linear_issue failed for %r: %s: %s",
                    title, type(e).__name__, e)
        # Say that nothing was saved. "I couldn't reach Linear" alone leaves
        # him believing it might have landed, which is worse than either fact.
        return "I couldn't get that into Linear just then, so nothing was saved."
    if not (data.get("issueCreate") or {}).get("success"):
        log.warning("write_linear_issue: Linear declined to create %r", title)
        return "Linear turned that down, so nothing was saved."
    return f"Written down in Linear: {title}."


@function_tool
@owner_named
async def comment_on_linear_issue(ctx: RunContext, query: str,
                                  comment: str) -> str:
    """Add a note to one of the owner's existing Linear tickets. This CHANGES it.

    Use it for "add a note to the demo video ticket", "comment on that one",
    "put on there that I already sent it". To make a NEW ticket, use
    write_linear_issue.

    Only when that specific change is what he asked for.

    query: words from the title of the ticket to write on.
    comment: what to put on it, in his words.
    """
    if not _LINEAR_KEY:
        return _NO_LINEAR
    comment = " ".join((comment or "").split())
    if not comment:
        return "I need something to write on it first."

    async def _do() -> str:
        data = await _gql(_FIND_ID_QUERY, {"q": query, "n": _FIND_N})
        nodes = (data.get("issues") or {}).get("nodes") or []
        if not nodes:
            return f"I couldn't find a ticket with {query} in the title."
        if len(nodes) > 1:
            # A stage direction, not a line to read out. Writing to the wrong
            # ticket is not undoable from a phone call.
            names = "; ".join(t for n in nodes[:3] if (t := _title(n)))
            return (f"More than one ticket matches {query} -- {names}. Ask {OWNER} "
                    f"which one he means, then call this again with more of "
                    f"the title. Do not guess.")
        got = await _gql(_COMMENT_QUERY,
                         {"issueId": nodes[0]["id"], "body": comment})
        if not (got.get("commentCreate") or {}).get("success"):
            return "Linear turned that down, so nothing was saved."
        return f"Added to {_title(nodes[0])}."

    try:
        return await asyncio.wait_for(_do(), timeout=LINEAR_WRITE_S)
    except Exception as e:
        log.warning("comment_on_linear_issue failed for %r: %s: %s",
                    query, type(e).__name__, e)
        return "I couldn't reach Linear just then, so nothing was saved."


_TAGS = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    return " ".join(_TAGS.sub("", text or "").split())


def _mine(payload: dict) -> list[str]:
    """Everything in the response that could hold the answer.

    The old version read `web.results` and nothing else, then kept three of the
    four hits it had paid for. Asked for tomorrow's weather in Portland it
    returned four sentences about weather websites and not one number -- while
    the SAME response carried an infobox with the forecast in it. With nothing
    to say, the only route the model had left was a background task, which cost
    money and rang the owner's phone ninety seconds later to tell him it would be hot.

    Ordered by answer density: the joined string is truncated from the end, and
    the model reads it top down.
    """
    out: list[str] = []

    # Structured facts first. `attributes` is where a forecast high/low, a
    # score, a price or a height actually lives.
    for box in ((payload.get("infobox") or {}).get("results") or []):
        bits = [_clean(box.get("title")), _clean(box.get("description")),
                _clean(box.get("long_desc"))]
        for attr in (box.get("attributes") or []):
            if isinstance(attr, (list, tuple)) and len(attr) >= 2:
                bits.append(f"{_clean(str(attr[0]))}: {_clean(str(attr[1]))}")
        if joined := " ".join(b for b in bits if b):
            out.append(joined)

    # A direct question/answer pair, usually with the number already in it.
    for faq in ((payload.get("faq") or {}).get("results") or []):
        q, a = _clean(faq.get("question")), _clean(faq.get("answer"))
        if q or a:
            out.append(f"{q} {a}".strip())

    # `age` matters for "tomorrow"/"today" questions and nothing else carries it.
    for item in ((payload.get("news") or {}).get("results") or []):
        title, desc = _clean(item.get("title")), _clean(item.get("description"))
        age = _clean(item.get("age"))
        if title or desc:
            out.append(f"{title} ({age}): {desc}" if age else f"{title}: {desc}")

    # The baseline -- ALL of them. `extra_snippets` is more sentences from the
    # same page, and is often where the number is when the description is
    # boilerplate.
    for hit in ((payload.get("web") or {}).get("results") or []):
        title, desc = _clean(hit.get("title")), _clean(hit.get("description"))
        extra = " ".join(_clean(x) for x in (hit.get("extra_snippets") or []))
        line = f"{title}: {desc}"
        if extra:
            line = f"{line} {extra}"
        out.append(line)

    for item in ((payload.get("discussions") or {}).get("results") or []):
        title, desc = _clean(item.get("title")), _clean(item.get("description"))
        if title or desc:
            out.append(f"{title}: {desc}")

    # Brave's `summarizer` block in a /web/search response is normally just a
    # KEY for a second request to /summarizer/search -- a different endpoint on
    # a different plan tier. We do not make that call: it would double the
    # network time inside a conversational turn for something that may not even
    # be enabled. If a future plan starts returning inline text, this picks it
    # up for free.
    for sm in ((payload.get("summarizer") or {}).get("results") or []):
        if text := _clean(sm.get("text") if isinstance(sm, dict) else None):
            out.insert(0, text)

    return [s for s in (line.strip(" :") for line in out) if s]


# Questions whose answer is a number. If one of these comes back with no digit
# anywhere in it, the search did not answer the question -- which is exactly
# what "what's the weather in Portland tomorrow" did.
_NUMBERY = re.compile(
    r"\b(weather|forecast|temperature|temp|degrees|hot|cold|rain|snow"
    r"|score|won|beat|final|standings|price|cost|how much|how many|how far"
    r"|how tall|how long|what time|when|hours|opens?|closes?|rate|stock)\b", re.I)
_DIGIT = re.compile(r"\d")


def _thin(query: str, lines: list[str]) -> bool:
    """Did this come back without the thing the question was asking for?"""
    if not lines:
        return True
    body = " ".join(lines)
    if len(body) < 120:
        return True
    return bool(_NUMBERY.search(query)) and not _DIGIT.search(body)


_FILLER = {"what", "whats", "what's", "hey", "can", "you", "could", "please",
           "tell", "me", "the", "a", "an", "is", "are", "of", "for", "about",
           "do", "know", "right", "now", "so", "just", "like", "s"}

# Terms that pull Brave's structured blocks (infobox, FAQ) into the response.
_SHARPEN = (
    (re.compile(r"\b(weather|forecast|temperature|temp|hot|cold|rain|snow)\b", re.I),
     "forecast high low temperature"),
    (re.compile(r"\b(score|won|beat|final|game)\b", re.I), "final score"),
    (re.compile(r"\b(price|cost|how much)\b", re.I), "price"),
    (re.compile(r"\b(hours|opens?|closes?)\b", re.I), "opening hours today"),
)


def _reformulate(query: str) -> str:
    """A sharper second query, built WITHOUT asking the model.

    The old rule told the model to delegate rather than search a third time,
    which is right -- repeating a similar query never works. But it made a thin
    FIRST search a reason to open a paid worker session that would ring the owner's
    phone. So the retry moved in here: from the model's side this is still one
    tool call, and it cannot be talked out of it.

    Strips the conversational framing, then adds the words that pull Brave's
    structured blocks in. Returns the query UNCHANGED when there is nothing to
    sharpen -- the caller must then not retry, because asking the identical
    question twice is the thing we already know does not work.
    """
    core = " ".join(w for w in re.findall(r"[\w']+", query)
                    if w.lower() not in _FILLER)
    core = core or query
    for pattern, extra in _SHARPEN:
        if pattern.search(query):
            return f"{core} {extra}"
    return query


async def _search_once(query: str) -> list[str]:
    r = await _brave.get("/web/search",
                         params={"q": query, "count": 4, "extra_snippets": 1})
    r.raise_for_status()
    return _mine(r.json() or {})


@function_tool
@owner_named
async def web_search(ctx: RunContext, query: str) -> str:
    """Search the web and get back SHORT SNIPPETS from a results page.

    This does not open any website. It cannot read a schedule, menu, listing or
    availability, cannot sign in, and cannot click anything. It is only useful
    when a one-line snippet already contains the answer: a score, the weather,
    who won, what time something generally opens, a price.

    It retries ITSELF once, internally, with a sharper query, whenever the first
    pass comes back without the thing you asked for. So a thin answer here is
    NOT a reason to search again and NOT a reason to start a background task --
    two passes have already happened by the time you read this. Use
    start_background_task only when a page has to actually be OPENED.

    If the owner named a site, said "use the browser", "go look at", "check their
    site" or "pull it up", or the answer needs a page OPENED -- a schedule, a
    menu, a listing, availability, a logged-in account, a purchase, a booking --
    this is the WRONG tool. Use start_background_task, without searching first.

    If he is asking about work you already did for him, this is also the wrong
    tool: the findings are already in get_task_status.
    """
    async def _both() -> tuple[str, list[str]]:
        lines = await _search_once(query)
        asked = query
        if _thin(query, lines):
            sharper = _reformulate(query)
            if sharper.strip().lower() != query.strip().lower():
                log.info("web_search: thin result for %r; retrying as %r",
                         query, sharper)
                second = await _search_once(sharper)
                if (not _thin(sharper, second)
                        or len(" ".join(second)) > len(" ".join(lines))):
                    lines, asked = second, sharper
        return asked, lines

    try:
        asked, lines = await asyncio.wait_for(_both(), timeout=WEB_SEARCH_TOTAL_S)
    except Exception as e:
        # The message, not type(e).__name__: the bare class name once turned a
        # wiring bug into a bland "no input" and hid it for two calls. A
        # transport failure is NOT a thin result and is never retried -- doubling
        # the dead air on a broken network helps nobody.
        log.warning("web_search failed for %r: %s: %s", query, type(e).__name__, e)
        return "I couldn't reach search just then."

    # Trimmed, but not as hard as it used to be: this string goes to the MODEL,
    # not to the ear. It still answers in two sentences; more context here is
    # what lets it find the number.
    body = " | ".join(line[:300] for line in lines)[:1000]

    if _thin(asked, lines):
        # BOTH passes came back without the thing that was asked for. Handing
        # the model the thin prose and nothing else is exactly where the
        # incident started -- with no number to state, the only move the prompt
        # left was a background task, which cost money and rang the owner's phone
        # ninety seconds later. So say what happened and close the topic. The
        # snippets still go along: they are context, not an answer.
        dead_end = (f"Two searches for {asked} turned up nothing that answers "
                    f"it. Tell {OWNER} you could not find it and stop there -- do "
                    f"NOT start a background task unless he asks for a page to "
                    f"be opened.")
        return f"{dead_end} What did come back: {body}" if body else dead_end

    return body
