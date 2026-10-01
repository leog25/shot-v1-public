"""L1/L2 — the delegate seam: registry row -> live worker session -> terminal state.

Deliberately uses a task the agent can answer with no tools, so it never pauses
on the bash always_ask gate and the whole test costs a few cents.
"""

from __future__ import annotations

import time

import pytest
import sqlalchemy as sa

from shot_supervisor.registry import (
    create_task,
    get_task,
    open_tasks,
    refresh_from_anthropic,
    slugify,
    start_session,
)

from support import TEST_OWNER

OWNER = TEST_OWNER
pytestmark = pytest.mark.contract


def test_slugify_is_speakable_and_unique():
    assert slugify("Order groceries from Loblaws", set()) == "order-groceries-from-lob"
    assert slugify("Order groceries", {"order-groceries"}) == "order-groceries-2"
    assert slugify("!!!", set()) == "task"


def test_delegate_loop_reaches_terminal(db):
    with db.begin() as cx:
        t = create_task(cx, owner=OWNER, goal=(
            "Reply with exactly: All set. Use no tools at all."),
            title="smoke check")
        assert t.state == "queued"
        assert t.ref == "smoke-check"

    # POST /v1/sessions returns with the session ALREADY running — this is the
    # background-task primitive, and it must not block the caller.
    t0 = time.monotonic()
    with db.begin() as cx:
        sid = start_session(cx, t.id)
    create_latency = time.monotonic() - t0
    assert sid.startswith("sesn_"), sid
    assert create_latency < 10, f"session create took {create_latency:.1f}s; must not block a call"

    with db.begin() as cx:
        assert get_task(cx, OWNER, "smoke-check").session_id == sid
        assert [x.ref for x in open_tasks(cx, OWNER)] == ["smoke-check"]

    # status is read via plain GETs; never by messaging the running session
    state = "running"
    for _ in range(60):
        with db.begin() as cx:
            state = refresh_from_anthropic(cx, t.id)
        if state in {"succeeded", "failed", "budget_reached"}:
            break
        time.sleep(2)

    assert state == "succeeded", f"ended in {state}"
    with db.begin() as cx:
        done = get_task(cx, OWNER, "smoke-check")
        assert done.result_summary, "no spoken summary captured"
        assert len(done.result_summary) <= 400
        assert not open_tasks(cx, OWNER), "terminal task still listed as open"
    print(f"\n  session={sid} create={create_latency:.2f}s summary={done.result_summary!r}")


def test_start_session_is_idempotent(db):
    with db.begin() as cx:
        t = create_task(cx, owner=OWNER, goal="Reply: ok. No tools.", title="idem check")
        first = start_session(cx, t.id)
        second = start_session(cx, t.id)   # must not fork a second session
    assert first == second

    with db.begin() as cx:
        n = cx.execute(sa.text(
            "SELECT count(*) FROM shot.tasks WHERE session_id=:s"), {"s": first}).scalar_one()
    assert n == 1
