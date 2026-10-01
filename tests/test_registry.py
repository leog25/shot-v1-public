"""L2 integration — the correctness that stops the phone ringing twice."""

from __future__ import annotations

import concurrent.futures as cf
import uuid

import pytest
import sqlalchemy as sa

from shot_core.db import cancel_callback, claim_callback

from support import TEST_OWNER

OWNER = TEST_OWNER


def _mk_callback(db, *, task_id=None) -> str:
    with db.begin() as cx:
        return str(cx.execute(sa.text("""
            INSERT INTO shot.callbacks (owner_phone, task_id, reason, brief, due_at)
            VALUES (:o, :t, 'task_done', 'your thing is done', now())
            RETURNING id"""), {"o": OWNER, "t": task_id}).scalar_one())


def _mk_task(db, ref, *, state="running", session_id=None,
             finished=False) -> str:
    with db.begin() as cx:
        return str(cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    session_id, finished_at)
            VALUES (:o, :r, :r, 'do the thing', CAST(:s AS shot.task_state),
                    :sid, CASE WHEN :fin THEN now() ELSE NULL END)
            RETURNING id"""),
            {"o": OWNER, "r": ref, "s": state, "sid": session_id,
             "fin": finished}).scalar_one())


def test_cas_guard_exactly_one_winner_under_contention(db):
    """Three firing paths converge on claim_callback. Exactly one may win.

    A double-fire means the phone rings twice and the user is told the same
    thing twice — with money attached if the callback triggers an action.
    """
    cid = _mk_callback(db)

    def attempt() -> bool:
        with db.begin() as cx:
            return claim_callback(cx, cid)

    with cf.ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda _: attempt(), range(20)))

    assert sum(results) == 1, f"{sum(results)} winners; expected exactly 1"

    with db.begin() as cx:
        attempt_n = cx.execute(
            sa.text("SELECT attempt FROM shot.callbacks WHERE id=:c"), {"c": cid}).scalar_one()
    # attempt increments inside the same atomic statement, so it cannot be
    # double-counted even though 20 threads raced.
    assert attempt_n == 1


def test_cas_guard_refuses_cancelled(db):
    cid = _mk_callback(db)
    with db.begin() as cx:
        cx.execute(sa.text("UPDATE shot.callbacks SET cancelled_at=now() WHERE id=:c"), {"c": cid})
    with db.begin() as cx:
        assert claim_callback(cx, cid) is False


def test_one_pending_callback_per_task(db):
    """The database, not application logic, forbids two live callbacks per task."""
    with db.begin() as cx:
        tid = str(cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal)
            VALUES (:o,'r','t','g') RETURNING id"""), {"o": OWNER}).scalar_one())
    _mk_callback(db, task_id=tid)
    with pytest.raises(sa.exc.IntegrityError):
        _mk_callback(db, task_id=tid)


def test_summary_length_cap_is_enforced_by_the_database(db):
    """>1200 chars flips gpt-realtime to text-only replies — silently.

    No exception, no log, just an agent that stops talking. The CHECK is the
    only place that survives a refactor of the summarizer.
    """
    with db.begin() as cx:
        cx.execute(sa.text(
            "INSERT INTO shot.summaries (owner_phone, summary) VALUES (:o,:s)"),
            {"o": OWNER, "s": "x" * 1200})
    with pytest.raises(sa.exc.IntegrityError):
        with db.begin() as cx:
            cx.execute(sa.text("UPDATE shot.summaries SET summary=:s WHERE owner_phone=:o"),
                       {"o": OWNER, "s": "x" * 1201})


def test_session_id_unique_makes_create_idempotent(db):
    """tasks.session_id UNIQUE is what makes a retried sessions.create safe."""
    sid = f"sesn_{uuid.uuid4().hex[:12]}"
    with db.begin() as cx:
        for ref in ("a", "b"):
            cx.execute(sa.text("""
                INSERT INTO shot.tasks (owner_phone, ref, title, goal)
                VALUES (:o,:r,'t','g')"""), {"o": OWNER, "r": ref})
    with db.begin() as cx:
        cx.execute(sa.text("UPDATE shot.tasks SET session_id=:s WHERE ref='a'"), {"s": sid})
    with pytest.raises(sa.exc.IntegrityError):
        with db.begin() as cx:
            cx.execute(sa.text("UPDATE shot.tasks SET session_id=:s WHERE ref='b'"), {"s": sid})


def test_callback_dial_target_defaults_to_owner(db):
    """Normal callbacks are unaffected by the override existing."""
    cid = _mk_callback(db)
    with db.begin() as cx:
        row = cx.execute(sa.text("""
            SELECT owner_phone, COALESCE(to_number, owner_phone) AS dial
              FROM shot.callbacks WHERE id=:c"""), {"c": cid}).mappings().one()
    assert row["dial"] == row["owner_phone"] == OWNER


def test_callback_can_be_pointed_away_from_a_real_phone(db):
    """A probe must be able to dial somewhere harmless.

    The absence of this is why a durability test rang a real mobile twice:
    the only way to schedule a callback was one that used the owner number.
    """
    PLAY = "+16504894546"
    with db.begin() as cx:
        cid = str(cx.execute(sa.text("""
            INSERT INTO shot.callbacks (owner_phone, reason, brief, due_at, to_number)
            VALUES (:o,'task_done','probe', now(), :n) RETURNING id"""),
            {"o": OWNER, "n": PLAY}).scalar_one())
        row = cx.execute(sa.text("""
            SELECT owner_phone, COALESCE(to_number, owner_phone) AS dial
              FROM shot.callbacks WHERE id=:c"""), {"c": cid}).mappings().one()
    # identity stays the owner (FK, PIN, allowlist); only the dial target moves
    assert row["owner_phone"] == OWNER
    assert row["dial"] == PLAY


def test_finished_task_is_notified_regardless_of_who_finished_it(db):
    """Regression: a task marked terminal by the STATUS ENDPOINT (i.e. because
    the owner asked "how's it going" mid-call) was invisible to the reconcile sweep,
    which only looked at open tasks -- so no callback was ever scheduled and
    the agent silently never rang back."""
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    result_summary, finished_at, session_id)
            VALUES (:o,'done-elsewhere','t','g','succeeded','all good',
                    now(), 'sesn_x')"""), {"o": OWNER})
        unreported = cx.execute(sa.text("""
            SELECT count(*) FROM shot.tasks
             WHERE owner_phone=:o AND finished_at IS NOT NULL
               AND notified_at IS NULL"""), {"o": OWNER}).scalar_one()
    assert unreported == 1, "the partial index this relies on found nothing"


def test_notification_claim_is_idempotent(db):
    """notified_at is claimed in the same statement that selects the row, so two
    concurrent sweeps cannot both schedule a callback for one task."""
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    result_summary, finished_at)
            VALUES (:o,'claim-once','t','g','succeeded','x', now())"""), {"o": OWNER})
    claimed = []
    for _ in range(3):
        with db.begin() as cx:
            rows = cx.execute(sa.text("""
                UPDATE shot.tasks SET notified_at = now()
                 WHERE id IN (SELECT id FROM shot.tasks
                               WHERE finished_at IS NOT NULL AND notified_at IS NULL
                               FOR UPDATE SKIP LOCKED)
             RETURNING ref""")).scalars().all()
        claimed.extend(rows)
    assert claimed.count("claim-once") == 1


def test_task_ref_lookup_forgives_how_it_was_spoken(db):
    """The agent says refs aloud, so it hands back "jazz bistro schedule" for
    `jazz-bistro-schedule`. Strict matching made every status check during a
    live call answer "I have no task called that" while the work was running."""
    from shot_supervisor.registry import get_task
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal)
            VALUES (:o,'jazz-bistro-schedule','Jazz Bistro schedule','g')"""), {"o": OWNER})
    for spoken in ("jazz-bistro-schedule", "jazz bistro schedule",
                   "Jazz Bistro Schedule", " jazz  bistro schedule "):
        with db.begin() as cx:
            assert get_task(cx, OWNER, spoken) is not None, spoken
    with db.begin() as cx:
        assert get_task(cx, OWNER, "something else entirely") is None


def test_worker_markdown_is_made_speakable():
    """Workers write for a file; this is read aloud by a TTS voice."""
    from shot_supervisor.registry import speakable
    out = speakable(
        "Done. Summary written to `/mnt/session/outputs/jazz.md`.\n\n"
        "**What I found** — three shows today:\n\n"
        "| Time | Show | Cover |\n|---|---|---|\n"
        "| 5pm | Gretzinger Trio | no cover |\n| 9pm | Hank Quartet | $20 |\n"
        "| 8:30 pm | Ragged row with no trailing pipe")
    assert "*" not in out and "|" not in out and "/mnt/" not in out
    assert "written to" not in out.lower()
    assert "Gretzinger Trio" in out and "Hank Quartet" in out


def test_a_call_record_survives_every_column_being_unknown(db):
    """The shape `_report_call` sends when a dial fails before AMD runs.

    shot.calls has NOT NULL columns that a minimal report does not fill --
    room_name, and pin_verified, which has a default that an explicit NULL
    overrides. Both were discovered by watching real callbacks fail to record.
    """
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.calls (direction, owner_phone, room_name, livekit_job_id,
                                    callback_id, amd_category, amd_reason, amd_delay_ms,
                                    amd_speech_s, pin_verified, sip_status_code,
                                    failure_kind, started_at, ended_at, end_reason)
            VALUES ('outbound', :o, 'cb-test', NULL, NULL, NULL, NULL, NULL, NULL,
                    false, NULL, 'trunk_error', now(), now(), 'trunk_error')"""),
            {"o": OWNER})
        row = cx.execute(sa.text(
            "SELECT end_reason, pin_verified FROM shot.calls")).one()
    assert row.end_reason == "trunk_error"
    assert row.pin_verified is False


# ------------------------------------------------------------ cancellation

def test_cancel_callback_is_the_writer_that_never_existed(db):
    """`cancelled_at` had four readers -- the CAS, both partial indexes, the due
    sweep, ops.status -- and its only writer in the whole repo was a line inside
    a test. So a scheduled callback could not be stopped."""
    cid = _mk_callback(db)
    with db.begin() as cx:
        assert cancel_callback(cx, cid) is True
    with db.begin() as cx:
        assert cancel_callback(cx, cid) is False, "cancelling twice is a no-op"
    with db.begin() as cx:
        assert claim_callback(cx, cid) is False, "a cancelled callback must not dial"


def test_cancel_callback_refuses_one_already_fired(db):
    """The "cannot un-ring" fact the spoken answer depends on."""
    cid = _mk_callback(db)
    with db.begin() as cx:
        assert claim_callback(cx, cid) is True
    with db.begin() as cx:
        assert cancel_callback(cx, cid) is False


def test_cancelling_a_task_kills_its_pending_callback(db):
    """The incident, as one test: work nobody wanted, finished, with a callback
    already scheduled to ring his phone about it."""
    from shot_supervisor.registry import cancel_task

    tid = _mk_task(db, "portland-tomorrow")
    cid = _mk_callback(db, task_id=tid)
    with db.begin() as cx:
        c = cancel_task(cx, OWNER, "portland-tomorrow")

    assert c.found and c.was_open
    assert c.callbacks_cancelled == 1
    assert c.callback_already_fired is False
    with db.begin() as cx:
        assert claim_callback(cx, cid) is False


def test_cancel_is_idempotent(db):
    from shot_supervisor.registry import cancel_task

    _mk_task(db, "twice")
    with db.begin() as cx:
        cancel_task(cx, OWNER, "twice")
    with db.begin() as cx:
        c = cancel_task(cx, OWNER, "twice")
    assert c.found and c.was_open is False and c.already == "cancelled"
    assert c.callbacks_cancelled == 0


def test_cancel_cannot_unring_a_callback_already_claimed(db):
    from shot_supervisor.registry import cancel_task, spoken_cancel

    tid = _mk_task(db, "already-dialling")
    cid = _mk_callback(db, task_id=tid)
    with db.begin() as cx:
        claim_callback(cx, cid)
    with db.begin() as cx:
        c = cancel_task(cx, OWNER, "already-dialling")

    assert c.callback_already_fired is True
    assert "can't pull that one back" in spoken_cancel(c)


def test_cancelling_a_finished_task_still_kills_its_callback(db):
    """The exact incident shape: the work is done, the callback is pending, and
    he says he does not want it."""
    from shot_supervisor.registry import cancel_task, spoken_cancel

    tid = _mk_task(db, "done-but-unwanted", state="succeeded", finished=True)
    cid = _mk_callback(db, task_id=tid)
    with db.begin() as cx:
        c = cancel_task(cx, OWNER, "done-but-unwanted")

    assert c.was_open is False, "it was not running any more"
    assert c.callbacks_cancelled == 1, "but the call about it was still stoppable"
    assert "won't ring you" in spoken_cancel(c)
    with db.begin() as cx:
        assert claim_callback(cx, cid) is False


def test_a_cancelled_task_is_never_read_back_to_him(db):
    """He said stop. Every path that decides what to tell him has to agree.

    The one that matters most is the notify claim: without the guard, cancelling
    a task that had ALREADY finished still left the sweep free to schedule a
    fresh callback about it -- which is the exact call he asked us not to make.
    """
    import inspect

    from shot_supervisor import callbacks as cb_mod
    from shot_supervisor.context import unreported
    from shot_supervisor.registry import cancel_task, finished_tasks

    _mk_task(db, "unwanted", state="succeeded", finished=True)
    with db.begin() as cx:
        assert finished_tasks(cx, OWNER), "sanity: visible before the cancel"
        cancel_task(cx, OWNER, "unwanted")

    with db.begin() as cx:
        assert finished_tasks(cx, OWNER) == [], "not read back as finished work"
        assert unreported(cx, OWNER) == [], "and not led with on his next call"

    # The notify claim is a DBOS workflow that SCHEDULES REAL CALLBACKS, so it
    # is asserted by source rather than executed -- same approach as
    # test_verify_is_constant_time_compare.
    src = inspect.getsource(cb_mod.notify_finished_tasks)
    assert "cancelled_at IS NULL" in src, (
        "the sweep would schedule a callback for a task he cancelled")


def test_cancelling_releases_the_one_pending_per_task_index(db):
    """The partial unique index is keyed on cancelled_at IS NULL, so a cancel
    must leave room for a later callback rather than poisoning the task."""
    tid = _mk_task(db, "room-for-more")
    _mk_callback(db, task_id=tid)
    from shot_supervisor.registry import cancel_task
    with db.begin() as cx:
        cancel_task(cx, OWNER, "room-for-more")
    _mk_callback(db, task_id=tid)          # must not raise IntegrityError


def test_a_refresh_racing_a_cancel_cannot_resurrect_it(db):
    """A cancel can land between the Anthropic retrieve and the write-back.
    Without the guard the sweep sets a task the owner just killed to 'succeeded' --
    which then schedules the callback he refused."""
    from shot_supervisor.registry import cancel_task

    _mk_task(db, "racy")
    with db.begin() as cx:
        cancel_task(cx, OWNER, "racy")
    with db.begin() as cx:
        cx.execute(sa.text("""
            UPDATE shot.tasks
               SET state='succeeded', updated_at=now(),
                   finished_at=COALESCE(finished_at, now())
             WHERE ref='racy' AND cancelled_at IS NULL"""))
    with db.begin() as cx:
        state = cx.execute(sa.text(
            "SELECT state FROM shot.tasks WHERE ref='racy'")).scalar_one()
    assert state == "cancelled"


def test_a_session_started_after_a_cancel_does_not_go_back_to_running(db):
    """start_session records the session id unconditionally -- losing the handle
    means a paid session nobody can cap -- but must not undo the cancel."""
    from shot_supervisor.registry import cancel_task

    tid = _mk_task(db, "started-late", state="queued")
    with db.begin() as cx:
        cancel_task(cx, OWNER, "started-late")
    with db.begin() as cx:
        row = cx.execute(sa.text("""
            UPDATE shot.tasks
               SET session_id='sess_x',
                   state = CASE WHEN cancelled_at IS NULL THEN 'running' ELSE state END,
                   updated_at=now()
             WHERE id=CAST(:i AS uuid) AND session_id IS NULL
         RETURNING cancelled_at"""), {"i": tid}).first()
    assert row is not None and row[0] is not None, "the cancel must be visible"
    with db.begin() as cx:
        state, sid = cx.execute(sa.text(
            "SELECT state, session_id FROM shot.tasks WHERE ref='started-late'")).one()
    assert state == "cancelled"
    assert sid == "sess_x", "the handle must be kept so spend can still be capped"


def test_cancel_forgives_how_the_ref_was_spoken(db):
    from shot_supervisor.registry import cancel_task

    _mk_task(db, "jazz-bistro-schedule")
    with db.begin() as cx:
        c = cancel_task(cx, OWNER, "jazz bistro schedule")
    assert c.found and c.was_open


def test_cancelling_something_that_does_not_exist_says_so(db):
    from shot_supervisor.registry import cancel_task, spoken_cancel

    with db.begin() as cx:
        c = cancel_task(cx, OWNER, "no such thing")
    assert c.found is False
    assert "no task called" in spoken_cancel(c)


def test_the_spoken_answer_never_promises_more_than_happened():
    """Pure. "Stopped, I won't call you back" about a call that is already
    ringing is the kind of confident wrong answer he would act on."""
    from shot_supervisor.registry import Cancellation, spoken_cancel

    stopped = spoken_cancel(Cancellation(found=True, ref="r", was_open=True))
    ringing = spoken_cancel(Cancellation(found=True, ref="r", was_open=True,
                                         callback_already_fired=True))
    done = spoken_cancel(Cancellation(found=True, ref="r", already="succeeded"))

    assert "won't call you back" in stopped
    assert "can't pull that one back" in ringing
    assert "already finished" in done
    assert len({stopped, ringing, done}) == 3, "three truths, three sentences"


def test_nothing_cancels_a_task_except_leo():
    """USER-INITIATED ONLY, enforced structurally.

    An automatic cancel is indistinguishable from a bug that silently drops his
    work, and by construction there is no callback left to tell him it happened.
    The call graph is: voice tool -> endpoint -> registry, and nothing else.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "packages"
    # registry defines it, app exposes it, tools calls the endpoint, agent
    # merely lists it among the tools the model may choose. Nothing else may
    # even mention it.
    hits = {f.name for f in root.rglob("*.py")
            if "cancel_task" in f.read_text()}
    assert hits == {"registry.py", "app.py", "tools.py", "agent.py"}, hits

    # And the modules that make decisions on their own must not invoke it: an
    # interruption, a timeout, or the flow deciding the work was a mistake are
    # all indistinguishable from silently losing his work.
    for name in ("callback_flow.py", "worker.py", "callbacks.py"):
        f = next(root.rglob(name))
        assert "cancel_task" not in f.read_text(), (
            f"{name} must never cancel on its own judgement")

    agent = (next(root.rglob("agent.py"))).read_text()
    assert "cancel_task(" not in agent, (
        "agent.py may offer the tool, never call it")


# --------------------------------------------------- capping a live session

class _FakeSessions:
    """Mimics the one behaviour that broke: the API refuses a ceiling at or
    below what the session has already consumed."""

    def __init__(self, spent: int, *, creep: int = 0):
        self.spent, self.creep = spent, creep
        self.budget: int | None = None
        self.interrupts = 0
        self.events = self

    def send(self, *, session_id, events):
        self.interrupts += 1

    def retrieve(self, sid):
        import types
        got = self.spent
        self.spent += self.creep          # it keeps spending while we read
        return types.SimpleNamespace(
            usage=types.SimpleNamespace(
                list_cost=types.SimpleNamespace(amount=str(got))))

    def update(self, sid, *, budget):
        want = int(budget["max_list_cost"]["amount"])
        if want <= self.spent:
            raise ValueError("`budget.max_list_cost` must be greater than the "
                             "session's consumed list cost")
        self.budget = want


def _fake_client(sessions):
    import types
    return types.SimpleNamespace(beta=types.SimpleNamespace(sessions=sessions))


def test_the_ceiling_is_set_above_what_was_already_spent(monkeypatch):
    """A flat one-cent ceiling 400s on any session that has spent anything.

    It did, on the first real cancellation: the exception propagated out of
    stop_session_spend, the interrupt after it never went out, and a task the owner
    had cancelled carried on to $2.35.
    """
    from shot_supervisor import registry

    fake = _FakeSessions(spent=235)
    monkeypatch.setattr(registry, "_client", lambda: _fake_client(fake))
    registry.stop_session_spend("sesn_x")

    assert fake.budget is not None and fake.budget > 235, (
        "the ceiling must clear what the session has already spent")


def test_a_session_still_spending_is_capped_on_the_retry(monkeypatch):
    """It spends while we read it, so the first ceiling can be stale."""
    from shot_supervisor import registry

    # It spends 8 cents between the two reads -- more than the first margin
    # allows, so the first ceiling is stale and gets rejected.
    fake = _FakeSessions(spent=100, creep=8)
    monkeypatch.setattr(registry, "_client", lambda: _fake_client(fake))
    registry.stop_session_spend("sesn_x")

    assert fake.budget is not None, "a moving target must still get capped"
    assert fake.budget > fake.spent - fake.creep, "and capped above what it spent"


def test_the_interrupt_goes_out_before_the_ceiling(monkeypatch):
    """The interrupt is what stops the turn running RIGHT NOW, so it must not be
    skipped when the ceiling call fails -- which is how the first one leaked."""
    from shot_supervisor import registry

    class _Broken(_FakeSessions):
        def update(self, sid, *, budget):
            raise ValueError("nope")

    fake = _Broken(spent=10)
    monkeypatch.setattr(registry, "_client", lambda: _fake_client(fake))
    with pytest.raises(ValueError):
        registry.stop_session_spend("sesn_x")

    assert fake.interrupts == 1, "the interrupt must already have gone out"
