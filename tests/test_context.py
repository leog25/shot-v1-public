"""L0/L2 — cross-call memory stays inside the limit that keeps the model speaking."""

from __future__ import annotations

import sqlalchemy as sa

from shot_supervisor.context import MAX_SUMMARY, _ago, build_summary, save_summary

from support import TEST_OWNER

OWNER = TEST_OWNER


def test_empty_history_yields_no_summary(db):
    with db.begin() as cx:
        assert build_summary(cx, OWNER) == ""


def test_summary_recalls_finished_and_open_work(db):
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    result_summary, finished_at)
            VALUES (:o,'lakers','check the Lakers schedule','g','succeeded',
                    'Next home game is October thirteenth.', now() - interval '2 hours')"""),
            {"o": OWNER})
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state, created_at)
            VALUES (:o,'groceries','order groceries','g','running',
                    now() - interval '4 minutes')"""), {"o": OWNER})
        s = build_summary(cx, OWNER)
    assert "check the Lakers schedule" in s
    assert "October thirteenth" in s
    assert "Still working on order groceries" in s
    assert "groceries" in s
    assert len(s) <= MAX_SUMMARY


def test_summary_never_exceeds_the_cap_that_keeps_audio_on(db):
    """Over 1200 chars flips gpt-realtime to text-only replies, silently."""
    with db.begin() as cx:
        for i in range(40):
            cx.execute(sa.text("""
                INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                        result_summary, finished_at)
                VALUES (:o,:r,:t,'g','succeeded',:s, now())"""),
                {"o": OWNER, "r": f"t{i}", "t": "x" * 90, "s": "y" * 300})
        s = build_summary(cx, OWNER)
        assert len(s) <= MAX_SUMMARY
        save_summary(cx, OWNER, s)     # the CHECK would reject anything longer


def test_summary_versions_are_kept(db):
    with db.begin() as cx:
        save_summary(cx, OWNER, "first")
        save_summary(cx, OWNER, "second")
        cur = cx.execute(sa.text(
            "SELECT summary, version FROM shot.summaries WHERE owner_phone=:o"),
            {"o": OWNER}).mappings().one()
        hist = cx.execute(sa.text(
            "SELECT count(*) FROM shot.summary_history WHERE owner_phone=:o"),
            {"o": OWNER}).scalar_one()
    assert cur["summary"] == "second" and cur["version"] == 2
    assert hist == 2


def test_ago_is_speakable():
    assert _ago(10) == "just now"
    assert _ago(600) == "10 minutes ago"
    assert _ago(7200) == "2 hours ago"
    assert _ago(200000) == "2 days ago"


def test_finished_work_is_not_described_as_in_progress(db):
    """The agent told the owner a task was already running when none was.

    The old phrasing, "you asked me to Check jazz dinner times", reads as
    ongoing. Finished work must say so.
    """
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    result_summary, finished_at)
            VALUES (:o,'jazz','check the Jazz Bistro listings','g','succeeded',
                    'Three shows tonight.', now())"""), {"o": OWNER})
        s = build_summary(cx, OWNER)
    assert s.lower().startswith("finished")
    assert "still working" not in s.lower()


def test_failed_work_is_reported_as_failed_not_pending(db):
    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state, finished_at)
            VALUES (:o,'x','check something','g','failed', now())"""), {"o": OWNER})
        s = build_summary(cx, OWNER)
    assert "did not complete" in s and "failed" in s


# ---------------------------------------------------------------- speakable

def test_speakable_strips_report_furniture():
    """Workers write documents; this has to sound like a person talking."""
    from shot_supervisor.registry import speakable

    out = speakable(
        "## What I found — Jazz Bistro has three shows tonight.\n\n"
        "**Notes worth flagging:** the late set runs past one.\n"
        "Full write-up saved to /mnt/session/outputs/report.md, and I logged "
        "the venue details to memory.")
    assert out
    low = out.lower()
    for furniture in ("what i found", "notes worth flagging", "outputs",
                      "write-up", "logged", "**", "##", "/mnt"):
        assert furniture not in low and furniture not in out, f"{furniture!r} survived: {out!r}"
    assert "three shows tonight" in low
    assert "late set runs past one" in low


def test_speakable_never_cuts_mid_word():
    """A hard slice produced '(Night 2 is Sat', and the agent read it aloud."""
    from shot_supervisor.registry import speakable

    long_clause = ("Jazz Bistro has three shows today and the second act is a "
                   "two night run with the late jam going until half past one")
    out = speakable(long_clause, limit=60)
    assert out and len(out) <= 62
    assert out.endswith(".")
    # every word in the output must be a whole word from the input
    assert all(w in long_clause.split() for w in out.rstrip(".").split())


def test_greeting_recap_stays_well_under_the_text_only_cliff(db):
    """Three finished tasks must not fill the 1200-char CHECK on their own.

    At 400 chars each they nearly did, and the greeting became a recitation of
    a whole write-up instead of a reminder.
    """
    report = ("Jazz Bistro has three shows tonight. " * 40)[:2000]
    with db.begin() as cx:
        for i in range(3):
            cx.execute(sa.text("""
                INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                        result_summary, created_at, finished_at)
                VALUES (:o, :r, :t, :t, 'succeeded', :s, now(), now())"""),
                {"o": OWNER, "r": f"t{i}", "t": f"Task {i}", "s": report})
        summary = build_summary(cx, OWNER)
    assert len(summary) < MAX_SUMMARY * 0.75, f"recap is {len(summary)} chars"
    assert summary.count("Finished") == 3


# ------------------------------------------------- told vs merely scheduled

def _finished_task(cx, ref, *, notified=True, reported=False, summary="Three shows tonight."):
    cx.execute(sa.text("""
        INSERT INTO shot.tasks (owner_phone, ref, title, goal, state, result_summary,
                                created_at, finished_at, notified_at, reported_at)
        VALUES (:o, :r, :t, :t, 'succeeded', :s, now(), now(),
                CASE WHEN :n THEN now() END, CASE WHEN :p THEN now() END)"""),
        {"o": OWNER, "r": ref, "t": ref.replace("-", " "), "s": summary,
         "n": notified, "p": reported})


def test_a_scheduled_callback_is_not_the_same_as_having_been_told(db):
    """The bug: both Jazz tasks were stamped notified_at at the exact moment
    their callbacks failed (pin_failed, uncertain), so the system believed the owner
    had been told about work he never heard a word about."""
    from shot_supervisor.context import unreported

    with db.begin() as cx:
        _finished_task(cx, "jazz-bistro", notified=True, reported=False)
        news = unreported(cx, OWNER)
    assert [n["ref"] for n in news] == ["jazz-bistro"], \
        "a failed callback must leave the task still owed to the owner"
    assert news[0]["gist"], "the news needs something speakable in it"


def test_once_told_it_stops_coming_up(db):
    from shot_supervisor.context import mark_reported, unreported

    with db.begin() as cx:
        _finished_task(cx, "jazz-bistro", notified=True, reported=False)
        [task] = unreported(cx, OWNER)
        assert mark_reported(cx, [task["id"]]) == 1
        assert unreported(cx, OWNER) == []
        # and re-marking is a no-op, not a second claim
        assert mark_reported(cx, [task["id"]]) == 0


def test_unfinished_work_is_not_news(db):
    from shot_supervisor.context import unreported

    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state, created_at)
            VALUES (:o, 'running-one', 'Running one', 'g', 'running', now())"""),
            {"o": OWNER})
        assert unreported(cx, OWNER) == []


def test_the_table_is_the_answer_and_must_survive_truncation():
    """A worker reported three shows as a markdown table. The table was
    stripped for speech and appended after the prose, where the length cap ate
    it — leaving "has three shows today:" with nothing after the colon. The owner
    asked what they were and the agent started a NEW browsing task to re-find
    what it had already been told."""
    from shot_supervisor.registry import speakable

    raw = ("## What I found — Jazz Bistro (251 Victoria St) has three shows today:\n\n"
           "| Time | Act |\n|---|---|\n"
           "| 7:00 pm | The Andrew Scott Trio |\n"
           "| 9:30 pm | Senor McNasty |\n"
           "| 11:45 pm | Late Night Jam |\n\n"
           "**Notes worth flagging:** "
           + "the room fills up early and the bar stays open late. " * 6)

    out = speakable(raw)
    assert out and len(out) <= 400, f"{len(out)} chars, over the limit"
    for showtime in ("7:00", "9:30", "11:45"):
        assert showtime in out, f"{showtime} was truncated away: {out!r}"
    assert "Andrew Scott Trio" in out
    # The lead-in has to survive too: budgeting it away made the agent open
    # with a bare timestamp, and trimming it to fit produced "the Señor."
    assert out.startswith("Jazz Bistro"), f"lost the lead-in: {out!r}"
    assert ":." not in out and ".." not in out, f"punctuation artifact: {out!r}"


def test_speakable_respects_its_limit_with_a_table():
    """The row-fitting branch undercounted the ". " joins and overshot: a real
    task produced 410 chars against a limit of 400."""
    from shot_supervisor.registry import speakable

    raw = ("Jazz Bistro has three shows today, Friday August 28, 2026:\n\n"
           "| Time | Act |\n|---|---|\n"
           "| 5:00 pm | Gretzinger, Simpson and Botos, alto sax bass and drums, fifteen dollars |\n"
           "| 8:30 pm | Senor McNasty Soul Jazz Experience, night one of two, twenty five |\n"
           "| 11:30 pm | Late Night Jazz Jam hosted by Jonathan Meyer, pay what you can |\n\n"
           "The late jam keeps the kitchen open until half past one. "
           "The earlier sets are seated listening room shows. "
           "Night two of the McNasty run is Saturday at half past eight. ")
    for limit in (200, 400, 900):
        out = speakable(raw, limit=limit)
        assert out and len(out) <= limit, f"limit {limit}: got {len(out)} chars"


# ------------------------------------------- what the agent can actually see

def test_finished_work_is_visible_not_just_running_work(db):
    """The owner called in and asked what had come back. The agent answered "I can
    only see what's running right now, and that list is empty" -- because
    /internal/tasks called open_tasks(), which filters on the OPEN states. A
    finished task could only be reached by guessing its exact reference."""
    from shot_supervisor.registry import finished_tasks, open_tasks

    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    result_summary, created_at, finished_at)
            VALUES (:o, 'jazz-bistro', 'Jazz Bistro schedule', 'g', 'succeeded',
                    'Three shows tonight at seven, nine and eleven.', now(), now())"""),
            {"o": OWNER})
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state, created_at)
            VALUES (:o, 'still-going', 'Still going', 'g', 'running', now())"""),
            {"o": OWNER})

        assert [t.ref for t in open_tasks(cx, OWNER)] == ["still-going"]

        done = finished_tasks(cx, OWNER)
        assert [d["ref"] for d in done] == ["jazz-bistro"]
        assert done[0]["title"] == "Jazz Bistro schedule"
        assert "three shows" in done[0]["gist"].lower(), "it must carry the answer"
        assert done[0]["ago"] == "just now"
        assert done[0]["told"] is False


def test_unfinished_work_is_not_listed_as_finished(db):
    from shot_supervisor.registry import finished_tasks

    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state, created_at)
            VALUES (:o, 'running-one', 'Running one', 'g', 'running', now())"""),
            {"o": OWNER})
        assert finished_tasks(cx, OWNER) == []


def test_a_failed_task_is_still_reported_back(db):
    """"It didn't work" is an answer; silence is not."""
    from shot_supervisor.registry import finished_tasks

    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    created_at, finished_at)
            VALUES (:o, 'broke', 'Check dinner times', 'g', 'failed', now(), now())"""),
            {"o": OWNER})
        [t] = finished_tasks(cx, OWNER)
        assert t["state"] == "failed"


def test_a_task_leo_cancelled_is_not_in_his_greeting(db):
    """He said stop. Opening his next call with "a task to X did not complete
    (cancelled)" is the system arguing with him about a decision he made out
    loud thirty seconds earlier -- and build_summary was the one consumer with
    no state filter at all."""
    from shot_supervisor.registry import cancel_task

    with db.begin() as cx:
        cx.execute(sa.text("""
            INSERT INTO shot.tasks (owner_phone, ref, title, goal, state,
                                    result_summary, finished_at)
            VALUES (:o,'unwanted','Unwanted thing','g','succeeded','found it',
                    now())"""), {"o": OWNER})
        assert build_summary(cx, OWNER), "sanity: it is there before the cancel"
        cancel_task(cx, OWNER, "unwanted")

    with db.begin() as cx:
        assert "Unwanted thing" not in build_summary(cx, OWNER)


def test_the_callback_brief_carries_the_findings_not_the_preamble():
    """A callback read the owner a MENU of his finished tasks instead of his forecast.

    The brief came from result_summary, which is speakable() trimmed to a
    sentence boundary -- and a worker's opening sentence is what it DID, not
    what it FOUND: "I pulled the National Weather Service forecast for Ann
    Arbor, plus the hourly breakdown and the severe weather outlook." 140
    characters, no forecast. DELIVER_CALLBACK then told the model to read out
    every fact, there were none, so it called list_my_tasks and read a menu.

    A turn played, so delivered=True was recorded and reported_at stamped -- a
    forecast he had never heard, filed as told to him. He rang back to ask.
    """
    from shot_supervisor.callbacks import brief_for

    preamble = ("I pulled the National Weather Service forecast for Portland, "
                "issued this evening, plus the hourly breakdown and the severe "
                "weather outlook.")
    findings = ("Thursday looks like the better of the two days: mostly sunny, "
                "a high near ninety, but humid enough to feel like the mid "
                "nineties, and the rain risk stays under fifteen percent.")

    brief = brief_for({"ref": "portland-forecast", "state": "succeeded",
                       "result_summary": preamble,
                       "result_raw": f"{preamble}\n\n{findings}"})

    assert "ninety" in brief, "the callback must read out what was FOUND"
    assert len(brief) > len(preamble), "the preamble alone is not a brief"


def test_the_callback_brief_falls_back_when_there_is_no_raw():
    from shot_supervisor.callbacks import brief_for

    assert "only this" in brief_for(
        {"ref": "r", "state": "succeeded",
         "result_summary": "only this", "result_raw": None})
    assert "finished as failed" in brief_for(
        {"ref": "some-task", "state": "failed",
         "result_summary": None, "result_raw": None})


def test_a_start_failure_is_translated_before_it_is_spoken():
    """The 2026-09-10 regression: a start failure is READ ALOUD verbatim.

    A task that never started has no result_raw, so brief_for falls back to
    result_summary and the phone gets whatever is in it. The owner was read "Error
    code four hundred. Type error. The error type is invalid request error.
    The message says MCP server hosts blocked by environment network
    policy..." before it stopped mid-word.
    """
    from shot_supervisor.callbacks import speakable_failure

    raw = ("Error code: 400 - {'type': 'error', 'error': {'type': "
           "'invalid_request_error', 'message': 'MCP server host(s) blocked by "
           "environment network policy: \"linear\" (mcp.linear.app). Add these "
           "hosts to the environment\'s allowed_hosts, or set "
           "allow_mcp_servers=true.'}, 'request_id': 'req_011CevWvQcGnjsJ4'}")
    said = speakable_failure(raw)
    for shape in ("400", "{", "}", "'", "invalid_request_error", "request_id",
                  "allowed_hosts", "mcp.linear.app"):
        assert shape not in said, f"{shape!r} is not something to say out loud"
    assert said.endswith("set up")


def test_an_unrecognised_start_failure_is_vague_rather_than_raw():
    """The fallback must not leak the exception. An unknown failure is a bug to
    read in `error`, not JSON to recite at someone."""
    from shot_supervisor.callbacks import speakable_failure

    assert speakable_failure("KeyError: 'sessions' at line 41 {'a': 1}") == (
        "the worker session could not be started")


def test_out_of_credit_is_still_its_own_sentence():
    """It was the ONE translated case before, and it stays distinguishable:
    an account with no money is a different action from a broken config."""
    from shot_supervisor.callbacks import speakable_failure

    assert "out of credit" in speakable_failure(
        "Error code: 400 - your credit balance is too low to access the API")


def test_the_callback_brief_fits_inside_the_playout_budget():
    """A ~400-char brief reads aloud in 35-45s against BRIEF_PLAYOUT_S of 100."""
    from shot_core.budget import BRIEF_PLAYOUT_S
    from shot_supervisor.callbacks import brief_for

    brief = brief_for({"ref": "r", "state": "succeeded", "result_summary": None,
                       "result_raw": "Sentence about the weather. " * 200})
    assert len(brief) <= 700
    assert len(brief) / 10 < BRIEF_PLAYOUT_S, "roughly 10 chars a second spoken"
