"""Cross-call memory.

Built from structured DB state, NOT from a transcript. Two reasons:

  1. Loading extensive history makes gpt-realtime start replying in TEXT ONLY,
     silently — no exception, no log, just an agent that stops talking.
  2. Assistant turns must be `output_text`, and a summary sidesteps the whole
     content-type question.

Hard-capped at 1200 chars by a CHECK constraint in the schema, so the cap
survives a refactor of this function.
"""

from __future__ import annotations

import sqlalchemy as sa

from shot_supervisor.registry import speakable

MAX_SUMMARY = 1200

# Per finished task, inside the greeting context. Deliberately far shorter than
# the 400 chars a task's own result_summary gets: this is a REMINDER, not the
# report. Three finished tasks at 400 each nearly hit the 1200 CHECK, which is
# the ceiling that keeps gpt-realtime speaking instead of silently going
# text-only -- and it made the agent open a call by reciting a whole write-up.
# The full answer is still one get_task_status away.
RECAP = 170


def _ago(seconds: int | None) -> str:
    if seconds is None:
        return "recently"
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes ago"
    if seconds < 172800:
        return f"{round(seconds / 3600)} hours ago"
    return f"{round(seconds / 86400)} days ago"


def build_summary(cx: sa.Connection, owner: str) -> str:
    """A few hundred tokens of what the owner would expect us to remember."""
    done = cx.execute(sa.text("""
        SELECT ref, title, result_summary, state,
               EXTRACT(EPOCH FROM now()-finished_at)::int AS ago
          FROM shot.tasks
         WHERE owner_phone=:o AND finished_at IS NOT NULL
           -- Matching finished_tasks() and unreported(). Without it a task the owner
           -- TOLD us to stop came back in his next greeting as "a task to X did
           -- not complete (cancelled)" -- the system arguing with him about a
           -- decision he had made out loud thirty seconds earlier.
           AND state IN ('succeeded','failed','budget_reached')
           AND cancelled_at IS NULL
         ORDER BY finished_at DESC LIMIT 3"""), {"o": owner}).mappings().all()
    open_ = cx.execute(sa.text("""
        SELECT ref, title, state,
               EXTRACT(EPOCH FROM now()-created_at)::int AS ago
          FROM shot.tasks
         WHERE owner_phone=:o
           AND state IN ('queued','starting','running','waiting_on_user','idle')
         ORDER BY updated_at DESC LIMIT 4"""), {"o": owner}).mappings().all()

    parts: list[str] = []
    for t in done:
        # Say FINISHED explicitly. "you asked me to X" reads as still-in-progress,
        # and the agent told the owner a task was already running when none was.
        when = _ago(t["ago"])
        # Do NOT lowercase: titles carry proper nouns ("Jazz Bistro").
        title = (t["title"] or "").strip() or "something"
        if t["state"] != "succeeded":
            parts.append(f"A task to {title} did not complete ({t['state']}), {when}.")
        elif t["result_summary"]:
            gist = speakable(t["result_summary"], limit=RECAP) or ""
            parts.append(f"Finished {when} — {title}: {gist}" if gist
                         else f"Finished {when}: {title}.")
        else:
            parts.append(f"Finished {when}: {title}.")
    for t in open_:
        parts.append(f"Still working on {t['title']} (started {_ago(t['ago'])}), "
                     f"reference {t['ref'].replace('-', ' ')}.")

    if not parts:
        return ""
    return " ".join(parts)[:MAX_SUMMARY]


def save_summary(cx: sa.Connection, owner: str, summary: str,
                 call_id: str | None = None) -> None:
    summary = summary[:MAX_SUMMARY]
    ver = cx.execute(sa.text("""
        INSERT INTO shot.summaries (owner_phone, summary, version, updated_at)
        VALUES (:o,:s,1,now())
        ON CONFLICT (owner_phone) DO UPDATE
              SET summary=EXCLUDED.summary,
                  version=shot.summaries.version+1,
                  updated_at=now()
        RETURNING version"""), {"o": owner, "s": summary}).scalar_one()
    cx.execute(sa.text("""
        INSERT INTO shot.summary_history (owner_phone, version, summary, call_id)
        VALUES (:o,:v,:s,:c) ON CONFLICT DO NOTHING"""),
        {"o": owner, "v": ver, "s": summary, "c": call_id})


def unreported(cx: sa.Connection, owner: str) -> list[dict]:
    """Finished work the owner has not actually heard about yet.

    Deliberately keyed on reported_at, not notified_at: notified_at only means
    a callback was scheduled, and the two Jazz Bistro tasks were stamped
    notified at the moment their callbacks failed. Backed by
    tasks_owner_unreported_idx.
    """
    rows = cx.execute(sa.text("""
        SELECT id, ref, title, state, result_summary,
               EXTRACT(EPOCH FROM now()-finished_at)::int AS ago
          FROM shot.tasks
         WHERE owner_phone=:o AND finished_at IS NOT NULL AND reported_at IS NULL
           AND state IN ('succeeded','failed','budget_reached')
           AND cancelled_at IS NULL
         ORDER BY finished_at DESC LIMIT 3"""), {"o": owner}).mappings().all()
    return [{"id": str(r["id"]), "ref": r["ref"],
             "title": (r["title"] or "").strip() or "something",
             "ok": r["state"] == "succeeded",
             "gist": speakable(r["result_summary"] or "", limit=RECAP) or "",
             "when": _ago(r["ago"])} for r in rows]


def mark_reported(cx: sa.Connection, task_ids: list[str]) -> int:
    if not task_ids:
        return 0
    return cx.execute(sa.text("""
        UPDATE shot.tasks SET reported_at = now()
         WHERE id = ANY(:ids) AND reported_at IS NULL"""),
        {"ids": task_ids}).rowcount
