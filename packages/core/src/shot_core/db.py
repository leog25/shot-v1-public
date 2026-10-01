"""Postgres access. Imported by the supervisor only — never on the voice import path."""

from __future__ import annotations

import pathlib

import sqlalchemy as sa
from sqlalchemy import Engine

SCHEMA_SQL = pathlib.Path(__file__).with_name("schema.sql")


def make_engine(url: str) -> Engine:
    # psycopg3 driver; the URL in .env is bare postgresql://
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return sa.create_engine(url, pool_pre_ping=True, future=True)


def apply_schema(engine: Engine) -> None:
    with engine.begin() as cx:
        cx.execute(sa.text(SCHEMA_SQL.read_text()))


def claim_callback(cx: sa.Connection, callback_id: str) -> bool:
    """Compare-and-set: exactly one caller wins, ever.

    Three firing paths converge on this (DBOS delayed enqueue, the due-callback
    sweep, and manual retrigger). `attempt` increments inside the same atomic
    statement so backoff can't be double-counted either.
    """
    row = cx.execute(sa.text("""
        UPDATE shot.callbacks
           SET fired_at = now(), attempt = attempt + 1
         WHERE id = :cid
           AND fired_at IS NULL
           AND cancelled_at IS NULL
     RETURNING id
    """), {"cid": callback_id}).first()
    return row is not None


def cancel_callback(cx: sa.Connection, callback_id: str) -> bool:
    """Stop a callback before it dials. The mirror image of claim_callback.

    Every READER of `cancelled_at` has existed since day one -- the CAS above,
    both partial indexes, the due sweep, ops.status -- and the only writer in
    the whole repo was one line inside a test. So there was no way to stop a
    callback that was already scheduled, which is exactly what happened when a
    background task nobody wanted finished and rang the owner ninety seconds after he
    had objected to it out loud.

    Race-safe by construction: the same three predicates as claim_callback, so
    exactly one of {fire, cancel} can win. Setting it makes claim_callback
    return False, drops the row out of callbacks_due_idx so the sweep stops
    re-enqueuing it, and RELEASES callbacks_one_pending_per_task so a later
    callback for the same task is still insertable.

    Returns False when the callback has already fired -- a truth the caller has
    to say out loud, because a dial already claimed cannot be recalled.
    """
    row = cx.execute(sa.text("""
        UPDATE shot.callbacks
           SET cancelled_at = now()
         WHERE id = :cid
           AND fired_at IS NULL
           AND cancelled_at IS NULL
     RETURNING id
    """), {"cid": callback_id}).first()
    return row is not None
