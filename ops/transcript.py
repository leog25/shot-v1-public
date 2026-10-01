"""What was said on a call, read back.

The last callback failure was pinned down to "the model called end_call six
seconds in" and no further, because nothing kept the words. `shot.transcripts`
does now, and this reads it.

    uv run python -m ops.transcript              # the most recent call
    uv run python -m ops.transcript 3            # the last 3
    uv run python -m ops.transcript cb-8e94a9e8  # one room, by prefix
"""

from __future__ import annotations

import sys

import sqlalchemy as sa

from shot_core.db import make_engine
from shot_core.settings import get_settings

DIM, AGENT, USER, NOTE, CUT, END = (
    "\033[2m", "\033[36m", "\033[32m", "\033[35m", "\033[33m", "\033[0m")

# "tool" is four characters so it fits the 9-wide role column below;
# "tool_result" would misalign every line in the file.
ROLE_COLOUR = {"assistant": AGENT, "user": USER, "note": NOTE, "tool": CUT}


def main(arg: str = "1") -> int:
    eng = make_engine(get_settings().database_url)
    if arg.isdigit():
        where, params = "", {"n": int(arg)}
    else:
        where, params = "WHERE t.room_name LIKE :room", {"n": 20, "room": f"{arg}%"}

    with eng.begin() as cx:
        rows = cx.execute(sa.text(f"""
            SELECT t.room_name, t.direction, t.turns, t.created_at,
                   c.end_reason, c.heard_chars, c.interrupted
              FROM shot.transcripts t
              LEFT JOIN shot.calls c ON c.room_name = t.room_name
             {where}
             ORDER BY t.created_at DESC LIMIT :n"""), params).mappings().all()

    if not rows:
        print("no transcripts recorded yet" if not where else f"no call matching {arg!r}")
        return 0

    for r in rows:
        outcome = r["end_reason"] or "?"
        print(f"\n{DIM}{'─' * 78}{END}")
        print(f"{r['created_at']:%Y-%m-%d %H:%M:%S}  {r['direction']}  "
              f"{outcome}  {DIM}{r['room_name']}{END}")
        # The one number that says whether he heard his results, right above the
        # words that prove it either way.
        if r["heard_chars"] is not None:
            cut = f" {CUT}(cut off){END}" if r["interrupted"] else ""
            print(f"{DIM}heard {r['heard_chars']} chars of the brief{END}{cut}")
        print(f"{DIM}{'─' * 78}{END}")

        for t in r["turns"] or []:
            role = t.get("role", "?")
            colour = ROLE_COLOUR.get(role, DIM)
            mark = f" {CUT}[interrupted]{END}" if t.get("interrupted") else ""
            print(f"{DIM}{t.get('at', 0):>6.1f}s{END} {colour}{role:>9}{END}{mark}  "
                  f"{t.get('text', '')}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "1"))
