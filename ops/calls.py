"""What happened on the last N calls.

Every callback failure this system has had was diagnosed by grepping
/tmp/shot_worker.log -- which is deleted on restart, and which for one failure
was being written by an orphaned worker to a file that no longer existed. The
`shot.calls` table had the right columns from day one and was never written to.
Now it is, and this reads it.

    uv run python -m ops.calls          # last 10
    uv run python -m ops.calls 30
"""

from __future__ import annotations

import sys

import sqlalchemy as sa

from shot_core.db import make_engine
from shot_core.settings import get_settings

DIM, OK, BAD, WARN, END = "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"

# Outcomes that are nobody's fault and need no action.
BENIGN = {"no_answer", "busy", "machine-vm", "machine-ivr", "machine-unavailable"}


def main(limit: int = 10) -> int:
    eng = make_engine(get_settings().database_url)
    with eng.begin() as cx:
        rows = cx.execute(sa.text("""
            SELECT started_at, direction, end_reason, amd_category, amd_reason,
                   amd_delay_ms, amd_speech_s, pin_verified, pin_result,
                   sip_status_code,
                   failure_kind IS NULL AS delivered,
                   failure_kind, callback_id, heard_chars, interrupted
              FROM shot.calls ORDER BY started_at DESC LIMIT :n"""),
            {"n": limit}).mappings().all()

    if not rows:
        print("no calls recorded yet")
        return 0

    print(f"{'when':<9} {'outcome':<16} {'AMD':<16} {'why':<20} "
          f"{'delay':>6} {'speech':>7}  {'pin':<6} heard")
    print("─" * 100)
    for r in rows:
        outcome = r["end_reason"] or "?"
        colour = OK if r["delivered"] else (DIM if outcome in BENIGN else BAD)
        # pin_verified is NOT NULL with a default of false, so it says only
        # whether, never why. `silent` and `wrong` are different calls with
        # different answers: nobody keying anything is very likely a machine or
        # a handset in a pocket; a wrong code is a person fumbling. Pad the TEXT
        # before wrapping it in colour -- padding the coloured string counts the
        # escape bytes and the column drifts.
        if outcome not in ("human", "pin_failed", "abandoned"):
            pin_txt, pin_col = "-", DIM        # never asked
        elif r["pin_verified"]:
            pin_txt, pin_col = "ok", OK
        elif r["pin_result"] == "no_input":
            pin_txt, pin_col = "silent", WARN
        elif r["pin_result"] == "wrong":
            pin_txt, pin_col = "wrong", BAD
        elif r["pin_result"] == "abandoned":
            pin_txt, pin_col = "hungup", DIM   # he left; not an identity failure
        else:
            pin_txt, pin_col = "no", BAD       # rows written before pin_result
        pin = f"{pin_col}{pin_txt:<6}{END}"
        delay = f'{r["amd_delay_ms"]}ms' if r["amd_delay_ms"] is not None else "-"
        speech = f'{r["amd_speech_s"]:.2f}s' if r["amd_speech_s"] is not None else "-"
        # 0.0s speech is the tell for the answer-and-wait deadlock; >9s is the
        # livekit/agents#6996 stall.
        if r["amd_speech_s"] == 0.0:
            speech = f"{WARN}{speech}{END}"
        if (r["amd_delay_ms"] or 0) > 9000:
            delay = f"{BAD}{delay}{END}"
        # How much of the brief he actually got. "cut" is a conversation
        # starting, not a failure -- it used to be answered by hanging up on
        # him. An empty heard on an undelivered human call is the real one.
        if r["heard_chars"] is None:
            heard = f"{DIM}-{END}"
        elif r["delivered"]:
            heard = f'{OK}{r["heard_chars"]}c{END}'
        elif r["interrupted"]:
            heard = f'{WARN}{r["heard_chars"]}c cut{END}'
        else:
            heard = f"{BAD}nothing{END}"
        print(f'{r["started_at"]:%H:%M:%S} {colour}{outcome:<16}{END} '
              f'{r["amd_category"] or "-":<16} {(r["amd_reason"] or "-")[:20]:<20} '
              f'{delay:>6} {speech:>7}  {pin} {heard}')

    # "reached the owner" means the brief actually finished playing -- NOT that AMD
    # said human. A call once recorded outcome=human while the agent never read
    # the brief at all, and the summary line called it a success.
    delivered = sum(1 for r in rows if r["delivered"])
    print(f"\n{delivered}/{len(rows)} actually heard the brief")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
