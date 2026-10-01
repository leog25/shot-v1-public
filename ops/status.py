"""One command to see what is actually live.

    uv run python -m ops.status

Checks every tier and every external dependency, and reports anything that
could ring the owner's phone. Read-only: never creates, cancels, or dials.
"""

from __future__ import annotations

import os
import subprocess
import sys

import httpx

from shot_core.settings import get_settings

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(state: str, label: str, detail: str = "") -> None:
    print(f"[{state}] {label:<26} {detail}")


def _systemd() -> bool:
    """True on the deployed VM, where both tiers run as systemd units."""
    try:
        out = subprocess.run(["systemctl", "is-enabled", "shot-worker"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return False            # macOS: no systemd at all
    return out in ("enabled", "static", "disabled")


def _worker_roots() -> list[int]:
    """Worker processes that are not the child of another worker.

    `uv run` spawns a child matching the same pattern, so a naive pgrep count
    always reads double. Matches `shot_voice.worker` without the subcommand:
    the laptop runs `dev`, the VM runs `start`.
    """
    out = subprocess.run(["pgrep", "-f", "shot_voice.worker"],
                         capture_output=True, text=True).stdout.split()
    pids = {int(p) for p in out if p.isdigit()}
    roots = []
    for pid in sorted(pids):
        ppid = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                              capture_output=True, text=True).stdout.strip()
        if not ppid.isdigit() or int(ppid) not in pids:
            roots.append(pid)
    return roots


# The two lines the LiveKit SDK writes either side of a connection drop. It
# logs the failure first, then the success when the retry lands -- often in the
# SAME second.
_REGISTERED = "registered worker"
_LOST = "failed to connect to livekit"


def _worker_registered(text: str) -> bool | None:
    """Is the worker registered RIGHT NOW, per its own log? None = cannot tell.

    The LAST of the two markers wins. Mere presence of `registered worker` is
    not enough and mere presence of a failure is not either: the worker drops
    and re-registers in place, without restarting -- six times in the eleven
    days to 2026-09-22, each reconnecting inside the same second.

    This used to be `"registered worker" in journalctl -n 200`, which is a
    presence test over a FIXED LINE WINDOW, and that window is the bug. Two
    callbacks write ~250 lines, so on 2026-09-10 the registration line aged out
    and `ops.status` reported `running but not registered` about a worker that
    had accepted five jobs inside that very window. The phone line was fine;
    the check was not, and it was read as an outage and "fixed" by a restart
    that only wrote a fresh registration line. The same cap fails the other
    way too: after a genuine drop it would keep reporting OK until 200 more
    lines had pushed the stale success out.

    Pure, and the whole reason this is a function: the failure it exists to
    prevent is a reading error, which is exactly what a test can pin.
    """
    last = None
    for ln in text.splitlines():
        # Order matters within a line only in theory; these never co-occur.
        if _REGISTERED in ln:
            last = True
        elif _LOST in ln:
            last = False
    return last


def main() -> int:
    s = get_settings()
    bad = 0

    print("── local ──────────────────────────────────────────────")
    try:
        h = httpx.get(f"{s.supervisor_base_url}/healthz", timeout=4).json()
        missing = h.get("missing_resource_ids") or []
        line(OK if not missing else WARN, "supervisor",
             f"{s.supervisor_base_url}" + (f"  missing={missing}" if missing else ""))
    except Exception as e:
        line(BAD, "supervisor", f"down ({type(e).__name__}) — start it, see RUNBOOK.md")
        bad += 1

    pg = subprocess.run(["docker", "ps", "--filter", "name=shot-pg", "--format", "{{.Status}}"],
                        capture_output=True, text=True).stdout.strip()
    line(OK if pg else BAD, "postgres", pg or "container not running")
    bad += 0 if pg else 1

    # The log is not proof of life. This said "registered as shot-voice" for
    # two and a half hours after the worker had exited -- it was reading a
    # registration line written before the process died, while the owner's calls went
    # nowhere. Ask the OS first, then the log.
    roots = _worker_roots()
    if _systemd():
        # journald is the log on the VM; /tmp/shot_worker.log only exists on the
        # laptop, and reading it there reported a live worker for hours after
        # the process had exited.
        #
        # The WHOLE unit journal, with no `-n` cap: the cap is what made this
        # check wrong (see _worker_registered). grep-shaped work on one unit is
        # cheap next to the API calls this command already makes.
        reg = _worker_registered(subprocess.run(
            ["journalctl", "-u", "shot-worker", "--no-pager"],
            capture_output=True, text=True).stdout)
        log = "journalctl -u shot-worker"
        fixit = "sudo systemctl restart shot-worker"
    else:
        log = "/tmp/shot_worker.log"
        reg = (_worker_registered(open(log, errors="ignore").read())
               if os.path.exists(log) else None)
        fixit = "run ops/restart.sh"
    if not roots:
        why = "PROCESS IS GONE"
        if not _systemd() and os.path.exists(log):
            txt = open(log, errors="ignore").read()
            if "failed to connect to livekit" in txt:
                why = "PROCESS IS GONE — it lost LiveKit and gave up retrying"
        line(BAD, "voice worker", f"{why}; {fixit}")
        bad += 1
    elif reg is False:
        line(BAD, "voice worker",
             f"running but LOST its LiveKit connection (check {log})")
        bad += 1
    elif reg is None:
        # Not a failure. The process is up and nothing in the log says either
        # way -- a rotated journal looks exactly like this, and calling that an
        # outage is the mistake this whole function exists to stop making.
        line(WARN, "voice worker",
             f"running; the log says nothing either way (check {log})")
    else:
        line(OK, "voice worker", "registered as shot-voice")

    # More than one worker is not redundancy, it is a coin flip. LiveKit
    # load-balances jobs across every registered worker, so an orphan left over
    # from a restart keeps answering real calls with whatever code it started
    # with. One survived for an hour and forty minutes behind a stale pid file
    # and took a live callback, which read exactly like the fixes not working.
    fixit = "sudo systemctl restart shot-worker" if _systemd() else "run ops/restart.sh"
    line(OK if len(roots) == 1 else BAD, "worker count",
         "1 (correct)" if len(roots) == 1
         else f"{len(roots)} running — pids {roots}; {fixit}")
    bad += 0 if len(roots) == 1 else 1

    print("\n── work in flight (can this ring the owner?) ────────────────")
    try:
        tasks = httpx.get(f"{s.supervisor_base_url}/internal/tasks", timeout=8).json()["tasks"]
        line(OK if not tasks else WARN, "open tasks",
             "none" if not tasks else ", ".join(f"{t['ref']}={t['state']}" for t in tasks))
        import sqlalchemy as sa

        from shot_core.db import make_engine
        with make_engine(s.database_url).begin() as cx:
            pend = cx.execute(sa.text("""
                SELECT COALESCE(to_number, owner_phone), to_char(due_at,'HH24:MI')
                  FROM shot.callbacks
                 WHERE fired_at IS NULL AND cancelled_at IS NULL""")).all()
        line(OK if not pend else WARN, "pending callbacks",
             "none" if not pend else "; ".join(f"{n} at {t}" for n, t in pend))
    except Exception as e:
        line(WARN, "work in flight", f"unreadable ({type(e).__name__})")

    print("\n── telephony ──────────────────────────────────────────")
    line(OK, "agent number", s.twilio_phone_number)
    line(OK, "owner (allowlisted)", s.owner_phone_number)
    # Defaults rather than fails, so nothing else would say it is missing.
    named = s.owner_name != type(s).model_fields["owner_name"].default
    line(OK if named else WARN, "owner name",
         s.owner_name if named else "OWNER_NAME unset -- the agent calls him \"the owner\"")
    line(OK, "livekit sip host", s.livekit_sip_host)
    try:
        r = httpx.get(f"https://trunking.twilio.com/v1/Trunks/{s.twilio_trunk_sid}/OriginationUrls",
                      auth=(s.twilio_api_key_sid, s.twilio_api_key_secret.get_secret_value()),
                      timeout=8).json()["origination_urls"]
        url = r[0]["sip_url"] if r else ""
        good = s.livekit_sip_host in url
        line(OK if good else BAD, "twilio origination",
             url + ("" if good else "  <-- does NOT match LIVEKIT_SIP_HOST"))
        bad += 0 if good else 1
    except Exception as e:
        line(WARN, "twilio origination", f"unreadable ({type(e).__name__})")

    print("\n── anthropic ──────────────────────────────────────────")
    try:
        import anthropic
        c = anthropic.Anthropic(api_key=s.anthropic_api_key.get_secret_value())
        live = list(c.beta.sessions.list(statuses=["running"], limit=20))
        line(OK if not live else WARN, "running worker sessions",
             "none" if not live else f"{len(live)} — these are spending money")
        spend = 0.0
        for sess in c.beta.sessions.list(limit=100, include_archived=True):
            u = getattr(sess, "usage", None)
            if u and getattr(u, "list_cost", None):
                spend += float(u.list_cost.amount) / 100
        line(OK, "lifetime session spend", f"${spend:,.2f}")
        line(OK, "per-session cap", f"${s.session_budget_cents / 100:.2f}")

        # Presence only, never a health check: MCP auth/connection failures
        # surface only as a session.error event on a running session, which
        # nothing here reads. This just answers "is the credential
        # registered at all" -- eyeball, not proof it still works.
        from shot_core.linear import MCP_URL as LINEAR_MCP_URL
        creds = list(c.beta.vaults.credentials.list(s.anthropic_vault_id))
        linear_cred = next((cr for cr in creds
                            if getattr(cr.auth, "type", None) == "static_bearer"
                            and getattr(cr.auth, "mcp_server_url", None) == LINEAR_MCP_URL),
                           None)
        line(OK if linear_cred else WARN, "linear vault credential",
             "registered" if linear_cred else "MISSING — worker has no Linear access")
    except Exception as e:
        msg = str(e)
        if "credit balance is too low" in msg:
            line(BAD, "anthropic", "OUT OF CREDIT — tasks will fail to start")
            bad += 1
        else:
            line(BAD, "anthropic", f"{type(e).__name__}: {msg[:70]}")
            bad += 1

    print("\n── linear (voice tier) ────────────────────────────────")
    # Tier 3's credential is checked above, in the Anthropic vault. Nothing
    # checked tier 1, which reads LINEAR_API_KEY_VOICE directly -- so a revoked
    # or mistyped key first surfaced as check_my_day quietly dropping half its
    # answer on a live call.
    if not s.linear_api_key_voice.get_secret_value():
        line(WARN, "linear key (voice)", "unset — check_my_day answers tasks only")
    else:
        try:
            from shot_core.budget import LINEAR_QUERY_S
            # The RAW key, no "Bearer " prefix. Sending Bearer here is a 401
            # that reads exactly like a bad key. worker.py used to send Bearer
            # for this same secret against the MCP endpoint -- two conventions
            # for one key in one repo -- and no longer does: tier 1 is GraphQL
            # only, and Bearer is tier 3's business alone.
            r = httpx.post(
                "https://api.linear.app/graphql",
                headers={"Authorization": s.linear_api_key_voice.get_secret_value()},
                json={"query": "query { viewer { displayName } }"},
                timeout=LINEAR_QUERY_S)
            body = r.json() if r.status_code == 200 else {}
            # A GraphQL failure is a 200 with a top-level `errors` array.
            who = ((body.get("data") or {}).get("viewer") or {}).get("displayName")
            if who:
                line(OK, "linear key (voice)", f"authenticates as {who}")
            else:
                err = (body.get("errors") or [{}])[0].get("message") or r.status_code
                line(BAD, "linear key (voice)", f"rejected — {str(err)[:60]}")
                bad += 1
        except Exception as e:
            line(BAD, "linear key (voice)", f"{type(e).__name__}: {str(e)[:60]}")
            bad += 1

    print("\n── dashboards ─────────────────────────────────────────")
    for label, url in [
        # Project ids come from .env so no tracked file names the account.
        # The SIP host is the LiveKit project id minus its `p_` prefix.
        ("livekit", "https://cloud.livekit.io/projects/p_"
                    + s.livekit_sip_host.split(".")[0]),
        ("anthropic sessions", "https://platform.claude.com/settings/workspaces/default"),
        ("anthropic billing", "https://console.anthropic.com/settings/billing"),
        ("twilio calls", "https://console.twilio.com/us1/monitor/logs/calls"),
        ("browserbase", "https://www.browserbase.com/sessions"),
        ("cloud run (hooks)", f"https://console.cloud.google.com/run?project={s.gcp_project_id}"),
    ]:
        print(f"       {label:<22} {url}")

    print()
    print("all good" if not bad else f"{bad} problem(s) above")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
