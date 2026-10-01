"""Smoke-test the tier-3 worker end to end, with the payment gate live.

Proves: session create returns immediately, the vault substitutes the Browserbase
key at egress, `limited` networking still lets `browse` out, and the
bash-always_ask gate round-trips through user.tool_confirmation.

The approver here is the seed of the supervisor's permission policy.

    uv run python -m ops.smoke_worker "your task"
"""

from __future__ import annotations

import sys

import anthropic

from shot_core.settings import get_settings

# Auto-approve read-only browsing. Anything else escalates to a human — which in
# the real system means a spoken confirmation plus the PIN.
ALLOW = ("browse cloud search", "browse cloud fetch", "browse skills show",
         "browse open", "browse snapshot", "browse get", "browse screenshot",
         "browse cloud projects", "browse cloud contexts get", "node --version",
         "npm ls", "which browse", "browse --version")
DENY_HINT = ("checkout", "place order", "pay", "purchase", "confirm order")


def _approve(cmd: str) -> tuple[bool, str]:
    low = " ".join(cmd.lower().split())
    if any(h in low for h in DENY_HINT):
        return False, "spends money; needs spoken confirmation + PIN"
    if any(low.startswith(a) for a in ALLOW):
        return True, ""
    return False, f"not on the allowlist: {cmd[:60]}"


def main(goal: str) -> None:
    s = get_settings()
    c = anthropic.Anthropic(api_key=s.anthropic_api_key.get_secret_value())

    session = c.beta.sessions.create(
        agent=s.anthropic_agent_id,
        environment_id=s.anthropic_environment_id,
        vault_ids=[s.anthropic_vault_id],
        resources=[{"type": "memory_store",
                    "memory_store_id": s.anthropic_memory_store_id,
                    "access": "read_write"}],
        title=goal[:120],
        metadata={"task_id": "smoke-1", "owner": s.owner_phone_number},
        budget={"type": "limit",
                "max_list_cost": {"amount": "60", "currency": "USD"}},  # 60c ceiling
        initial_events=[{"type": "user.message",
                         "content": [{"type": "text", "text": goal}]}],
    )
    print(f"session {session.id}  status={session.status}\n")

    pending: dict[str, str] = {}
    with c.beta.sessions.events.stream(session_id=session.id) as stream:
        for ev in stream:
            t = getattr(ev, "type", "")

            if t == "agent.message":
                for blk in getattr(ev, "content", []) or []:
                    if getattr(blk, "text", None):
                        print(f"[agent] {blk.text.strip()[:400]}")

            elif t == "agent.tool_use":
                cmd = ""
                inp = getattr(ev, "input", None) or {}
                if isinstance(inp, dict):
                    cmd = inp.get("command") or inp.get("file_path") or ""
                print(f"[tool ] {getattr(ev,'name','?')}: {str(cmd)[:110]}")
                pending[ev.id] = str(cmd)

            elif t == "session.status_idle":
                sr = getattr(ev, "stop_reason", None)
                kind = getattr(sr, "type", None)
                if kind == "requires_action":
                    for eid in getattr(sr, "event_ids", []) or []:
                        cmd = pending.get(eid, "")
                        ok, why = _approve(cmd)
                        print(f"[gate ] {'ALLOW' if ok else 'DENY '} {cmd[:70]}"
                              + (f"   ({why})" if why else ""))
                        c.beta.sessions.events.send(session_id=session.id, events=[{
                            "type": "user.tool_confirmation",
                            "tool_use_id": eid,
                            "result": "allow" if ok else "deny",
                            **({} if ok else {"deny_message": why}),
                        }])
                else:
                    print(f"\n=== idle: stop_reason={kind} ===")
                    break

            elif t == "session.status_terminated":
                print("\n=== terminated ===")
                break

            elif t == "session.error":
                print(f"[ERROR] {getattr(ev, 'error', ev)}")

    final = c.beta.sessions.retrieve(session.id)
    cost = getattr(getattr(final, "usage", None), "list_cost", None)
    print(f"\nstatus={final.status}  list_cost={cost}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "Find out when the next Los Angeles Lakers home game is. "
         "Reply with just the date, opponent, and time.")
