"""Provision the Anthropic environment, vault + credential, memory store, and agent.

Idempotent: each resource is looked up by name before creating. Safe to re-run.

    uv run python -m ops.bootstrap.anthropic_res [--dry-run]
"""

from __future__ import annotations

import io
import pathlib
import sys
from urllib.parse import urlparse

import anthropic

from ops.bootstrap.envfile import write_env
from shot_core.linear import MCP_URL as LINEAR_MCP_URL
from shot_core.settings import get_settings

ENV_NAME = "shot-worker-cloud"
VAULT_NAME = "shot-worker"
STORE_NAME = "shot-owner-memory"
AGENT_NAME = "shot-worker"

# From .env, like the voice prompts, so no tracked file names him. With the
# same OWNER_NAME this is byte-identical to the deployed prompt, so a re-run
# reports the agent unchanged rather than pushing an update.
OWNER = get_settings().owner_name

SYSTEM = f"""\
You are the background worker for {OWNER}'s phone assistant. You are given one task \
and you run it to completion without a human watching.

Use the `browse` CLI for anything on the web. It drives a real Chrome on \
Browserbase. Start with `browse skills show` if you are unsure of a command.

For anything behind a login -- an account page, an order, a checkout -- attach \
to {OWNER}'s signed-in browser profile:

    CDP=$(sh /workspace/bb_connect.sh)
    browse open https://example.com --cdp "$CDP"

Pass `--cdp "$CDP"` on every browse command in that task; without it you get a \
fresh signed-out browser. Run bb_connect.sh as often as you like -- it returns \
the same browser for the whole task.

If a site still shows you signed out, the saved login has expired. Say so \
plainly in your final answer -- {OWNER} re-does it in about a minute -- and do NOT \
try to log in yourself. You have no passwords and are not meant to have any.

Write durable facts about {OWNER} to /mnt/memory/ and read it before you start. \
Never write credentials, PINs, or tokens there.

Your FINAL message is read aloud down a phone line by a text-to-speech voice. \
It is not a document and nobody will ever see it.

- No markdown at all: no tables, no asterisks, no headings, no bullet points.
- No file paths, no URLs, no IDs. Never say you wrote a file somewhere.
- Write the actual answer in prose, the way you would say it to someone \
  driving. Numbers as words where natural. Two or three sentences.

Good: "Jazz Bistro has three shows tonight. The Gretzinger Trio at five, no \
cover, then the Hank Quartet at nine for twenty dollars."
Bad: "**What I found** | Time | Show | see /mnt/session/outputs/jazz.md"

Before any action that spends money or is hard to undo, stop and ask. Your bash \
calls are gated: a human may be asked to approve them.\
"""

# Module-level, not inlined in _agent(), because the reuse branch has to push
# the exact same shape via agents.update() -- see _agent().
MCP_SERVERS = [
    {"type": "url", "name": "linear", "url": LINEAR_MCP_URL},
]

TOOLS = [
    {
        "type": "agent_toolset_20260401",
        "default_config": {"enabled": True,
                           "permission_policy": {"type": "always_allow"}},
        "configs": [
            # Browserbase Fetch/Search via `browse` instead — keeps every
            # web hop inside the one audited egress path.
            {"name": "web_fetch", "enabled": False},
            {"name": "web_search", "enabled": False},
            # Payment gate DISABLED for now, at the owner's request. With
            # always_ask the session parks on requires_action for every
            # single command and a delegated task stalls forever unless the
            # supervisor answers -- which made the first demo unusable.
            # shot_supervisor.approvals still holds the policy; flip this
            # back to always_ask before pointing the agent at a saved card.
            {"name": "bash", "permission_policy": {"type": "always_allow"}},
        ],
    },
    {
        "type": "mcp_toolset",
        "mcp_server_name": "linear",
        # No allowlist and always_allow: full read+write, no approval
        # prompts, at the owner's explicit request. The platform default here is
        # always_ask, which — same failure as bash above — would park every
        # Linear call on requires_action with nothing answering it.
        "default_config": {"permission_policy": {"type": "always_allow"}},
    },
]


# Module-level for the same reason MCP_SERVERS is: the reuse branch has to push
# the exact same shape back via environments.update(). See _environment().
ENV_CONFIG = {
    "type": "cloud",
    "packages": {"npm": ["browse"]},
    "networking": {
        "type": "limited",
        # The browser runs ON Browserbase, so the sandbox itself never
        # needs open-web egress. This also structurally prevents
        # `browse --local` from being usable.
        "allowed_hosts": ["*.browserbase.com"],
        # REQUIRED whenever `packages` is set, else 400 — even if the
        # registry hosts are listed in allowed_hosts.
        "allow_package_managers": True,
        # The hosts in MCP_SERVERS are checked against this policy when a
        # SESSION is created, not when the agent is declared. So adding Linear
        # to the agent while this was False (its default) left a config that
        # passed every check in _verify and then rejected every task with
        # `MCP server host(s) blocked by environment network policy: "linear"
        # (mcp.linear.app)`. Both background tasks of 2026-09-10 died that
        # way, a third of a second after creation, with no session, no
        # session_id, and nothing to read back down the phone but the 400.
        #
        # This rather than putting mcp.linear.app in allowed_hosts: that list
        # is the SANDBOX's own egress, and holding it to Browserbase alone is
        # what makes `browse --local` structurally unusable. An MCP host is
        # dialled by the platform, not from inside the sandbox, so it does not
        # belong in it -- and widening the sandbox's egress to fix a
        # platform-side connection would trade a real property away for
        # nothing.
        "allow_mcp_servers": True,
    },
}


def _environment(c: anthropic.Anthropic, s) -> str:
    for e in c.beta.environments.list():
        if e.name == ENV_NAME and not getattr(e, "archived_at", None):
            # Pushed UNCONDITIONALLY on reuse, exactly like _agent() pushes
            # mcp_servers and for the same reason: this branch used to return
            # the id and change nothing at all, so an environment created
            # before a config change kept the old policy forever while every
            # future re-run printed `reuse` and looked like it had applied.
            # Editing ENV_CONFIG without this is a no-op against a live
            # workspace.
            c.beta.environments.update(e.id, config=ENV_CONFIG)
            print(f"  environment   reuse {e.id} (config synced)")
            return e.id
    e = c.beta.environments.create(name=ENV_NAME, config=ENV_CONFIG)
    print(f"  environment   created {e.id}")
    return e.id


def _vault(c: anthropic.Anthropic, s) -> str:
    vault_id = None
    for v in c.beta.vaults.list():
        if v.display_name == VAULT_NAME and not getattr(v, "archived_at", None):
            vault_id = v.id
            print(f"  vault         reuse {vault_id}")
            break
    if vault_id is None:
        v = c.beta.vaults.create(display_name=VAULT_NAME,
                                 metadata={"managed_by": "shot-bootstrap"})
        vault_id = v.id
        print(f"  vault         created {vault_id}")

    have = {cr.auth.secret_name for cr in c.beta.vaults.credentials.list(vault_id)
            if getattr(cr.auth, "secret_name", None)}
    if "BROWSERBASE_API_KEY" not in have:
        # environment_variable, NOT static_bearer: we drive the `browse` CLI, not
        # the hosted MCP. The sandbox holds an opaque placeholder; the real key is
        # substituted at the network egress boundary, so the agent cannot leak it
        # even under a full prompt-injection takeover.
        c.beta.vaults.credentials.create(
            vault_id=vault_id,
            display_name="Browserbase API key",
            auth={
                "type": "environment_variable",
                "secret_name": "BROWSERBASE_API_KEY",
                "secret_value": s.browserbase_api_key.get_secret_value(),
                "networking": {"type": "limited", "allowed_hosts": ["*.browserbase.com"]},
                # header-only is the narrower config; the request body is the
                # broader exposure surface.
                "injection_location": {"header": True},
            },
        )
        print("  credential    created BROWSERBASE_API_KEY")
    else:
        print("  credential    reuse BROWSERBASE_API_KEY")

    # static_bearer, not environment_variable: this credential authenticates
    # the platform's OWN request to the Linear MCP server, matched by
    # `mcp_server_url` — a different mechanism from the env-var injection
    # above, which the sandbox reads directly. static_bearer credentials
    # carry no `secret_name`, so they cannot share the `have` set above; a
    # naive reuse of `getattr(cr.auth, "secret_name", None)` silently skips
    # every credential of this type.
    have_urls = {cr.auth.mcp_server_url for cr in c.beta.vaults.credentials.list(vault_id)
                if getattr(cr.auth, "type", None) == "static_bearer"}
    if LINEAR_MCP_URL not in have_urls:
        c.beta.vaults.credentials.create(
            vault_id=vault_id,
            display_name="Linear API key (worker)",
            auth={
                "type": "static_bearer",
                "mcp_server_url": LINEAR_MCP_URL,
                "token": s.linear_api_key_worker.get_secret_value(),
            },
        )
        print("  credential    created Linear static_bearer")
    else:
        print("  credential    reuse Linear static_bearer")
    return vault_id


HELPER_PATH = "/workspace/bb_connect.sh"
HELPER_NAME = "bb_connect.sh"


def _browser_helper(c: anthropic.Anthropic, s) -> str:
    """Upload the CDP connect helper the worker uses to reach the signed-in profile.

    `browse --remote` builds its own Browserbase session and cannot attach a
    Context -- v0.9.6 sets only `browserSettings.verified` and reads no context
    env var -- so simply setting BROWSERBASE_CONTEXT_ID does nothing at all.
    The session is created by this script instead, and browse attaches to it
    with `--cdp`.

    Mounted read-only on every session, so the model does not have to get a
    multi-step curl right from a prompt.
    """
    if not s.browserbase_context_id:
        print("  browser cred  SKIP (no BROWSERBASE_CONTEXT_ID)")
        return ""
    tmpl = (pathlib.Path(__file__).parent / "sandbox" / "bb_connect.sh.tmpl").read_text()
    body = (tmpl.replace("__PROJECT_ID__", s.browserbase_project_id)
                .replace("__CONTEXT_ID__", s.browserbase_context_id))
    f = c.beta.files.upload(file=(HELPER_NAME, io.BytesIO(body.encode()), "text/x-shellscript"))
    print(f"  browser cred  uploaded {f.id} (context {s.browserbase_context_id[:8]})")
    return f.id


def _memory_store(c: anthropic.Anthropic, s) -> str:
    # NOTE: memory-store endpoints use agent-memory-2026-07-22 INSTEAD of the
    # managed-agents header. Sending both is a 400. The SDK handles it.
    matches = [m for m in c.beta.memory_stores.list()
               if m.name == STORE_NAME and not getattr(m, "archived_at", None)]
    if len(matches) > 1:
        raise SystemExit(f"{len(matches)} memory stores named {STORE_NAME}; resolve by hand")
    if matches:
        print(f"  memory store  reuse {matches[0].id}")
        return matches[0].id
    m = c.beta.memory_stores.create(
        name=STORE_NAME,
        # This description is shown to the model — write it for the model.
        description=(f"Durable facts about {OWNER}: preferences, addresses, standing orders, "
                     "recurring vendors, and outcomes of prior delegated tasks. Read this "
                     "before starting any task. Never write credentials, PINs, or tokens here."),
    )
    print(f"  memory store  created {m.id}")
    return m.id


def _agent(c: anthropic.Anthropic, s, env_id: str) -> str:
    for a in c.beta.agents.list():
        if a.name == AGENT_NAME and not getattr(a, "archived_at", None):
            # mcp_servers/tools are pushed UNCONDITIONALLY on every reuse, not
            # diffed like system below. They're structured API response
            # objects, not a string, so there's no cheap equality check
            # against what this file would construct locally -- and the old
            # code never pushed them at all outside creation. That meant an
            # agent that already existed before Linear was added would keep
            # mcp_servers=[] forever, on every future bootstrap re-run,
            # while looking like the change had taken effect. update() only
            # touches the fields you pass, so passing them every time is
            # what makes this actually idempotent rather than create-only.
            changed = (a.system or "") != SYSTEM
            c.beta.agents.update(a.id, system=SYSTEM, mcp_servers=MCP_SERVERS, tools=TOOLS)
            print(f"  agent         reuse {a.id}"
                  + (" (system prompt updated)" if changed else "")
                  + " (mcp servers/tools synced)")
            return a.id
    a = c.beta.agents.create(
        name=AGENT_NAME,
        model={"id": "claude-opus-5", "effort": "high"},
        description="Tier-3 worker for the phone-callable agent.",
        system=SYSTEM,
        mcp_servers=MCP_SERVERS,
        tools=TOOLS,
        metadata={"managed_by": "shot-bootstrap", "tier": "3"},
    )
    print(f"  agent         created {a.id}")
    return a.id


def _verify(c: anthropic.Anthropic, env_id: str, vault_id: str,
            store_id: str, agent_id: str) -> None:
    e = c.beta.environments.retrieve(env_id)
    net = e.config.networking
    assert net.type == "limited", net
    assert "*.browserbase.com" in net.allowed_hosts
    assert net.allow_package_managers is True

    # The invariant nothing checked, and the one that actually stops a task:
    # every MCP host declared on the agent has to be reachable under THIS
    # environment's policy, or sessions 400 on create while all three asserts
    # above still pass. That is exactly what shipped on 2026-09-09 and what
    # both tasks on 2026-09-10 hit. Assert the pairing, not either half.
    for srv in MCP_SERVERS:
        host = urlparse(srv["url"]).hostname or ""
        assert net.allow_mcp_servers is True or host in net.allowed_hosts, (
            f"MCP host {host!r} is declared on the agent but blocked by the "
            f"environment network policy — every session will 400 on create")

    creds = list(c.beta.vaults.credentials.list(vault_id))
    bb = [x for x in creds if getattr(x.auth, "secret_name", None) == "BROWSERBASE_API_KEY"]
    assert len(bb) == 1, f"{len(bb)} Browserbase credentials"
    assert bb[0].auth.injection_location.header is True

    linear_creds = [x for x in creds if getattr(x.auth, "type", None) == "static_bearer"
                    and getattr(x.auth, "mcp_server_url", None) == LINEAR_MCP_URL]
    assert len(linear_creds) == 1, f"{len(linear_creds)} Linear vault credentials"

    assert c.beta.memory_stores.retrieve(store_id).name == STORE_NAME

    a = c.beta.agents.retrieve(agent_id)
    assert a.model.id == "claude-opus-5", a.model
    ts = next(t for t in a.tools if t.type == "agent_toolset_20260401")
    by = {cfg.name: cfg for cfg in ts.configs}
    # always_allow while the payment gate is disabled at the owner's request;
    # see shot_supervisor.approvals for the policy to re-enable.
    assert by["bash"].permission_policy.type in ("always_ask", "always_allow")
    assert by["web_search"].enabled is False

    servers = {srv.name: srv for srv in (a.mcp_servers or [])}
    assert "linear" in servers, "linear MCP server not declared on the agent"
    assert servers["linear"].url == LINEAR_MCP_URL
    mcp_ts = next(t for t in a.tools if getattr(t, "type", None) == "mcp_toolset"
                  and t.mcp_server_name == "linear")
    # Full read+write, no approval prompts -- bypassed at the owner's explicit
    # request, same as bash above.
    assert mcp_ts.default_config.permission_policy.type == "always_allow"
    print("  verified      all assertions passed")


def main(dry_run: bool = False) -> None:
    s = get_settings()
    c = anthropic.Anthropic(api_key=s.anthropic_api_key.get_secret_value())
    env_id = _environment(c, s)
    vault_id = _vault(c, s)
    store_id = _memory_store(c, s)
    helper_id = _browser_helper(c, s)
    agent_id = _agent(c, s, env_id)
    _verify(c, env_id, vault_id, store_id, agent_id)
    for line in write_env({
        "ANTHROPIC_ENVIRONMENT_ID": env_id,
        "ANTHROPIC_VAULT_ID": vault_id,
        "ANTHROPIC_MEMORY_STORE_ID": store_id,
        "ANTHROPIC_BROWSER_HELPER_FILE_ID": helper_id,
        "ANTHROPIC_AGENT_ID": agent_id,
    }, dry_run=dry_run):
        print("  ", line)


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
