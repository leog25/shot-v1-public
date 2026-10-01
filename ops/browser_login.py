"""Sign in to a site once, by hand, so the worker's browser stays signed in.

Credentials are NOT stored anywhere in this system. What persists is a
Browserbase Context -- a saved Chromium user-data dir (cookies, localStorage,
IndexedDB). You log in through a real browser window; the agent inherits the
cookie. There is no password in .env, in the vault, in the sandbox, or in a log,
because none was ever collected.

    uv run python -m ops.browser_login            # open the profile to log in
    uv run python -m ops.browser_login --check    # what is in there / when touched

Two-step form, for when nobody is sitting at this terminal to press ENTER
(an agent driving it, or a browser opened from another machine):

    uv run python -m ops.browser_login --open           # prints a session id
    uv run python -m ops.browser_login --finish <id>    # release + verify

One context holds every site. Run this once per site you want the agent signed
in to (a delivery app, a store, ...), in the same context, and they accumulate.

`persist: true` is set here and ONLY here -- that is what writes the profile
back. Worker sessions attach with persist: false so a confused agent cannot
clobber a login you set up by hand.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from shot_core.settings import get_settings

API = "https://api.browserbase.com"
# Long enough to find the password, get the 2FA text, and mistype it twice.
LOGIN_WINDOW_S = 1800

DIM, OK, WARN, BAD, END = "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def _call(key: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-BB-API-Key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"browserbase {method} {path} -> {e.code}: "
                         f"{e.read()[:300].decode(errors='replace')}") from None


def _context(key: str, ctx_id: str) -> dict:
    return _call(key, "GET", f"/v1/contexts/{ctx_id}")


def _cookie_domains(key: str, s, ctx_id: str) -> dict[str, int]:
    """Open a throwaway browser on the profile and see what cookies it carries.

    This is the ONLY honest proof that a login persisted. `updatedAt` on the
    context looks like the obvious signal and is not one: it tracks metadata
    edits, never profile syncs, so it sits at createdAt forever while
    persistence works perfectly. Trusting it reported a successful
    login as "nothing was saved".

    persist:false -- reading the profile must never be able to write to it.
    """
    import websockets  # not needed for --open; keep the import off that path

    sess = _call(key, "POST", "/v1/sessions", {
        "projectId": s.browserbase_project_id,
        "browserSettings": {"context": {"id": ctx_id, "persist": False}},
        "timeout": 120, "keepAlive": True})

    async def read() -> list[dict]:
        async with websockets.connect(sess["connectUrl"], max_size=None) as ws:
            await ws.send(json.dumps({"id": 1, "method": "Storage.getCookies"}))
            while True:
                m = json.loads(await ws.recv())
                if m.get("id") == 1:
                    return m.get("result", {}).get("cookies", [])

    try:
        cookies = asyncio.run(read())
    finally:
        try:
            _call(key, "POST", f"/v1/sessions/{sess['id']}",
                  {"status": "REQUEST_RELEASE"})
        except SystemExit:
            pass

    out: dict[str, int] = {}
    for c in cookies:
        out[c.get("domain", "?").lstrip(".")] = out.get(c.get("domain", "?").lstrip("."), 0) + 1
    return out


def check(key: str, ctx_id: str, s) -> int:
    """What is this browser actually signed in to?"""
    print(f"context   {ctx_id}")
    print(f"{DIM}opening a throwaway browser on the profile to read it back…{END}")
    domains = _cookie_domains(key, s, ctx_id)
    if not domains:
        print(f"{WARN}no cookies: this profile is empty. Run without --check "
              f"to sign in.{END}")
        return 1
    print(f"{OK}{sum(domains.values())} cookie(s) across "
          f"{len(domains)} domain(s):{END}")
    for d, n in sorted(domains.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {d}")
    return 0


def _open_session(key: str, s, ctx_id: str) -> tuple[str, str]:
    """Start a browser on the profile and return (session id, live view url)."""
    sess = _call(key, "POST", "/v1/sessions", {
        "projectId": s.browserbase_project_id,
        # persist:true is the whole point of this script.
        "browserSettings": {"context": {"id": ctx_id, "persist": True}},
        "timeout": LOGIN_WINDOW_S,
        "keepAlive": True,
    })
    live = _call(key, "GET", f"/v1/sessions/{sess['id']}/debug")
    return sess["id"], (live.get("debuggerFullscreenUrl") or live.get("debuggerUrl"))


def _finish(key: str, ctx_id: str, sid: str, s) -> int:
    """Close the browser, then prove the profile kept what was just done in it."""
    _call(key, "POST", f"/v1/sessions/{sid}", {"status": "REQUEST_RELEASE"})
    print(f"  {DIM}session released; letting the profile sync…{END}")
    time.sleep(8)          # the upload happens after the session ends
    return check(key, ctx_id, s)


def main(argv: list[str]) -> int:
    s = get_settings()
    key = s.browserbase_api_key.get_secret_value()
    ctx_id = s.browserbase_context_id
    if not ctx_id:
        raise SystemExit("BROWSERBASE_CONTEXT_ID is not set; create one with "
                         "`browse cloud contexts create` and put it in .env")

    if "--check" in argv:
        return check(key, ctx_id, s)

    if "--finish" in argv:
        sid = argv[argv.index("--finish") + 1]
        # `before` is the state the profile had when the session opened; a
        # session that logged in has moved it by now, so compare against the
        # value from --open. Re-reading here would compare a timestamp with
        # itself and always report failure.
        return _finish(key, ctx_id, sid, s)

    if "--open" in argv:
        sid, url = _open_session(key, s, ctx_id)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print(f"\n  {OK}Browser open on the agent's profile.{END} Log in, then:\n")
        print(f"    uv run python -m ops.browser_login --finish {sid}\n")
        print(f"  {url}\n")
        return 0

    sid, url = _open_session(key, s, ctx_id)

    print(f"\n  {OK}Opening a browser you control, on the agent's profile.{END}\n")
    print(f"  {url}\n")
    print(f"  {DIM}Sign in to whatever you want the agent signed in to — "
          f"delivery apps, stores,\n  anything. Do the 2FA. Add a card if you want "
          f"one saved.\n\n  Nothing you type here is recorded by this system: "
          f"only the resulting\n  cookies are, and only inside Browserbase.\n\n"
          f"  Press ENTER when you are done.{END}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\n  (no ENTER — releasing the session anyway)")

    return _finish(key, ctx_id, sid, s)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
