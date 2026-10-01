"""L1 contract tests — real APIs, cheap, non-interactive.

These exist because the failures they catch are otherwise invisible until a
phone is ringing. Each one encodes a specific trap found during research.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from shot_core.settings import get_settings

pytestmark = pytest.mark.contract


def test_settings_load_and_derive():
    s = get_settings()
    # The SIP host is CONFIGURED, never derived. Deriving it from LIVEKIT_URL
    # produced shot-if1tuw2o.sip.livekit.cloud, which resolves (wildcard DNS),
    # accepts TCP, answers SIP, and returns `404 No trunk found` -- so every
    # inbound call failed while looking like a trunk misconfiguration. The real
    # host is the project id minus its `p_` prefix.
    assert s.livekit_sip_host, "LIVEKIT_SIP_HOST unset; inbound calls will 404"
    assert not s.livekit_sip_host.startswith(
        s.livekit_url.split("://")[1].split(".")[0]), \
        "SIP host looks derived from LIVEKIT_URL; read it from Project settings"
    assert s.twilio_phone_number.startswith("+1")
    assert s.owner_phone_number.startswith("+1")
    # never `minimal`: -7.7pts agentic / -2.1pts task success to save 240ms,
    # on the metric that is literally this agent's job.
    assert s.openai_reasoning_effort != "minimal"


def test_ringing_timeout_beats_sdk_default():
    """The Python SDK pins DEFAULT_RINGING_TIMEOUT=30.0 when the field is unset.

    A US/CA mobile rolls to voicemail at ~20-25s, so an unset value means we
    reliably talk to voicemail instead of failing fast and retrying.
    """
    assert get_settings().ringing_timeout_seconds < 30


@pytest.mark.asyncio
async def test_realtime_model_string_is_real():
    """`gpt-realtime-2.1` is absent from the plugin's RealtimeModels literal.

    It is typed `RealtimeModels | str`, so a bare string passes at runtime and
    NOTHING validates it until the first live call. Note that
    POST /v1/realtime/client_secrets does NOT validate the model either — it
    happily accepts `gpt-realtime-9.9-nope`. Only opening a session does.
    """
    websockets = pytest.importorskip("websockets")
    s = get_settings()

    async def session_model(model: str) -> str | None:
        url = f"wss://api.openai.com/v1/realtime?model={model}"
        hdr = {"Authorization": f"Bearer {s.openai_api_key.get_secret_value()}"}
        async with websockets.connect(url, additional_headers=hdr, open_timeout=20) as ws:
            for _ in range(4):
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                if ev.get("type") == "session.created":
                    return ev["session"].get("model")
                if ev.get("type") == "error":
                    return None
        return None

    assert await session_model(s.openai_realtime_model) == s.openai_realtime_model
    # negative control: proves the assertion above can actually fail
    assert await session_model("gpt-realtime-9.9-nope") is None


def test_browserbase_context_is_pinnable():
    """The whole browser strategy rests on a saved login surviving sessions.

    The hosted Browserbase MCP cannot carry a contextId (local-server flag only),
    which is why we drive `browse` instead. This asserts the context still exists
    and is addressable; the login itself is populated by ops/browserbase_login.py.
    """
    import subprocess

    s = get_settings()
    assert s.browserbase_context_id, "run ops/bootstrap to create the context"
    out = subprocess.run(
        ["npx", "--yes", "browse@0.9.6", "cloud", "contexts", "get", s.browserbase_context_id],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "BROWSERBASE_API_KEY": s.browserbase_api_key.get_secret_value()},
    )
    assert s.browserbase_context_id in out.stdout, out.stderr[-400:]


def test_amd_uses_streaming_stt_not_the_6996_fallback():
    """With no DEEPGRAM_API_KEY, AMD falls back to the STT configuration that
    reproduces livekit/agents#6996 (~7% of answered calls stall 17-20s). This
    asserts we are NOT on that path — measured at 2.0s delay against a real
    Twilio Play call, vs the 9s outer belt.
    """
    from shot_voice.outbound import _amd_stt
    stt = _amd_stt()
    assert stt is not None, "DEEPGRAM_API_KEY unset; AMD is on the #6996 fallback"
    assert "deepgram" in type(stt).__module__
