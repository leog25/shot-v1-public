"""Provision the LiveKit SIP inbound trunk, outbound trunk, and dispatch rule.

Idempotent: each resource is looked up by name before creating. Safe to re-run.

    uv run python -m ops.bootstrap.livekit_sip [--dry-run]
"""

from __future__ import annotations

import asyncio
import json
import sys

from google.protobuf.duration_pb2 import Duration
from livekit import api

from ops.bootstrap.envfile import write_env
from shot_core.settings import get_settings

INBOUND_NAME = "shot-inbound"
OUTBOUND_NAME = "shot-outbound"
RULE_NAME = "shot-inbound-owner"
AGENT_NAME = "shot-voice"  # must equal @server.rtc_session(agent_name=...)
ROOM_PREFIX = "shot-call-"


async def _inbound(lk: api.LiveKitAPI, s) -> str:
    existing = await lk.sip.list_inbound_trunk(
        api.ListSIPInboundTrunkRequest(numbers=[s.twilio_phone_number]))
    mine = [t for t in existing.items if t.name == INBOUND_NAME]
    if len(existing.items) > len(mine):
        raise SystemExit(f"another inbound trunk claims {s.twilio_phone_number}; resolve by hand")
    if mine:
        print(f"  inbound   reuse {mine[0].sip_trunk_id}")
        return mine[0].sip_trunk_id

    trunk = api.SIPInboundTrunkInfo(
        name=INBOUND_NAME,
        metadata=json.dumps({"managed_by": "shot-bootstrap"}),
        numbers=[s.twilio_phone_number],
        # allowlist layer 1. There is no auth on a phone call otherwise.
        allowed_numbers=[s.owner_phone_number],
        # Krisp goes on the AGENT (telephony-tuned model); stacking two passes
        # on one stream degrades it. Trunk-side is the plain NC model only.
        krisp_enabled=False,
        # Twilio sends STIR/SHAKEN as X-Twilio-VerStat, NOT an Identity header.
        # Absent != failed: it's only present when the call carried a PASSporT,
        # and the framework is deployed only in the US and France.
        headers_to_attributes={"X-Twilio-VerStat": "caller.attestation"},
        include_headers=api.SIPHeaderOptions.SIP_NO_HEADERS,
        ringing_timeout=Duration(seconds=s.ringing_timeout_seconds),
        max_call_duration=Duration(seconds=1800),
        # NOTE: no auth_username/password. The credential-list pair authenticates
        # TERMINATION (outbound); Twilio sends no credentials on origination.
        # LiveKit only requires inbound auth when `numbers` is empty.
    )
    info = await lk.sip.create_inbound_trunk(
        api.CreateSIPInboundTrunkRequest(trunk=trunk))
    print(f"  inbound   created {info.sip_trunk_id}")
    return info.sip_trunk_id


async def _outbound(lk: api.LiveKitAPI, s) -> str:
    existing = await lk.sip.list_outbound_trunk(
        api.ListSIPOutboundTrunkRequest(numbers=[s.twilio_phone_number]))
    mine = [t for t in existing.items if t.name == OUTBOUND_NAME]
    if mine:
        print(f"  outbound  reuse {mine[0].sip_trunk_id}")
        return mine[0].sip_trunk_id

    # hostname only — the proto is explicit that this is NOT a SIP URI
    addr = s.twilio_trunk_termination_uri
    addr = addr.removeprefix("sips:").removeprefix("sip:").split(";")[0].split("@")[-1]
    assert "://" not in addr and "/" not in addr, addr

    trunk = api.SIPOutboundTrunkInfo(
        name=OUTBOUND_NAME,
        metadata=json.dumps({"managed_by": "shot-bootstrap"}),
        address=addr,
        # where the CALL TERMINATES (the callee), not where our number is from.
        destination_country="CA",
        # must match the ;transport=tcp on Twilio's origination URL.
        # NOT TLS: the Twilio trunk has secure=False.
        transport=api.SIPTransport.SIP_TRANSPORT_TCP,
        numbers=[s.twilio_phone_number],
        auth_username=s.twilio_trunk_auth_username,
        auth_password=s.twilio_trunk_auth_password.get_secret_value(),
    )
    info = await lk.sip.create_outbound_trunk(
        api.CreateSIPOutboundTrunkRequest(trunk=trunk))
    print(f"  outbound  created {info.sip_trunk_id}")
    return info.sip_trunk_id


async def _rule(lk: api.LiveKitAPI, s, inbound_id: str) -> str:
    existing = await lk.sip.list_dispatch_rule(
        api.ListSIPDispatchRuleRequest(trunk_ids=[inbound_id]))
    mine = [r for r in existing.items if r.name == RULE_NAME]
    if len(existing.items) > len(mine):
        raise SystemExit("a wildcard/duplicate dispatch rule would race ours; resolve by hand")
    if mine:
        print(f"  rule      reuse {mine[0].sip_dispatch_rule_id}")
        return mine[0].sip_dispatch_rule_id

    info = api.SIPDispatchRuleInfo(
        # Individual, not callee: the pre-warm-by-call-id pattern needs control of
        # the SIP To header, which Elastic SIP Trunking doesn't give us.
        rule=api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix=ROOM_PREFIX)),
        name=RULE_NAME,
        trunk_ids=[inbound_id],          # never leave a wildcard rule
        inbound_numbers=[s.owner_phone_number],   # allowlist layer 2
        numbers=[s.twilio_phone_number],
        hide_phone_number=False,         # agent reads sip.phoneNumber as layer 3
        krisp_enabled=False,             # second place it can be enabled; see inbound
        metadata=json.dumps({"managed_by": "shot-bootstrap"}),
        attributes={"caller.role": "owner"},
        # Without this, explicit dispatch means NO agent ever joins an inbound call.
        room_config=api.RoomConfiguration(
            agents=[api.RoomAgentDispatch(
                agent_name=AGENT_NAME,
                metadata=json.dumps({"tier": "voice"}),
            )]),
    )
    rule = await lk.sip.create_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(dispatch_rule=info))
    print(f"  rule      created {rule.sip_dispatch_rule_id}")
    return rule.sip_dispatch_rule_id


async def _verify(lk: api.LiveKitAPI, s, inb: str, outb: str, rule: str) -> None:
    t = (await lk.sip.list_inbound_trunk(
        api.ListSIPInboundTrunkRequest(trunk_ids=[inb]))).items[0]
    assert list(t.numbers) == [s.twilio_phone_number], t.numbers
    assert list(t.allowed_numbers) == [s.owner_phone_number], t.allowed_numbers
    assert t.krisp_enabled is False
    assert t.headers_to_attributes.get("X-Twilio-VerStat") == "caller.attestation"
    assert not t.auth_username, "inbound must not carry the termination credential"

    o = (await lk.sip.list_outbound_trunk(
        api.ListSIPOutboundTrunkRequest(trunk_ids=[outb]))).items[0]
    assert not o.address.startswith("sip"), o.address
    assert o.destination_country.upper() == "CA", o.destination_country
    assert o.transport == api.SIPTransport.SIP_TRANSPORT_TCP
    assert o.auth_username == s.twilio_trunk_auth_username

    r = (await lk.sip.list_dispatch_rule(
        api.ListSIPDispatchRuleRequest(dispatch_rule_ids=[rule]))).items[0]
    assert r.rule.WhichOneof("rule") == "dispatch_rule_individual"
    assert list(r.inbound_numbers) == [s.owner_phone_number]
    assert r.room_config.agents[0].agent_name == AGENT_NAME

    # the assertion that matters most: exactly one rule may match this trunk,
    # or a leftover wildcard silently competes for our calls.
    matching = await lk.sip.list_dispatch_rule(
        api.ListSIPDispatchRuleRequest(trunk_ids=[inb]))
    assert len(matching.items) == 1, f"{len(matching.items)} rules match the inbound trunk"
    print("  verified  all assertions passed")


async def main(dry_run: bool = False) -> None:
    s = get_settings()
    async with api.LiveKitAPI(s.livekit_url, s.livekit_api_key,
                              s.livekit_api_secret.get_secret_value()) as lk:
        inb = await _inbound(lk, s)
        outb = await _outbound(lk, s)
        rule = await _rule(lk, s, inb)
        await _verify(lk, s, inb, outb, rule)

    for line in write_env({
        "LIVEKIT_SIP_INBOUND_TRUNK_ID": inb,
        "LIVEKIT_SIP_OUTBOUND_TRUNK_ID": outb,
        "LIVEKIT_SIP_DISPATCH_RULE_ID": rule,
    }, dry_run=dry_run):
        print("  ", line)


if __name__ == "__main__":
    asyncio.run(main("--dry-run" in sys.argv))
