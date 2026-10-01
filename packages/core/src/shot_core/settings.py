"""The only reader of .env.

Deliberately dependency-light: the voice worker forks a subprocess per job and
pays this module's import cost against `initialize_process_timeout` (10s default),
so nothing heavy may be imported at module scope here.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

E164 = re.compile(r"^\+[1-9]\d{7,14}$")

# Provisioned by ops/bootstrap/*. Blank in .env until then; `missing_resource_ids()`
# reports them and ops/bootstrap/verify.py refuses to start the supervisor without.
_RESOURCE_KEYS = (
    "livekit_sip_inbound_trunk_id",
    "livekit_sip_outbound_trunk_id",
    "livekit_sip_dispatch_rule_id",
    "anthropic_agent_id",
    "anthropic_environment_id",
    "anthropic_memory_store_id",
    "anthropic_vault_id",
    "anthropic_browser_helper_file_id",
    "browserbase_context_id",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    # ── GCP ───────────────────────────────────────────────────────────────
    gcp_project_id: str
    # No defaults that name a region: nothing in Python reads these, and ops
    # scripts take theirs from .env and refuse when they are unset.
    gcp_region: str = ""
    gcp_zone: str = ""
    cloud_tasks_queue: str = "callbacks"
    cloud_tasks_location: str = ""

    # ── supervisor ────────────────────────────────────────────────────────
    database_url: str
    public_base_url: str = ""
    supervisor_base_url: str = "http://127.0.0.1:8090"

    # ── voice ─────────────────────────────────────────────────────────────
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: SecretStr
    livekit_sip_host: str = ""
    openai_api_key: SecretStr
    # Passed through as a bare string; the plugin has no allowlist to validate it
    # against, so a typo first surfaces on a ringing phone. See tests/test_model_string.
    openai_realtime_model: str = "gpt-realtime-2.1"
    # `low` is the staff-confirmed default. NOT `minimal`: it saves 240ms and costs
    # 7.7pts of agentic performance on the metric that is literally this agent's job.
    openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    deepgram_api_key: SecretStr = SecretStr("")  # AMD's streaming STT; see R1

    # ── telephony ─────────────────────────────────────────────────────────
    twilio_account_sid: str
    twilio_api_key_sid: str
    twilio_api_key_secret: SecretStr
    twilio_trunk_sid: str = ""
    twilio_trunk_termination_uri: str = ""
    twilio_trunk_auth_username: str = ""
    twilio_trunk_auth_password: SecretStr = SecretStr("")
    twilio_phone_number: str
    owner_phone_number: str
    # Who Shot works for, by name -- the persona, the greeting, the worker's
    # system prompt. From .env so no tracked file names him. NO validator, for
    # the reason owner_timezone has none: a missing name is not worth failing
    # settings import for, so it reads as "the owner" and the call still works.
    # deploy.sh refuses a fetched .env without it, and ops.status warns.
    owner_name: str = "the owner"
    # Read by BOTH the time-of-day greeting and echo_time, through
    # shot_core.clock -- it was a hardcoded ZoneInfo(...) inside echo_time,
    # so the greeting and the tool could only ever have agreed by
    # coincidence. Deliberately has NO validator: a bad zone falls back to UTC
    # with a warning in now_local(), because a wrong greeting is not worth
    # failing settings import for, and a failed import takes the phone number
    # down. A bad owner_phone_number could dial a stranger; this cannot.
    # UTC, not a city: the default must not say where he lives. deploy.sh
    # refuses a fetched .env without OWNER_TIMEZONE, so this is never live.
    owner_timezone: str = "UTC"
    # Cap is 80s, but a US/CA mobile rolls to voicemail at ~20-25s and the Python SDK
    # silently pins 30s when wait_until_answered=True unless we set this explicitly.
    ringing_timeout_seconds: int = Field(default=20, ge=5, le=80)
    # Answering-machine detection. These MIRROR shot_core.budget.AMD_* -- that
    # table is the documented source of truth, production reads THESE, and
    # test_budget.py asserts the two agree so they cannot drift. outbound.py
    # hardcoded 8.0/6.0/9.0 for months while budget.py claimed to own them.
    # Env-overridable so tuning the no-speech cutoff is a config change rather
    # than a redeploy; ge=1.0 admits going lower once M3 says it is safe.
    amd_detection_timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)
    amd_no_speech_seconds: float = Field(default=3.0, ge=1.0, le=30.0)
    amd_wait_seconds: float = Field(default=9.0, ge=1.0, le=40.0)

    # ── worker agent ──────────────────────────────────────────────────────
    anthropic_api_key: SecretStr
    anthropic_webhook_signing_key: SecretStr = SecretStr("")
    anthropic_webhook_id: str = ""
    session_budget_cents: int = Field(default=300, gt=0)
    min_balance_cents: int = Field(default=100, ge=0)

    # ── browser / tools ───────────────────────────────────────────────────
    browserbase_api_key: SecretStr = SecretStr("")
    browserbase_project_id: str = ""
    brave_api_key: SecretStr = SecretStr("")
    # Separate keys per tier on purpose: independent revocation, and Linear's
    # own activity log can tell which tier made a given change. voice is read
    # directly by shot_voice.worker as an HTTP bearer header; worker is never
    # read at runtime -- ops.bootstrap.anthropic_res pushes it into the
    # Anthropic vault once, as a static_bearer credential, and the platform
    # injects it from there.
    linear_api_key_voice: SecretStr = SecretStr("")
    linear_api_key_worker: SecretStr = SecretStr("")

    # ── policy ────────────────────────────────────────────────────────────
    pin_policy: Literal["off", "callbacks_only", "always"] = "callbacks_only"

    def pin_required(self, *, direction: str) -> bool:
        """Whether this call must pass the PIN.

        An OUTBOUND call always does, under every policy. Caller ID proves
        nothing about who picked up: we dialled a number and *something*
        answered, and there is no configuration under which reading the owner's
        results to that something is acceptable. This matters more now than it
        used to -- the AMD gate no longer withholds the greeting, so the PIN is
        the only thing standing between a stranger and the brief.

        PIN_POLICY therefore governs INBOUND only, where the number allowlist is
        already the identity check. That makes `off` and `callbacks_only`
        identical today; `off` is kept a legal value so a stale .env cannot fail
        validation at import and take the phone number down with it.

        Lives here because BOTH tiers need it and neither may import the other:
        the supervisor decides when dispatching a callback, the voice job obeys.
        It previously lived in shot_voice.pin with no production caller at all,
        while the supervisor hardcoded `require_pin: True` -- so PIN_POLICY was
        a knob that did nothing. Do not re-split it.

        Note this is only the *identity* check. The AMD gate is separate, is not
        switchable, and still blocks a detected machine from the results.
        """
        if direction == "outbound":
            return True
        return self.pin_policy == "always"
    @model_validator(mode="after")
    def _amd_windows_nest(self) -> Settings:
        """no_speech < detect < wait, checked at import.

        A belt shorter than the thing it belts is just a timeout that fires
        first, and a ringing phone is the wrong place to discover it. The
        equivalent constants in budget.py have had a test asserting this since
        they were written; the settings that production actually reads had
        nothing until they became overridable.
        """
        if not (self.amd_no_speech_seconds
                < self.amd_detection_timeout_seconds
                < self.amd_wait_seconds):
            raise ValueError(
                f"AMD windows must nest: no_speech({self.amd_no_speech_seconds})"
                f" < detect({self.amd_detection_timeout_seconds})"
                f" < wait({self.amd_wait_seconds})")
        return self

    callback_pin: SecretStr = SecretStr("")

    # ── provisioned resource ids (blank until ops/bootstrap runs) ─────────
    livekit_sip_inbound_trunk_id: str = ""
    livekit_sip_outbound_trunk_id: str = ""
    livekit_sip_dispatch_rule_id: str = ""
    anthropic_agent_id: str = ""
    anthropic_environment_id: str = ""
    anthropic_memory_store_id: str = ""
    anthropic_vault_id: str = ""
    # Mounted into every worker session; see ops/bootstrap/sandbox/bb_connect.sh.tmpl
    anthropic_browser_helper_file_id: str = ""
    browserbase_context_id: str = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """.env outranks the process environment.

        Inverted from the pydantic-settings default on purpose. Claude Code
        exports ANTHROPIC_API_KEY=dev-key... into every shell it spawns, which
        silently outranked the real key and produced a 401 that looked like a
        bad credential. Ambient env vars from tooling are a bigger hazard here
        than a stale .env. Deployment is unaffected: with no .env present the
        dotenv source contributes nothing and env vars are used as normal.
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    @field_validator("twilio_phone_number", "owner_phone_number")
    @classmethod
    def _e164(cls, v: str) -> str:
        if not E164.match(v):
            raise ValueError(f"must be E.164 (e.g. +15555550100), got {v!r}")
        return v

    @field_validator("livekit_url")
    @classmethod
    def _ws(cls, v: str) -> str:
        if not v.startswith(("ws://", "wss://")):
            raise ValueError(f"must be a ws:// or wss:// URL, got {v!r}")
        return v

    def missing_resource_ids(self) -> list[str]:
        return [k.upper() for k in _RESOURCE_KEYS if not getattr(self, k)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
