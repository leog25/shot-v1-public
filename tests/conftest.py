from __future__ import annotations

import functools
import os

import pytest
import sqlalchemy as sa
from pydantic import SecretStr

import shot_core.settings as settings_mod
from shot_core.db import apply_schema, make_engine

from support import TEST_NAME, TEST_OWNER, TEST_PIN

_live_settings = settings_mod.get_settings


@functools.lru_cache(maxsize=1)
def _test_settings():
    """The live settings with TEST_PIN and TEST_NAME in place of the real ones.

    Installed HERE, at import, not in a fixture: prompts.py interpolates the
    owner's name when it is first imported, which is during collection, before
    any fixture runs. `pin.verify` and `CallbackAgent`'s bare `Identity()` read
    the PIN through it too -- without that the callback tests could only pass
    by keying the real PIN, which is how it got written into them.
    """
    return _live_settings().model_copy(update={
        "callback_pin": SecretStr(TEST_PIN), "owner_name": TEST_NAME})


settings_mod.get_settings = _test_settings

TEST_DB = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:shot@localhost:5433/shot_test")


@pytest.fixture(scope="session")
def engine():
    eng = make_engine(TEST_DB)
    try:
        with eng.connect() as cx:
            cx.execute(sa.text("select 1"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no test Postgres at {TEST_DB}: {type(e).__name__}")
    apply_schema(eng)
    return eng


@pytest.fixture
def db(engine):
    """Clean slate per test; owners row pre-seeded."""
    with engine.begin() as cx:
        cx.execute(sa.text(
            "TRUNCATE shot.callbacks, shot.calls, shot.tasks, shot.summaries, "
            "shot.summary_history, shot.webhook_events, shot.owners CASCADE"))
        cx.execute(sa.text("INSERT INTO shot.owners (phone) VALUES (:p)"), {"p": TEST_OWNER})
    return engine


@pytest.fixture(scope="session", autouse=True)
def _env_from_settings():
    """Mirror resolved settings into os.environ.

    Several vendor plugins read credentials straight from os.environ
    (inference.TurnDetector wants LIVEKIT_API_KEY, RealtimeModel wants
    OPENAI_API_KEY) and never see our .env. Worse, Claude Code exports a
    placeholder ANTHROPIC_API_KEY into every shell, which is exactly why
    settings rank .env above env in the first place -- so mirror in that
    direction too, and let the resolved value win here as well.
    """
    from shot_core.settings import get_settings

    s = get_settings()
    for key, val in {
        "LIVEKIT_URL": s.livekit_url,
        "LIVEKIT_API_KEY": s.livekit_api_key,
        "LIVEKIT_API_SECRET": s.livekit_api_secret.get_secret_value(),
        "OPENAI_API_KEY": s.openai_api_key.get_secret_value(),
        "DEEPGRAM_API_KEY": s.deepgram_api_key.get_secret_value(),
        "ANTHROPIC_API_KEY": s.anthropic_api_key.get_secret_value(),
    }.items():
        if val:
            os.environ[key] = val
