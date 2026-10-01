"""L0 — the clock, which three readers have to agree on.

`echo_time` held the only clock in the system with ZoneInfo("America/New_York")
hardcoded inside it. The moment the greeting also needed to know the time, the
two could only ever have agreed by coincidence — which is the failure budget.py
exists to prevent, one domain over. So there is one clock, and these pin it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from shot_core.clock import now_local, part_of_day, spoken_now

TZ = ZoneInfo("America/New_York")


def _at(hour: int) -> datetime:
    return datetime(2026, 9, 9, hour, 14, tzinfo=TZ)


@pytest.mark.parametrize("hour,expected", [
    (0, "night"), (4, "night"),
    (5, "morning"), (8, "morning"), (11, "morning"),
    (12, "afternoon"), (16, "afternoon"),
    (17, "evening"), (21, "evening"),
    (22, "night"), (23, "night"),
])
def test_the_part_of_day_boundaries_are_written_down(hour, expected):
    """Handing the model "8:14 AM" and hoping it picks the right greeting is the
    same gap that produced "It's done -- if you want, I can read the highlights
    next": a prompt asserting the model knows something never interpolated."""
    assert part_of_day(_at(hour)) == expected


def test_every_hour_of_the_day_has_a_word():
    assert {part_of_day(_at(h)) for h in range(24)} == {
        "morning", "afternoon", "evening", "night"}


def test_the_spoken_time_names_the_city_not_the_abbreviation():
    """%Z gives "EDT", an ID-shaped token the session prompt forbids reading
    aloud -- and the old hardcoded "Eastern" becomes a lie the moment the zone
    is configurable."""
    said = spoken_now(_at(8))
    assert "New York" in said
    assert "EDT" not in said and "Eastern" not in said
    assert "Wednesday" in said and "8:14 AM" in said


def test_the_spoken_time_reads_a_multiword_zone_as_words():
    said = spoken_now(datetime(2026, 9, 9, 8, 14, tzinfo=ZoneInfo("America/New_York")))
    assert "New York" in said and "New_York" not in said


def test_the_spoken_time_takes_its_zone_from_the_datetime_not_settings():
    """Pure, so it cannot disagree with the datetime it was handed."""
    assert "UTC" in spoken_now(datetime(2026, 9, 9, 8, 14, tzinfo=UTC))


def test_now_local_reads_the_setting(monkeypatch):
    class _S:
        owner_timezone = "Asia/Tokyo"

    monkeypatch.setattr("shot_core.settings.get_settings", lambda: _S())
    assert now_local().tzinfo == ZoneInfo("Asia/Tokyo")


def test_a_garbage_timezone_does_not_take_the_greeting_down(monkeypatch, caplog):
    """There is deliberately no validator on OWNER_TIMEZONE. A bad value can at
    worst say "good evening" in the morning; failing settings import kills
    shot-worker and takes the phone number down with it."""
    class _S:
        owner_timezone = "America/New_Yor"

    monkeypatch.setattr("shot_core.settings.get_settings", lambda: _S())
    now = now_local()
    assert now.tzinfo is UTC
    assert now.utcoffset() is not None, "must stay timezone-aware"
