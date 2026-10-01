"""What time it is for the owner, in one place.

Three call sites need this and they must agree: the greeting says "good
morning", `echo_time` answers "what time is it", and `check_my_day` decides
whether a ticket is due today. `echo_time` used to hold the only clock in the
system, with `ZoneInfo("America/New_York")` hardcoded inside it -- so the moment
a second reader existed the two could only ever have agreed by coincidence.
That is the failure `budget.py` was written to prevent, one domain over.

It lives in `shot_core` rather than beside its readers because `shot_voice.tools`
needs it and `shot_voice.agent` already imports `tools` -- a helper in `agent.py`
would be an import cycle.

Standard library only. The voice worker forks a subprocess per job and pays this
module's import cost against a 10s `initialize_process_timeout`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("shot.clock")

# Boundaries written down rather than left to the model. Handing it "8:14 AM"
# and hoping it picks the right greeting is the same gap that produced "It's
# done -- if you want, I can read the highlights next": a prompt asserting the
# model knows something that was never interpolated.
_MORNING = range(5, 12)     # 05:00-11:59
_AFTERNOON = range(12, 17)  # 12:00-16:59
_EVENING = range(17, 22)    # 17:00-21:59
                            # everything else is night


def now_local() -> datetime:
    """Now, in the owner's timezone. NEVER raises.

    A bad OWNER_TIMEZONE falls back to UTC with a warning rather than throwing,
    and there is deliberately no validator on the settings field: a bad value
    here can at worst say "good evening" in the morning, while failing settings
    import kills `shot-worker` and takes the phone number down with it. Same
    reasoning as PIN_POLICY keeping `off` a legal value.
    """
    from shot_core.settings import get_settings

    name = get_settings().owner_timezone
    try:
        return datetime.now(ZoneInfo(name))
    except (ZoneInfoNotFoundError, ValueError) as e:
        log.warning("OWNER_TIMEZONE %r is not a usable zone (%s); using UTC",
                    name, e)
        return datetime.now(UTC)


def part_of_day(dt: datetime) -> str:
    """morning | afternoon | evening | night. Pure, so it tests without a clock."""
    hour = dt.hour
    if hour in _MORNING:
        return "morning"
    if hour in _AFTERNOON:
        return "afternoon"
    if hour in _EVENING:
        return "evening"
    return "night"


def spoken_now(dt: datetime) -> str:
    """The date and time as a phone can read it out.

    Names the CITY, not the abbreviation. `%Z` gives "EDT", which is an
    ID-shaped token the session prompt forbids reading aloud -- and the old
    hardcoded "Eastern" becomes a lie the moment the zone is configurable.

    Pure: the zone comes off `dt`, not off settings, so this cannot disagree
    with the datetime it was handed.
    """
    tz = dt.tzinfo
    where = tz.key.split("/")[-1].replace("_", " ") if isinstance(tz, ZoneInfo) else "UTC"
    return f"{dt:%A %-d %B %Y, %-I:%M %p} in {where}"
