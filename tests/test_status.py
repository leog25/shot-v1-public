"""L0 -- `ops.status` reads the worker's log correctly. No network, no systemd.

This exists because the check it covers was wrong in production and read as an
outage. `ops.status` decided "is the worker registered with LiveKit" by looking
for `registered worker` in the last 200 journald lines -- a presence test over
a fixed line window. Two callbacks write about 250 lines, so on 2026-09-10 the
registration line aged out and the command reported `running but not
registered` about a worker that had accepted five jobs inside that same window.

The worker really does drop and re-register without restarting -- six times in
the eleven days to 2026-09-22 -- so both markers appear in any long-lived log
and only their ORDER carries the answer.
"""

from __future__ import annotations

from ops.status import _worker_registered

REG = ('{"message": "registered worker", "level": "INFO", '
       '"name": "livekit.agents", "agent_name": "shot-voice"}')
LOST = ('{"message": "failed to connect to livekit, retrying in 0s", '
        '"level": "WARNING", "name": "livekit.agents"}')
NOISE = '{"message": "turn assistant: hello", "name": "shot.transcript"}'


def test_a_fresh_registration_reads_as_registered():
    assert _worker_registered(REG) is True


def test_registration_survives_a_busy_call_that_would_overflow_a_line_window():
    """THE regression. 250 lines of call traffic after the registration is
    exactly what made the old `-n 200` check report a false outage."""
    log = "\n".join([REG] + [NOISE] * 250)
    assert _worker_registered(log) is True


def test_a_drop_after_registration_reads_as_lost():
    """The other half: presence of `registered worker` must not outvote a
    later failure, or a real outage reads as healthy."""
    assert _worker_registered("\n".join([REG, LOST])) is False


def test_a_reconnect_reads_as_registered_again():
    """Drop then recover, which is the common case -- the SDK logs the failure
    first and the success in the same second."""
    assert _worker_registered("\n".join([REG, LOST, REG])) is True


def test_six_drops_and_six_recoveries_read_as_registered():
    """Eleven days of real log shape."""
    assert _worker_registered("\n".join([REG] + [LOST, NOISE, REG] * 6)) is True


def test_a_log_with_neither_marker_is_unknown_not_failed():
    """`None` is a third answer on purpose. A rotated journal says nothing
    either way, and reporting that as an outage is the original mistake."""
    assert _worker_registered("\n".join([NOISE] * 20)) is None
    assert _worker_registered("") is None


def test_a_retry_storm_that_never_recovers_reads_as_lost():
    assert _worker_registered("\n".join([REG] + [LOST] * 16)) is False
