"""L0 — the worker must not decline a real call because `apt` was busy.

The SDK's default load_fnc is a 2.5s moving average of MACHINE-WIDE cpu_percent
(`livekit/agents/utils/hw/cpu.py` -- psutil, or the ROOT cgroup's cpu.max, never
this service's own). At or above load_threshold the worker answers
`available = False` in `_answer_availability` and DECLINES the job: no queue, no
retry, and `ops.status` requires exactly one registered worker, so nobody else
takes it. Nothing is logged either -- no `job start`, no room, no `shot.calls`
row. It reads exactly like the phone number being broken.

Measured on the VM over eleven days: nine excursions over the threshold, 35s
total. Six were deploy prewarm and one was during a call, both harmless. Two
were `apt-daily-upgrade` on 2026-09-09, which burned 54s of CPU on 2 vCPU and
made the number unreachable for 12.5s at 02:26 local -- while completely idle,
with no call and no restart anywhere near it. The other ten daily apt runs cost
2-7s of CPU and crossed nothing, which is why this is a hole you find once and
can never reproduce on demand.
"""

from __future__ import annotations

import inspect

import pytest

from shot_voice import worker
from shot_voice.worker import MAX_CONCURRENT_CALLS, _call_load

# livekit/agents/worker.py:148 -- ServerEnvOption(dev_default=inf, prod_default=0.7).
# Hard-coded here on purpose: if the SDK moves it, this file should go red and be
# read, not silently follow.
PROD_THRESHOLD = 0.7


class _Server:
    """Only the one attribute _call_load reads."""

    def __init__(self, jobs: int) -> None:
        self.active_jobs = [object()] * jobs


def test_an_idle_worker_is_available_whatever_the_machine_is_doing():
    """THE regression guard. Nothing that is not a call may move this number, so
    a nightly apt upgrade cannot take the phone number down for twelve seconds
    ever again."""
    assert _call_load(_Server(0)) == 0.0


def test_the_first_call_is_never_declined():
    """_get_effective_load adds `_reserved_slots * job_load_estimate`, and with
    no active jobs that estimate is `load_threshold / num_idle_processes`. If
    that sum reached the threshold, the very first call of the day would be
    refused -- so the arithmetic is pinned, not assumed."""
    num_idle = 2  # AgentServer(num_idle_processes=2) in worker.py
    reserved_estimate = PROD_THRESHOLD / num_idle
    assert _call_load(_Server(0)) + reserved_estimate < PROD_THRESHOLD


@pytest.mark.parametrize("jobs", range(MAX_CONCURRENT_CALLS))
def test_it_accepts_up_to_the_cap(jobs):
    assert _call_load(_Server(jobs)) < PROD_THRESHOLD


def test_it_declines_past_the_cap():
    """An unbounded worker on 2 vCPU is how you get 5-20s of dead air on every
    call. The cap is a safety net, not a limit we expect to brush."""
    assert _call_load(_Server(MAX_CONCURRENT_CALLS)) >= PROD_THRESHOLD
    assert _call_load(_Server(MAX_CONCURRENT_CALLS + 5)) >= PROD_THRESHOLD


def test_the_cap_leaves_room_for_a_callback_during_an_inbound_call():
    """The owner is one person, so the real number is one -- but the supervisor can
    dial him while he is already on a call, and that must not be refused."""
    assert MAX_CONCURRENT_CALLS >= 2
    assert _call_load(_Server(2)) < PROD_THRESHOLD


def test_the_server_actually_uses_it():
    """Passing load_fnc is the entire fix; dropping the kwarg silently restores
    the machine-CPU default and the failure comes back invisible."""
    from shot_voice.worker import server

    assert server.load_fnc is _call_load


def test_the_worker_builds_no_linear_mcp_toolset():
    """Pinned absent, because nothing could see it go.

    The four wiring tests that used to guard this built their own fake Toolset
    and asserted ShotAgent/CallbackAgent forwarded it -- they never touched
    worker.py, so deleting `_linear_toolset()` and both call sites passed the
    entire suite green. The thing worth asserting was always the call site.

    It is gone on purpose: 65 tools and 78KB of schema, re-charged against the
    account's 40,000 TPM realtime ceiling on EVERY response, cost 14,923 of it
    and left room for two responses a minute. A tool-using turn needs two, so
    the one that speaks the result came back `rate_limit_exceeded` and the owner got
    silence. tools.py carries three hand-written Linear tools instead.
    """
    assert not hasattr(worker, "_linear_toolset")
    src = inspect.getsource(worker)
    assert "MCPToolset" not in src
    assert "linear_toolset" not in src
