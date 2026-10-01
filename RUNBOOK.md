# shot — runbook

## Deployed on a GCE VM (`GCP_VM`, `GCP_ZONE`, `GCP_PROJECT_ID` in .env)
```bash
./ops/deploy.sh                                   # push code + restart both tiers
set -a && . ./.env && set +a      # the VM, zone and project live in .env
gcloud compute ssh "${GCP_VM:?}" --zone "${GCP_ZONE:?}" --project "${GCP_PROJECT_ID:?}" --tunnel-through-iap
sudo systemctl status shot-worker shot-supervisor
sudo journalctl -u shot-worker -f
```

## Local (offline dev only — the VM is authoritative)
```bash
docker start shot-pg                              # Postgres 17 on :5433
uv run python -m ops.bootstrap.db                 # schema + owner (idempotent)
uv run python -m ops.bootstrap.livekit_sip        # SIP trunks + dispatch rule (idempotent)
uv run python -m ops.bootstrap.anthropic_res      # env, vault, memory, agent (idempotent)

./ops/restart.sh                                  # both tiers, kills by pattern
```

**Always restart with `ops/restart.sh`, never by pid file.** A worker started at
17:05 outlived every restart for an hour and forty minutes because a later
launch overwrote the pid file, and LiveKit load-balances jobs across every
*registered* worker — so that orphan kept answering real calls with pre-fix
code. `ops.status` now fails if the worker count is not 1.

## Tests
The DB-backed tests need a local Postgres, which is stopped now that the VM is
authoritative — `docker start shot-pg` first, or **37** of them skip silently
(all of `test_registry.py` and `test_context.py`). Without it the run reports
357 passed / 37 skipped, which reads green.
```bash
docker start shot-pg
uv run pytest -m "not contract"     # 394, offline, ~8s, no network at all
uv run --with websockets pytest     # 419 total, ~4 min, cents
```

The callback path is fully covered offline, including the real `Identity`
collector and real DTMF — an `AgentSession` needs no room. Anthropic spend in the suite is
confined to `test_delegate_loop.py`.

## Stopping something he did not want
```bash
# On the VM -- /internal/* is localhost-only there.
curl -sX POST localhost:8090/internal/tasks/portland-tomorrow/cancel
```
Cancels the task and any callback that has not dialled yet, and caps the worker
session at one cent. It cannot recall a call already ringing; the `spoken` field
in the response says which of those happened. `ops.status` will still show the
session as `running` until the ceiling bites — that is honest, not a bug.

## What happened on the last calls
```bash
uv run python -m ops.calls          # reads shot.calls; the log is not evidence
uv run python -m ops.transcript     # what was SAID on the last call
uv run python -m ops.transcript 3   # the last three
uv run python -m ops.transcript cb-8e94a9e8   # one room, by prefix
uv run python -m ops.status         # every tier; fails if 2 workers are up
```

**`voice worker: running but LOST its LiveKit connection`** is usually not an
outage to act on. The worker drops and re-registers in place several times a
week without restarting, normally inside the same second; the longest seen is
95s. Re-run `ops.status` before touching anything, and only restart if it stays
LOST. `running; the log says nothing either way` is a rotated journal, not a
fault. (This check used to read the last 200 journald lines, which a single
busy call overflows -- it reported a false outage on 2026-09-10 about a worker
that was taking jobs at the time.)

## Verified end to end
- inbound: agent dispatched into a real room, publishes audio 0.5s after a caller joins
- outbound: dialled Twilio Play (+16504894546) through the trunk; ringing -> active in 8s;
  AMD transcribed real carrier audio and classified `machine-ivr`; disclosure withheld
- delegate: task row -> Managed Agents session (0.92s) -> `succeeded` -> spoken summary
- worker agent drove a real Browserbase browser and wrote to its memory store
- AMD on Deepgram nova-3 streaming: 2.0s delay on a real call (9s belt), full
  transcript captured -- NOT the livekit/agents#6996 fallback path

- PIN over DTMF or speech, 3 attempts then goodbye (offline-covered; the
  real-handset leg is MANUAL_E2E M2)
- durable callbacks: a 90s timer survived `kill -9` + a 20s outage and still fired
- autonomous dial: DBOS sweep claimed, dialled Play, classified, reported, retried
- cross-call context assembled from DB state (230 chars, cap 1200)

## Known gaps
- `AMD_NO_SPEECH_S` is 3.0, down from 6.0. It only affects lines silent for that
  long, and every box on record classified on 3.7-5.7s of speech -- but MANUAL_E2E
  M3 against real carrier voicemail is the evidence, and it has not been run yet.
- `PinResult.ABANDONED` now has a producer (`participant_disconnected`), so the
  `no_input` count in `shot.calls.pin_result` should stop over-reporting. Not yet
  observed on a real call.
- The worker's final `agent.message` is all that is captured as a task result.
  A worker that signs off with "done" stores exactly that -- `api-smoke`'s whole
  result is the word `done`.
- Webhook receiver still the Cloud Run placeholder. The reconcile sweep (every 2 min)
  covers correctness; wiring the webhook only cuts callback latency to ~5s.
- `ops/bootstrap/verify.py` preflight not written.

See MANUAL_E2E.md for the five steps automation cannot reach.
