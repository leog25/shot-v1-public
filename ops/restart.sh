#!/usr/bin/env bash
# Restart both tiers. Kills by PATTERN, not by pid file.
#
# A pid file is not enough: a worker started at 17:05 survived every restart
# for an hour and forty minutes because the pid file had been overwritten by a
# later launch. LiveKit load-balances jobs across every REGISTERED worker, so
# that orphan kept taking real calls and answering them with pre-fix code --
# which made live behaviour a coin flip between old and new logic and looked
# like the fixes had not worked.
set -euo pipefail
cd "$(dirname "$0")/.."

# The system runs on the VM now. Starting a worker here as well would register
# a SECOND worker with LiveKit, which load-balances jobs across every
# registered worker -- so calls would land on the laptop or the VM at random,
# and the laptop's would answer with whatever code happened to be checked out.
# That exact split (one stale worker, one fresh) once made live behaviour a
# coin flip for an hour and forty minutes.
if [ "${SHOT_LOCAL:-}" != "1" ]; then
    # From .env, like deploy.sh. Unset must REFUSE: an ssh that cannot run
    # reads as "not serving", which is the one answer that starts a second worker.
    env_get() { { grep -E "^$1=" .env || true; } | tail -1 | cut -d= -f2-; }
    PROJECT=$(env_get GCP_PROJECT_ID); ZONE=$(env_get GCP_ZONE); VM=$(env_get GCP_VM)
    : "${PROJECT:?GCP_PROJECT_ID is not set in .env; cannot check the VM}"
    : "${ZONE:?GCP_ZONE is not set in .env; cannot check the VM}"
    : "${VM:?GCP_VM is not set in .env; cannot check the VM}"
    if gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet \
         --tunnel-through-iap --command "systemctl is-active shot-worker" 2>/dev/null \
         | grep -q active; then
        echo "REFUSING: the VM is serving. Deploy with ops/deploy.sh instead." >&2
        echo "  (SHOT_LOCAL=1 ops/restart.sh to override for offline dev)" >&2
        exit 1
    fi
fi

pkill -f "uvicorn shot_supervisor" 2>/dev/null || true
pkill -f "shot_voice.worker"       2>/dev/null || true
sleep 3

for pattern in "uvicorn shot_supervisor" "shot_voice.worker"; do
    if pgrep -f "$pattern" >/dev/null; then
        echo "REFUSING TO START: '$pattern' survived SIGTERM" >&2
        pgrep -fl "$pattern" >&2
        exit 1
    fi
done

set -a && . ./.env && set +a

# Without this Python block-buffers stdout into a non-tty, so a fresh log shows
# nothing at all until ~8KB accumulates -- a whole call can happen and
# `tail -f /tmp/shot_worker.log` stays empty, which reads exactly like the
# worker never got the job.
export PYTHONUNBUFFERED=1

nohup uv run uvicorn shot_supervisor.app:app --host 127.0.0.1 --port 8090 \
      > /tmp/shot_sup.log 2>&1 &
nohup uv run python -m shot_voice.worker dev > /tmp/shot_worker.log 2>&1 &

for _ in $(seq 1 40); do curl -sf localhost:8090/healthz >/dev/null 2>&1 && break; sleep 1; done
for _ in $(seq 1 60); do grep -q "registered worker" /tmp/shot_worker.log && break; sleep 1; done

# `uv run` spawns a child that matches the same pattern, so count the roots:
# processes whose parent is not itself a worker.
n=$(pgrep -f "shot_voice.worker dev" | wc -l | tr -d ' ')
echo "supervisor: $(curl -sf localhost:8090/healthz >/dev/null && echo up || echo DOWN)"
echo "workers:    $(grep -c "registered worker" /tmp/shot_worker.log) registration(s) in this log, $n matching processes"
