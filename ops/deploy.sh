#!/usr/bin/env bash
# Push local code to the VM and restart both tiers.
#
# The WORKING TREE is the artifact -- this does not pull from git, so a commit
# is not a deploy and a deploy is not a commit. `.env` is NOT
# shipped -- the VM pulls it from Secret Manager, so a credential change is
# `gcloud secrets versions add "$GCP_ENV_SECRET" --data-file=.env` plus a restart.
set -euo pipefail
cd "$(dirname "$0")/.."

# Project, zone, VM and secret come from .env, so no tracked file names the
# GCP account or what is in it. Fail closed: an empty --project falls back to
# whichever project `gcloud config set` last chose, and a second project
# shares that config.
env_get() { { grep -E "^$1=" .env || true; } | tail -1 | cut -d= -f2-; }
PROJECT=$(env_get GCP_PROJECT_ID)
ZONE=$(env_get GCP_ZONE)
VM=$(env_get GCP_VM)
SECRET=$(env_get GCP_ENV_SECRET)
: "${PROJECT:?GCP_PROJECT_ID is not set in .env}" "${ZONE:?GCP_ZONE is not set in .env}"
: "${VM:?GCP_VM is not set in .env}" "${SECRET:?GCP_ENV_SECRET is not set in .env}"

tar --exclude=.venv --exclude=.git --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='*.pyc' --exclude='.env' -czf /tmp/shot-code.tar.gz .
gcloud compute scp /tmp/shot-code.tar.gz "$VM:/tmp/shot-code.tar.gz" \
  --zone "$ZONE" --project "$PROJECT" --quiet --tunnel-through-iap

gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --tunnel-through-iap --command '
set -e
sudo tar -xzf /tmp/shot-code.tar.gz -C /opt/shot
sudo chown -R shot:shot /opt/shot
sudo -u shot bash -lc "cd /opt/shot && export PATH=\$HOME/.local/bin:\$PATH && uv sync --frozen -q"
# Pull .env from Secret Manager. Nothing did this, so `gcloud secrets versions
# add` + deploy -- the documented rotate -- left the VM on whatever .env it was
# built with. A new key reads as an empty string via settings.py and the feature
# it gates silently does not exist, which is indistinguishable from a code bug.
# Written via a temp file so a failed fetch cannot truncate a working .env.
sudo -u shot gcloud secrets versions access latest --secret '"$SECRET"' \
  --project '"$PROJECT"' > /tmp/dotenv.new
grep -q "^DATABASE_URL=" /tmp/dotenv.new || { echo "refused: fetched .env looks wrong"; exit 1; }
# These two have code defaults ("the owner", UTC) so a missing one cannot fail
# settings import -- which means nothing else would notice it was missing.
# The agent would greet him in the wrong timezone and stop using his name.
for key in OWNER_NAME OWNER_TIMEZONE; do
  grep -q "^$key=." /tmp/dotenv.new || {
    echo "refused: $key is not in Secret Manager; add it to .env and push the secret first"; exit 1; }
done
sudo install -m 600 -o shot -g shot /tmp/dotenv.new /opt/shot/.env
rm -f /tmp/dotenv.new
echo ".env refreshed from Secret Manager"
# Schema BEFORE the restart. It is idempotent (CREATE ... IF NOT EXISTS, ALTER
# ... ADD COLUMN IF NOT EXISTS), and skipping it ships code that writes columns
# the database does not have -- which fails inside the try/except in
# record_call, so the call is simply never recorded and nothing says why.
sudo -u shot bash -lc "cd /opt/shot && export PATH=\$HOME/.local/bin:\$PATH && uv run python -m ops.bootstrap.db"
# supervisor first: the worker talks to it on localhost
sudo systemctl restart shot-supervisor
sleep 8
sudo systemctl restart shot-worker
sleep 15
echo "supervisor: $(systemctl is-active shot-supervisor)   worker: $(systemctl is-active shot-worker)"
'
