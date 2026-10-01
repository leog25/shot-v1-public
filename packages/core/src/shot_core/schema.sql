-- Schema `shot`. DBOS owns schema `dbos` and migrates it itself; never touch it here.
CREATE SCHEMA IF NOT EXISTS shot;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS shot.owners (
  phone       text PRIMARY KEY,                       -- E.164
  pin_hash    text,                                   -- scrypt; NULL disables the PIN
  created_at  timestamptz NOT NULL DEFAULT now()
);

DO $$ BEGIN
  CREATE TYPE shot.task_state AS ENUM (
    'queued','starting','running','waiting_on_user','idle',
    'succeeded','failed','cancelled','budget_reached');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE shot.callback_reason AS ENUM (
    'task_done','task_needs_input','user_requested','budget_reached','task_failed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS shot.tasks (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_phone      text NOT NULL REFERENCES shot.owners(phone),
  ref              text NOT NULL,                     -- speakable slug: "lakers-game"
  title            text NOT NULL,
  goal             text NOT NULL,
  state            shot.task_state NOT NULL DEFAULT 'queued',
  stop_reason      text,                              -- from the latest session.status_idle
  session_id       text UNIQUE,                       -- sesn_... UNIQUE => create is idempotent
  workflow_id      text UNIQUE,
  dedupe_key       text UNIQUE,
  result_summary   text,                              -- <=400 chars, written to be SPOKEN
  error            jsonb,
  list_cost_cents  integer NOT NULL DEFAULT 0,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  finished_at      timestamptz,
  notified_at      timestamptz,
  UNIQUE (owner_phone, ref)
);

-- THE phone->work index. Partial, so it stays tiny forever.
CREATE INDEX IF NOT EXISTS tasks_owner_open_idx ON shot.tasks (owner_phone, updated_at DESC)
  WHERE state IN ('queued','starting','running','waiting_on_user','idle');
CREATE INDEX IF NOT EXISTS tasks_owner_unreported_idx ON shot.tasks (owner_phone, finished_at)
  WHERE finished_at IS NOT NULL AND notified_at IS NULL;
-- There is NO metadata filter on GET /v1/sessions, so this index is load-bearing,
-- not a convenience: it is the only way back from a session id to a task.
CREATE INDEX IF NOT EXISTS tasks_session_idx ON shot.tasks (session_id)
  WHERE session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS shot.callbacks (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_phone   text NOT NULL REFERENCES shot.owners(phone),
  task_id       uuid REFERENCES shot.tasks(id) ON DELETE CASCADE,
  reason        shot.callback_reason NOT NULL,
  brief         text NOT NULL,                        -- opening line for the callback
  due_at        timestamptz NOT NULL,
  fired_at      timestamptz,                          -- <- the CAS column
  attempt       smallint NOT NULL DEFAULT 0,
  max_attempts  smallint NOT NULL DEFAULT 3,
  outcome       text,
  room_name     text,
  cancelled_at  timestamptz,
  created_at    timestamptz NOT NULL DEFAULT now()
);
-- at most one pending callback per task, enforced by the database
-- Dial target, separate from owner_phone (which is identity + FK). NULL means
-- "the owner", so normal callbacks are unaffected. Exists so a test can point a
-- callback somewhere harmless instead of at a real person's phone.
ALTER TABLE shot.callbacks ADD COLUMN IF NOT EXISTS to_number text;

CREATE UNIQUE INDEX IF NOT EXISTS callbacks_one_pending_per_task ON shot.callbacks (task_id)
  WHERE fired_at IS NULL AND cancelled_at IS NULL AND task_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS callbacks_due_idx ON shot.callbacks (due_at)
  WHERE fired_at IS NULL AND cancelled_at IS NULL;

CREATE TABLE IF NOT EXISTS shot.calls (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  direction       text NOT NULL CHECK (direction IN ('inbound','outbound')),
  owner_phone     text NOT NULL,
  room_name       text NOT NULL,
  livekit_job_id  text,
  callback_id     uuid REFERENCES shot.callbacks(id),
  amd_category    text,        -- human|machine-ivr|machine-vm|machine-unavailable|uncertain
  amd_reason      text,
  amd_delay_ms    integer,     -- issue #6996 canary: alert if p95 > 9000
  amd_speech_s    double precision,  -- 0.0 => no VAD boundary ever reached the classifier
  pin_verified    boolean NOT NULL DEFAULT false,
  sip_status_code integer,     -- NULL on ring-out: LiveKit sends CANCEL, not a SIP code
  failure_kind    text,        -- ring_timeout|sip_error|dispatch_error|pin_failed
  started_at      timestamptz NOT NULL DEFAULT now(),
  answered_at     timestamptz,
  ended_at        timestamptz,
  end_reason      text
);
CREATE INDEX IF NOT EXISTS calls_owner_started_idx ON shot.calls (owner_phone, started_at DESC);

-- How much of the brief reached him before he spoke over it. `failure_kind`
-- says only "not delivered", which made "he heard nothing" and "he interrupted
-- me eighty percent of the way through" the same row -- and the code answered
-- both by hanging up on him.
ALTER TABLE shot.calls ADD COLUMN IF NOT EXISTS heard_chars integer;
ALTER TABLE shot.calls ADD COLUMN IF NOT EXISTS interrupted boolean;

-- WHY the PIN failed. `pin_verified boolean` says only "not verified", and
-- end_reason says `pin_failed` for both -- so "nobody keyed anything" (very
-- likely a machine, or a handset in a pocket) and "he keyed the wrong thing"
-- (a person fumbling) were the same row with the same answer. The voice tier
-- has carried the distinction on CallRecord.pin_result all along and dropped it
-- on the floor at process exit, while a test asserted "shot.calls must tell
-- silence from a bad code" about a table that had nowhere to put it.
-- verified|wrong|no_input|abandoned. NULL on rows written before this column.
ALTER TABLE shot.calls ADD COLUMN IF NOT EXISTS pin_result text;

-- What was actually said. Every callback failure so far was reconstructed from
-- Twilio durations and AMD categories because the words were kept nowhere: the
-- room closes, the realtime session goes with it, `enable_recording` leaves no
-- egress, and journald has no transcript at INFO. One row per call.
--
-- `turns` is [{at, role, text, interrupted}]. `text` on an interrupted
-- assistant turn is truncated at the PLAYBACK position -- what he heard, not
-- what was generated -- which is the whole reason this is worth keeping.
CREATE TABLE IF NOT EXISTS shot.transcripts (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  room_name   text NOT NULL,
  direction   text NOT NULL CHECK (direction IN ('inbound','outbound')),
  callback_id uuid REFERENCES shot.callbacks(id),
  turns       jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);
-- One row per room: a job that flushes twice (shutdown callback plus a retry)
-- must update, not duplicate.
CREATE UNIQUE INDEX IF NOT EXISTS transcripts_room_idx ON shot.transcripts (room_name);
CREATE INDEX IF NOT EXISTS transcripts_created_idx ON shot.transcripts (created_at DESC);

-- cross-call memory. The length cap is NOT a style rule: exceeding it flips
-- gpt-realtime to text-only replies, and a CHECK is the only place that
-- survives a well-meaning refactor of the summarizer.
CREATE TABLE IF NOT EXISTS shot.summaries (
  owner_phone text PRIMARY KEY REFERENCES shot.owners(phone),
  summary     text NOT NULL CHECK (length(summary) <= 1200),
  version     integer NOT NULL DEFAULT 1,
  updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS shot.summary_history (
  owner_phone text NOT NULL,
  version     integer NOT NULL,
  summary     text NOT NULL,
  call_id     uuid REFERENCES shot.calls(id),
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (owner_phone, version)
);

CREATE TABLE IF NOT EXISTS shot.webhook_events (
  event_id     text PRIMARY KEY,          -- whe_... == webhook-id, stable across retries
  data_type    text NOT NULL,
  resource_id  text NOT NULL,
  occurred_at  timestamptz NOT NULL,      -- payload created_at, NOT webhook-timestamp
  received_at  timestamptz NOT NULL DEFAULT now(),
  processed_at timestamptz,
  payload      jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS webhook_events_unprocessed_idx ON shot.webhook_events (received_at)
  WHERE processed_at IS NULL;

-- `notified_at` means "a callback was scheduled", which is NOT the same as
-- "the owner heard it". Two finished tasks were stamped notified at the exact moment
-- their callbacks failed (pin_failed, uncertain), so the system believed he had
-- been told about work he never heard a word about. This column means spoken,
-- out loud, to a verified human -- set on a delivered callback or when the
-- inbound greeting leads with it.
ALTER TABLE shot.tasks ADD COLUMN IF NOT EXISTS reported_at timestamptz;

CREATE INDEX IF NOT EXISTS tasks_owner_unreported_idx
    ON shot.tasks (owner_phone, finished_at DESC)
 WHERE reported_at IS NULL AND finished_at IS NOT NULL;

-- `result_summary` is the ~400-char SPOKEN headline. Keeping only that threw
-- the answer away: a worker reported three shows as a markdown table, the
-- table was stripped for speech, and when the owner asked "what are they?" the agent
-- had nothing left and started a whole new browsing task to re-find what it
-- had already been told. Keep the worker's report verbatim so a follow-up is a
-- read, not another session.
ALTER TABLE shot.tasks ADD COLUMN IF NOT EXISTS result_raw text;

-- The owner said stop. This is NOT `finished_at` (which says only "it stopped
-- running") and it is emphatically NOT `reported_at` (which means he HEARD the
-- result -- he heard nothing here, and conflating those two is the mistake that
-- filed two finished tasks as told while both callbacks had failed).
--
-- state='cancelled' says WHAT happened; this says WHEN, and its presence is
-- what makes cancelling idempotent and what stops a concurrent
-- refresh_from_anthropic resurrecting a task he already killed.
ALTER TABLE shot.tasks ADD COLUMN IF NOT EXISTS cancelled_at timestamptz;
