# Manual end-to-end test

Everything below the PSTN boundary is covered offline -- the whole callback
decision tree, the digit collector, real DTMF -- none of which needs a network.
These five steps cover what no test can reach. ~25 minutes.

## Before you start

**The stack runs on the VM, not on this laptop. Do not start a worker here.**
LiveKit load-balances jobs across every *registered* worker, so a local one
makes live behaviour a coin flip between two checkouts — which reads exactly
like a fix not working. `ops/restart.sh` refuses for this reason; do not
hand-run the uvicorn/worker commands to get around it.

```bash
uv run python -m ops.status          # must say "worker count 1 (correct)"
```

`/internal/*` binds to 127.0.0.1 **on the VM**, so anything curling it has to
run there:

```bash
set -a && . ./.env && set +a      # the VM, zone and project live in .env
gcloud compute ssh "${GCP_VM:?}" --zone "${GCP_ZONE:?}" --project "${GCP_PROJECT_ID:?}" --tunnel-through-iap
sudo journalctl -u shot-worker -f    # journald, not /tmp/shot_worker.log
```

Worth grepping: `dial outcome=`, `amd prediction`, `executing tool`,
`caller present`.

**After every call, read the record — not the log:**

```bash
uv run python -m ops.calls           # outcome, PIN, and how much he HEARD
uv run python -m ops.transcript      # what was actually said, both sides
```

`ops.calls`'s `heard` column is the one to look at: `nothing` is a real
delivery failure, `Nc cut` means he talked over it, which is a conversation
starting rather than a fault. The `pin` column now says *why* a code failed:
`-` never asked · `ok` · `silent` nobody keyed anything · `wrong` a bad code ·
`hungup` he left mid-challenge.

Your PIN is `CALLBACK_PIN` and the agent's number is `TWILIO_PHONE_NUMBER`, both
in `.env` and nowhere else. It will only answer calls from `OWNER_PHONE_NUMBER`
and will only call that number back.

```bash
grep -E '^(CALLBACK_PIN|TWILIO_PHONE_NUMBER|OWNER_PHONE_NUMBER)=' .env
```

---

## M1 — Inbound cold call · 5 min

Call the agent's number from your mobile. Talk for ~30 s. Then:

> "Find out when the next Lakers home game is and call me back."

Then say goodbye and let it hang up.

**Proves:** Twilio *inbound* origination actually routes to LiveKit (tests assert the
config string; only a real call proves Twilio honours it) · `allowed_numbers` accepts
your real ANI as the carrier formats it · agent join latency on the ring · real handset
echo and barge-in behaviour.

**Watch for:** how long between the last ring and the agent speaking. If it is >3 s,
that is the LiveKit plan cold start, and you want the Ship plan.

**Before any of this, check `uv run python -m ops.status` says
`worker count  1 (correct)`.** Two registered workers means LiveKit
load-balances your call between them, and a stale one will answer with old code.

**Then check:** `curl -s localhost:8090/internal/tasks | python3 -m json.tool`
— the task should be there, `running`, with a speakable ref.

---

## M2 — The callback, answered · 5 min

Wait. When the worker finishes, the reconcile sweep (every 2 min) schedules a
callback and the phone rings on its own.

Answer it, then **stay quiet for a few seconds**. That is the case that broke
repeatedly: you wait for the agent to speak, AMD waits for you to speak, and the
call used to die in that deadlock. The agent greets you first regardless now,
and `uncertain` no longer withholds — it goes to the PIN like any other answer.

**It must NOT say what it found.** The greeting says who is calling and asks for
your code, and nothing else. Then **say or type your PIN**, and only then does it
name the work and read it out.

**Proves:** the whole hang-up-and-call-back loop · AMD classifying a live human
"hello" over μ-law at 8 kHz · the PIN over real audio · that a silent answer is
greeted rather than abandoned · that the agent opens with context from the
earlier call.

**The criterion that matters most on this call:** *the task title must not be
audible before you key the code.* Check `uv run python -m ops.transcript` — the
first agent turn must contain nothing about the subject, and the title should
appear only in the turn that carries the results. This is the ONLY place the
model's actual obedience to that instruction can be observed; every offline test
can assert what the prompt *says* and nothing more.

**Two more things to try before you hang up**, both of which were broken:

- After it delivers, say **"that's all, thanks"**. It must say goodbye and hang
  up. It must not greet you again or ask what you want to talk about — that is
  exactly what it did at 02:51, ninety seconds into the call, over an explicit
  dismissal.
- Or instead: after it delivers, **say nothing at all for three minutes**. It
  must speak a short sign-off and then end the call. Silently deleting the room
  is how a callback becomes a mystery on your end.

**Then talk over it.** Halfway through the brief, say something — "wait, which
one?" — and it must **stop and answer you**. It must not hang up. That is the
bug that cost three calls in ninety seconds: an interruption was read as "the
brief did not play" and the room was deleted on the same millisecond, while
`_holding` was still set, so he spoke, got silence, and the line dropped.

**Pass criterion:** `ops.calls` shows `outcome=human`, `pin=ok`, and
`N/N actually heard the brief`. If you interrupted, expect `Nc cut` in `heard`
and the task left *unreported* — deliberate, so you are told again next call.
`ops.transcript` should show your interruption and the agent's reply to it.

**If you want it sooner** — from the VM, since `/internal/*` is localhost-only:
```bash
# rings YOUR phone
curl -sX POST localhost:8090/internal/callbacks -H 'content-type: application/json' \
  -d '{"brief":"Test callback. Say hello.","delay_s":0}'

# rings Twilio's test line instead -- use this for anything that is not
# deliberately testing your own handset
curl -sX POST localhost:8090/internal/callbacks -H 'content-type: application/json' \
  -d '{"brief":"probe","delay_s":0,"to_number":"+16504894546"}'
```

The response echoes `to_number` so you can see which one you just aimed at.

---

## M3 — The callback, declined to voicemail · 5 min

Trigger a callback as above, then **decline it** so it rolls to your carrier's
voicemail.

**Proves the highest-consequence path in the system.** AMD must return
`machine-vm` and the gate must withhold — against *your carrier's actual greeting*,
which nothing else can synthesise.

**Pass criterion**, from `uv run python -m ops.calls`:

```
when      outcome          AMD              why          delay  speech  pin
23:30:24  machine-vm       machine-vm       llm          872ms   3.74s   -
```

`outcome` and `AMD` both `machine-vm`, and **`pin` shows `-`** — a machine must
never even be prompted for a code, and must never be left waiting out the
twenty-five second code window. If `pin` shows `ok`, `no`, `wrong` or `silent`,
the gate let a machine through and that is a stop-everything bug.

**Then play the voicemail back.** A detected machine now hears the greeting —
deliberately, because AMD is not infallible and a real person misclassified as a
machine must get words rather than dead air and a dropped line. The recording
must contain a short greeting (who is calling, and a request for a code) and a
short apology, and **nothing else**. If it contains the title of the task, or
any of the findings, stop everything. It should also be SHORT — around ten
seconds, not most of a minute.

**Also record `amd_category` from this call.** `AMD_NO_SPEECH_S` went from 6.0
to 3.0, halving the dead air a silent answerer waits before the agent may speak.
It only affects lines that are silent for that long, and real boxes talk early
(the ones on record classified on 3.7–5.7s of speech) — but this is the test
that proves it. If your carrier's voicemail still comes back `machine-vm`, the
value is safe and could go lower; if it comes back `uncertain`, put it back up
via `AMD_NO_SPEECH_SECONDS` in `.env`.

(This previously told you to look for a log line reading
`disclosure withheld (outcome=machine-vm)`. No code has ever emitted it, so the
highest-stakes manual test had no pass criterion at all.)

**This is the one to watch closely.** If AMD says `human` here, the agent would read
your task results into a voicemail box.

**Nothing redials, ever.** A callback that does not reach you leaves its task
unreported, and you get the result on your next inbound call. Automatic retries
rang this phone three times in one evening while a systematic bug went unfixed.
Trigger another by hand with `POST /internal/callbacks` when you want one.

---

## M4 — Number reputation · 5 min

Call a friend from the agent's number and ask what their handset displays.

**Proves:** the number isn't carrying spam labels from its previous owner. There is
no API for this, and it regresses silently over time.

---

## M5 — Linear over the phone, and the token ceiling · 5 min

This is the one that broke on 2026-09-10, and no offline test can reach it: the
failure was OpenAI rejecting the model's responses, not our code.

Call the agent's number and, without pausing much between them:

1. *"What's on today?"* — one answer, one turn.
2. *"Look up the founder profile ticket."* — titles and status, no ID read out.
3. *"Write down: try the new callback flow."*
4. *"What's the weather in Portland?"*

**The point is asking for four things in a row.** Each tool turn costs TWO
model responses half a second apart, and the ceiling is 40,000 tokens per
ROLLING MINUTE — so a burst is what finds it, not a long slow call. Every one
must answer without dead air.

**Pass criteria:**

```bash
uv run python -m ops.transcript      # every request answered on its own turn
sudo journalctl -u shot-worker --since "10 min ago" | grep -c rate_limit_exceeded   # must be 0
sudo journalctl -u shot-worker --since "10 min ago" | grep "tokens:"
```

That last line is new and is the one to write down — `peak input N (M cached)`.
Anything approaching 40,000 means the headroom is gone again. Measured on the
bench, one response costs **4,575** of the bucket with these tools and **17,351**
with the Linear MCP toolset that used to be attached.

**If a response IS rejected**, `ops.transcript` now shows it as a `note` row
reading *"the model could not answer: ..."* in line with the words. Before this
change it appeared nowhere at all — the agent just went quiet, which reads
exactly like a routing bug.

**Then check Linear itself** for the ticket from step 3. `write_linear_issue`
has no approval gate, at your explicit request, so the only thing between a
passing remark and a real ticket is the model's judgement — worth eyeballing
once. Also try *"add a note to that one saying it worked"* and confirm it
comments rather than creating a second ticket.

**Last, listen to the shape of the replies.** They should end on the answer. If
they still trail off into "if you want, I can also…", the turn-length rule in
`INSTRUCTIONS` did not take.

---

## What to send back

The `dial outcome=` line from each call, plus `amd_delay_ms`. If any is >9000 the
Deepgram STT path is not being used; check `DEEPGRAM_API_KEY`.
