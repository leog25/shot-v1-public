"""L0 — the prompts are pinned, and the pin is real.

`prompts.py` has always said "pinned so an accidental prompt edit is a red CI
run". Nothing read INSTRUCTIONS_SHA256 -- not a test, not CI, not an ops script
-- so an accidental edit was in fact completely silent. Now it is not.
"""

from __future__ import annotations

from shot_voice import prompts

# Update DELIBERATELY, in the same commit as the prompt change, having read the
# diff. A surprise failure here means someone edited the session prompt without
# meaning to. Taken with support.TEST_NAME, which conftest installs as the
# owner's name -- so it pins the template, and says nothing about the real name.
INSTRUCTIONS_DIGEST = "6fd5b6c15d91cfbfdc201b22a83e6d9a792b18d7c764d2ba411b28487f667b58"


def test_the_instructions_are_pinned():
    assert prompts.INSTRUCTIONS_SHA256 == INSTRUCTIONS_DIGEST, (
        "the session prompt changed -- if that was deliberate, update "
        "INSTRUCTIONS_DIGEST in this file with the new value")


def test_neither_callback_opening_takes_a_format_argument():
    """A literal {what} went out on a live call once.

    Both openings used to interpolate the title; now neither names the work at
    all, so the whole class of bug is dead by construction rather than fixed.
    `.format()` on a string with no placeholders is a no-op, but a string WITH
    one that nobody formats reaches the model verbatim.
    """
    for opening in (prompts.OPENING_CALLBACK, prompts.OPENING_CALLBACK_VERIFIED):
        assert "{what}" not in opening
        assert "{brief}" not in opening


def test_the_greeting_forbids_naming_the_work():
    """Withholding has to be the MANDATORY half of the instruction.

    OPENING_INBOUND said the agent *may* reference earlier context but *must*
    ask what the owner needs, and it reliably did the mandatory half and skipped the
    other. So this is written as a prohibition, not a preference.
    """
    text = prompts.OPENING_CALLBACK.lower()
    assert "say nothing about what you looked into" in text
    assert "four digit code" in text


def test_only_the_delivery_turn_names_the_work():
    assert "{what}" in prompts.DELIVER_CALLBACK
    assert "{brief}" in prompts.DELIVER_CALLBACK


def test_the_wind_down_prompts_carry_the_tool_guidance():
    """AFTER_CALLBACK* replace the SESSION prompt for the rest of the call, so
    they have to wrap INSTRUCTIONS. Built from _SPEECH instead, the agent would
    be unable to start a task at the exact moment he is most likely to ask."""
    for prompt in (prompts.AFTER_CALLBACK,
                   prompts.AFTER_CALLBACK_CUT.format(brief="x")):
        assert prompts.INSTRUCTIONS in prompt
        assert "do NOT ask" in prompt


# ------------------------------------------------------- one shared persona

def test_the_shared_voice_is_in_both_personas():
    """INSTRUCTIONS and _SPEECH used to be two independent descriptions of the
    same agent, so the inbound and callback voices could drift apart silently.
    Now there is one string and they cannot."""
    assert prompts._VOICE in prompts.INSTRUCTIONS
    assert prompts._VOICE in prompts._SPEECH


def test_the_shared_voice_makes_no_claim_about_who_placed_the_call():
    """_SPEECH adds that; _VOICE must not, or INSTRUCTIONS asserts something
    false on every inbound call."""
    assert "placed this call" not in prompts._VOICE
    assert "placed this call" in prompts._SPEECH


def test_the_shared_voice_carries_no_literal_another_file_pins_absent():
    """_SPEECH is embedded in nine prompts and several of them assert these
    ABSENT -- test_callback_flow.py counts "Open the call." across every spoken
    script, so one occurrence here would make it six and fail in a way that
    reads like a flow bug rather than a prompt bug."""
    for literal in ("anything else", "Open the call.", "four digit code",
                    "in your own words", "ONE natural turn",
                    "do not mention codes"):
        assert literal not in prompts._SPEECH, literal


def test_the_shared_voice_sets_no_turn_length():
    """It is embedded in DELIVER_CALLBACK, whose whole point is "keeping every
    fact: every item, every time, every price. Do not summarise it and do not
    shorten it." The old _SPEECH said "brief" right next to that."""
    for length in ("brief", "two sentences", "one sentence", "keep turns short",
                   "shorten", "or fewer per turn"):
        assert length not in prompts._VOICE.lower(), length
    # ...and INSTRUCTIONS, which governs a free conversation, still sets one.
    assert "keep turns short" in prompts.INSTRUCTIONS.lower()


# ------------------------------------------------------------- the traps

def test_the_session_prompt_survives_format():
    """_after_prompt runs AFTER_CALLBACK_CUT.format(brief=...) over a string
    that embeds the whole of INSTRUCTIONS. One literal brace raises KeyError
    INSIDE hand_back's try/except, which logs and silently leaves the inbound
    prompt in force mid-callback -- the original bug, back and invisible."""
    assert "{" not in prompts.INSTRUCTIONS and "}" not in prompts.INSTRUCTIONS
    # The thing that would actually blow up, exercised.
    prompts.AFTER_CALLBACK_CUT.format(brief="x")


def test_the_session_prompt_does_not_greet():
    """INSTRUCTIONS is the session prompt underneath OPENING_INBOUND_NEWS, whose
    contract test bans these phrasings -- and it is embedded VERBATIM in
    AFTER_CALLBACK, installed live ninety seconds into a callback. Greeting
    energy here re-opens a conversation the owner is already having."""
    low = " ".join(prompts.INSTRUCTIONS.split()).lower()
    for banned in ("how can i help", "what can i help", "what do you need",
                   "what would you like", "greet him", "say hello"):
        assert banned not in low, banned


def test_the_session_prompt_never_names_the_cancel_tool():
    """test_registry.py pins the exact set of files mentioning it, and
    prompts.py is not one. Cancel guidance rides on the tool description."""
    assert "cancel_task" not in prompts.INSTRUCTIONS


def test_the_session_prompt_still_carries_every_routing_rule():
    """The persona rewrite must not have quietly dropped a rule that cost money
    to learn. Each of these traces to a real incident."""
    # Whitespace-normalised: these are wrapped strings whose line continuations
    # keep the next line's indent, so a phrase that matters can arrive with a
    # double space in the middle of it.
    low = " ".join(prompts.INSTRUCTIONS.split()).lower()
    assert "never answer a question about the current world from memory" in low
    assert "rings his phone" in low
    assert "use the browser" in low and "opened" in low
    assert "do not search again and do not delegate" in low
    assert "impatience is not a cancellation" in low
    assert "check_my_day" in prompts.INSTRUCTIONS
    assert "list_my_tasks" in prompts.INSTRUCTIONS
    # The three Linear tools, by name. The paragraph they replaced promised
    # "his full Linear tool set" and "when he asks you to write something
    # down" -- and after the MCP toolset came off tier 1 there was no
    # write-capable tool on ShotAgent at all, so the model would either
    # invent a confirmation or route it to a background task that rings him.
    assert "find_linear_issue" in prompts.INSTRUCTIONS
    assert "write_linear_issue" in prompts.INSTRUCTIONS
    assert "comment_on_linear_issue" in prompts.INSTRUCTIONS
    assert "full Linear tool set" not in prompts.INSTRUCTIONS


def test_the_session_prompt_concedes_the_clock_without_conceding_the_world():
    """A time is now seeded into the ChatContext, so "you do not know today's
    date" became false -- and an instruction the model can see is false is
    worse than none. The prohibition it was carrying still has to hold."""
    low = " ".join(prompts.INSTRUCTIONS.split()).lower()
    assert "you do not know today's date" not in low
    assert "told the day and the time" in low
    assert "even when you feel certain" in low


# --------------------------------------------------------- the two openings

def test_the_inbound_openings_take_the_time_of_day():
    for opening in (prompts.OPENING_INBOUND, prompts.OPENING_INBOUND_NEWS):
        assert "{part}" in opening and "{when}" in opening


def test_the_news_turn_asks_nothing_because_the_offer_turn_does():
    """A question at the tail of the news invites him to answer over it, which
    marks work he heard IN FULL as undelivered and re-announces it next call."""
    news = prompts.OPENING_INBOUND_NEWS
    assert "Do NOT ask him anything at all in this turn" in news
    assert "Do NOT ask whether he would like to hear it" in news
    assert "offer" in prompts.OPENING_INBOUND_OFFER.lower()


def test_the_offer_turn_does_not_go_and_fetch_the_answer():
    """Told to talk about "what else is on today", the model otherwise reaches
    for check_my_day and turns an offer into an unrequested monologue."""
    assert "Do not call any tool" in prompts.OPENING_INBOUND_OFFER
    assert "Do not call any tool" not in prompts.OPENING_INBOUND_NEWS
    assert "without calling a tool first" in prompts.OPENING_INBOUND
