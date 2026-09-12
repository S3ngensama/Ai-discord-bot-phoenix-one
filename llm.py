"""Calls Claude to reason about the driver's problem and produce a short, beginner-friendly diagnosis + suggested fix."""

import logging
import re

from anthropic import AsyncAnthropic

import config

log = logging.getLogger("phoenix-one")

_client = None

REASONING_MAX_TOKENS = 3000
FAST_MAX_TOKENS = 700
SUMMARY_MAX_TOKENS = 400

PROBLEM_LINE = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(.*)$")
MAX_EXTRACTED_PROBLEMS = 4      # a real intake has one to three
MAX_PROBLEM_LENGTH = 160        # longer than this is prose, not a symptom

FAILED_DIAGNOSIS = "Sorry, I had trouble thinking that through — mind trying again?"
FAILED_ANSWER = "Ask me that again, human — I lost the thread of it."

REPLY_TYPES = (
    "confirmed_better",
    "confirmed_partial",
    "confirmed_worse_or_same",
    "correction",
    "revert_done",
    "question",
    "new_problem",
    "skipped",
)

RESULT_PENDING = "pending"
RESULT_HELPED = "helped"
RESULT_PARTIAL = "partial"
RESULT_WORSE = "worse"
RESULT_REVERTED = "reverted"
RESULT_UNTESTED = "untested"
RESULT_SUPERSEDED = "superseded"

RESULT_LABELS = {
    RESULT_PENDING: "just suggested, driver hasn't reported back yet",
    RESULT_HELPED: "HELPED — this fixed the problem, it stays on the car",
    RESULT_PARTIAL: "PARTLY HELPED — improved things but didn't finish the job, it stays on the car",
    RESULT_WORSE: "DID NOT WORK — this was tried and did not help (or made things worse)",
    RESULT_REVERTED: "REVERTED — was tried, didn't work, and has been put back to where it was",
    RESULT_UNTESTED: "NEVER TESTED — suggested but the driver did not try it, so it is not on the car",
    RESULT_SUPERSEDED: "SUPERSEDED — no result was recorded; no longer awaiting a test. Do not infer success or failure",
}


def _get_client():
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def change_text(change) -> str:
    """The suggestion itself, from any storage shape. Entries used to be plain strings and older sessions still hold those."""
    if isinstance(change, dict):
        return change.get("text", "")
    return change or ""


def change_applied(change) -> bool:
    """Whether the driver actually put this change on the car. Legacy string entries are treated as applied, since that was the old assumption and re-reading them any other way would rewrite history."""
    if isinstance(change, dict):
        return bool(change.get("applied")) and change.get("result") != RESULT_REVERTED
    return True


def change_result(change) -> str:
    """What happened when the driver tried it. Entries written before outcome tracking have no result field, so they fall back to applied state — which is all that was known about them."""
    if isinstance(change, dict):
        result = change.get("result")
        if result in RESULT_LABELS:
            return result
        return RESULT_PENDING if change.get("applied") else RESULT_UNTESTED
    return RESULT_PENDING


def applied_changes(changes_tried) -> list:
    """Only the changes actually on the car. The revert logic counts these — an untested suggestion is not an adjustment."""
    if not changes_tried:
        return []
    return [c for c in changes_tried if change_applied(c)]


def failed_changes(changes_tried) -> list:
    """Changes with a known bad outcome. These are never fresh options again, however much else has changed since."""
    if not changes_tried:
        return []
    return [c for c in changes_tried if change_result(c) in (RESULT_WORSE, RESULT_REVERTED)]


def _extract_text(response, label: str = "call") -> str:
    """Finds the actual text reply in a response, regardless of whether a thinking block or anything else comes before it in the content list. Returns "" if there is no text at all — callers treat an empty result as a failed call, which is tidier than raising."""
    try:
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
    except Exception as e:
        log.error(f"Could not read response content ({label}): {e}")
        return ""

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        thinking = None
        try:
            details = getattr(response.usage, "output_tokens_details", None)
            thinking = getattr(details, "thinking_tokens", None)
        except Exception:
            pass
        log.error(
            f"{label}: ran out of tokens before writing a reply "
            f"(stop_reason=max_tokens, thinking_tokens={thinking}). "
            f"The token ceiling for this call is too low for the amount "
            f"of context it is being given."
        )
    else:
        log.warning(f"{label}: response contained no text block (stop_reason={stop_reason})")
    return ""


def _failed(text: str = FAILED_DIAGNOSIS) -> dict:
    """A reply the driver can read, marked as not-real-advice so it never gets recorded as a setup change."""
    return {"text": text, "is_question": False, "ok": False}


def _parse_response(raw: str) -> dict:
    """Detects whether Claude asked a clarifying question instead of giving a real diagnosis. Returns {"text", "is_question", "ok"} with the prefix stripped either way. An empty response counts as a failure — _extract_text has already logged why."""
    stripped = (raw or "").strip()
    if not stripped:
        return _failed()
    if stripped.upper().startswith("CLARIFYING QUESTION:"):
        text = stripped.split(":", 1)[1].strip()
        if not text:
            return _failed()
        return {"text": text, "is_question": True, "ok": True}
    # Reverts carry an explicit reference to the numbered change list.
    # This is stripped before replying to the driver.
    if stripped.upper().startswith("REVERT CHANGE:"):
        match = re.match(r"REVERT CHANGE:\s*(\d+)\s*\n(.+)", stripped, re.I | re.S)
        if not match or not match.group(2).strip():
            return _failed()
        return {"text": match.group(2).strip(), "is_question": False,
                "ok": True, "kind": "revert", "revert_target": int(match.group(1)) - 1}
    return {"text": stripped, "is_question": False, "ok": True}


SYSTEM_PROMPT = """PLACEHOLDER — the engineering persona, the diagnostic
rules and the hard constraints that make this assistant behave like a race
engineer are not included in this public repository.

Supply your own system prompt here. What it needs to establish:
  - who the assistant is and how it talks
  - that it suggests exactly ONE change at a time
  - what it must never recommend, for your simulator and car classes
  - how it explains the cost of every change it proposes
  - the order it works through adjustments when diagnosing

The surrounding code does not care what this string says. It only assumes
the model returns a short reply addressed to a driver.
"""


def _context_blocks(current_setup, telemetry) -> str:
    """The driver's uploaded data, or an explicit statement that there isn't any. The explicit absence matters: left silent, the model has filled the gap with invented telemetry and cited it as evidence."""
    blocks = ""
    if current_setup:
        blocks += f"\n\n{current_setup}"
    else:
        blocks += (
            "PLACEHOLDER — task instructions removed. "
        )
    if telemetry:
        blocks += f"\n\n{telemetry}"
    else:
        blocks += (
            "PLACEHOLDER — task instructions removed. "
        )
    return blocks


def _changes_block(changes_tried) -> str:
    """Everything suggested this session, each with what came of it."""
    if not changes_tried:
        return ""
    lines = []
    for number, c in enumerate(changes_tried, start=1):
        label = RESULT_LABELS.get(change_result(c), RESULT_LABELS[RESULT_PENDING])
        kind = "UNDO INSTRUCTION; " if isinstance(c, dict) and c.get("kind") == "revert" else ""
        lines.append(f"- Change {number}: [{kind}{label}] {change_text(c)}")
    block = "\n\nChanges suggested for this car in this session:\n" + "\n".join(lines)

    if failed_changes(changes_tried):
        block += (
            "PLACEHOLDER — task instructions removed. "
        )
    return block


def _memory_blocks(history, summary) -> str:
    """The thread's memory: a condensed account of the older part of the conversation, then the recent messages word-for-word."""
    blocks = ""
    if summary:
        blocks += f"\n\nWhat's happened in this thread so far:\n{summary}"
    if history:
        lines = []
        for m in history:
            who = "Driver" if m.get("role") == "driver" else "You"
            lines.append(f"{who}: {m.get('content')}")
        blocks += "\n\nMost recent messages:\n" + "\n".join(lines)
    return blocks


def _knowledge_block(knowledge) -> str:
    """The retrieved knowledge, each entry labelled with the layer it came from."""
    if not knowledge:
        return ""
    entries = []
    for item in knowledge:
        pillar = item.get("pillar") or item.get("source_type") or "GENERAL"
        car = item.get("car") or "not tagged"
        track = item.get("track") or "not tagged"
        entries.append(f"- [{pillar}; car: {car}; track: {track}] {item['content']}")
    return (
        f"\n\nRelevant info from the community knowledge base "
        f"(use this if it's actually relevant, ignore it if not). The tag "
        f"on each entry says which layer it came from: CONFIRMED_FIX is "
        f"something that actually worked for a real driver, PHYSICS "
        f"explains why something happens, COMPONENT and CORRELATION say "
        f"what to change and what it costs, CLASS is car-class specific, "
        f"SAFETY and PACE cover priorities:\n" + "\n".join(entries)
    )


async def get_diagnosis(car, track, problem: str, knowledge: list = None, current_setup: str = None, telemetry: str = None, history: list = None, summary: str = None, changes_tried: list = None) -> dict:
    """The first read on a problem. changes_tried is optional but worth passing on a re-diagnosis, so the model doesn't suggest something already tried."""
    if not config.ANTHROPIC_API_KEY:
        return _failed(
            "(AI reasoning isn't set up yet — an admin needs to add an "
            "Anthropic API key.)"
        )

    user_message = (
        f"Car: {car or 'not specified'}\n"
        f"Track: {track or 'not specified'}\n"
        f"Driver's description: {problem}"
        f"{_memory_blocks(history, summary)}"
        f"{_knowledge_block(knowledge)}"
        f"{_changes_block(changes_tried)}"
        f"{_context_blocks(current_setup, telemetry)}"
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return _parse_response(_extract_text(response, "diagnosis"))
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return _failed(
            "Sorry, I had trouble thinking that through just now — mind "
            "trying again in a moment?"
        )


async def answer_question(car, track, current_problem, question: str, knowledge: list = None, last_suggestion: str = None, current_setup: str = None, telemetry: str = None, changes_tried: list = None, history: list = None, summary: str = None) -> dict:
    """Answers a driver's question — what a setting does, why a fix works, whether something else would be better."""
    if not config.ANTHROPIC_API_KEY:
        return _failed("(AI reasoning isn't set up yet — an admin needs to add an Anthropic API key.)")

    user_message = (
        f"Car: {car or 'not specified'}\n"
        f"Track: {track or 'not specified'}\n"
        f"Problem currently being worked on: {current_problem or 'not specified'}\n"
        f"Fix most recently suggested: {last_suggestion or 'none yet'}\n"
        f"The driver is ASKING A QUESTION, not reporting a result: {question}"
        f"{_memory_blocks(history, summary)}"
        f"{_knowledge_block(knowledge)}"
        f"{_changes_block(changes_tried)}"
        f"{_context_blocks(current_setup, telemetry)}\n\n"
        "PLACEHOLDER — task instructions removed. "
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        result = _parse_response(_extract_text(response, "question"))
        if not result["ok"]:
            return _failed(FAILED_ANSWER)
        return result
    except Exception as e:
        log.error(f"LLM question call failed: {e}")
        return _failed(FAILED_ANSWER)


async def get_followup(car, track, problem, last_suggestion, feedback: str, current_setup: str = None, telemetry: str = None, changes_tried: list = None, history: list = None, summary: str = None, partial: bool = False, knowledge: list = None) -> dict:
    """Called when the driver's feedback isn't a clean 'better' — either the fix didn't work, or it helped without finishing the job."""
    if not config.ANTHROPIC_API_KEY:
        return _failed("Got it, noted — let's keep working on it once AI reasoning is set up.")

    revert_block = ""
    if not partial and len(applied_changes(changes_tried)) >= 2:
        revert_block = (
            "PLACEHOLDER — task instructions removed. "
        )

    if partial:
        closing = (
            "PLACEHOLDER — task instructions removed. "
        )
    else:
        closing = (
            "PLACEHOLDER — task instructions removed. "
        )

    user_message = (
        f"Car: {car or 'not specified'}\n"
        f"Track: {track or 'not specified'}\n"
        f"Original problem: {problem}\n"
        f"Fix I already suggested: {last_suggestion}\n"
        f"Driver's feedback after testing it: {feedback}"
        f"{_memory_blocks(history, summary)}"
        f"{_knowledge_block(knowledge)}"
        f"{_changes_block(changes_tried)}"
        f"{_context_blocks(current_setup, telemetry)}"
        f"{revert_block}\n\n"
        f"{closing}"
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return _parse_response(_extract_text(response, "follow-up"))
    except Exception as e:
        log.error(f"LLM follow-up call failed: {e}")
        return _failed()


async def get_post_revert_suggestion(car, track, problem, reverted_change, current_setup: str = None, telemetry: str = None, changes_tried: list = None, history: list = None, summary: str = None, knowledge: list = None) -> dict:
    """Called when the driver has just put a setting back as instructed."""
    if not config.ANTHROPIC_API_KEY:
        return _failed("Got it, noted — let's keep working on it once AI reasoning is set up.")

    user_message = (
        f"Car: {car or 'not specified'}\n"
        f"Track: {track or 'not specified'}\n"
        f"Problem still to solve: {problem}\n"
        f"The driver has just PUT BACK this change, as you told them to: {reverted_change}"
        f"{_memory_blocks(history, summary)}"
        f"{_knowledge_block(knowledge)}"
        f"{_changes_block(changes_tried)}"
        f"{_context_blocks(current_setup, telemetry)}\n\n"
        "PLACEHOLDER — task instructions removed. "
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return _parse_response(_extract_text(response, "post-revert"))
    except Exception as e:
        log.error(f"LLM post-revert call failed: {e}")
        return _failed()


async def get_pace_suggestion(car, track, problem, last_suggestion, feedback: str, current_setup: str = None, telemetry: str = None, changes_tried: list = None, history: list = None, summary: str = None, knowledge: list = None) -> dict:
    """Suggests another pace adjustment using actual recorded outcomes."""
    if not config.ANTHROPIC_API_KEY:
        return _failed("Got it, noted — let's keep tuning once AI reasoning is set up.")

    user_message = (
        f"Car: {car or 'not specified'}\n"
        f"Track: {track or 'not specified'}\n"
        f"Previous handling problem: {problem}\n"
        f"Most recent suggestion (use its recorded outcome below): {last_suggestion}\n"
        f"Driver's request now: {feedback}"
        f"{_memory_blocks(history, summary)}"
        f"{_knowledge_block(knowledge)}"
        f"{_changes_block(changes_tried)}"
        f"{_context_blocks(current_setup, telemetry)}\n\n"
        "PLACEHOLDER — task instructions removed. "
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return _parse_response(_extract_text(response, "pace suggestion"))
    except Exception as e:
        log.error(f"LLM pace-suggestion call failed: {e}")
        return _failed()


async def summarise_thread(previous_summary: str, new_messages: list):
    """Folds new messages into the running summary."""
    if not config.ANTHROPIC_API_KEY or not new_messages:
        return None

    lines = []
    for m in new_messages:
        who = "Driver" if m.get("role") == "driver" else "Engineer"
        lines.append(f"{who}: {m.get('content')}")
    transcript = "\n".join(lines)

    previous_block = (
        f"Existing summary:\n{previous_summary}\n\n" if previous_summary else ""
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_FAST_MODEL,
            max_tokens=SUMMARY_MAX_TOKENS,
            system=(
                "PLACEHOLDER — task prompt removed from the public "
                "repository. See README_PUBLIC.md."
            ),
            messages=[{"role": "user", "content": f"{previous_block}New messages:\n{transcript}"}],
        )
        updated = _extract_text(response, "summarise").strip()
        return updated or None
    except Exception as e:
        log.error(f"Thread summarisation failed: {e}")
        return None


async def tuning_intent(feedback: str) -> str:
    """In the tuning stage, works out what the driver actually wants. Returns MORE, DONE, BROKEN, QUESTION, FEEDBACK, SKIPPED or ERROR."""
    if not config.ANTHROPIC_API_KEY:
        return "ERROR"

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_FAST_MODEL,
            max_tokens=FAST_MAX_TOKENS,
            system=(
                "PLACEHOLDER — task prompt removed from the public "
                "repository. See README_PUBLIC.md."
            ),
            messages=[{"role": "user", "content": feedback}],
        )
        raw = _extract_text(response, "tuning intent").strip().upper()
        for candidate in ("BROKEN", "QUESTION", "FEEDBACK", "MORE", "DONE", "SKIPPED"):
            if raw.strip(" .`*_\n") == candidate:
                return candidate
        return "ERROR"
    except Exception as e:
        log.error(f"Tuning intent check failed: {e}")
        return "ERROR"


def _clean_problem(raw: str) -> str:
    """Strips markdown off one extracted problem line and returns "" if what remains is not a symptom."""
    cleaned = raw.strip()
    cleaned = cleaned.strip("*_`").strip()
    cleaned = cleaned.lstrip("-•–— ").strip()
    cleaned = cleaned.strip("*_`").strip()

    if not any(character.isalpha() for character in cleaned):
        return ""
    if cleaned.endswith(":"):
        return ""
    if len(cleaned) > MAX_PROBLEM_LENGTH:
        return ""
    return cleaned


async def extract_car_track_and_problems(text: str):
    """Pulls car, track, and a LIST of distinct problems out of the driver's free-text intake message. Returns (car, track, problems)."""
    if not config.ANTHROPIC_API_KEY:
        return None, None, [text]

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_FAST_MODEL,
            max_tokens=FAST_MAX_TOKENS,
            system=(
                "PLACEHOLDER — task prompt removed from the public "
                "repository. See README_PUBLIC.md."
            ),
            messages=[{"role": "user", "content": text}],
        )
        raw = _extract_text(response, "extraction")
        car, track = None, None
        problems = []
        in_problems = False
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("CAR:"):
                value = stripped.split(":", 1)[1].strip()
                car = None if value.lower() == "unknown" else value
            elif stripped.upper().startswith("TRACK:"):
                value = stripped.split(":", 1)[1].strip()
                track = None if value.lower() == "unknown" else value
            elif stripped.upper().startswith("PROBLEMS"):
                in_problems = True
            elif in_problems:
                # Blank lines are separators, not terminators: the requested
                # format puts one after the last problem.
                if not stripped:
                    continue

                match = PROBLEM_LINE.match(stripped)
                if match is None:
                    if problems:
                        break
                    # Nothing collected yet: tolerate a stray line between
                    # the header and the first entry rather than giving up.
                    continue

                cleaned = _clean_problem(match.group(2))
                if not cleaned:
                    if problems:
                        break
                    continue

                problems.append(cleaned)
                if len(problems) >= MAX_EXTRACTED_PROBLEMS:
                    break

        if not problems:
            problems = [text]
        return car, track, problems
    except Exception as e:
        log.error(f"Car/track/problems extraction failed: {e}")
        return None, None, [text]


async def interpret_reply(car, track, current_problem, last_suggestion, driver_reply: str, history: list = None):
    """Figures out what the driver's reply actually IS. Returns {"type": one of REPLY_TYPES}."""
    if not config.ANTHROPIC_API_KEY:
        return {"type": "confirmed_worse_or_same", "ok": False}

    user_message = (
        f"Car: {car or 'not specified'}\n"
        f"Track: {track or 'not specified'}\n"
        f"Current problem being worked on: {current_problem}\n"
        f"Fix suggested: {last_suggestion}\n"
        f"Driver's reply: {driver_reply}"
        f"{_memory_blocks(history, None)}"
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.ANTHROPIC_FAST_MODEL,
            max_tokens=FAST_MAX_TOKENS,
            system=(
                "PLACEHOLDER — task prompt removed from the public "
                "repository. See README_PUBLIC.md."
            ),
            messages=[{"role": "user", "content": user_message}],
        )
        raw = _extract_text(response, "interpret reply")

        for line in raw.splitlines():
            stripped = line.strip().strip("`*_ ").rstrip(".")
            if not stripped:
                continue
            if stripped.upper().startswith("TYPE:"):
                stripped = stripped.split(":", 1)[1].strip()
            value = stripped.lower()
            if value in REPLY_TYPES:
                return {"type": value, "ok": True}

        lowered = raw.lower()
        for candidate in sorted(REPLY_TYPES, key=len, reverse=True):
            if candidate in lowered:
                log.info(f"Reply type {candidate!r} recovered from surrounding text")
                return {"type": candidate, "ok": True}

        log.error(f"Reply interpretation returned no valid TYPE: {raw[:200]!r}")
        return {"type": "confirmed_worse_or_same", "ok": False}
    except Exception as e:
        log.error(f"Reply interpretation failed: {e}")
        return {"type": "confirmed_worse_or_same", "ok": False}
