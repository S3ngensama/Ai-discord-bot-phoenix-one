"""Handles the setup-engineer conversation in forum threads: greeting, initial diagnosis, a feedback loop, and closure."""

import asyncio
import json
import logging
import os
import re
import tempfile
import weakref

import discord
from discord.ext import commands

import config
from storage import db
from engine import llm, knowledge, svm_parser, telemetry_parser

log = logging.getLogger("phoenix-one")

ROUND_GENERAL = 1
ROUND_CLASS = 2

HISTORY_WINDOW = 6
SUMMARY_EVERY = 10

MIN_STARTER_LENGTH = 15

# What the driver's reply means for the change they were testing.
RESULT_FOR_REPLY = {
    "confirmed_better": llm.RESULT_HELPED,
    "confirmed_partial": llm.RESULT_PARTIAL,
    "confirmed_worse_or_same": llm.RESULT_WORSE,
}

CLASS_PATTERNS = [
    ("Hypercar", ("hypercar", "hyper car", "lmh", "lmdh")),
    ("LMP2", ("lmp2", "lmp 2")),
    ("LMP3", ("lmp3", "lmp 3")),
    ("GT3", ("gt3", "gt 3", "lmgt3")),
]

CLASS_NAMES = {name.lower() for name, _ in CLASS_PATTERNS}

_CLOSE_PHRASES = [
    r"yes", r"yeah", r"yep", r"yup", r"ya", r"sure", r"okay", r"ok",
    r"correct", r"confirm", r"confirmed", r"agreed", r"affirmative",
    r"done", r"all done", r"we're done", r"were done", r"i'm done",
    r"im done", r"finished", r"all good", r"good to go", r"that's it",
    r"thats it", r"go ahead", r"please do", r"close it", r"close",
    r"you can close it", r"close the thread", r"close it please",
    r"yes please", r"yes close it", r"yes we're done", r"yes im done",
    r"yes i'm done", r"yeah close it", r"ok close it", r"okay close it",
    r"sounds good", r"nothing else", r"that's all", r"thats all",
    r"thanks", r"thank you", r"cheers",
]

# Optional politeness either side, so "yes thanks" or "ok thanks mate"
# still count without opening the door to arbitrary text.
_CLOSE_FILLER = r"(?:\s*(?:please|thanks|thank you|cheers|mate|man|bud|for now))*"

CLOSE_CONFIRMATION = re.compile(
    rf"(?:{'|'.join(_CLOSE_PHRASES)}){_CLOSE_FILLER}",
    re.IGNORECASE,
)

CLASS_QUESTION = (
    "Still no car, human. Fine — just give me the class and I'll work "
    "with that: **LMP2**, **LMP3**, **GT3**, or **Hypercar**?"
)

FORCE_DIAGNOSIS_DIRECTIVE = (
    "\n\nIMPORTANT: you have already asked this driver a clarifying "
    "question and this is their answer. Do NOT ask another question. "
    "Commit to your best diagnosis and ONE fix using whatever you know, "
    "even if the picture is incomplete. If you are working from an "
    "assumption, say so in a few words as part of your normal reply."
)

NO_CAR_DIRECTIVE = (
    "\n\nNOTE: the driver has not told you the car or even its class, "
    "and will not be asked again. Give the most generally-applicable "
    "advice you can, using only adjustments available on every class, "
    "and state plainly in a few words that this is generic guidance "
    "because you don't know the car."
)

CLASS_ONLY_DIRECTIVE = (
    "\n\nNOTE: you know only the CLASS of car, not the specific model. "
    "Reason at class level and say in a few words that you're assuming "
    "typical behaviour for that class."
)

LATE_CAR_DIRECTIVE = (
    "\n\nNOTE: the driver has just told you the car (or its class) after "
    "you had already given advice without knowing it. Redo the diagnosis "
    "properly now. Acknowledge the new information in a few words — "
    "dryly is fine — and give ONE fix. Do not ask another question."
)

CORRECTION_DIRECTIVE = (
    "\n\nIMPORTANT: you read this problem wrong, and the driver has just "
    "corrected you. The description above is THEIR corrected version — "
    "work from it, not from what you assumed before. They have NOT tested "
    "your last suggestion, so do not ask how it went and do not tell them "
    "to undo it. Acknowledge the correction in a few words — dryly is "
    "fine, and owning it is better than glossing over it — then give ONE "
    "fix for what they've actually described. Do not ask another question."
)

BROKEN_AGAIN_DIRECTIVE = (
    "\n\nNOTE: this car was working and the driver had moved on to chasing "
    "lap time, but they are now reporting a handling problem again. Stop "
    "chasing pace — this is a handling fix. Some of the changes listed "
    "above were made for speed rather than stability, so consider whether "
    "one of them is what broke the car before adding anything new."
)

WELCOME_MESSAGE = (
    "Another human with car problems, thrilling. Give me:\n\n"
    "**1. Car**\n**2. Track**\n**3. What it's doing wrong**\n\n"
    "Got more than one issue? List them all, I'll work through them one "
    "at a time. Setup file (.svm) or telemetry (.duckdb) welcome too, "
    "if you actually have data instead of vibes."
)

FEEDBACK_PROMPT = (
    "\n\n—\nTest it out, then let me know how it went. You can just reply:\n"
    "🟢 **better**  🔴 **worse**  ⚪ **no change**\n"
    "...or describe it in your own words, ask a question, or mention "
    "something else you've noticed — I'll figure out what you mean."
)

QUEUED_PROMPT = (
    "\n\n—\nStill worth testing the last change first — tell me how it "
    "went and I'll come back to this one."
)

ALL_PROBLEMS_DONE = (
    "🏁 That's everything on the list, human. Want to push for more pace "
    "from here, or leave it where it is? Just tell me either way."
)

CONFIRM_CLOSE_MESSAGE = (
    "Sounds like we're done here, human. Say the word and I'll close this "
    "thread off — it stays readable, but I won't be answering in it after "
    "that, so you'd need a fresh post next time. Anything else first?"
)

DONE_MESSAGE = (
    "Closed off, human. A car that works is worth more than one more "
    "clever change — go and drive it. Start a new post if something "
    "goes wrong down the line."
)

CLOSED_MESSAGE = (
    "This one's closed, human. Start a new post and I'll take a look with "
    "fresh eyes."
)


def is_yes(text: str) -> bool:
    """True only for an explicit, unambiguous confirmation to close."""
    if not text:
        return False

    # Phones substitute curly apostrophes silently. Normalise before
    # anything else so "don’t" and "don't" are the same string.
    cleaned = text.strip().lower().replace("\u2019", "'").replace("\u02bc", "'")

    if "?" in cleaned:
        return False

    # Strip surrounding punctuation and collapse whitespace, so
    # "yes!" and "  yes. " both reduce to "yes".
    cleaned = re.sub(r"[^\w\s']", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return False

    return bool(CLOSE_CONFIRMATION.fullmatch(cleaned))


def load_problems(session: dict) -> list:
    raw = session.get("problems")
    if not raw:
        return [session.get("problem")] if session.get("problem") else []
    try:
        return json.loads(raw)
    except Exception:
        return [raw]


def load_changes(session: dict) -> list:
    """Every suggestion made so far this session. Entries are dicts of {"text", "applied", "result"}; older sessions hold plain strings or dicts without a result, which the llm helpers read sensibly."""
    raw = session.get("changes_tried")
    if not raw:
        return []
    try:
        changes = json.loads(raw)
        if not isinstance(changes, list):
            return []
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                continue
            if change.get("result") == llm.RESULT_REVERTED:
                change["applied"] = False
            if change.get("result") == llm.RESULT_PENDING and index < len(changes) - 1:
                change["result"] = llm.RESULT_SUPERSEDED
        return changes
    except Exception:
        return []


def record_change(changes: list, result: dict, problem=None) -> list:
    """Adds a suggestion — but only if it was a real one. Questions aren't changes, and neither are error fallbacks. Starts as pending and not applied: the driver hasn't tested it, so it isn't on the car and nothing is known about whether it works."""
    if result.get("ok") and not result.get("is_question"):
        if _last_pending(changes) is not None:
            raise ValueError("An earlier adjustment is still awaiting a result")
        entry = {
            "text": result["text"],
            "applied": False,
            "result": llm.RESULT_PENDING,
            "kind": result.get("kind", "adjustment"),
            "problem": problem,
        }
        if entry["kind"] == "revert":
            target = result.get("revert_target")
            if (not isinstance(target, int) or target < 0 or target >= len(changes)
                    or not llm.change_applied(changes[target])
                    or (isinstance(changes[target], dict) and changes[target].get("kind") == "revert")):
                raise ValueError("The undo instruction has no valid applied target")
            entry["revert_target"] = target
        changes.append(entry)
    return changes


def _last_pending(changes: list):
    """The suggestion the driver was asked to test — the most recent one still waiting on a result."""
    for change in reversed(changes):
        if isinstance(change, dict) and change.get("result") == llm.RESULT_PENDING:
            return change
    return None


def record_result(changes: list, outcome: str) -> list:
    """Records what came of the change the driver just tested. Testing it is also what puts it on the car, so applied is set here too."""
    change = _last_pending(changes)
    if change is not None:
        if change.get("kind") == "revert":
            mark_reverted(changes)
            return changes
        change["applied"] = True
        change["result"] = outcome
    return changes


def mark_reverted(changes: list):
    """Resolve an undo and its target, including older untyped records."""
    pending = _last_pending(changes)
    target = pending.get("revert_target") if pending else None
    if target is None:
        target = next((i for i in range(len(changes) - 1, -1, -1)
                       if llm.change_applied(changes[i])
                       and not (isinstance(changes[i], dict) and changes[i].get("kind") == "revert")), None)
    if not isinstance(target, int) or target < 0 or target >= len(changes):
        return ""
    original = changes[target]
    if not llm.change_applied(original):
        return ""
    if not isinstance(original, dict):
        original = {"text": llm.change_text(original)}
        changes[target] = original
    original.update(applied=False, result=llm.RESULT_REVERTED)
    if pending is not None and pending is not original:
        if pending.get("kind") in (None, "revert"):
            pending.update(kind="revert", applied=False, result=llm.RESULT_REVERTED,
                           revert_target=target)
        else:
            pending["result"] = llm.RESULT_SUPERSEDED
    return original.get("text", "")


def mark_untested(changes: list) -> list:
    """Flags suggestions the driver never tried. Used when they correct the diagnosis instead of testing the fix."""
    for change in changes:
        if isinstance(change, dict) and change.get("result") == llm.RESULT_PENDING:
            change["applied"] = False
            change["result"] = llm.RESULT_UNTESTED
    return changes


def detect_car_class(text: str):
    """Finds a car class named anywhere in the driver's reply. Checked here rather than left to the extractor so a bare answer like "gt3" is reliably recognised as the class question being answered."""
    if not text:
        return None
    lowered = text.lower()
    for canonical, needles in CLASS_PATTERNS:
        for needle in needles:
            if needle in lowered:
                return canonical
    return None


def is_class_only(car) -> bool:
    """True when all we know is the class (e.g. "GT3"), not a model."""
    return bool(car) and car.strip().lower() in CLASS_NAMES


def car_confidence_note(car) -> str:
    """How certain the car context was, recorded alongside a confirmed fix so a guess doesn't enter shared knowledge looking verified."""
    if not car:
        return " | car unknown — generic advice, treat with low confidence"
    if is_class_only(car):
        return f" | class-level only ({car}), specific model not given"
    return ""


class SetupEngineer(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._thread_locks = weakref.WeakValueDictionary()

    def thread_lock(self, thread_id: int):
        """One conversation turn at a time, including intake and uploads."""
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_locks[thread_id] = lock
        return lock

    async def remember(self, thread_id: int, role: str, content: str):
        """Logs a message to the thread's memory. Failures are logged but never raised — losing one line of history is not worth breaking the driver's conversation over."""
        try:
            await db.add_message(thread_id, role, content)
        except Exception as e:
            log.error(f"Failed to log message for thread {thread_id}: {e}")

    async def recall(self, thread_id: int, session: dict = None):
        """Recent messages plus EVERY message not covered by the summary."""
        try:
            if session is None:
                session = await db.get_session(thread_id) or {}
            recent, unsummarised = await asyncio.gather(
                db.get_recent_messages(thread_id, HISTORY_WINDOW),
                db.get_messages_since(thread_id, session.get("summarised_upto") or 0),
            )
            messages = {m["id"]: m for m in recent + unsummarised}
            return [messages[key] for key in sorted(messages)]
        except Exception as e:
            log.error(f"Failed to load history for thread {thread_id}: {e}")
            return []

    async def previous_telemetry(self, thread_id: int):
        """The last telemetry upload in this thread, parsed back into a dict, or None. A failure here costs the comparison only — the new upload is still read and used on its own."""
        try:
            run = await db.get_last_telemetry_run(thread_id)
            if not run or not run.get("payload"):
                return None
            return json.loads(run["payload"])
        except Exception as e:
            log.error(f"Couldn't load previous telemetry for thread {thread_id}: {e}")
            return None

    async def store_telemetry_run(self, thread_id: int, car, track, data: dict):
        """Keeps this upload so the NEXT one has something to compare against. Never raised: losing the baseline for a future comparison is not worth failing the current upload over."""
        try:
            await db.add_telemetry_run(thread_id, car, track, json.dumps(data))
        except Exception as e:
            log.error(f"Couldn't store telemetry run for thread {thread_id}: {e}")

    async def find_knowledge(self, query: str, car: str = None):
        """Knowledge base lookup that never breaks the conversation. A retrieval failure should cost reference material, not the whole reply."""
        try:
            return await knowledge.search(query or "", car=car)
        except Exception as e:
            log.error(f"Knowledge search failed: {e}")
            return []

    async def refresh_summary(self, thread_id: int, session: dict, force: bool = False):
        """Folds anything new into the running summary. Called when a problem is confirmed fixed, or once enough messages have piled up — not every turn, since each refresh is an API call."""
        try:
            last_id = session.get("summarised_upto") or 0
            pending = await db.get_messages_since(thread_id, last_id)
            if not pending:
                return
            if not force and len(pending) < SUMMARY_EVERY:
                return

            updated = await llm.summarise_thread(session.get("summary"), pending)

            if updated is None:
                log.warning(
                    f"Summary refresh failed for thread {thread_id} — "
                    f"holding checkpoint, {len(pending)} message(s) still pending"
                )
                return

            await db.update_session(
                thread_id,
                summary=updated,
                summarised_upto=pending[-1]["id"],
            )
            log.info(f"Summary refreshed for thread {thread_id} ({len(pending)} new message(s))")
        except Exception as e:
            log.error(f"Failed to refresh summary for thread {thread_id}: {e}")

    async def say(self, message: discord.Message, text: str):
        """Replies and records what was said, so the bot's own words are part of the memory rather than lost."""
        for start in range(0, len(text), 1900):
            await message.reply(text[start:start + 1900])
        await self.remember(message.channel.id, "bot", text)

    async def starter_body(self, thread: discord.Thread) -> str:
        """The body of the post that opened the thread."""
        try:
            starter = thread.starter_message
            if starter is None:
                starter = await thread.fetch_message(thread.id)
            return (starter.content or "").strip()
        except Exception as e:
            log.warning(f"Couldn't read starter message for thread {thread.id}: {e}")
            return ""

    async def update(self, session: dict, **fields):
        await db.update_session(session['thread_id'], **fields)
        session.update(fields)

    @staticmethod
    def advice_request(session: dict) -> dict:
        try:
            request = json.loads(session.get('advice_request') or '{}')
            return request if isinstance(request, dict) else {}
        except (TypeError, ValueError):
            return {}

    async def pending_prompt(self, message):
        await self.say(
            message,
            "Before I pile another change on top — did you try the last one? "
            "Tell me how it went, or say you skipped it and I'll leave it out of the picture.",
        )

    async def request_advice(self, message, session, request, history, intro=''):
        """Persist the exact task before calling the model, then save usable advice."""
        changes = load_changes(session)
        if _last_pending(changes) is not None:
            await self.pending_prompt(message)
            return
        request = dict(request)
        request.setdefault('kind', 'diagnosis')
        request.setdefault('return_stage', 'awaiting_feedback')
        request.setdefault('last_suggestion', session.get('last_suggestion'))
        await self.update(
            session, stage='awaiting_advice', advice_request=json.dumps(request),
            changes_tried=json.dumps(changes), last_suggestion=None,
        )
        feedback = request.get('feedback', '')
        problem = session.get('problem') or ''
        directive = request.get('directive', '')
        if request.get('force'):
            directive += FORCE_DIAGNOSIS_DIRECTIVE
        if not session.get('car') and request.get('force'):
            directive += NO_CAR_DIRECTIVE
        elif is_class_only(session.get('car')):
            directive += CLASS_ONLY_DIRECTIVE
        kind = request['kind']
        query = f"{problem} {feedback}"
        if kind == 'pace':
            query = f"lap time pace {session.get('track') or ''} {feedback}"
        async with message.channel.typing():
            relevant = await self.find_knowledge(query, session.get('car'))
            call_history = list(history or [])
            if message.content and (not call_history or call_history[-1].get('role') != 'driver'
                                    or call_history[-1].get('content') != message.content):
                # Extraction retains car/track/problems, but may omit other
                # crucial details such as rain or a missing menu setting.
                call_history.append({'role': 'driver', 'content': message.content})
            common = dict(
                car=session.get('car'), track=session.get('track'),
                current_setup=session.get('current_setup'),
                telemetry=session.get('telemetry_summary'), changes_tried=changes,
                history=call_history, summary=session.get('summary'), knowledge=relevant,
            )
            if kind == 'followup':
                result = await llm.get_followup(
                    **common, problem=problem + directive,
                    last_suggestion=request.get('last_suggestion'),
                    feedback=feedback, partial=request.get('partial', False),
                )
            elif kind == 'post_revert':
                goal = problem if request['return_stage'] != 'tuning' else 'Improve lap time'
                result = await llm.get_post_revert_suggestion(
                    **common, problem=goal + directive,
                    reverted_change=request.get('reverted_change'),
                )
            elif kind == 'pace':
                result = await llm.get_pace_suggestion(
                    **common, problem=problem,
                    last_suggestion=request.get('last_suggestion'),
                    feedback=feedback + directive,
                )
            else:
                result = await llm.get_diagnosis(**common, problem=problem + directive)

        if result.get('ok') and result.get('is_question') and request.get('force'):
            # A model that ignores the clarification cap has not delivered
            # an adjustment. Do not present its question as a testable fix.
            result = llm._failed("I couldn't produce a usable adjustment yet — ask me to try again, human.")
        if (result.get('ok') and not result.get('is_question')
                and request.get('partial') and result.get('kind') == 'revert'):
            log.warning('Rejected undo advice after partial improvement in thread %s', session['thread_id'])
            result = llm._failed(
                "That adjustment helped, so keep it on the car, human. "
                "I couldn't find a suitable next change this time — ask me to try again."
            )
        if result.get('ok') and not result.get('is_question'):
            try:
                record_change(changes, result, problem=problem)
            except ValueError as error:
                log.error('Unusable advice in thread %s: %s', session['thread_id'], error)
                result = llm._failed("I couldn't turn that into a usable adjustment — ask me to try again, human.")

        if not result.get('ok'):
            await self.say(message, intro + result['text'])
            return
        if result.get('is_question'):
            await self.update(
                session, stage='awaiting_clarification', last_suggestion=result['text'],
                clarify_count=ROUND_GENERAL,
            )
            await self.say(message, intro + result['text'])
            return
        await self.update(
            session, stage=request['return_stage'], last_suggestion=result['text'],
            changes_tried=json.dumps(changes), advice_request=None, clarify_count=0,
        )
        await self.say(message, intro + result['text'] + FEEDBACK_PROMPT)

    async def answer(self, message, session, history, feedback_prompt=False):
        async with message.channel.typing():
            relevant = await self.find_knowledge(message.content, session.get('car'))
            result = await llm.answer_question(
                car=session.get('car'), track=session.get('track'),
                current_problem=session.get('problem'), question=message.content,
                knowledge=relevant, last_suggestion=session.get('last_suggestion'),
                current_setup=session.get('current_setup'), telemetry=session.get('telemetry_summary'),
                changes_tried=load_changes(session), history=history, summary=session.get('summary'),
            )
        suffix = FEEDBACK_PROMPT if feedback_prompt and result.get('ok') else ''
        await self.say(message, result['text'] + suffix)

    async def refine_problem(self, message, session, correction=False):
        """Keep the active complaint aligned with the adjustment it will earn."""
        if correction:
            context = (f"Original description: {session.get('problem')}\n"
                       f"Driver's correction: {message.content}")
        else:
            context = (
                f"Previous problem: {session.get('problem')}\n"
                f"Driver's result: {message.content}\n\n"
                "List only symptoms that remain or newly appeared. Do not list a symptom "
                "the driver says is resolved. If they give no new symptom detail, retain "
                "the previous problem."
            )
        car, track, extracted = await llm.extract_car_track_and_problems(context)
        has_refined_problems = bool(extracted) and extracted != [context]
        problems = extracted if has_refined_problems else [message.content or session.get('problem')]
        queue = load_problems(session)
        index = session.get('problem_index') or 0
        if index >= len(queue):
            queue, index = [], 0
        if queue:
            queue[index] = problems[0]
        else:
            queue = [problems[0]]
        for extra in problems[1:]:
            if extra and extra not in queue[index:]:
                queue.append(extra)
        await self.update(
            session, problem=problems[0], problems=json.dumps(queue), problem_index=index,
            car=car or session.get('car'), track=track or session.get('track'),
        )

    async def next_problem(self, message, session, history, confirmed=False):
        queue = load_problems(session)
        index = (session.get('problem_index') or 0) + 1
        if index >= len(queue):
            return False
        await self.update(session, problem=queue[index], problem_index=index)
        intro = (f"🏁 That's #{index} sorted. On to #{index + 1}: {queue[index]}\n\n"
                 if confirmed else f"On to #{index + 1}: {queue[index]}\n\n")
        await self.request_advice(message, session, {'kind': 'diagnosis'}, history, intro)
        return True

    async def confirmed(self, message, session, history, tested, tuning):
        if not tuning and tested:
            note = (
                f"[Patch {config.PATCH_VERSION}] Confirmed fix — "
                f"problem: {tested.get('problem') or session.get('problem')} | "
                f"fix that worked: {tested['text']}"
            )
            if session.get('telemetry_evidence'):
                note += f" | {session['telemetry_evidence']}"
            note += car_confidence_note(session.get('car'))
            try:
                await knowledge.add_document(
                    note, source_type='confirmed_fix', car=session.get('car'), track=session.get('track'),
                )
            except Exception:
                log.exception('Could not store confirmed fix for thread %s', session['thread_id'])
        if await self.next_problem(message, session, history, confirmed=True):
            return
        if tuning:
            await self.request_advice(message, session, {
                'kind': 'pace', 'return_stage': 'tuning', 'feedback': message.content,
            }, history)
        else:
            await self.update(session, stage='tuning', advice_request=None)
            await self.say(message, ALL_PROBLEMS_DONE)

    async def handle_feedback(self, message, session, history, result=None, tuning=False, handling_report=False):
        if result is None:
            result = await llm.interpret_reply(
                car=session.get('car'), track=session.get('track'),
                current_problem=session.get('problem'), last_suggestion=session.get('last_suggestion'),
                driver_reply=message.content, history=history,
            )
        if not result.get('ok', True):
            await self.say(message,
                "Something went wrong at my end and I couldn't read that properly — nothing's "
                "been recorded. Did the change make it **better**, **worse**, or **no change**?")
            return
        reply_type = result.get('type')
        if reply_type == 'question':
            await self.answer(message, session, history, feedback_prompt=not tuning)
            return
        changes = load_changes(session)
        pending = _last_pending(changes)

        if reply_type == 'skipped':
            # An unavailable setting or a skipped test has no result.
            # Release the pending slot without claiming it was on the car.
            if pending is None:
                await self.say(message, "There isn't an outstanding adjustment to skip, human. What do you want to work on?")
                return
            mark_untested(changes)
            await self.update(session, changes_tried=json.dumps(changes), last_suggestion=None)
            if tuning and await self.next_problem(message, session, history):
                return
            await self.request_advice(message, session, {
                'kind': 'pace' if tuning and not handling_report else 'diagnosis',
                'return_stage': 'tuning' if tuning and not handling_report else 'awaiting_feedback',
                'feedback': message.content,
                'directive': '\nThe driver skipped the previous adjustment: ' + message.content
                             + '\nIt was NOT tested. Give one usable alternative; respect any stated menu limit.',
            }, history)
            return

        if reply_type == 'revert_done':
            reverted = mark_reverted(changes)
            if not reverted:
                await self.say(message, "I can't identify an applied change to mark as put back. Which setting did you undo?")
                return
            await self.update(session, changes_tried=json.dumps(changes))
            await self.request_advice(message, session, {
                'kind': 'post_revert', 'reverted_change': reverted,
                'return_stage': 'tuning' if tuning and not handling_report else 'awaiting_feedback',
            }, history)
            return

        if reply_type == 'correction':
            await self.refine_problem(message, session, correction=True)
            mark_untested(changes)
            await self.update(session, changes_tried=json.dumps(changes))
            await self.request_advice(message, session, {
                'kind': 'pace' if tuning and not handling_report else 'diagnosis',
                'return_stage': 'tuning' if tuning and not handling_report else 'awaiting_feedback',
                'feedback': message.content, 'directive': CORRECTION_DIRECTIVE,
                'force': True,
            }, history)
            return

        if reply_type == 'new_problem':
            _, _, extracted = await llm.extract_car_track_and_problems(message.content)
            additions = extracted or [message.content]
            if tuning and pending is None:
                await self.start_handling(message, session, history)
                return
            queue = load_problems(session)
            for problem in additions:
                if problem not in queue:
                    queue.append(problem)
            await self.update(session, problems=json.dumps(queue))
            await self.say(message, f"Noted, human — that one goes on the list.{QUEUED_PROMPT}")
            return

        if reply_type not in RESULT_FOR_REPLY:
            await self.say(message, "I couldn't read that as a test result. Nothing's been recorded — can you clarify what you meant?")
            return
        if pending is None:
            # Also recovers old sessions stranded in feedback by a failed
            # diagnosis. There is no testable advice to attach this result to.
            await self.request_advice(message, session, {
                'kind': 'pace' if tuning else 'diagnosis',
                'return_stage': 'tuning' if tuning else 'awaiting_feedback',
                'feedback': message.content,
            }, history, "I don't have an outstanding test to attach that result to. Let me pick up the current problem.\n\n")
            return
        tested = dict(pending)
        record_result(changes, RESULT_FOR_REPLY[reply_type])
        await self.update(session, changes_tried=json.dumps(changes))
        if reply_type == 'confirmed_better':
            await self.confirmed(message, session, history, tested, tuning)
            return

        partial = reply_type == 'confirmed_partial'
        if tuning and not handling_report and not partial:
            await self.request_advice(message, session, {
                'kind': 'pace', 'return_stage': 'tuning', 'feedback': message.content,
            }, history)
            return
        await self.refine_problem(message, session)
        await self.request_advice(message, session, {
            'kind': 'followup', 'feedback': message.content, 'partial': partial,
        }, history)

    async def start_handling(self, message, session, history):
        car, track, problems = await llm.extract_car_track_and_problems(message.content)
        problems = problems or [message.content]
        await self.update(
            session, car=car or session.get('car'), track=track or session.get('track'),
            problem=problems[0], problems=json.dumps(problems), problem_index=0,
        )
        await self.request_advice(message, session, {
            'kind': 'diagnosis', 'directive': BROKEN_AGAIN_DIRECTIVE,
        }, history)

    async def handle_tuning(self, message, session, history):
        intent = await llm.tuning_intent(message.content)
        if intent == 'DONE':
            # Closure is permanent, so this only asks for confirmation.
            await self.update(session, stage='awaiting_close')
            await self.say(message, CONFIRM_CLOSE_MESSAGE)
            return
        if intent == 'QUESTION':
            await self.answer(message, session, history)
            return
        if intent == 'SKIPPED':
            await self.handle_feedback(message, session, history,
                                       result={'type': 'skipped', 'ok': True}, tuning=True)
            return
        if intent == 'FEEDBACK' or (intent == 'BROKEN' and _last_pending(load_changes(session))):
            # The second classifier's category is authoritative. Corrections,
            # questions and new symptoms must never default to a failed test.
            await self.handle_feedback(message, session, history, tuning=True,
                                       handling_report=intent == 'BROKEN')
            return
        if intent == 'BROKEN':
            # A new handling complaint takes priority over chasing pace.
            await self.start_handling(message, session, history)
            return
        if intent == 'MORE':
            if _last_pending(load_changes(session)):
                await self.pending_prompt(message)
                return
            if await self.next_problem(message, session, history):
                return
            await self.request_advice(message, session, {
                'kind': 'pace', 'return_stage': 'tuning', 'feedback': message.content,
            }, history)
            return
        await self.say(message, "Something went wrong reading that request — nothing's changed. Can you try again, human?")

    async def handle_clarification(self, message, session, history):
        context = f"{session.get('problem')}\nAdditional detail from driver: {message.content}"
        extracted_car, extracted_track, extracted = await llm.extract_car_track_and_problems(context)
        car = extracted_car or session.get('car') or detect_car_class(message.content)
        track = extracted_track or session.get('track')
        problem = extracted[0] if extracted else session.get('problem')
        problems = load_problems(session)
        index = session.get('problem_index') or 0
        if problems and index < len(problems):
            problems[index] = problem
        else:
            problems, index = [problem], 0
        for extra in extracted[1:]:
            if extra and extra not in problems:
                problems.append(extra)
        await self.update(session, car=car, track=track, problem=problem,
                          problems=json.dumps(problems), problem_index=index)
        rounds = session.get('clarify_count') or 0
        if not car and rounds == ROUND_GENERAL:
            # Ask for the class only. Keep the clarification stage so "GT3"
            # answers the missing-car question instead of entering feedback.
            await self.update(session, stage='awaiting_clarification', clarify_count=ROUND_CLASS,
                              last_suggestion=CLASS_QUESTION)
            await self.say(message, CLASS_QUESTION)
            return
        request = self.advice_request(session)
        request.setdefault('kind', 'diagnosis')
        request.setdefault('return_stage', 'awaiting_feedback')
        previous_feedback = request.get('feedback', '')
        request['feedback'] = f'{previous_feedback}\n{context}' if previous_feedback else context
        request['force'] = request.get('force', False) or rounds >= ROUND_GENERAL
        await self.request_advice(message, session, request, history)

    async def handle_attachments(self, message, session):
        for attachment in message.attachments:
            filename = attachment.filename.lower()
            try:
                if filename.endswith('.svm'):
                    raw = await attachment.read()
                    settings = svm_parser.parse_svm(raw.decode('utf-8-sig', errors='ignore'))
                    if not settings:
                        await self.say(message, "Hmm, couldn't read any settings from that file.")
                        continue
                    await self.update(session, current_setup=svm_parser.format_for_prompt(settings))
                    await self.say(message, f"📄 Got your setup file — read {len(settings)} settings. "
                                           "I'll factor this into what I suggest from here.")
                elif filename.endswith('.duckdb'):
                    raw = await attachment.read()
                    # Every operation, including download/write/cleanup, is
                    # inside the guard. A failed file doesn't skip later files.
                    with tempfile.TemporaryDirectory(prefix=f'phoenix-{message.id}-') as directory:
                        path = os.path.join(directory, 'telemetry.duckdb')
                        with open(path, 'wb') as stream:
                            stream.write(raw)
                        data = await asyncio.to_thread(telemetry_parser.parse_telemetry, path)
                    summary = telemetry_parser.format_for_diagnosis(data)
                    evidence = telemetry_parser.format_for_evidence(data)
                    # Read the old run BEFORE storing this upload, or the
                    # comparison silently becomes a comparison with itself.
                    previous = await self.previous_telemetry(session['thread_id'])
                    note = ''
                    if previous:
                        comparison = telemetry_parser.compare_runs(previous, data)
                        if comparison:
                            summary += '\n\n' + comparison
                        reason = telemetry_parser.comparison_unavailable_reason(previous, data)
                        note = f' No comparison: {reason}' if reason else ' Comparing it against your last upload.'
                    meta = data.get('metadata') or {}
                    await self.store_telemetry_run(session['thread_id'], meta.get('car'), meta.get('track'), data)
                    await self.update(session, telemetry_summary=summary, telemetry_evidence=evidence or None)
                    await self.say(message, f"📊 Got your telemetry — {len(data.get('lap_times', []))} lap(s) recorded."
                                           f"{note} I'll factor this into what I suggest from here.")
                else:
                    await self.say(message, f"I can't read `{attachment.filename}` as setup data. Upload an `.svm` setup or `.duckdb` telemetry file.")
            except Exception:
                log.exception('Upload failed in thread %s: %s', session['thread_id'], attachment.filename)
                await self.say(message, f"I couldn't read or save `{attachment.filename}`. Please upload it again.")

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        if not config.FORUM_CHANNEL_ID or thread.parent_id != int(config.FORUM_CHANNEL_ID):
            return
        async with self.thread_lock(thread.id):
            await db.create_session(thread.id, thread.owner_id)
            session = await db.get_session(thread.id)
            if session and session['stage'] != 'awaiting_intake':
                return
            if len(await self.starter_body(thread)) >= MIN_STARTER_LENGTH:
                return
            for attempt in range(5):
                try:
                    await thread.send(WELCOME_MESSAGE)
                    await self.remember(thread.id, 'bot', WELCOME_MESSAGE)
                    return
                except discord.Forbidden:
                    await asyncio.sleep(1)
            log.warning('Gave up trying to greet thread: %s', thread.name)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not isinstance(message.channel, discord.Thread):
            return
        thread = message.channel
        if not config.FORUM_CHANNEL_ID or thread.parent_id != int(config.FORUM_CHANNEL_ID):
            return
        async with self.thread_lock(thread.id):
            try:
                session = await db.get_session(thread.id)
                if not session:
                    # Starter-message and thread-created events may arrive in
                    # either order. Never drop an upload before a row exists.
                    await db.create_session(thread.id, thread.owner_id)
                    session = await db.get_session(thread.id)
                if not session:
                    raise RuntimeError('Session could not be created')
                if session['stage'] == 'closed':
                    await message.reply(CLOSED_MESSAGE)
                    return
                await self.handle_attachments(message, session)
                if not message.content.strip():
                    return
                # Read prior history before logging this message. Advice
                # receives the current driver message separately, once.
                history = await self.recall(thread.id, session)
                await self.remember(thread.id, 'driver', message.content)
                stage = session['stage']
                problem_index_before = session.get('problem_index') or 0
                if stage == 'awaiting_intake':
                    car, track, problems = await llm.extract_car_track_and_problems(message.content)
                    car = car or detect_car_class(message.content)
                    problems = problems or [message.content]
                    await self.update(session, car=car, track=track, problems=json.dumps(problems),
                                      problem_index=0, problem=problems[0])
                    intro = ''
                    if len(problems) > 1:
                        numbered = '\n'.join(f'{i+1}. {p}' for i, p in enumerate(problems))
                        intro = (f"Got it — I count {len(problems)} separate issues here. "
                                 f"I'll tackle them one at a time:\n{numbered}\n\nStarting with #1:\n\n")
                    await self.request_advice(message, session, {'kind': 'diagnosis'}, history, intro)
                elif stage == 'awaiting_advice':
                    request = self.advice_request(session) or {'kind': 'diagnosis'}
                    retry_history = history + [{'role': 'driver', 'content': message.content}]
                    await self.request_advice(message, session, request, retry_history)
                elif stage == 'awaiting_clarification':
                    await self.handle_clarification(message, session, history)
                elif stage == 'awaiting_feedback':
                    if not session.get('car') and detect_car_class(message.content):
                        # A late car/class answer is context we already
                        # asked for, not a new problem or a test result.
                        changes = mark_untested(load_changes(session))
                        await self.update(session, car=detect_car_class(message.content),
                                          changes_tried=json.dumps(changes))
                        await self.request_advice(message, session, {
                            'kind': 'diagnosis', 'force': True, 'directive': LATE_CAR_DIRECTIVE,
                        }, history + [{'role': 'driver', 'content': message.content}])
                    else:
                        await self.handle_feedback(message, session, history)
                elif stage == 'tuning':
                    await self.handle_tuning(message, session, history)
                elif stage == 'awaiting_close':
                    if is_yes(message.content):
                        await self.update(session, stage='closed', advice_request=None)
                        await self.say(message, DONE_MESSAGE)
                    else:
                        await self.update(session, stage='tuning')
                        await self.say(message, "Still going then, human. What's the car doing?")
                else:
                    await self.say(message, "I couldn't identify where this conversation stopped. Please ask an admin to check this thread.")
                await self.refresh_summary(thread.id, session,
                                           force=(session.get('problem_index') or 0) != problem_index_before
                                           or (session['stage'] in ('tuning', 'awaiting_close', 'closed')
                                               and stage != session['stage']))
            except Exception:
                log.exception('Conversation failed in thread %s', thread.id)
                await self.say(message, "Something went wrong at my end. Please try again — I haven't treated that error as a test result.")


async def setup(bot: commands.Bot):
    await bot.add_cog(SetupEngineer(bot))
