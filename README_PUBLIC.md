# Phoenix One — setup assistant (public code snapshot)

A Discord bot that acts as a virtual race engineer. Drivers open a thread,
describe what the car is doing wrong, and the bot diagnoses iteratively —
one change at a time, tracking whether each change worked before suggesting
another.

This is a **frozen snapshot of the code**, published so people can see how it
works and build their own. It is not maintained here and will not receive
updates. It is not a product, and it will not run usefully without the parts
described below.

---

## What is NOT in this repository

**The prompts.** Every prompt has been replaced with a placeholder. That
includes the system prompt that defines the assistant's persona and its
engineering rules, and the task prompts for extraction, classification,
summarisation and intent detection. The code that assembles context around
them is intact; the instructions themselves are not.

**The knowledge base.** The curated engineering documents, their structure
and their tagging are not included. The retrieval code, chunker and schema
are.

**The evaluation scenarios.** `eval/cases.py` is an empty shell. The
structural regression tests in `eval/test_regressions.py` are intact.

**Anything operational.** No credentials, no database, no deployment config.

---

## What IS in this repository

- The conversation state machine: pending, confirmed, partial, skipped,
  reverted, and the rules for what each transition does
- Per-thread concurrency control, so two messages arriving together cannot
  corrupt thread state
- Retrieval over pgvector, with section-aware chunking, per-document tagging
  and diversity capping on results
- Parsers for simulator setup exports and telemetry files
- Persistence layer and schema
- Admin commands for inspecting and correcting stored knowledge
- The regression suite covering the state machine and parsers

---

## Running it

You will need: a Discord bot token, a PostgreSQL database with `pgvector`, an
Anthropic API key, an OpenAI API key for embeddings, and somewhere to host it.

You will also need to write your own prompts and supply your own knowledge
documents. Those are the parts that determine whether the assistant gives good
advice, and they are the parts not published here.

---

## Licence and attribution

Provided as-is, with no warranty and no support. If you build something from
it, an acknowledgement is appreciated but not required.

*Phoenix One is a sim racing team focused on accessibility and driver
development.*
