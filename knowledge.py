"""Handles turning text into embeddings and storing/searching them in the knowledge base."""

import asyncio
import logging
import re
import time

from openai import AsyncOpenAI

import config
from storage import db

log = logging.getLogger("phoenix-one")

_client = None

CHUNK_SIZE = 500        # fallback size for unstructured text (no section markers)
MAX_SECTION_SIZE = 1500  # safety cap — an oversized section still gets sub-split
MIN_SECTION_SIZE = 60    # pieces shorter than this get merged into the next one

RESULT_LIMIT = 5

EMBED_BATCH_TOKENS = 100_000  # estimated tokens per request
EMBED_BATCH_MAX = 256         # texts per request regardless of size
EMBED_MIN_INTERVAL = 0.2      # seconds between requests
EMBED_MAX_RETRIES = 5         # attempts per batch before giving up
EMBED_BACKOFF = 5.0           # base seconds to wait after a rate-limit rejection

CHARS_PER_TOKEN = 4

# Matches a line that is ONLY equals signs, four or more. The minimum
# length is deliberate — see the note at the top of this file.
SECTION_MARKER = re.compile(r"^\s*={4,}\s*$", re.MULTILINE)
KNOWLEDGE_HEADER = re.compile(r"\bKNOWLEDGE_ID:\s*([A-Za-z0-9_]+)", re.IGNORECASE)

# Kept in step with CLASS_PATTERNS in cogs/setup_engineer.py. Duplicated
# rather than imported to avoid the engine depending on a cog.
CLASS_PATTERNS = [
    ("Hypercar", ("hypercar", "hyper car", "lmh", "lmdh")),
    ("LMP2", ("lmp2", "lmp 2")),
    ("LMP3", ("lmp3", "lmp 3")),
    ("GTE", ("gte", "gt e")),
    ("GT3", ("gt3", "gt 3")),
]

# When the last embedding request went out, so pacing survives across
# separate add_document calls rather than resetting each time.
_last_request_at = 0.0


def detect_class(car: str):
    """The class named in a car string, if any. "Example Car 01" gives "GT3", so a chunk tagged only with the class still matches a driver who named a specific model."""
    if not car:
        return None
    lowered = car.lower()
    for canonical, needles in CLASS_PATTERNS:
        for needle in needles:
            if needle in lowered:
                return canonical
    return None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _estimate_tokens(text: str) -> int:
    """Rough token count for batch sizing."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _batch_chunks(chunks: list) -> list:
    """Groups chunks into requests that stay under the per-request token budget and count cap. A single chunk larger than the budget goes out on its own rather than being dropped."""
    batches = []
    current = []
    current_tokens = 0
    for chunk in chunks:
        tokens = _estimate_tokens(chunk)
        too_many = len(current) >= EMBED_BATCH_MAX
        too_big = current and (current_tokens + tokens) > EMBED_BATCH_TOKENS
        if too_many or too_big:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


async def _embed_batch(texts: list) -> list:
    """One embedding request, paced against the rate limit and retried with backoff if rejected anyway. Returns a list of vectors in the same order as the input."""
    global _last_request_at
    client = _get_client()

    for attempt in range(EMBED_MAX_RETRIES):
        elapsed = time.monotonic() - _last_request_at
        if elapsed < EMBED_MIN_INTERVAL:
            await asyncio.sleep(EMBED_MIN_INTERVAL - elapsed)

        try:
            _last_request_at = time.monotonic()
            response = await client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            name = type(e).__name__
            is_rate_limit = "RateLimit" in name or "rate limit" in str(e).lower()
            if not is_rate_limit or attempt == EMBED_MAX_RETRIES - 1:
                raise
            wait = EMBED_BACKOFF * (attempt + 1)
            log.warning(
                f"Embedding rate limit hit (attempt {attempt + 1}/"
                f"{EMBED_MAX_RETRIES}), waiting {wait:.0f}s"
            )
            await asyncio.sleep(wait)

    raise RuntimeError("Embedding failed after retries")


def _chunk_text(text: str) -> list[str]:
    """Splits text into chunks for embedding. Prefers splitting on the section dividers used in structured knowledge docs, so each numbered section stays intact as one chunk instead of being cut mid-sentence by blind character counting. Falls back to plain character-based chunking for unstructured text..."""
    text = text.strip()
    if not text:
        return []

    pieces = [p.strip() for p in SECTION_MARKER.split(text)]
    pieces = [p for p in pieces if p]

    # No section markers found — unstructured text, use the simple
    # character-based approach.
    if len(pieces) <= 1:
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    merged = []
    carry = ""
    for piece in pieces:
        combined = f"{carry}\n\n{piece}".strip() if carry else piece
        if len(combined) < MIN_SECTION_SIZE:
            carry = combined
            continue
        merged.append(combined)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{carry}".strip()
        else:
            merged.append(carry)

    # Safety cap: sub-split any single section that's still very large.
    chunks = []
    for section in merged:
        if len(section) <= MAX_SECTION_SIZE:
            chunks.append(section)
        else:
            for i in range(0, len(section), MAX_SECTION_SIZE):
                chunks.append(section[i:i + MAX_SECTION_SIZE])

    return chunks


def count_chunks(content: str) -> int:
    """How many chunks this content will produce, without embedding anything. Lets a caller report the size of an upload before it starts."""
    return len(_chunk_text(content))


def chunk_pillars(chunks: list[str], source_type: str) -> list[str]:
    """Carry a document's pillar across its sections within this upload."""
    current = source_type.upper()
    pillars = []
    for chunk in chunks:
        header = KNOWLEDGE_HEADER.search(chunk)
        if header and source_type in ("document", "manual"):
            current = header.group(1).split("_", 1)[0].upper()
        pillars.append(current)
    return pillars


async def add_document(content: str, source_type: str, car: str = None, track: str = None, progress=None):
    """Chunks, embeds, and stores a piece of knowledge."""
    if not config.OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — skipping ingestion")
        return 0

    chunks = _chunk_text(content)
    if not chunks:
        return 0

    batches = _batch_chunks(chunks)
    pillars = chunk_pillars(chunks, source_type)
    log.info(
        f"Ingesting {len(chunks)} chunk(s) as {source_type} "
        f"in {len(batches)} batch(es)"
    )

    stored = 0
    for i, batch in enumerate(batches, start=1):
        embeddings = await _embed_batch(batch)
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding service returned an incomplete batch")
        for chunk, embedding in zip(batch, embeddings):
            await db.add_knowledge_chunk(
                source_type, car, track, chunk, embedding, pillar=pillars[stored]
            )
            stored += 1
        log.info(f"  batch {i}/{len(batches)} stored ({stored}/{len(chunks)} chunks)")
        if progress:
            try:
                await progress(stored, len(chunks))
            except Exception as e:
                log.warning(f"Progress callback failed: {e}")

    log.info(f"Ingested {stored} chunk(s) as {source_type}")
    return stored


async def search(query: str, limit: int = RESULT_LIMIT, car: str = None) -> list[dict]:
    """Finds the most relevant stored knowledge for a driver's problem."""
    if not config.OPENAI_API_KEY:
        return []

    embeddings = await _embed_batch([query])
    query_embedding = embeddings[0]

    return await db.search_knowledge(
        query_embedding,
        limit=limit,
        car=car,
        car_class=detect_class(car),
    )
