"""Watches a dedicated admin-only channel. Anything posted there gets chunked, embedded, and stored in the knowledge base automatically."""

import logging
import os
import re

import discord
from discord.ext import commands
import duckdb

import config
from engine import knowledge, svm_parser
from storage import db

log = logging.getLogger("phoenix-one")

PROGRESS_THRESHOLD = 40

TAG_PATTERN = re.compile(
    r"^car:\s*(?P<car>.*?)\s*\|\s*track:\s*(?P<track>.*?)\s*\n(?P<rest>.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_tags(content: str):
    match = TAG_PATTERN.match(content.strip())
    if match:
        return match.group("car").strip(), match.group("track").strip(), match.group("rest").strip()
    return None, None, content.strip()


def inspect_duckdb(path: str) -> str:
    """Connects to a .duckdb file and returns a plain-text report of its real table and column structure — no assumptions, just what's actually there. Table names may contain spaces (e.g. "Ambient Temperature"), so they're always double-quoted in queries."""
    lines = []
    con = duckdb.connect(path, read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        lines.append(f"Tables found: {len(table_names)}")
        lines.append("All table names:")
        lines.append(", ".join(table_names))

        # Full metadata dump (small table, worth seeing all of it) —
        # this is where car/track/session info likely lives.
        if "metadata" in table_names:
            lines.append('\n=== "metadata" (FULL) ===')
            meta_rows = con.execute('SELECT * FROM "metadata"').fetchall()
            for row in meta_rows:
                lines.append(f"  {row}")

        priority = [
            "channelsList", "eventsList",
            "Lap Time", "Lap Dist", "GPS Speed", "Throttle Pos",
            "Brake Pos", "Steering Pos", "G Force Lat",
            "TyresPressure", "TyresCarcassTemp",
        ]
        sample_tables = [t for t in priority if t in table_names]
        for table_name in sample_tables:
            safe_name = table_name.replace('"', '""')
            lines.append(f'\n=== "{table_name}" ===')
            try:
                columns = con.execute(f'DESCRIBE "{safe_name}"').fetchall()
                for col in columns:
                    lines.append(f"  {col}")
            except Exception as e:
                lines.append(f"  (describe failed: {e})")
            try:
                count = con.execute(f'SELECT COUNT(*) FROM "{safe_name}"').fetchone()[0]
                lines.append(f"  (rows: {count})")
            except Exception as e:
                lines.append(f"  (row count failed: {e})")
            try:
                sample = con.execute(f'SELECT * FROM "{safe_name}" LIMIT 3').fetchall()
                lines.append(f"  sample rows: {sample}")
            except Exception as e:
                lines.append(f"  (sample fetch failed: {e})")
    finally:
        con.close()
    return "\n".join(lines)


class KnowledgeAdmin(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def ingest_text(self, message: discord.Message, text: str, label: str) -> int:
        """Chunks and stores a text document, reporting progress in the channel while it works. Returns the number of chunks stored."""
        car, track, content = parse_tags(text)
        expected = knowledge.count_chunks(content)

        if expected == 0:
            await message.reply(
                f"`{label}` contained no readable text — nothing to store."
            )
            return 0

        status = None
        if expected >= PROGRESS_THRESHOLD:
            status = await message.reply(
                f"Ingesting `{label}` — {expected} chunks. This will take a moment."
            )

        async def report(done, total):
            if status and (done % 200 == 0 or done == total):
                try:
                    await status.edit(
                        content=f"Ingesting `{label}` — {done}/{total} chunks stored."
                    )
                except Exception:
                    pass  # editing is a nicety, never worth failing over

        stored = await knowledge.add_document(
            content,
            source_type="document",
            car=car,
            track=track,
            progress=report,
        )

        tag_note = ""
        if car or track:
            tag_note = f" (tagged car={car or 'none'}, track={track or 'none'})"

        if stored == 0:
            await message.reply(
                f"⚠️ `{label}` produced {expected} chunks but stored none. "
                f"Check the logs — this usually means the embedding service "
                f"isn't configured or rejected the request."
            )
            return 0

        summary = f"✅ `{label}` — stored {stored} chunk(s).{tag_note}"
        if stored != expected:
            summary = (
                f"⚠️ `{label}` — stored {stored} of {expected} expected chunk(s). "
                f"Some batches failed; check the logs before relying on this document."
            )

        if status:
            try:
                await status.edit(content=summary)
                return stored
            except Exception:
                pass  # fall through and send a fresh message instead

        await message.reply(summary)
        return stored

    async def show_retrieval(self, message: discord.Message, query: str):
        """Shows exactly which chunks a query retrieves, and from which pillar."""
        car = None
        match = re.search(r"car\s*=\s*(.+)$", query, re.IGNORECASE)
        if match:
            car = match.group(1).strip()
            query = query[: match.start()].strip()

        try:
            if hasattr(knowledge, "search"):
                results = await knowledge.search(query, car=car)
            else:
                available = [n for n in dir(knowledge) if not n.startswith("_")]
                await message.reply(
                    f"⚠️ `knowledge.search` isn't there. Module exposes: "
                    f"`{', '.join(available)}`. Falling back to a direct lookup."
                )
                embeddings = await knowledge._embed_batch([query])
                results = await db.search_knowledge(
                    embeddings[0],
                    car=car,
                    car_class=knowledge.detect_class(car) if car else None,
                )
        except Exception as e:
            log.error(f"Retrieval inspection failed: {e}", exc_info=True)
            await message.reply(f"Lookup failed: `{type(e).__name__}: {e}`")
            return

        if not results:
            await message.reply("Nothing came back for that.")
            return

        header = f"**Retrieved for** `{query[:80]}`"
        if car:
            header += f" *(car: {car})*"
        lines = [header, ""]
        for i, row in enumerate(results, start=1):
            pillar = row.get("pillar") or row.get("source_type") or "?"
            content = " ".join(row["content"].split())
            lines.append(f"**{i}. [{pillar}]** {content[:220]}")
            lines.append("")

        text = "\n".join(lines)
        for i in range(0, len(text), 1900):
            await message.reply(text[i:i + 1900])

    async def find_chunks(self, message: discord.Message, needle: str):
        """Shows stored chunks containing some text, with their ids, so something that shouldn't be in the knowledge base can be found and removed."""
        try:
            rows = await db.find_chunks(needle)
        except Exception as e:
            log.error(f"Chunk search failed: {e}", exc_info=True)
            await message.reply(f"Search failed: `{type(e).__name__}: {e}`")
            return

        if not rows:
            await message.reply(f"Nothing stored contains `{needle[:60]}`.")
            return

        lines = [f"**{len(rows)} chunk(s) containing** `{needle[:60]}`:", ""]
        for row in rows:
            preview = row["preview"].replace("\n", " ")
            lines.append(
                f"`{row['id']}` [{row['source_type']}] {row['len']} chars — {preview}"
            )
        lines.append("")
        lines.append("Remove one with `!forget <id>`.")

        text = "\n".join(lines)
        for i in range(0, len(text), 1900):
            await message.reply(text[i:i + 1900])

    async def forget_chunk(self, message: discord.Message, chunk_id: int):
        """Deletes one chunk by id."""
        try:
            gone = await db.delete_chunk(chunk_id)
        except Exception as e:
            log.error(f"Chunk delete failed: {e}", exc_info=True)
            await message.reply(f"Delete failed: `{type(e).__name__}: {e}`")
            return

        if gone:
            log.info(f"Deleted knowledge chunk {chunk_id}")
            await message.reply(f"✅ Chunk `{chunk_id}` deleted.")
        else:
            await message.reply(f"No chunk with id `{chunk_id}` — nothing deleted.")

    async def report_pillars(self, message: discord.Message):
        """Shows what a pillar backfill WOULD write. Reads only."""
        try:
            by_pillar, by_document, untagged, other = await db.preview_pillar_backfill()
        except Exception as e:
            log.error(f"Pillar preview failed: {e}", exc_info=True)
            await message.reply(
                f"Couldn't work out the pillars: `{type(e).__name__}: {e}`\n"
                f"Nothing was written."
            )
            return

        lines = ["**Pillar backfill — preview only, nothing written.**", "", "__By pillar__"]
        for name, count in by_pillar:
            lines.append(f"  {name}: {count}")

        if other:
            lines.append("")
            lines.append("__Not documents (labelled from source_type)__")
            for name, count in other:
                lines.append(f"  {name}: {count}")

        lines.append("")
        if untagged:
            lines.append(
                f"⚠️ {untagged} document chunk(s) would get NO pillar. Those sit "
                f"before the first KNOWLEDGE_ID header in the table."
            )
        else:
            lines.append("✅ Every document chunk resolves to a pillar.")

        lines.append("")
        lines.append("__By document__ (check these against the upload counts)")
        for name, count in by_document:
            lines.append(f"  {name}: {count}")

        text = "\n".join(lines)
        # Discord rejects anything over 2000 characters.
        for i in range(0, len(text), 1900):
            await message.reply(text[i:i + 1900])

    async def apply_pillars(self, message: discord.Message):
        """Fills missing legacy pillars, preserving existing ingestion tags."""
        try:
            updated = await db.apply_pillar_backfill()
        except Exception as e:
            log.error(f"Pillar backfill failed: {e}", exc_info=True)
            await message.reply(
                f"Backfill failed: `{type(e).__name__}: {e}`\n"
                f"It runs in a transaction, so nothing was changed."
            )
            return
        log.info(f"Pillar backfill applied to {updated} chunk(s)")
        await message.reply(
            f"✅ Missing pillar filled for {updated} document chunk(s). "
            f"Run `!pillars` again to see the result."
        )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not config.KNOWLEDGE_CHANNEL_ID:
            return
        if message.channel.id != int(config.KNOWLEDGE_CHANNEL_ID):
            return

        command = message.content.strip().lower()
        if command.startswith("!"):
            words = command.lstrip("!").split()
            head = words[0] if words else ""
            rest = words[1] if len(words) > 1 else ""

            if head in ("pillar", "pillars"):
                if rest in ("", "check", "preview"):
                    await self.report_pillars(message)
                    return
                if rest == "apply":
                    await self.apply_pillars(message)
                    return
                await message.reply(
                    "Not sure what you want doing with the pillars. "
                    "`!pillars` to check, `!pillars apply` to write them. "
                    "Nothing was stored."
                )
                return

            if head in ("recall", "retrieve"):
                query = message.content.strip()[len(words[0]) + 1:].strip()
                if not query:
                    await message.reply(
                        "Give me something to look up: `!recall rear spins on "
                        "power out of slow corner`. Add `car=Example Car 02` at the "
                        "end to see car-aware ranking."
                    )
                    return
                await self.show_retrieval(message, query)
                return

            if head in ("find", "search"):
                needle = message.content.strip()[len(words[0]) + 1:].strip()
                if not needle:
                    await message.reply("Give me something to look for: `!find some text`")
                    return
                await self.find_chunks(message, needle)
                return

            if head in ("forget", "delete"):
                if not rest.isdigit():
                    await message.reply(
                        "Give me a chunk id: `!forget 1234`. "
                        "Use `!find <text>` to get the id first."
                    )
                    return
                await self.forget_chunk(message, int(rest))
                return

            await message.reply(
                f"Don't know the command `{command[:60]}`. Nothing was stored. "
                f"Known commands: `!pillars`, `!pillars apply`, `!find <text>`, "
                f"`!forget <id>`, `!recall <query>`."
            )
            return

        total_chunks = 0
        handled_special = False
        reported = False

        # Plain message text, when there is any beyond an attachment.
        if message.content.strip():
            try:
                car, track, content = parse_tags(message.content)
                total_chunks += await knowledge.add_document(
                    content, source_type="manual", car=car, track=track
                )
            except Exception as e:
                log.error(f"Failed to ingest message text: {e}", exc_info=True)
                await message.reply(f"Couldn't store that text: `{type(e).__name__}: {e}`")
                return

        for attachment in message.attachments:
            filename = attachment.filename.lower()
            log.info(
                f"Attachment received: {attachment.filename!r} "
                f"({attachment.size} bytes)"
            )

            try:
                if filename.endswith(".svm"):
                    handled_special = True
                    raw = await attachment.read()
                    text = raw.decode("utf-8", errors="ignore")
                    settings = svm_parser.parse_svm(text)
                    if settings:
                        if message.content.strip():
                            car, track, _ = parse_tags(message.content)
                        else:
                            car, track = None, None
                        summary = svm_parser.format_for_prompt(settings)
                        stored = await knowledge.add_document(
                            summary, source_type="svm", car=car, track=track
                        )
                        total_chunks += stored
                        await message.reply(
                            f"✅ `{attachment.filename}` — read {len(settings)} "
                            f"settings, stored {stored} chunk(s)."
                        )
                    else:
                        await message.reply(
                            f"Couldn't read any settings from `{attachment.filename}`."
                        )
                    reported = True

                elif filename.endswith(".duckdb"):
                    handled_special = True
                    raw = await attachment.read()
                    tmp_path = f"/tmp/{message.id}.duckdb"
                    with open(tmp_path, "wb") as f:
                        f.write(raw)
                    try:
                        report = inspect_duckdb(tmp_path)
                        log.info(f"Inspected .duckdb from message {message.id}")
                        # Discord messages cap at 2000 chars — split up
                        for i in range(0, len(report), 1800):
                            await message.reply(f"```\n{report[i:i + 1800]}\n```")
                        reported = True
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                else:
                    raw = await attachment.read()
                    text = raw.decode("utf-8", errors="ignore")
                    if not text.strip():
                        await message.reply(
                            f"`{attachment.filename}` didn't decode as text "
                            f"({attachment.size} bytes read). If it's a binary "
                            f"file, it can't be stored as knowledge."
                        )
                        reported = True
                        continue
                    total_chunks += await self.ingest_text(
                        message, text, attachment.filename
                    )
                    reported = True

            except Exception as e:
                log.error(
                    f"Failed to process attachment {attachment.filename!r}: {e}",
                    exc_info=True,
                )
                await message.reply(
                    f"Something went wrong with `{attachment.filename}`:\n"
                    f"`{type(e).__name__}: {e}`"
                )
                return

        if reported:
            return
        if total_chunks:
            await message.reply(f"✅ Ingested — stored as {total_chunks} chunk(s).")
        elif not handled_special:
            await message.reply(
                "Nothing to ingest — post text or attach a file. "
                "(Check the logs if you attached something and see this.)"
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(KnowledgeAdmin(bot))
