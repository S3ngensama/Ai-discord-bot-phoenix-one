"""This is the file that actually starts the bot."""

import asyncio
import logging

import discord
from discord.ext import commands

import config
from storage import db

# Basic logging so you can see what the bot is doing
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("phoenix-one")

# "Intents" tell Discord which kinds of events your bot wants to receive.
# message_content is required to read what people type in ticket threads.
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# List of cogs (feature files) to load. As we build more steps, we'll add
# to this list — nothing else in this file needs to change.
INITIAL_COGS = [
    "cogs.setup_engineer",
    "cogs.knowledge_admin",
]


@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (id: {bot.user.id})")

    # Sync slash commands to your specific server (instant) rather than
    # globally (which can take up to an hour to show up everywhere).
    if config.GUILD_ID:
        guild = discord.Object(id=int(config.GUILD_ID))
        bot.tree.copy_global_to(guild=guild)
        synced = await bot.tree.sync(guild=guild)
        log.info(f"Synced {len(synced)} slash command(s) to guild {config.GUILD_ID}")
    else:
        synced = await bot.tree.sync()
        log.info(f"Synced {len(synced)} slash command(s) globally")


async def main():
    await db.init_db()
    log.info("Database connected")
    async with bot:
        for cog in INITIAL_COGS:
            await bot.load_extension(cog)
            log.info(f"Loaded cog: {cog}")
        await bot.start(config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
