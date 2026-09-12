"""This file's only job: read your secret keys/settings from environment variables (set in Railway's dashboard, or from a local .env file if you're running on your own machine) and make them available to the rest of the bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Discord settings ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
GUILD_ID = os.getenv("GUILD_ID")  # your server's ID, used to sync commands instantly
FORUM_CHANNEL_ID = os.getenv("FORUM_CHANNEL_ID")  # the forum channel where drivers post their problems
KNOWLEDGE_CHANNEL_ID = os.getenv("KNOWLEDGE_CHANNEL_ID")

# --- Anthropic (Claude) settings ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

ANTHROPIC_FAST_MODEL = os.getenv("ANTHROPIC_FAST_MODEL") or ANTHROPIC_MODEL

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# Kept so an older deploy can still be rolled back to without editing
# this file. Not used while OPENAI_API_KEY is set.
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

PATCH_VERSION = os.getenv("PATCH_VERSION", "unknown")

if not DISCORD_TOKEN:
    raise RuntimeError(
        "DISCORD_TOKEN is missing. Set it in Railway's Variables tab "
        "(or in a local .env file if running on your own machine)."
    )
