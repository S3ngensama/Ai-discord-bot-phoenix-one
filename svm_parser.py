"""Parses the simulator .svm setup files."""

import re

LINE_PATTERN = re.compile(r"^(\w+)\s*=\s*(-?\d+)\s*//\s*(.+)$")
SECTION_PATTERN = re.compile(r"^\[([^\]]+)\]\s*(?://.*)?$")


def parse_svm(raw_text: str) -> dict:
    """Returns section-qualified settings with their real display values."""
    settings = {}
    section = ""
    for line in raw_text.splitlines():
        header = SECTION_PATTERN.match(line.strip())
        if header:
            section = header.group(1).strip()
            continue
        match = LINE_PATTERN.match(line.strip())
        if not match:
            continue
        name, index, value = match.groups()
        key = f"{section}.{name}" if section else name
        settings[key] = {"index": int(index), "value": value.strip()}
    return settings


def format_for_prompt(settings: dict) -> str:
    """Turns parsed settings into a short, readable summary for Claude — only real, observed values, nothing invented."""
    if not settings:
        return ""
    lines = [f"{name}: {info['value']}" for name, info in settings.items()]
    return "Driver's current setup (from uploaded .svm file):\n" + "\n".join(lines)
