"""Parses the simulator's native .duckdb telemetry export."""

import json
import logging

import duckdb

log = logging.getLogger("phoenix-one")

WHEEL_LABELS = ["FL", "FR", "RL", "RR"]  # assumed order — see module docstring

# Setup keys worth surfacing in human-readable form, mapped to a short label.
SETUP_KEYS = {
    "VM_BRAKE_BALANCE": "Brake balance",
    "VM_FRONT_ANTISWAY": "Front ARB",
    "VM_REAR_ANTISWAY": "Rear ARB",
    "VM_FRONT_TOEIN": "Front toe",
    "VM_REAR_TOEIN": "Rear toe",
    "VM_FRONT_WING": "Front wing",
    "VM_REAR_WING": "Rear wing",
    "VM_DIFF_PRELOAD": "Diff preload",
    "VM_TRACTIONCONTROLMAP": "Traction control",
    "WM_CAMBER-W_FL": "Camber FL",
    "WM_CAMBER-W_FR": "Camber FR",
    "WM_CAMBER-W_RL": "Camber RL",
    "WM_CAMBER-W_RR": "Camber RR",
    "WM_PRESSURE-W_FL": "Tire pressure FL",
    "WM_PRESSURE-W_FR": "Tire pressure FR",
    "WM_PRESSURE-W_RL": "Tire pressure RL",
    "WM_PRESSURE-W_RR": "Tire pressure RR",
    "WM_SPRING-W_FL": "Spring FL",
    "WM_SPRING-W_RL": "Spring RL",
    "WM_RIDEHEIGHT-W_FL": "Ride height FL",
    "WM_RIDEHEIGHT-W_RL": "Ride height RL",
}

SUMMARY_CHANNELS = [
    "Throttle Pos",
    "Brake Pos",
    "Steering Pos",
    "G Force Lat",
    "G Force Long",
    "G Force Vert",
    "Ground Speed",
    "FrontRideHeight",
    "RearRideHeight",
    "Front3rdDeflection",
    "Fuel Level",
]

# 4-wheel channels worth summarizing per-wheel. The three tread
# temperatures are read separately on purpose — see the module docstring.
WHEEL_CHANNELS = [
    "TyresPressure",
    "TyresCarcassTemp",
    "TyresRimTemp",
    "TyresTempLeft",
    "TyresTempCentre",
    "TyresTempRight",
    "Tyres Wear",
]

# The three tread positions, used to derive a spread per wheel.
TREAD_CHANNELS = ["TyresTempLeft", "TyresTempCentre", "TyresTempRight"]

COMPARE_CHANNELS = {
    "Ground Speed": 1.0,
    "FrontRideHeight": 0.5,
    "RearRideHeight": 0.5,
    "Front3rdDeflection": 0.5,
    "Brake Pos": 1.0,
    "Throttle Pos": 1.0,
    "Steering Pos": 1.0,
    "G Force Lat": 0.05,
    "G Force Long": 0.05,
}

# Per-wheel channels worth diffing, with the same noise floor idea.
COMPARE_WHEEL_CHANNELS = {
    "TyresPressure": 1.0,
    "TyresCarcassTemp": 2.0,
    "TyresTempLeft": 2.0,
    "TyresTempCentre": 2.0,
    "TyresTempRight": 2.0,
}


def _safe(name: str) -> str:
    return name.replace('"', '""')


def _round(value, places=2):
    return round(value, places) if value is not None else None


def parse_telemetry(path: str) -> dict:
    """Reads a .duckdb file and returns a structured summary — real aggregates computed in SQL, not raw row dumps."""
    con = duckdb.connect(path, read_only=True)
    try:
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        result = {
            "metadata": {},
            "setup": {},
            "setup_range": {},
            "lap_times": [],
            "channel_stats": {},
            "wheel_stats": {},
            "tread_spread": {},
            "rake": {},
        }

        if "metadata" in tables:
            rows = con.execute('SELECT key, value FROM "metadata"').fetchall()
            meta = dict(rows)
            result["metadata"] = {
                "track": meta.get("TrackName"),
                "car": meta.get("CarName"),
                "car_class": meta.get("CarClass"),
                "driver": meta.get("DriverName"),
                "session_type": meta.get("SessionType"),
                "weather": meta.get("WeatherConditions"),
            }
            car_setup_raw = meta.get("CarSetup")
            if car_setup_raw:
                try:
                    setup_json = json.loads(car_setup_raw)
                    for key, label in SETUP_KEYS.items():
                        entry = setup_json.get(key)
                        if not entry or "stringValue" not in entry:
                            continue
                        result["setup"][label] = entry["stringValue"]

                        index = entry.get("value")
                        low = entry.get("minValue")
                        high = entry.get("maxValue")
                        if index is None or low is None or high is None:
                            continue
                        limit = None
                        if index <= low:
                            limit = "AT MINIMUM — cannot go lower"
                        elif index >= high:
                            limit = "AT MAXIMUM — cannot go higher"
                        result["setup_range"][label] = {
                            "index": index,
                            "min": low,
                            "max": high,
                            "limit": limit,
                        }
                except Exception as e:
                    log.error(f"Failed to parse CarSetup JSON: {e}")

        if "Lap Time" in tables:
            try:
                rows = con.execute('SELECT value FROM "Lap Time" WHERE value > 0 ORDER BY ts').fetchall()
                result["lap_times"] = [round(r[0], 3) for r in rows]
            except Exception as e:
                log.error(f"Failed to read Lap Time: {e}")

        for channel in SUMMARY_CHANNELS:
            if channel not in tables:
                continue
            safe_name = _safe(channel)
            try:
                row = con.execute(
                    f'SELECT AVG(value), MIN(value), MAX(value) FROM "{safe_name}"'
                ).fetchone()
                if row and row[0] is not None:
                    result["channel_stats"][channel] = {
                        "avg": round(row[0], 2),
                        "min": round(row[1], 2),
                        "max": round(row[2], 2),
                    }
            except Exception as e:
                log.error(f"Failed to summarize {channel}: {e}")

        for channel in WHEEL_CHANNELS:
            if channel not in tables:
                continue
            safe_name = _safe(channel)
            try:
                row = con.execute(
                    f'SELECT AVG(value1), AVG(value2), AVG(value3), AVG(value4) FROM "{safe_name}"'
                ).fetchone()
                if row and row[0] is not None:
                    result["wheel_stats"][channel] = {
                        label: round(val, 2) for label, val in zip(WHEEL_LABELS, row)
                    }
            except Exception as e:
                log.error(f"Failed to summarize {channel}: {e}")

        result["tread_spread"] = _tread_spread(result["wheel_stats"])
        result["rake"] = _rake(result["channel_stats"])

        return result
    finally:
        con.close()


def _tread_spread(wheel_stats: dict) -> dict:
    """Hottest minus coldest across the three tread positions, per wheel."""
    available = [c for c in TREAD_CHANNELS if c in wheel_stats]
    if len(available) < 2:
        return {}

    spread = {}
    for wheel in WHEEL_LABELS:
        values = [
            wheel_stats[channel][wheel]
            for channel in available
            if wheel in wheel_stats[channel]
        ]
        if len(values) >= 2:
            spread[wheel] = round(max(values) - min(values), 2)
    return spread


def _rake(channel_stats: dict) -> dict:
    """Rear ride height minus front, as a proxy for rake."""
    front = channel_stats.get("FrontRideHeight")
    rear = channel_stats.get("RearRideHeight")
    if not front or not rear:
        return {}
    return {
        "avg": round(rear["avg"] - front["avg"], 2),
        # Independent minima are useful ground-clearance observations. Their
        # difference is NOT a minimum rake: they can occur at different times.
        "front_min": front["min"],
        "rear_min": rear["min"],
    }


def format_for_diagnosis(data: dict) -> str:
    """Full, readable summary for use in live diagnosis — as much real context as is useful for THIS driver's current session."""
    lines = ["Driver's telemetry summary (from uploaded .duckdb session):"]

    meta = data.get("metadata", {})
    if meta.get("car") or meta.get("track"):
        lines.append(
            f"Car: {meta.get('car', 'unknown')} | Track: {meta.get('track', 'unknown')} | "
            f"Weather: {meta.get('weather', 'unknown')}"
        )

    if data.get("lap_times"):
        laps = data["lap_times"]
        lines.append(f"Laps completed: {len(laps)} | Best: {min(laps)}s | Latest: {laps[-1]}s")

    if data.get("setup"):
        lines.append("Key setup values:")
        ranges = data.get("setup_range") or {}
        for label, value in data["setup"].items():
            info = ranges.get(label) or {}
            note = ""
            if info.get("limit"):
                note = f"  <-- {info['limit']}"
            elif info.get("index") is not None:
                note = f"  (position {info['index']} of {info['min']}-{info['max']})"
            lines.append(f"  {label}: {value}{note}")
        if any(r.get("limit") for r in ranges.values()):
            lines.append(
                "  NOTE: settings marked AT MINIMUM or AT MAXIMUM cannot be "
                "moved further in that direction. Do NOT suggest a change that "
                "the driver has no way to make — the displayed value is the "
                "setting's real-world figure, not its position in the range, "
                "so a number well above zero can still be the lowest available."
            )

    if data.get("channel_stats"):
        lines.append(
            "Session driving stats (averaged over the WHOLE session, not per "
            "corner — these cannot locate a problem at a specific corner):"
        )
        for channel, stats in data["channel_stats"].items():
            lines.append(f"  {channel}: avg {stats['avg']}, range {stats['min']}-{stats['max']}")

    if data.get("rake"):
        rake = data["rake"]
        lines.append(
            f"Ride height difference (rear minus front, a PROXY for rake — "
            f"units unstated in the file, so trend and sign are meaningful, "
            f"the absolute number is not): avg {rake['avg']}. Independent "
            f"session minima: front {rake['front_min']}, rear {rake['rear_min']}. "
            f"These minima need not occur together; no minimum rake is inferred."
        )

    if data.get("wheel_stats"):
        lines.append("Per-wheel averages (FL/FR/RL/RR order is assumed from CarSetup key order, not explicitly labelled in the channels):")
        for channel, wheels in data["wheel_stats"].items():
            values = ", ".join(f"{label}={val}" for label, val in wheels.items())
            lines.append(f"  {channel}: {values}")

    if data.get("tread_spread"):
        spread = data["tread_spread"]
        values = ", ".join(f"{label}={val}" for label, val in spread.items())
        lines.append(
            f"Tread temperature spread per wheel, hottest minus coldest of the "
            f"three positions ({values}). A LARGE spread means the tyre is "
            f"working unevenly across its width; a small one means it is even, "
            f"whatever the overall temperature. Which physical edge is hot is "
            f"NOT established here — read the Left/Centre/Right values above "
            f"and note that which edge is outer depends on the side of the car."
        )

    return "\n".join(lines)


def format_for_evidence(data: dict) -> str:
    """A SHORT, strictly factual snippet for the shared knowledge base — only used when a fix is CONFIRMED working. Never a generalized claim, just what the numbers showed for this one session. Kept narrow on purpose: this becomes shared context for other drivers, so it must not overreach beyond what wa..."""
    parts = []

    if data.get("wheel_stats"):
        for channel, wheels in data["wheel_stats"].items():
            values = ", ".join(f"{label}={val}" for label, val in wheels.items())
            parts.append(f"{channel} observed: {values}")

    if data.get("channel_stats"):
        for channel in ("Brake Pos", "G Force Lat"):
            stats = data["channel_stats"].get(channel)
            if stats:
                parts.append(f"{channel} observed: avg {stats['avg']}, max {stats['max']}")

    if not parts:
        return ""

    return (
        "Telemetry evidence from this session (factual observation, not a "
        "generalized rule; FL/FR/RL/RR channel order is assumed): " + "; ".join(parts)
    )


def _setup_changes(previous: dict, current: dict) -> list:
    """Which setup values differ between two runs, as before/after pairs."""
    old = previous.get("setup") or {}
    new = current.get("setup") or {}
    changes = []
    for label, new_value in new.items():
        old_value = old.get(label)
        if old_value is not None and old_value != new_value:
            changes.append((label, old_value, new_value))
    return changes


def _channel_changes(previous: dict, current: dict) -> list:
    """Aggregates that moved by more than their noise floor."""
    old = previous.get("channel_stats") or {}
    new = current.get("channel_stats") or {}
    changes = []
    for channel, floor in COMPARE_CHANNELS.items():
        if channel not in old or channel not in new:
            continue
        delta = new[channel]["avg"] - old[channel]["avg"]
        if abs(delta) >= floor:
            changes.append((channel, old[channel]["avg"], new[channel]["avg"], round(delta, 2)))
    return changes


def _wheel_changes(previous: dict, current: dict) -> list:
    """Per-wheel values that moved by more than their noise floor."""
    old = previous.get("wheel_stats") or {}
    new = current.get("wheel_stats") or {}
    changes = []
    for channel, floor in COMPARE_WHEEL_CHANNELS.items():
        if channel not in old or channel not in new:
            continue
        moved = []
        for wheel in WHEEL_LABELS:
            if wheel not in old[channel] or wheel not in new[channel]:
                continue
            delta = new[channel][wheel] - old[channel][wheel]
            if abs(delta) >= floor:
                moved.append(f"{wheel} {old[channel][wheel]}->{new[channel][wheel]}")
        if moved:
            changes.append((channel, ", ".join(moved)))
    return changes


def comparison_unavailable_reason(previous: dict, current: dict) -> str:
    """Why two runs cannot be compared, or an empty string when verified."""
    old_meta = (previous or {}).get("metadata") or {}
    new_meta = (current or {}).get("metadata") or {}
    for key in ("track", "car"):
        old = (old_meta.get(key) or "").strip()
        new = (new_meta.get(key) or "").strip()
        if not old or not new:
            return f"The {key} is missing from one or both uploads, so a matching {key} cannot be verified."
        if old.lower() != new.lower():
            return f"The {key} differs: previously {old}, now {new}."
    return ""


def compare_runs(previous: dict, current: dict) -> str:
    """Diffs the driver's previous telemetry against the one just uploaded, and says plainly when the two cannot be compared."""
    if not previous or not current:
        return ""

    old_meta = previous.get("metadata") or {}
    new_meta = current.get("metadata") or {}

    reason = comparison_unavailable_reason(previous, current)
    if reason:
        return (
            f"{reason} These uploads are NOT comparable and no comparison "
            f"has been made. Do not compare them."
        )

    lines = ["Comparison with this driver's PREVIOUS telemetry upload (same car, same track):"]

    old_weather = old_meta.get("weather")
    new_weather = new_meta.get("weather")
    if old_weather and new_weather and old_weather != new_weather:
        lines.append(
            f"  CONDITIONS DIFFER: previously {old_weather}, now {new_weather}. "
            f"Weight any difference below accordingly — conditions alone can "
            f"account for it."
        )

    setup_changes = _setup_changes(previous, current)
    if setup_changes:
        lines.append("  Setup values that changed between the two runs:")
        for label, old_value, new_value in setup_changes:
            lines.append(f"    {label}: {old_value} -> {new_value}")
    else:
        lines.append(
            "  No difference was observed in the tracked setup values present "
            "in BOTH uploads. Only a subset of settings is read; omitted or "
            "missing settings may have changed. This does not establish "
            "whether the driver made the suggested adjustment."
        )

    old_laps = previous.get("lap_times") or []
    new_laps = current.get("lap_times") or []
    if old_laps and new_laps:
        old_best = min(old_laps)
        new_best = min(new_laps)
        delta = round(new_best - old_best, 3)
        direction = "faster" if delta < 0 else "slower"
        lines.append(
            f"  Best lap: {old_best}s over {len(old_laps)} lap(s) -> {new_best}s "
            f"over {len(new_laps)} lap(s), {abs(delta)}s {direction}."
        )

    channel_changes = _channel_changes(previous, current)
    if channel_changes:
        lines.append("  Session averages that moved:")
        for channel, old_value, new_value, delta in channel_changes:
            lines.append(f"    {channel}: {old_value} -> {new_value} ({delta:+})")

    wheel_changes = _wheel_changes(previous, current)
    if wheel_changes:
        lines.append("  Per-wheel values that moved:")
        for channel, detail in wheel_changes:
            lines.append(f"    {channel}: {detail}")

    lines.append(
        "  TREAT THIS AS EVIDENCE, NOT PROOF. Two runs differ for many "
        "reasons besides the setup: different lap counts, different fuel, "
        "and a driver who has had more practice. A value that moved "
        "alongside a setup change is worth citing as support for a "
        "diagnosis; it does not on its own establish that the change caused "
        "it. Absence of a tracked setup difference does not disprove a "
        "change outside the parser's selected settings."
    )

    return "\n".join(lines)
