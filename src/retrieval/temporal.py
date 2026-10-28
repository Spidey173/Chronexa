"""Temporal utilities using dateparser for relative and absolute date resolution.

Replaces hardcoded regexes with generic library-backed date parsing relative to question as_of.
Provides soft metadata boosting and records-on-date lookups. Never applies hard exclusions.
"""
import re
from datetime import datetime
from typing import List, Tuple, Set, Optional, Any
from zoneinfo import ZoneInfo
try:
    import dateparser
except ImportError:
    dateparser = None
from eval_harness.records import Unit

LOCAL_TZ = ZoneInfo("America/Los_Angeles")


def parse_iso_datetime(val: Any) -> datetime:
    """Parses an ISO datetime string or datetime object into an offset-aware datetime.
    
    If naive, explicitly assigns the workspace timezone (America/Los_Angeles).
    Supports strings with 'Z', offsets, or bare dates (YYYY-MM-DD).
    """
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=LOCAL_TZ)
        return val

    s = str(val).strip()
    if not s:
        raise ValueError("Cannot parse empty datetime string")
    
    # Handle YYYY-MM-DD bare dates by setting to end of that day
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        s += "T23:59:59"

    normalized = s.replace("Z", "+00:00")
    # Replace space with T if needed
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T")

    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt


DATE_INDICATORS = {
    "today", "yesterday", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june",
    "jul", "july", "aug", "august", "sep", "sept", "september", "oct", "october", "nov", "november", "dec", "december",
    "week", "month", "days", "day", "ago", "last", "past", "this", "next"
}


def resolve_dates_with_dateparser(query: str, as_of: datetime) -> List[str]:
    """Resolves relative and absolute temporal expressions in query against as_of.

    Returns a list of unique ISO date strings (YYYY-MM-DD) discovered.
    """
    base = as_of.replace(tzinfo=None) if as_of.tzinfo else as_of
    words = re.findall(r"[a-zA-Z0-9]+", query)
    resolved: Set[str] = set()

    # Look for ISO dates explicitly (e.g. 2026-09-23)
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", query):
        resolved.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    # Check 1- to 3-word spans containing a temporal indicator
    for span_len in (3, 2, 1):
        for i in range(len(words) - span_len + 1):
            span_words = words[i:i + span_len]
            if any(w.lower() in DATE_INDICATORS for w in span_words):
                phrase = " ".join(span_words)
                if dateparser is not None:
                    try:
                        dt = dateparser.parse(
                            phrase,
                            languages=["en"],
                            settings={
                                "RELATIVE_BASE": base,
                                "PREFER_DATES_FROM": "past"
                            }
                        )
                        if dt and abs((base - dt).days) <= 365:
                            # Ensure year is reasonable
                            if dt.year in (base.year, base.year - 1, base.year + 1):
                                # Avoid false positives on common words
                                if phrase.lower() not in {"to", "out", "we", "on", "in", "at", "did", "do", "how", "when", "day", "days"}:
                                    resolved.add(dt.strftime("%Y-%m-%d"))
                    except Exception:
                        pass

    return sorted(list(resolved))


def get_records_on_date(units: List[Unit], target_date: str) -> List[Unit]:
    """Returns units whose timestamp or text matches the target date (YYYY-MM-DD)."""
    matches = []
    for u in units:
        # Check delivery/event timestamp
        if u.time and u.time.strftime("%Y-%m-%d") == target_date:
            matches.append(u)
        # Check text content (e.g. calendar start dates formatted as YYYY-MM-DD)
        elif target_date in u.text:
            matches.append(u)
    return matches


def apply_date_boost(
    hits: List[Tuple[Unit, float]],
    target_dates: List[str],
    boost_weight: float = 0.2
) -> List[Tuple[Unit, float]]:
    """Applies a soft metadata boost to hits matching any target date.

    Never filters or excludes hits. Only increases scores of temporally relevant candidates.
    """
    if not target_dates or not hits:
        return hits

    target_set = set(target_dates)
    boosted = []
    for u, score in hits:
        bonus = 0.0
        u_date = u.time.strftime("%Y-%m-%d") if u.time else ""
        if u_date in target_set or any(d in u.text for d in target_set):
            bonus = boost_weight
        boosted.append((u, score + bonus))

    # Re-sort descending
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted
