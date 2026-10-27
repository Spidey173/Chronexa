"""High-performance caching data loader with optional passage enrichment and temporal boundary filtering.

Loads records into memory, then filters strictly by `as_of` timestamp:
- Guarantees zero future temporal leakage (delivery_time <= as_of)
- Masks deleted records
- Applies in-place message edits up to as_of
- Supports enriched contextual windowing (speaker turns, surrounding dialog, rich metadata)
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

# Add project root to sys.path so eval_harness can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_harness import records
from eval_harness.records import Unit
from src.config import DATA_DIR
from src.ingestion.chunker import load_enriched_records
from src.retrieval.temporal import parse_iso_datetime

_CACHE = {}
_ENRICHED_CACHE = {}


def _apply_edits(text: str, newer_text: str) -> str:
    """Safely replaces edited message text while preserving multi-line enriched headers."""
    if "\n" in text:
        lines = text.split("\n")
        prefix, sep, _ = lines[-1].partition(": ")
        if sep:
            lines[-1] = f"{prefix}: {newer_text} (edited)"
        else:
            lines[-1] = f"{newer_text} (edited)"
        return "\n".join(lines)
    else:
        prefix, sep, _ = text.partition(": ")
        if sep:
            return f"{prefix}: {newer_text} (edited)"
        return f"{newer_text} (edited)"


def _get_raw_records(data_dir: Path = DATA_DIR, enriched: bool = False):
    path_key = str(data_dir.resolve())
    if enriched:
        if path_key not in _ENRICHED_CACHE:
            units, deleted, edits = load_enriched_records(data_dir)
            _ENRICHED_CACHE[path_key] = (units, deleted, edits)
        return _ENRICHED_CACHE[path_key]
    else:
        if path_key not in _CACHE:
            units, deleted, edits = records.load(data_dir)
            _CACHE[path_key] = (units, deleted, edits)
        return _CACHE[path_key]


def get_visible_units(as_of: Any, data_dir: Path = DATA_DIR, enriched: bool = True) -> List[Unit]:
    """Returns all units visible at `as_of` without re-reading files from disk.

    Follows the strict temporal boundary logic of eval_harness/records.py:visible.
    Safely parses timezone-aware and naive datetimes.
    """
    as_of_dt = parse_iso_datetime(as_of)
    units, deleted, edits = _get_raw_records(data_dir, enriched=enriched)
    out = []
    for u in units:
        # u.time is aware (America/Los_Angeles or UTC offset)
        u_time = parse_iso_datetime(u.time) if isinstance(u.time, (str, datetime)) else u.time
        if u_time > as_of_dt or (u.id in deleted and parse_iso_datetime(deleted[u.id]) <= as_of_dt):
            continue
        newer = [txt for t, txt in edits.get(u.id, []) if parse_iso_datetime(t) <= as_of_dt]
        text = u.text if not newer else _apply_edits(u.text, newer[-1])
        out.append(Unit(
            id=u.id,
            record=u.record,
            time=u_time,
            text=text,
            source_type=getattr(u, "source_type", "unknown"),
            speaker=getattr(u, "speaker", ""),
            meeting_id=getattr(u, "meeting_id", ""),
            timestamp_iso=getattr(u, "timestamp_iso", "")
        ))
    return out
