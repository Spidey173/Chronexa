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
from typing import List, Tuple

# Add project root to sys.path so eval_harness can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_harness import records
from eval_harness.records import Unit
from src.config import DATA_DIR
from src.ingestion.chunker import load_enriched_records

_CACHE = {}
_ENRICHED_CACHE = {}


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


def get_visible_units(as_of: datetime, data_dir: Path = DATA_DIR, enriched: bool = True) -> List[Unit]:
    """Returns all units visible at `as_of` without re-reading files from disk.

    Follows the strict temporal boundary logic of eval_harness/records.py:visible.
    """
    units, deleted, edits = _get_raw_records(data_dir, enriched=enriched)
    out = []
    for u in units:
        if u.time > as_of or (u.id in deleted and deleted[u.id] <= as_of):
            continue
        newer = [txt for t, txt in edits.get(u.id, []) if t <= as_of]
        text = u.text if not newer else u.text.partition(": ")[0] + ": " + newer[-1] + " (edited)"
        out.append(Unit(
            id=u.id,
            record=u.record,
            time=u.time,
            text=text,
            source_type=getattr(u, "source_type", "unknown"),
            speaker=getattr(u, "speaker", ""),
            meeting_id=getattr(u, "meeting_id", ""),
            timestamp_iso=getattr(u, "timestamp_iso", "")
        ))
    return out
