"""Tests for temporal filtering, timezone handling, message edits, and deletions using pytest.
"""
from datetime import datetime

from eval_harness.records import Unit, load
from src.retrieval.temporal import parse_iso_datetime, LOCAL_TZ
from src.ingestion.loader import get_visible_units, _apply_edits
from src.config import DATA_DIR


def test_parse_iso_datetime_aware():
    s = "2026-09-18T18:00:00-07:00"
    dt = parse_iso_datetime(s)
    assert dt.tzinfo is not None
    assert dt.year == 2026
    assert dt.month == 9
    assert dt.day == 18


def test_parse_iso_datetime_naive():
    s = "2026-09-18T18:00:00"
    dt = parse_iso_datetime(s)
    assert dt.tzinfo is not None
    assert dt.tzinfo == LOCAL_TZ


def test_parse_iso_datetime_utc_z():
    s = "2026-09-18T18:00:00Z"
    dt = parse_iso_datetime(s)
    assert dt.tzinfo is not None


def test_parse_iso_datetime_bare_date():
    s = "2026-09-18"
    dt = parse_iso_datetime(s)
    assert dt.tzinfo is not None
    assert dt.day == 18
    assert dt.hour == 23
    assert dt.minute == 59


def test_compare_naive_against_aware_no_crash():
    naive_dt = datetime(2026, 9, 18, 12, 0)
    units = get_visible_units(naive_dt, data_dir=DATA_DIR)
    assert len(units) > 0
    # Verify no unit has time strictly greater than as_of
    target_dt = parse_iso_datetime(naive_dt)
    for u in units:
        assert u.time <= target_dt


def test_future_leakage_boundary():
    # Sep 9, 2026 at 09:00:00
    early_as_of = parse_iso_datetime("2026-09-09T09:00:00-07:00")
    early_units = get_visible_units(early_as_of, data_dir=DATA_DIR)
    for u in early_units:
        assert u.time <= early_as_of, f"Unit {u.id} from future leaked: {u.time} > {early_as_of}"

    # Later date has strictly more or equal visible records
    late_as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
    late_units = get_visible_units(late_as_of, data_dir=DATA_DIR)
    assert len(late_units) > len(early_units)


def test_message_deletion_masking():
    # Message was deleted: should not be in visible units after deletion timestamp
    units, deleted, _ = load(DATA_DIR)
    assert len(deleted) > 0, "Expected at least one deletion in dataset"
    target_del_id, del_time = next(iter(deleted.items()))

    # After deletion time, unit should NOT be visible
    visible_after = get_visible_units(del_time, data_dir=DATA_DIR)
    visible_after_ids = {u.id for u in visible_after}
    assert target_del_id not in visible_after_ids


def test_edit_preserves_enriched_headers():
    original = "[Slack #route-planner | Date: 2026-09-16 | User: Priya Patel | ID: SL-0916-1]\nPriya Patel: 60 of 64 passing"
    edited = _apply_edits(original, "61 of 64 passing")
    assert "Priya Patel: 61 of 64 passing (edited)" in edited
    assert "[Slack #route-planner | Date: 2026-09-16 | User: Priya Patel | ID: SL-0916-1]" in edited


def test_dateparser_missing_fallback_iso_dates_only():
    """Verifies reduced functionality when dateparser is unavailable: ISO dates resolve, relative dates skipped."""
    import src.retrieval.temporal as temporal_mod
    orig = temporal_mod.dateparser
    try:
        temporal_mod.dateparser = None
        as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
        # 1. Explicit ISO date should still be extracted via regex
        res_iso = temporal_mod.resolve_dates_with_dateparser("Status report for 2026-09-15 meeting", as_of)
        assert res_iso == ["2026-09-15"]

        # 2. Relative date expression cannot be parsed without dateparser, returns empty without crashing
        res_rel = temporal_mod.resolve_dates_with_dateparser("What blockers were discussed yesterday?", as_of)
        assert res_rel == []
    finally:
        temporal_mod.dateparser = orig


def test_dateparser_missing_date_boost_no_crash():
    """Verifies apply_date_boost safely operates with or without resolved target dates."""
    import src.retrieval.temporal as temporal_mod
    orig = temporal_mod.dateparser
    try:
        temporal_mod.dateparser = None
        # Should execute safely using regex extraction only
        u = Unit(id="U1", text="Discussion on 2026-09-15", time=datetime(2026, 9, 15, 10, 0), record="R1")
        boosted = temporal_mod.apply_date_boost([(u, 0.5)], ["2026-09-15"])
        assert len(boosted) == 1
        assert boosted[0][1] >= 0.5
    finally:
        temporal_mod.dateparser = orig
