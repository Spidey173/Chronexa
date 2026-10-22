"""Tests for temporal filtering, timezone handling, message edits, and deletions.
"""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from eval_harness.records import Unit, visible, load
from src.retrieval.temporal import parse_iso_datetime, LOCAL_TZ
from src.ingestion.loader import get_visible_units, _apply_edits
from src.config import DATA_DIR


class TestTemporalBoundaries(unittest.TestCase):
    def test_parse_iso_datetime_aware(self):
        s = "2026-09-18T18:00:00-07:00"
        dt = parse_iso_datetime(s)
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 9)
        self.assertEqual(dt.day, 18)

    def test_parse_iso_datetime_naive(self):
        s = "2026-09-18T18:00:00"
        dt = parse_iso_datetime(s)
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.tzinfo, LOCAL_TZ)

    def test_parse_iso_datetime_utc_z(self):
        s = "2026-09-18T18:00:00Z"
        dt = parse_iso_datetime(s)
        self.assertIsNotNone(dt.tzinfo)

    def test_parse_iso_datetime_bare_date(self):
        s = "2026-09-18"
        dt = parse_iso_datetime(s)
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.day, 18)
        self.assertEqual(dt.hour, 23)
        self.assertEqual(dt.minute, 59)

    def test_compare_naive_against_aware_no_crash(self):
        naive_dt = datetime(2026, 9, 18, 12, 0)
        units = get_visible_units(naive_dt, data_dir=DATA_DIR)
        self.assertTrue(len(units) > 0)
        # Verify no unit has time strictly greater than as_of
        target_dt = parse_iso_datetime(naive_dt)
        for u in units:
            self.assertLessEqual(u.time, target_dt)

    def test_future_leakage_boundary(self):
        # Sep 9, 2026 at 09:00:00
        early_as_of = parse_iso_datetime("2026-09-09T09:00:00-07:00")
        early_units = get_visible_units(early_as_of, data_dir=DATA_DIR)
        for u in early_units:
            self.assertLessEqual(u.time, early_as_of, f"Unit {u.id} from future leaked: {u.time} > {early_as_of}")

        # Later date has strictly more or equal visible records
        late_as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
        late_units = get_visible_units(late_as_of, data_dir=DATA_DIR)
        self.assertGreater(len(late_units), len(early_units))

    def test_message_deletion_masking(self):
        # Message was deleted: should not be in visible units after deletion timestamp
        units, deleted, _ = load(DATA_DIR)
        self.assertTrue(len(deleted) > 0, "Expected at least one deletion in dataset")
        target_del_id, del_time = next(iter(deleted.items()))
        
        # After deletion time, unit should NOT be visible
        visible_after = get_visible_units(del_time, data_dir=DATA_DIR)
        visible_after_ids = {u.id for u in visible_after}
        self.assertNotIn(target_del_id, visible_after_ids)

    def test_edit_preserves_enriched_headers(self):
        original = "[Slack #route-planner | Date: 2026-09-16 | User: Priya Patel | ID: SL-0916-1]\nPriya Patel: 60 of 64 passing"
        edited = _apply_edits(original, "61 of 64 passing")
        self.assertIn("Priya Patel: 61 of 64 passing (edited)", edited)
        self.assertIn("[Slack #route-planner | Date: 2026-09-16 | User: Priya Patel | ID: SL-0916-1]", edited)


if __name__ == "__main__":
    unittest.main()
