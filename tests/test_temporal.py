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

    def test_dateparser_missing_fallback_iso_dates_only(self):
        """Verifies reduced functionality when dateparser is unavailable: ISO dates resolve, relative dates skipped."""
        import src.retrieval.temporal as temporal_mod
        orig = temporal_mod.dateparser
        try:
            temporal_mod.dateparser = None
            as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
            # 1. Explicit ISO date should still be extracted via regex
            res_iso = temporal_mod.resolve_dates_with_dateparser("Status report for 2026-09-15 meeting", as_of)
            self.assertEqual(res_iso, ["2026-09-15"])

            # 2. Relative date expression cannot be parsed without dateparser, returns empty without crashing
            res_rel = temporal_mod.resolve_dates_with_dateparser("What blockers were discussed yesterday?", as_of)
            self.assertEqual(res_rel, [])
        finally:
            temporal_mod.dateparser = orig

    def test_dateparser_missing_date_boost_no_crash(self):
        """Verifies apply_date_boost safely operates with or without resolved target dates."""
        import src.retrieval.temporal as temporal_mod
        as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
        u1 = Unit(id="U1", text="Report on 2026-09-15", time=as_of, record="R1")
        u2 = Unit(id="U2", text="Other discussion", time=as_of, record="R2")
        hits = [(u1, 0.5), (u2, 0.4)]
        # Empty dates (e.g. from skipped relative parsing) should return unchanged hits
        boosted_empty = temporal_mod.apply_date_boost(hits, [])
        self.assertEqual(len(boosted_empty), 2)
        self.assertEqual(boosted_empty[0][1], 0.5)

        # Explicit ISO target date should boost matching unit
        boosted = temporal_mod.apply_date_boost(hits, ["2026-09-15"], boost_weight=0.3)
        self.assertAlmostEqual(boosted[0][1], 0.8)
        self.assertEqual(boosted[0][0].id, "U1")


if __name__ == "__main__":
    unittest.main()
