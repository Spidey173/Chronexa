"""Tests for calibrated abstention on unanswerable and missing memory queries.
"""
import unittest
from eval_harness.records import Unit
from src.generation.generator import AnswerGenerator
from src.retrieval.temporal import parse_iso_datetime


class TestAbstention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = AnswerGenerator()
        cls.generator.provider = "none"  # Test offline fallback guardrails

    def test_abstains_on_empty_retrieved_units(self):
        res = self.generator.generate("What is the company mission statement?", "2026-09-18T18:00:00-07:00", [])
        self.assertTrue(res["abstained"])
        self.assertTrue(res["answer"].startswith("Not in memory: nothing in the data covers this."))
        self.assertEqual(res["sources"], [])

    def test_abstains_on_unrecorded_salary_question(self):
        dummy_unit = Unit("SL-01", "SL-01", parse_iso_datetime("2026-09-15T10:00:00-07:00"), "[Slack #general] Dana: Shared the new design system files.")
        res = self.generator.generate("What is Dana's salary?", "2026-09-18T18:00:00-07:00", [dummy_unit])
        self.assertTrue(res["abstained"])
        self.assertTrue(res["answer"].startswith("Not in memory: nothing in the data covers this."))
        self.assertEqual(res["sources"], [])

    def test_abstains_on_soc2_question(self):
        dummy_unit = Unit("SL-02", "SL-02", parse_iso_datetime("2026-09-15T10:00:00-07:00"), "[Slack #general] Marcus: Harbor requested a demo call.")
        res = self.generator.generate("What did Harbor Logistics say about SOC 2?", "2026-09-18T18:00:00-07:00", [dummy_unit])
        self.assertTrue(res["abstained"])
        self.assertTrue(res["answer"].startswith("Not in memory: nothing in the data covers this."))

    def test_does_not_abstain_on_answerable_question(self):
        doc = Unit("SL-RP-0910-1", "SL-RP-0910-1", parse_iso_datetime("2026-09-10T09:42:05-07:00"), "[Slack #route-planner] Sarah Kim: QA found a severe geocoding regression.")
        res = self.generator.generate("Why did the launch slip from September 30?", "2026-09-18T18:00:00-07:00", [doc])
        self.assertFalse(res["abstained"])
        self.assertFalse(res["answer"].startswith("Not in memory"))
        self.assertEqual(res["sources"], ["SL-RP-0910-1"])


if __name__ == "__main__":
    unittest.main()
