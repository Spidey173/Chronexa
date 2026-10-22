"""Tests for BM25, hybrid retrieval, and zero future leakage during search.
"""
import unittest
from datetime import datetime
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.loader import get_visible_units
from src.retrieval.temporal import parse_iso_datetime
from src.config import DATA_DIR


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(enable_dense=True, enable_rerank=True)

    def test_bm25_search_finds_exact_terms(self):
        as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
        units = get_visible_units(as_of, data_dir=DATA_DIR)
        bm25 = BM25Index(units)
        hits = bm25.search("geocoding regression", top_k=5)
        self.assertTrue(len(hits) > 0)
        top_unit, score = hits[0]
        self.assertTrue("geocod" in top_unit.text.lower())

    def test_hybrid_search_temporal_boundary_zero_leakage(self):
        # As of Sept 9, 2026
        as_of_str = "2026-09-09T09:00:00-07:00"
        as_of_dt = parse_iso_datetime(as_of_str)
        units = get_visible_units(as_of_dt, data_dir=DATA_DIR)
        hits = self.retriever.search("When is Route Planner v2 launching?", units, top_k=20, as_of=as_of_str)
        
        self.assertTrue(len(hits) > 0)
        # Verify that no hit has time > as_of
        for u, score in hits:
            self.assertLessEqual(u.time, as_of_dt, f"Hit {u.id} leaked future record with time {u.time}")

    def test_synonym_expansion_retrieves_agreements(self):
        as_of = parse_iso_datetime("2026-09-18T18:00:00-07:00")
        units = get_visible_units(as_of, data_dir=DATA_DIR)
        hits = self.retriever.search("Acme agreement terms", units, top_k=10, as_of=as_of.isoformat())
        self.assertTrue(len(hits) > 0)
        hit_ids = [u.id for u, _ in hits]
        # Should retrieve the Acme pricing proposal email (EM-0915-ACME-PROP)
        self.assertTrue(any("ACME" in hid for hid in hit_ids))


if __name__ == "__main__":
    unittest.main()
