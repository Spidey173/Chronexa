"""Cross-encoder reranker using FastEmbed TextCrossEncoder.

Re-scores candidate passages from hybrid BM25 + dense search using deep cross-attention:
- Default Model: Xenova/ms-marco-MiniLM-L-6-v2 (fastest, lightweight ~80MB, superior MRR on dev)
- Candidate Pool: fused top 40 or 60 from hybrid_rrf
- Output: top 8 reranked passages
- Text Truncation: truncates passages exceeding 512 tokens (approx 2000 characters) to prevent OOM
"""
import time
from typing import List, Tuple, Optional, Dict, Any

from fastembed.rerank.cross_encoder import TextCrossEncoder
from eval_harness.records import Unit


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        max_chars: int = 2000,
        threads: Optional[int] = 4,
        batch_size: int = 40,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_chars = max_chars
        self.threads = threads
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        t0 = time.time()
        self.encoder = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir, threads=threads)
        self.load_time_sec = time.time() - t0
        
        self.truncation_count = 0
        self.total_candidates_processed = 0

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Unit, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Unit, float]]:
        """Reranks candidate (Unit, initial_score) pairs using cross-encoder cross-attention.

        Returns reranked (Unit, rerank_score) pairs. If top_k is None, returns the entire reranked pool.
        """
        if not candidates:
            return []

        doc_texts = []
        for u, _ in candidates:
            self.total_candidates_processed += 1
            text = u.text
            if len(text) > self.max_chars:
                self.truncation_count += 1
                text = text[:self.max_chars]
            doc_texts.append(text)

        # Cross-encoder inference with tuned batch size
        scores = list(self.encoder.rerank(query, doc_texts, batch_size=self.batch_size))

        # Re-sort candidates by cross-encoder score descending
        reranked = [
            (cand[0], float(score))
            for cand, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        ]

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def rerank_multi_queries(
        self,
        queries: List[str],
        candidates: List[Tuple[Unit, float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[Unit, float]]:
        """Reranks candidate (Unit, initial_score) pairs across multiple queries using cross-attention.

        Takes the maximum cross-encoder score across all queries for each document.
        """
        if not candidates or not queries:
            return candidates[:top_k] if top_k else candidates

        unique_queries = []
        seen_q = set()
        for q in queries:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen_q:
                seen_q.add(q_clean.lower())
                unique_queries.append(q_clean)

        if len(unique_queries) == 1:
            return self.rerank(unique_queries[0], candidates, top_k=top_k)

        doc_texts = []
        for u, _ in candidates:
            self.total_candidates_processed += 1
            text = u.text
            if len(text) > self.max_chars:
                self.truncation_count += 1
                text = text[:self.max_chars]
            doc_texts.append(text)

        max_scores = [-float("inf")] * len(candidates)
        for q in unique_queries[:4]:
            scores = list(self.encoder.rerank(q, doc_texts, batch_size=self.batch_size))
            for i, s in enumerate(scores):
                if float(s) > max_scores[i]:
                    max_scores[i] = float(s)

        reranked = [
            (candidates[i][0], max_scores[i])
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def get_truncation_stats(self) -> Dict[str, Any]:
        return {
            "total_processed": self.total_candidates_processed,
            "truncated": self.truncation_count,
            "truncation_rate": (
                self.truncation_count / self.total_candidates_processed
                if self.total_candidates_processed > 0
                else 0.0
            )
        }
