"""Benchmark cross-encoder rerankers on Dev Paraphrases.

Evaluates:
- Models:
  1. Xenova/ms-marco-MiniLM-L-6-v2
  2. jinaai/jina-reranker-v1-tiny-en
  3. BAAI/bge-reranker-base
- Candidate pool sizes: top 40 vs top 60 from hybrid_rrf
- Metrics: Recall@5, Recall@10 (complete_unit@10), MRR, Primary Score, Found None@20, Forbidden@10
- Latency per query (ms) and model load times (cold & warm)
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
EVAL_HARNESS_DIR = ROOT_DIR / "eval_harness"
if str(EVAL_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_HARNESS_DIR))

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.config import DATA_DIR
from src.ingestion.loader import get_visible_units
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
import eval_harness.records as records_lib
from eval_harness.score_retrieval import score as score_retrieval


def benchmark():
    ctx = records_lib.context(DATA_DIR)
    with open("eval/dev_paraphrases.jsonl") as f:
        dev_para = [json.loads(line) for line in f if line.strip()]

    dense = DenseRetriever(model_name="BAAI/bge-small-en-v1.5")
    hybrid = HybridRetriever(dense_retriever=dense, enable_dense=True)

    models = [
        "Xenova/ms-marco-MiniLM-L-6-v2",
        "jinaai/jina-reranker-v1-tiny-en",
        "BAAI/bge-reranker-base"
    ]

    pools = [40, 60]

    results = []

    # Pre-fetch candidate pools to keep benchmark fast and identical
    print("Pre-fetching candidate pools from hybrid_rrf for all dev queries...")
    query_candidates: Dict[str, List[Tuple[Any, float]]] = {}
    for item in dev_para:
        as_of_dt = datetime.fromisoformat(item["as_of"].replace("Z", "+00:00"))
        visible = get_visible_units(as_of_dt, data_dir=DATA_DIR, enriched=False)
        candidates = hybrid.search(item["question"], visible, top_k=60, rrf_k=60)
        query_candidates[item["id"]] = candidates

    print("Candidate generation complete.\n")

    for model_name in models:
        print(f"\n=======================================================")
        print(f"Testing Reranker: {model_name}")
        print("=======================================================")
        
        t0 = time.time()
        encoder = TextCrossEncoder(model_name=model_name)
        warm_load_time = time.time() - t0
        print(f"Model warm load time: {warm_load_time:.2f}s")

        for pool_size in pools:
            print(f"\n--- Evaluating Candidate Pool: Top {pool_size} ---")
            answers_top8 = {}
            answers_full_ranked = {}
            query_times = []
            
            for item in dev_para:
                qid = item["id"]
                q_text = item["question"]
                candidates = query_candidates[qid][:pool_size]
                
                # Extract texts, truncating if necessary to 512 tokens (~2000 chars) for models with 512 limit
                doc_texts = []
                for u, _ in candidates:
                    txt = u.text
                    if "tiny-en" not in model_name and len(txt) > 2000:
                        txt = txt[:2000]
                    doc_texts.append(txt)

                t_q0 = time.time()
                # Run reranker
                scores = list(encoder.rerank(q_text, doc_texts))
                t_q = (time.time() - t_q0) * 1000.0  # ms
                query_times.append(t_q)

                # Sort candidates by reranker score
                ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                ranked_units = [u.id for (u, _), s in ranked]
                
                # Top 8 as primary output
                answers_top8[qid] = {
                    "id": qid,
                    "retrieved": ranked_units[:8]
                }
                # Full ranked pool for rank analysis
                answers_full_ranked[qid] = {
                    "id": qid,
                    "retrieved": ranked_units
                }

            # Score retrieval with top 8
            res_top8 = score_retrieval(dev_para, answers_top8, ctx)
            sum_top8 = res_top8["summary"]

            # Score retrieval with full ranked pool (for complete @10 within pool)
            res_full = score_retrieval(dev_para, answers_full_ranked, ctx)
            sum_full = res_full["summary"]

            avg_latency = sum(query_times) / len(query_times)
            p95_latency = sorted(query_times)[int(len(query_times) * 0.95)]

            print(f"Avg Latency: {avg_latency:.1f}ms / query (p95: {p95_latency:.1f}ms)")
            print(f"Top 8 return: Recall@5: {sum_top8.get('complete_unit@5', 0.0):.1%}, Recall@10 (in top 8): {sum_top8.get('complete_unit@10', 0.0):.1%}, MRR: {sum_top8.get('mrr', 0.0):.4f}, Primary: {sum_top8['score']['mean']:.1%}, Found None@20: {sum_top8.get('found_none@20', 0.0):.1%}")
            print(f"Full Pool: Recall@10: {sum_full.get('complete_unit@10', 0.0):.1%}, MRR: {sum_full.get('mrr', 0.0):.4f}, Primary: {sum_full['score']['mean']:.1%}")

            results.append({
                "model": model_name,
                "pool_size": pool_size,
                "warm_load_sec": round(warm_load_time, 2),
                "avg_latency_ms": round(avg_latency, 1),
                "p95_latency_ms": round(p95_latency, 1),
                "top8_recall@5": sum_top8.get("complete_unit@5", 0.0),
                "top8_recall@10": sum_top8.get("complete_unit@10", 0.0),
                "top8_mrr": sum_top8.get("mrr", 0.0),
                "top8_primary": sum_top8["score"]["mean"],
                "top8_found_none": sum_top8.get("found_none@20", 0.0),
                "forbidden@10": sum_top8.get("questions_with_forbidden@10", 0),
                "full_recall@10": sum_full.get("complete_unit@10", 0.0),
                "full_mrr": sum_full.get("mrr", 0.0),
            })

    print("\n\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE (DEV PARAPHRASES):")
    print("="*80)
    print(f"{'Model':<32} | {'Pool':<4} | {'Latency':<8} | {'Rec@5':<7} | {'Rec@10(top8)':<12} | {'MRR':<7} | {'Primary':<7} | {'None@20':<7}")
    print("-" * 105)
    for r in results:
        print(f"{r['model']:<32} | {r['pool_size']:<4} | {r['avg_latency_ms']:>5.1f}ms | {r['top8_recall@5']:>6.1%} | {r['top8_recall@10']:>11.1%} | {r['top8_mrr']:>7.4f} | {r['top8_primary']:>7.1%} | {r['top8_found_none']:>7.1%}")


if __name__ == "__main__":
    benchmark()
