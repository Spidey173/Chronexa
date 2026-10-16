"""Unified Evaluation Harness for Chronexa Memory Retrieval & Generalization.

Evaluates retrieval configurations across Dev (strictly zero holdout access):
- v2_baseline: Historical baseline (lexical search with regex decomposition and synonyms)
- bm25_only: Clean BM25 without question-specific regexes
- dense_only: Dense semantic search (BAAI/bge-small-en-v1.5)
- hybrid_rrf: BM25 + Dense fused via Reciprocal Rank Fusion (k=60)
- hybrid_rrf_rerank: Cross-encoder reranked with Xenova/ms-marco-MiniLM-L-6-v2 (top 40 pool)
- +decomposition: Generic LLM/sub-query decomposition fused with RRF
- +decomposition+2hop: Iterative 2-hop retrieval based on hop-1 evidence
- +date_handling: dateparser relative date resolution and soft metadata boost
- synonyms_disabled: Workplace synonyms ablation on dev

Computes Recall@5, Recall@10, Recall@20, MRR, and Found None@20 identically across all configs.
Separately evaluates generator context tradeoff (top 8 vs top 10 vs top 12).
Evaluates end-to-end answer accuracy using score_memory.py rules.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
EVAL_HARNESS_DIR = ROOT_DIR / "eval_harness"
if str(EVAL_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_HARNESS_DIR))

from src.config import DATA_DIR, EVALS_DIR
from src.ingestion.loader import get_visible_units, _get_raw_records
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.decomposition import decompose_query, propose_hop2_query
from src.retrieval.temporal import resolve_dates_with_dateparser, apply_date_boost

import eval_harness.records as records_lib
from eval_harness.score_retrieval import score as score_retrieval
from eval_harness.score_memory import deterministic as score_memory_deterministic, POINTS

EVAL_DIR = ROOT_DIR / "eval"
RESULTS_MD = EVAL_DIR / "results.md"


def _parse_dt(dt_str: str) -> datetime:
    return datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))


class EvaluationRunner:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.ctx = records_lib.context(data_dir)
        self.dense_retrievers: Dict[str, DenseRetriever] = {}
        self.rerankers: Dict[str, CrossEncoderReranker] = {}

    def get_dense_retriever(self, model_name: str = "BAAI/bge-small-en-v1.5") -> DenseRetriever:
        if model_name not in self.dense_retrievers:
            d = DenseRetriever(model_name=model_name)
            all_units, _, _ = _get_raw_records(self.data_dir, enriched=False)
            d.warm_cache(all_units)
            self.dense_retrievers[model_name] = d
        return self.dense_retrievers[model_name]

    def get_reranker(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
        if model_name not in self.rerankers:
            self.rerankers[model_name] = CrossEncoderReranker(model_name=model_name, threads=4, batch_size=40)
        return self.rerankers[model_name]

    def _fuse_rrf(self, list_of_hit_lists: List[List[Tuple[records_lib.Unit, float]]], k: int = 60) -> List[Tuple[records_lib.Unit, float]]:
        scores = {}
        umap = {}
        for hit_list in list_of_hit_lists:
            for rank, (u, _) in enumerate(hit_list, 1):
                scores[u.id] = scores.get(u.id, 0.0) + 1.0 / (k + rank)
                umap[u.id] = u
        fused = sorted([(umap[uid], sc) for uid, sc in scores.items()], key=lambda x: x[1], reverse=True)
        return fused

    def retrieve(
        self,
        config: str,
        query: str,
        visible_units: list,
        as_of_dt: datetime,
        top_k: int = 20,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        rerank_pool: int = 40,
        use_synonyms: bool = True
    ) -> Tuple[List[Tuple[records_lib.Unit, float]], List[Tuple[records_lib.Unit, float]], int]:
        """Executes retrieval for a query.

        Returns: (top_hits_for_eval, full_candidates_pool, llm_calls_count)
        """
        if not visible_units:
            return [], [], 0

        as_of_str = as_of_dt.isoformat()
        llm_calls = 0

        if config == "v2_baseline":
            r = HybridRetriever(enable_dense=False, use_synonyms=True, use_legacy_regex=True)
            hits = r.search(query, visible_units, top_k=top_k, dense_weight=0.0)
            return hits, hits, 0

        elif config == "bm25_only":
            bm25_idx = BM25Index(visible_units, use_synonyms=use_synonyms)
            hits = bm25_idx.search(query, top_k=top_k, use_legacy_regex=False)
            return hits, hits, 0

        elif config == "dense_only":
            dense = self.get_dense_retriever(dense_model)
            hits = dense.search(query, visible_units, top_k=top_k)
            return hits, hits, 0

        elif config == "hybrid_rrf":
            dense = self.get_dense_retriever(dense_model)
            r = HybridRetriever(dense_retriever=dense, enable_dense=True, model_name=dense_model, use_synonyms=use_synonyms, use_legacy_regex=False)
            hits = r.search(query, visible_units, top_k=top_k, rrf_k=60, bm25_weight=1.0, dense_weight=1.0)
            return hits, hits, 0

        elif config == "hybrid_rrf_rerank":
            dense = self.get_dense_retriever(dense_model)
            r = HybridRetriever(dense_retriever=dense, enable_dense=True, model_name=dense_model, use_synonyms=use_synonyms, use_legacy_regex=False)
            candidates = r.search(query, visible_units, top_k=min(rerank_pool, len(visible_units)), rrf_k=60, bm25_weight=1.0, dense_weight=1.0)
            reranker = self.get_reranker(reranker_model)
            reranked_pool = reranker.rerank(query, candidates, top_k=None)
            return reranked_pool[:top_k], reranked_pool, 0

        elif config == "+decomposition":
            llm_calls += 1
            decomp = decompose_query(query, as_of_str)
            sub_queries = decomp.get("sub_queries", [query])
            dense = self.get_dense_retriever(dense_model)
            r = HybridRetriever(dense_retriever=dense, enable_dense=True, model_name=dense_model, use_synonyms=use_synonyms, use_legacy_regex=False)
            
            sub_hit_lists = []
            for sq in sub_queries:
                sq_hits = r.search(sq, visible_units, top_k=min(rerank_pool, len(visible_units)), rrf_k=60)
                sub_hit_lists.append(sq_hits)
            fused_candidates = self._fuse_rrf(sub_hit_lists, k=60)[:rerank_pool]
            
            reranker = self.get_reranker(reranker_model)
            reranked_pool = reranker.rerank(query, fused_candidates, top_k=None)
            return reranked_pool[:top_k], reranked_pool, llm_calls

        elif config in ("+decomposition+2hop", "+date_handling", "synonyms_disabled"):
            llm_calls += 1
            decomp = decompose_query(query, as_of_str)
            sub_queries = decomp.get("sub_queries", [query])
            dense = self.get_dense_retriever(dense_model)
            r = HybridRetriever(dense_retriever=dense, enable_dense=True, model_name=dense_model, use_synonyms=(use_synonyms and config != "synonyms_disabled"), use_legacy_regex=False)
            
            # Hop 1
            sub_hit_lists = []
            for sq in sub_queries:
                sq_hits = r.search(sq, visible_units, top_k=min(rerank_pool, len(visible_units)), rrf_k=60)
                sub_hit_lists.append(sq_hits)
            hop1_fused = self._fuse_rrf(sub_hit_lists, k=60)
            
            # Hop 2 proposal
            llm_calls += 1
            hop2 = propose_hop2_query(query, as_of_str, hop1_fused)
            all_lists = [hop1_fused]
            target_dates = []
            
            if hop2.get("needs_hop2") and hop2.get("follow_up_query"):
                h2_hits = r.search(hop2["follow_up_query"], visible_units, top_k=min(rerank_pool, len(visible_units)), rrf_k=60)
                all_lists.append(h2_hits)
                if hop2.get("target_date"):
                    target_dates.append(hop2["target_date"])

            # Date handling
            if config in ("+date_handling", "synonyms_disabled"):
                query_dates = resolve_dates_with_dateparser(query, as_of_dt)
                for qd in query_dates:
                    if qd not in target_dates:
                        target_dates.append(qd)

            fused_pool = self._fuse_rrf(all_lists, k=60)[:rerank_pool]
            
            if target_dates and config in ("+date_handling", "synonyms_disabled"):
                fused_pool = apply_date_boost(fused_pool, target_dates, boost_weight=0.25)

            reranker = self.get_reranker(reranker_model)
            rerank_q = query
            if target_dates:
                rerank_q = f"{query} ({target_dates[0]})"
            reranked_pool = reranker.rerank(rerank_q, fused_pool, top_k=None)
            return reranked_pool[:top_k], reranked_pool, llm_calls

        else:
            raise ValueError(f"Unknown config: {config}")

    def run_eval(
        self,
        questions: List[Dict[str, Any]],
        config: str,
        generate_answers: bool = False,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        rerank_pool: int = 40,
        use_synonyms: bool = True
    ) -> Dict[str, Any]:
        answers = {}
        candidate_ranks_by_q = {}
        generator = None
        if generate_answers:
            from src.generation.generator import AnswerGenerator
            generator = AnswerGenerator()

        total_llm_calls = 0
        context_tradeoff = {
            8: {"covered": 0, "chars": 0, "tokens": 0},
            10: {"covered": 0, "chars": 0, "tokens": 0},
            12: {"covered": 0, "chars": 0, "tokens": 0}
        }
        scored_q_count = 0

        start_time = time.time()
        for idx, item in enumerate(questions, start=1):
            q_id = item["id"]
            q_text = item["question"]
            as_of_str = item["as_of"]
            as_of_dt = _parse_dt(as_of_str)

            # Strictly visible units under as_of temporal boundary (guarantees forbidden@10 == 0)
            visible = get_visible_units(as_of_dt, data_dir=self.data_dir, enriched=False)

            hits, all_cands, calls = self.retrieve(
                config,
                q_text,
                visible,
                as_of_dt=as_of_dt,
                top_k=20,
                dense_model=dense_model,
                reranker_model=reranker_model,
                rerank_pool=rerank_pool,
                use_synonyms=use_synonyms
            )
            total_llm_calls += calls

            # Full reranked pool returned up to 20 for identical metric scoring
            retrieved_ids = [u.id for u, _ in hits[:20]]
            candidate_ranks_by_q[q_id] = {u.id: r for r, (u, _) in enumerate(all_cands, 1)}

            # Evaluate context tradeoff for generator at top 8, top 10, top 12
            groups = item.get("needed", [])
            is_scored = bool(groups if item.get("retrieval_scored", item.get("answerable", True)) else [])
            if is_scored:
                scored_q_count += 1
                for cutoff in (8, 10, 12):
                    top_slice = all_cands[:cutoff]
                    top_ids = {u.id for u, _ in top_slice}
                    top_recs = {self.ctx["record_of"].get(u.id, u.id) for u, _ in top_slice}
                    all_met = all(any(m in top_ids or (m not in self.ctx["record_of"] and m in top_recs) for m in g) for g in groups)
                    if all_met:
                        context_tradeoff[cutoff]["covered"] += 1
                    tot_chars = sum(len(u.text) for u, _ in top_slice)
                    context_tradeoff[cutoff]["chars"] += tot_chars
                    context_tradeoff[cutoff]["tokens"] += math.ceil(tot_chars / 4.0)

            # Generate answers if requested
            if generate_answers and generator:
                top_units_gen = [u for u, _ in hits[:8]]
                gen_res = generator.generate(q_text, as_of_str, top_units_gen)
                ans_text = gen_res["answer"]
                sources = gen_res["sources"]
                abstained = gen_res["abstained"]
            else:
                ans_text = ""
                sources = retrieved_ids[:3]
                abstained = False

            answers[q_id] = {
                "id": q_id,
                "answer": ans_text,
                "sources": sources,
                "retrieved": retrieved_ids,
                "abstained": abstained
            }

        elapsed = time.time() - start_time
        retrieval_res = score_retrieval(questions, answers, self.ctx)
        summary = retrieval_res["summary"]

        # Score end-to-end answers if generated
        ans_strict_scores = []
        ans_lenient_scores = []
        if generate_answers:
            for item in questions:
                ans = answers.get(item["id"])
                if ans:
                    verdict, reason, unverified = score_memory_deterministic(item, ans, self.ctx)
                    strict_pt = POINTS.get(verdict, 0.0) if not unverified else 0.0
                    lenient_pt = POINTS.get(verdict, 0.0)
                    ans_strict_scores.append(strict_pt)
                    ans_lenient_scores.append(lenient_pt)
            strict_acc = round(sum(ans_strict_scores) / len(ans_strict_scores), 4) if ans_strict_scores else 0.0
            lenient_acc = round(sum(ans_lenient_scores) / len(ans_lenient_scores), 4) if ans_lenient_scores else 0.0
        else:
            strict_acc = None
            lenient_acc = None

        # Build detailed failure list
        failures = []
        for q, item_res in zip(questions, retrieval_res["items"]):
            if item_res.get("scored") and item_res.get("score", 0.0) < 1.0:
                needed_info = q.get("needed", [])
                c_ranks = candidate_ranks_by_q.get(q["id"], {})
                unit_ranks = {}
                for g_idx, grp in enumerate(needed_info, 1):
                    for uid in grp:
                        rank_val = c_ranks.get(uid)
                        unit_ranks[uid] = f"#{rank_val}" if rank_val else ">pool"

                failures.append({
                    "id": q["id"],
                    "category": q.get("category", "unknown"),
                    "question": q["question"],
                    "recall_unit@10": item_res.get("recall_unit@10", 0.0),
                    "first_rank": item_res.get("first_rank"),
                    "needed": needed_info,
                    "needed_ranks": unit_ranks,
                    "retrieved_top5": answers[q["id"]]["retrieved"][:5]
                })

        # Calculate context tradeoff stats
        context_stats = {}
        if scored_q_count > 0:
            for cutoff in (8, 10, 12):
                context_stats[cutoff] = {
                    "coverage": round(context_tradeoff[cutoff]["covered"] / scored_q_count, 4),
                    "avg_chars": round(context_tradeoff[cutoff]["chars"] / scored_q_count),
                    "avg_tokens": round(context_tradeoff[cutoff]["tokens"] / scored_q_count)
                }

        return {
            "config": config,
            "dense_model": dense_model,
            "reranker_model": reranker_model,
            "n_total": len(questions),
            "n_scored": summary.get("scored_on_coverage", len(questions)),
            "elapsed_sec": round(elapsed, 2),
            "latency_ms": round((elapsed / len(questions)) * 1000, 1),
            "llm_calls_per_q": round(total_llm_calls / len(questions), 2),
            "complete_unit@5": summary.get("complete_unit@5", 0.0),
            "complete_unit@10": summary.get("complete_unit@10", 0.0),
            "complete_unit@20": summary.get("complete_unit@20", 0.0),
            "complete_record@10": summary.get("complete_record@10", 0.0),
            "mrr": summary.get("mrr", 0.0),
            "primary_score": summary["score"]["mean"],
            "found_none@20": summary.get("found_none@20", 0.0),
            "forbidden@10": summary.get("questions_with_forbidden@10", 0),
            "answer_strict_acc": strict_acc,
            "answer_lenient_acc": lenient_acc,
            "context_stats": context_stats,
            "failures": failures,
            "by_category": summary.get("by_category", {})
        }


def append_to_results_md(record: Dict[str, Any]):
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_MD.exists():
        header = (
            "# Chronexa Retrieval Evaluation Results Benchmarks\n\n"
            "| Split | Query Set | Config | N | Recall@5 | Recall@10 | Recall@20 | MRR | Found None@20 | Forbidden@10 | Latency/Q | Ans Strict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write(header)

    ans_str = f"{record['answer_strict_acc']:.1%}" if record.get('answer_strict_acc') is not None else "N/A"
    row = (
        f"| {record['split']} | {record['dataset']} | `{record['config']}` | {record['n_total']} | "
        f"{record['complete_unit@5']:.1%} | {record['complete_unit@10']:.1%} | {record.get('complete_unit@20', 0.0):.1%} | {record['mrr']:.4f} | "
        f"{record['found_none@20']:.1%} | {record['forbidden@10']} | {record['latency_ms']:.0f}ms | {ans_str} |\n"
    )
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(row)


def main():
    parser = argparse.ArgumentParser(description="Chronexa Evaluation Runner")
    parser.add_argument("--config", default="phase4_ablation",
                        choices=["v2_baseline", "bm25_only", "dense_only", "hybrid_rrf", "hybrid_rrf_rerank",
                                 "+decomposition", "+decomposition+2hop", "+date_handling", "synonyms_disabled",
                                 "compare_all_dev", "phase4_ablation"])
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--mode", default="paraphrases", choices=["paraphrases", "original", "both"])
    parser.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--rerank-model", default="Xenova/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-pool", type=int, default=40)
    parser.add_argument("--generate-answers", action="store_true", default=True)
    args = parser.parse_args()

    runner = EvaluationRunner()

    if args.config == "compare_all_dev":
        configs_to_run = ["v2_baseline", "bm25_only", "dense_only", "hybrid_rrf", "hybrid_rrf_rerank"]
    elif args.config == "phase4_ablation":
        configs_to_run = ["v2_baseline", "hybrid_rrf", "hybrid_rrf_rerank", "+decomposition", "+decomposition+2hop", "+date_handling", "synonyms_disabled"]
    else:
        configs_to_run = [args.config]

    eval_jobs = []
    if args.mode in ("paraphrases", "both"):
        eval_jobs.append(("dev", "paraphrases", EVAL_DIR / "dev_paraphrases.jsonl"))
    if args.mode in ("original", "both"):
        eval_jobs.append(("dev", "original", EVAL_DIR / "dev_original.jsonl"))

    for split_name, ds_name, path in eval_jobs:
        with open(path, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f if line.strip()]

        print(f"\n================================================================================")
        print(f" EVALUATION ON [{split_name.upper()}] [{ds_name.upper()}] (Total: {len(questions)} questions)")
        print(f"================================================================================")

        for cfg in configs_to_run:
            print(f"\n>>> Running Config: {cfg} ...")
            res = runner.run_eval(
                questions,
                cfg,
                generate_answers=args.generate_answers,
                dense_model=args.dense_model,
                reranker_model=args.rerank_model,
                rerank_pool=args.rerank_pool,
                use_synonyms=(cfg != "synonyms_disabled")
            )
            res["split"] = split_name
            res["dataset"] = ds_name
            append_to_results_md(res)

            ans_str = f"Strict={res['answer_strict_acc']:.1%}, Lenient={res['answer_lenient_acc']:.1%}" if res.get('answer_strict_acc') is not None else "N/A"
            print(f"Results for {cfg}:")
            print(f"  Recall@5:        {res['complete_unit@5']:.1%}")
            print(f"  Recall@10:       {res['complete_unit@10']:.1%}")
            print(f"  Recall@20:       {res['complete_unit@20']:.1%}")
            print(f"  MRR:             {res['mrr']:.4f}")
            print(f"  Found None@20:   {res['found_none@20']:.1%}")
            print(f"  Forbidden@10:    {res['forbidden@10']}")
            print(f"  Latency/Q:       {res['latency_ms']:.1f}ms (LLM calls: {res['llm_calls_per_q']}/q)")
            print(f"  Answer Acc:      {ans_str}")

            if res.get("context_stats"):
                cs = res["context_stats"]
                print(f"  Context Tradeoff: Top8={cs[8]['coverage']:.1%} ({cs[8]['avg_tokens']} tok) | Top10={cs[10]['coverage']:.1%} ({cs[10]['avg_tokens']} tok) | Top12={cs[12]['coverage']:.1%} ({cs[12]['avg_tokens']} tok)")

            if res["failures"]:
                print(f"  Failures @10 ({len(res['failures'])} / {len(questions)} missed):")
                for fail in res["failures"]:
                    ranks_detail = ", ".join(f"{uid}: {r}" for uid, r in fail.get("needed_ranks", {}).items())
                    print(f"   * {fail['id']} [{fail['category']}]: \"{fail['question']}\" -> recall@10={fail['recall_unit@10']:.1f} | ranks: [{ranks_detail}]")


if __name__ == "__main__":
    main()
