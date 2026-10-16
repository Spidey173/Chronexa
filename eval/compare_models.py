"""Compare Dense Embedding Models on DEV ONLY.

Tests:
1. sentence-transformers/all-MiniLM-L6-v2
2. BAAI/bge-small-en-v1.5
3. BAAI/bge-base-en-v1.5

Measures: Recall@5, Recall@10, MRR, Found None@20 on dev_paraphrases.jsonl.
Strictly restricted to DEV dataset.
"""
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
EVAL_HARNESS_DIR = ROOT_DIR / "eval_harness"
if str(EVAL_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_HARNESS_DIR))

from src.config import DATA_DIR
from src.retrieval.dense import DenseRetriever
from eval.run_eval import EvaluationRunner

EVAL_DIR = ROOT_DIR / "eval"
DEV_PATH = EVAL_DIR / "dev_paraphrases.jsonl"

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5"
]

def main():
    with open(DEV_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"===========================================================")
    print(f" EVALUATING DENSE EMBEDDING MODELS ON DEV ({len(questions)} queries)")
    print(f"===========================================================\n")

    results = []
    for model_name in MODELS:
        print(f"\n--- Testing Model: {model_name} ---")
        runner = EvaluationRunner(DATA_DIR)
        start = time.time()
        
        # Test dense_only config
        res = runner.run_eval(
            questions,
            config="dense_only",
            dense_model=model_name
        )
        elapsed = time.time() - start
        
        results.append({
            "model": model_name,
            "complete_unit@5": res["complete_unit@5"],
            "complete_unit@10": res["complete_unit@10"],
            "mrr": res["mrr"],
            "found_none@20": res["found_none@20"],
            "elapsed_sec": round(elapsed, 2)
        })
        
        print(f"Results for {model_name}:")
        print(f"  Recall@5:        {res['complete_unit@5']:.1%}")
        print(f"  Recall@10:       {res['complete_unit@10']:.1%}")
        print(f"  MRR:             {res['mrr']:.4f}")
        print(f"  Found None @ 20: {res['found_none@20']:.1%}")
        print(f"  Time:            {elapsed:.2f}s")

    print("\n\n===========================================================")
    print(" SUMMARY COMPARISON ON DEV PARAPHRASES")
    print("===========================================================")
    print(f"| Model | Recall@5 | Recall@10 | MRR | Found None@20 | Time (s) |")
    print(f"|---|---|---|---|---|---|")
    for r in results:
        print(f"| `{r['model']}` | {r['complete_unit@5']:.1%} | {r['complete_unit@10']:.1%} | {r['mrr']:.4f} | {r['found_none@20']:.1%} | {r['elapsed_sec']}s |")


if __name__ == "__main__":
    main()
