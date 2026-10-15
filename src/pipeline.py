"""End-to-end memory pipeline: Temporal Filtering -> Hybrid Retrieval -> Grounded Answer Generation.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, EVALS_DIR, OUTPUTS_DIR
from src.ingestion.loader import get_visible_units, _get_raw_records
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import AnswerGenerator


def _parse_dt(dt_str: str) -> datetime:
    return datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))


def run_pipeline(
    questions_path: Path,
    output_path: Path,
    data_dir: Path = DATA_DIR,
    mode: str = "hybrid",
    generate_answers: bool = True
):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    retriever = HybridRetriever() if mode == "hybrid" else None
    generator = AnswerGenerator() if generate_answers else None

    # Pre-warm embedding cache for all units if dense is active
    if retriever and hasattr(retriever, "dense") and retriever.dense is not None:
        all_raw_units, _, _ = _get_raw_records(data_dir)
        print(f"Pre-computing embeddings for {len(all_raw_units)} corpus units...")
        retriever.dense.warm_cache(all_raw_units)
        print("Embeddings ready.")

    output_lines = []
    for idx, item in enumerate(questions, start=1):
        q_id = item["id"]
        q_text = item["question"]
        as_of_str = item["as_of"]
        as_of_dt = _parse_dt(as_of_str)

        # 1. Strictly temporally visible units
        visible_units = get_visible_units(as_of_dt, data_dir=data_dir)

        # 2. Retrieve top 20 candidates
        if mode == "hybrid":
            results = retriever.search(q_text, visible_units, top_k=20, as_of=as_of_str)
        else:
            bm25_idx = BM25Index(visible_units)
            results = bm25_idx.search(q_text, top_k=20)

        retrieved_ids = [unit.id for unit, _ in results]
        retrieved_units = [unit for unit, _ in results]

        # 3. Answer Generation (Phase 3)
        if generate_answers:
            print(f"[{idx}/{len(questions)}] Generating answer for {q_id}...")
            gen_res = generator.generate(q_text, as_of_str, retrieved_units)
            answer_text = gen_res["answer"]
            sources_ids = gen_res["sources"]
            abstained_flag = gen_res["abstained"]
            import time
            time.sleep(3.5)
        else:
            answer_text = ""
            sources_ids = retrieved_ids[:3]
            abstained_flag = False

        record = {
            "id": q_id,
            "answer": answer_text,
            "sources": sources_ids,
            "retrieved": retrieved_ids,
            "abstained": abstained_flag
        }
        output_lines.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in output_lines:
            f.write(json.dumps(rec) + "\n")

    print(f"Pipeline complete! Output {len(output_lines)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Chronexa end-to-end memory runner")
    parser.add_argument("--questions", default=str(EVALS_DIR / "memory_train.jsonl"), help="Path to questions JSONL")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "memory_train_answers.jsonl"), help="Path to output JSONL")
    parser.add_argument("--data", default=str(DATA_DIR), help="Path to data directory")
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "hybrid"], help="Retrieval mode")
    parser.add_argument("--no-generate", action="store_true", help="Skip answer generation")
    args = parser.parse_args()

    run_pipeline(
        Path(args.questions),
        Path(args.output),
        Path(args.data),
        mode=args.mode,
        generate_answers=not args.no_generate
    )


if __name__ == "__main__":
    main()
