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
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.temporal import parse_iso_datetime
from src.generation.generator import AnswerGenerator


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
        as_of_dt = parse_iso_datetime(as_of_str)

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
            if generator.provider != "none":
                import time
                time.sleep(1.0)
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


def search_single(
    query: str,
    as_of: str = "2026-09-18T18:00:00-07:00",
    data_dir: Path = DATA_DIR,
    mode: str = "hybrid",
    generate_answer: bool = True,
    top_k: int = 10
):
    """Executes a single ad-hoc search query with temporal filtering and grounded answer generation."""
    as_of_dt = parse_iso_datetime(as_of)
    visible_units = get_visible_units(as_of_dt, data_dir=data_dir)

    if mode == "hybrid":
        retriever = HybridRetriever()
        results = retriever.search(query, visible_units, top_k=top_k, as_of=as_of)
    else:
        bm25_idx = BM25Index(visible_units)
        results = bm25_idx.search(query, top_k=top_k)

    retrieved_units = [unit for unit, _ in results]
    scores = [score for _, score in results]

    answer_text = ""
    sources = []
    abstained = False
    if generate_answer:
        generator = AnswerGenerator()
        gen_res = generator.generate(query, as_of, retrieved_units)
        answer_text = gen_res["answer"]
        sources = gen_res["sources"]
        abstained = gen_res["abstained"]

    return {
        "query": query,
        "as_of": as_of,
        "visible_count": len(visible_units),
        "answer": answer_text,
        "sources": sources,
        "abstained": abstained,
        "results": [
            {
                "id": u.id,
                "record": u.record,
                "time": u.time.isoformat() if hasattr(u.time, "isoformat") else str(u.time),
                "speaker": getattr(u, "speaker", ""),
                "source_type": getattr(u, "source_type", ""),
                "text": u.text,
                "score": float(sc)
            }
            for u, sc in zip(retrieved_units, scores)
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Chronexa end-to-end memory runner")
    parser.add_argument("--query", "-q", default=None, help="Run an ad-hoc single natural language query")
    parser.add_argument("--as-of", default="2026-09-18T18:00:00-07:00", help="Historical cutoff date for query (ISO-8601)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of evidence candidates to return")
    parser.add_argument("--questions", default=str(EVALS_DIR / "memory_train.jsonl"), help="Path to questions JSONL")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "memory_train_answers.jsonl"), help="Path to output JSONL")
    parser.add_argument("--data", default=str(DATA_DIR), help="Path to data directory")
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "hybrid"], help="Retrieval mode")
    parser.add_argument("--no-generate", action="store_true", help="Skip answer generation")
    args = parser.parse_args()

    if args.query:
        res = search_single(
            args.query,
            as_of=args.as_of,
            data_dir=Path(args.data),
            mode=args.mode,
            generate_answer=not args.no_generate,
            top_k=args.top_k
        )
        print(f"\n=======================================================")
        print(f" Query: {res['query']}")
        print(f" As-Of: {res['as_of']} ({res['visible_count']} units visible)")
        print(f"=======================================================")
        if res["answer"]:
            print(f"\nAnswer:\n{res['answer']}")
            print(f"\nCitations: {res['sources']}")
            print(f"Abstained: {res['abstained']}")
        print(f"\nTop Retrieved Evidence Units ({len(res['results'])}):")
        for i, item in enumerate(res['results'][:5], 1):
            speaker_str = f" [{item['speaker']}]" if item.get('speaker') else ""
            print(f" [{i}] {item['id']} ({item['record']}){speaker_str} | Date: {item['time'][:16]}")
            snippet = item['text'].replace('\n', ' ')[:140]
            print(f"     \"{snippet}...\"\n")
        return

    run_pipeline(
        Path(args.questions),
        Path(args.output),
        Path(args.data),
        mode=args.mode,
        generate_answers=not args.no_generate
    )


if __name__ == "__main__":
    main()
