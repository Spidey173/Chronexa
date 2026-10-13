import json
from pathlib import Path
import sys
sys.path.insert(0, ".")
from eval_harness import records

data_dir = Path("data")
with open("evals/memory_train.jsonl") as f:
    gold = {item["id"]: item for item in [json.loads(l) for l in f if l.strip()]}

with open("outputs/memory_train_bm25.jsonl") as f:
    answers = {item["id"]: item for item in [json.loads(l) for l in f if l.strip()]}

ctx = records.context(data_dir)
record_of = ctx["record_of"]

failed_ids = ["MEM-TR-01", "MEM-TR-02", "MEM-TR-03", "MEM-TR-04", "MEM-TR-05", "MEM-TR-12", "MEM-TR-14", "MEM-TR-20", "MEM-TR-21", "MEM-TR-25", "MEM-TR-26"]

for qid in failed_ids:
    g = gold[qid]
    a = answers[qid]
    retrieved = a.get("retrieved", [])
    needed = g.get("needed", [])
    print("=" * 70)
    print(f"ID: {qid} | Cat: {g['category']} | as_of: {g['as_of']}")
    print(f"Q: {g['question']}")
    print(f"Top 5 Retrieved: {retrieved[:5]}")
    for idx, grp in enumerate(needed):
        found_rank = None
        found_id = None
        for r_idx, r_id in enumerate(retrieved):
            if r_id in grp or record_of.get(r_id) in grp:
                found_rank = r_idx + 1
                found_id = r_id
                break
        print(f"  Group {idx+1}: {grp} -> Rank in top20: {found_rank} ({found_id})")
