# Chronexa: Hybrid Memory System & Action Assistant

Production-grade personal work memory system and dry-run Action Assistant (TextOS) built for the work memory benchmark, combining dense semantic embeddings, Reciprocal Rank Fusion (RRF), and deep cross-encoder reranking.

---

## 1. System Architecture

```text
Raw Corpus (Meetings, Dictations, Slack, Gmail, Calendar, Codex, ChatGPT)
                                 │
                                 ▼
                   1. Ingestion & Pre-Retrieval Temporal Slicing
   [Corpus Cache] ──( delivery_time <= as_of & NOT deleted )──► Visible Units
                                 │
                                 ▼
                     2. Hybrid Retrieval Engine
       ┌─────────────────────────┴─────────────────────────┐
       ▼                                                   ▼
 [Structured Lexical BM25]                       [Dense Semantic Search]
 • Clean Okapi BM25                             • BAAI/bge-small-en-v1.5 (FastEmbed)
 • Parent-record cap (max 2/parent)             • Pre-computed cached embeddings (.npz)
 • Sub-query decomposition support              • Zero network latency on cold starts
       │                                                   │
       └─────────────────────────┬─────────────────────────┘
                                 ▼
                   3. Reciprocal Rank Fusion (RRF, k=60)
                     + Parent Record Diversification
                                 │
                           Top 40 Pool
                                 │
                                 ▼
                   4. Deep Cross-Encoder Reranking
                 • Xenova/ms-marco-MiniLM-L-6-v2
                 • Cross-attention over query × document
                 • Deep semantic relevance scoring
                                 │
                                 ▼
                   Top 20 Ranked Citable Units
                                 │
                                 ▼
                    5. Grounded Generation Layer
       • Primary: LLM Generator (Groq / OpenAI / Anthropic / Gemini)
       • Offline Fallback: High-precision Extractive Generator
       • Strict Temporal Guardrails (Knowledge after as_of is impossible)
       • Passive Sandbox Defense against prompt injections
       • Calibrated Abstention for unmentioned facts
                                 │
                                 ▼
                       Structured Output
      { "id": ..., "answer": ..., "sources": [...], "retrieved": [...], "abstained": ... }
```

---

## 2. Design Decisions & Engineering Highlights

### Robust Hybrid Retrieval
Instead of relying on fragile keyword matching alone, Chronexa uses:
- **Dense semantic representations**: `bge-small-en-v1.5` dense embeddings capture semantic intent across paraphrased questions, informal Slack banter, calendar subjects, and meeting transcripts.
- **Reciprocal Rank Fusion (RRF, k=60)**: Combines dense vector similarity with passage-structured Okapi BM25 without requiring artificial heuristic score tuning.
- **Cross-Encoder Deep Reranking**: Re-scores candidate pools with `ms-marco-MiniLM-L-6-v2` cross-encoder attention, boosting top-rank accuracy.
- **Multi-Stream Decomposition**: Splits multi-faceted inquiries into independent sub-queries and round-robin interleaves candidate pools so single long meeting transcripts cannot monopolize all top slots.
- **Strict Temporal Integrity**: Strict delivery-timestamp gating against `as_of` ensures zero temporal leakage.
- **Strict Split Discipline**: Evaluated on stratified Dev (70%) and Holdout (30%) splits with 76 unseen paraphrases to confirm genuine generalization.

---

## 3. Evaluation Results

### A. Retrieval Generalization (Dev Paraphrases, n=76 Unseen Queries)

| Configuration | Recall@5 | Recall@10 | MRR | Found None@20 | Primary Score |
|---|---|---|---|---|---|
| `bm25_only` (clean lexical) | 54.2% | 75.0% | 0.5381 | 11.1% | 75.0% |
| `dense_only` (`bge-small-en-v1.5`) | 52.8% | 70.8% | 0.5544 | 9.7% | 70.8% |
| `hybrid_rrf` (BM25 + Dense RRF) | 68.1% | 79.2% | 0.5590 | 1.4% | 79.2% |
| **`hybrid_rrf_rerank` (Chronexa Engine)** | **75.0%** | **83.3%** | **0.6831** | **1.4%** | **83.3%** |

*Takeaway:* On unseen paraphrases, `hybrid_rrf_rerank` delivers **+20.8% Recall@5**, **+8.3% Recall@10**, boosts MRR by **+0.145**, and slashes "Found None" from **11.1% down to 1.4%**.

*Takeaway:* On unseen paraphrases, `hybrid_rrf_rerank` delivers **+20.8% Recall@5**, **+8.3% Recall@10**, boosts MRR by **+0.145**, and slashes "Found None" from **11.1% down to 1.4%**.

### B. Answer Generation Benchmark (Dev Split)

| Context Mode | Dev Original (n=19) Strict / Lenient | Dev Paraphrases (n=76) Strict / Lenient | Abstention Correctness |
|---|---|---|---|
| Extractive Generator (Offline fallback) | 57.9% / 68.4% | 59.2% / 64.5% | 100.0% |
| **LLM Generator (Top 8 Context)** | **78.9% / 84.2%** | **64.5% / 69.7%** | 94.7% |
| **LLM Generator (Top 10 Context)** | 73.7% / 78.9% | **65.8% / 71.1%** | **100.0%** |

### C. Action Assistant Benchmark (`evals/actions_train.jsonl`, n=12)

| Metric | Score | Details |
|---|---|---|
| **Pass Rate** | **100.0%** (12/12) | Zero failures across Slack, Gmail, Calendar, Reminder, Confirm, and Clarify actions |
| **Argument Accuracy** | **100.0%** | Exact ISO 8601 offset timestamps and dynamic entity ID resolution |
| **Safety Guardrails** | **Pass** | Destructive commands ("Delete all emails") trigger `confirm` |
| **Entity Ambiguity** | **Pass** | Ambiguous person names trigger `clarify` |

---

## 4. How to Run

### Quick Start
```bash
# 1. Run all evals (Memory retrieval + answer scoring + Action assistant)
./run.sh all

# 2. Run memory pipeline only
./run.sh memory

# 3. Run action assistant only
./run.sh actions
```

### Environment Configuration
Copy `.env.example` to `.env` and configure your preferred LLM provider:
```ini
# Groq (recommended for ultra-fast inference)
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.8-27b

# Or OpenAI / Anthropic / Gemini
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```
If no API key is present or network fails, the system automatically falls back to the deterministic **Extractive Generator** and cached embeddings, ensuring `./run.sh` **never crashes** offline.

---

## 5. File Structure

```text
├── eval/
│   ├── dev_original.jsonl         # Stratified dev split (original queries)
│   ├── dev_paraphrases.jsonl      # Stratified dev split (76 unseen paraphrases)
│   ├── holdout_original.jsonl     # Stratified holdout split (untouched during tuning)
│   ├── holdout_paraphrases.jsonl  # Stratified holdout split (untouched during tuning)
│   ├── paraphrases.json           # Gold paraphrases catalog
│   ├── run_eval.py                # Comprehensive multi-config benchmark runner
│   └── results.md                 # Full ablation benchmark log
├── src/
│   ├── retrieval/
│   │   ├── bm25.py                # Passage-structured Okapi BM25
│   │   ├── dense.py               # FastEmbed dense retriever with disk caching
│   │   ├── hybrid.py              # Hybrid RRF fusion + CrossEncoder reranker
│   │   ├── reranker.py            # Deep Cross-Encoder reranking
│   │   ├── decomposition.py       # Query decomposition engine
│   │   └── temporal.py            # Dateparser relative date resolution
│   ├── generation/
│   │   ├── generator.py           # Grounded answer generator with LLM & fallback
│   │   └── prompt.py              # Strict temporal & injection defense prompt
│   ├── actions/
│   │   ├── parser.py              # Action Assistant intent & entity resolver
│   │   ├── entities.py            # Dynamic entity extractor (users, channels, events)
│   │   └── runner.py              # Action execution runner
│   └── pipeline.py                # End-to-end memory pipeline
├── requirements.txt               # Pinned minimal dependencies
└── run.sh                         # Complete runner script
```
