# ⏱️ Chronexa: Workplace-Memory Search & Safe Action Previews

**Chronexa** is a prototype desktop workplace-memory assistant designed to search across scattered enterprise communications (Slack messages, Gmail threads, Google Calendar events, meeting transcripts, and voice dictations) with strict historical point-in-time ("as of") date filtering and generate safe, schema-validated dry-run action plans.

Built as an engineering portfolio project to tackle three common limitations in corporate search:
1. **Temporal Inconsistency & Future Leakage:** Standard RAG pipelines index all documents uniformly. If a user asks what the launch plan was on Sept 9, naive search leaks delays announced on Sept 16. Chronexa enforces pre-retrieval physical boundaries (`timestamp <= as_of`).
2. **Vocabulary & Identifier Mismatch:** Keyword search fails on colloquial paraphrasing; pure vector search fails on exact dates, project codenames, and employee IDs. Chronexa combines Okapi BM25 with local FastEmbed dense vectors via Reciprocal Rank Fusion (RRF).
3. **Unsafe Autonomous Execution:** AI assistants should never silently trigger destructive external side-effects. Chronexa translates natural language into structured dry-run JSON actions validated against strict allowlisted schemas, requesting confirmation for destructive operations and clarification for ambiguous entities.

---

## 🏗️ Architecture & Data Flow

```text
User Question / Command + Cutoff Date (as_of)
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Ingestion & Temporal Filtering (src/ingestion/)     │
│    - Loads records across 7 workplace connectors       │
│    - Normalizes ISO-8601 & timezone offsets (-07:00)   │
│    - Applies chronological message edits & deletions   │
│    - Discards any unit where timestamp > as_of         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Hybrid Retrieval Engine (src/retrieval/)            │
│    ├── Lexical Index: Pure Python Okapi BM25           │
│    └── Dense Vectors: Local FastEmbed BGE CPU          │
│    └── Rank Fusion: Reciprocal Rank Fusion (RRF k=60)  │
│    └── Cross-Encoder Reranker (ms-marco-MiniLM)        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Evidence Sandboxing & Defense (src/generation/)     │
│    - Encloses records in <evidence_records> XML tags   │
│    - Sanitizes breakout tags and prompt injection text │
│    - Calibrated abstention when evidence is missing    │
└──────────────────────────┬─────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Grounded Answer Synthesis │ │ Safe Action Preview       │
│ - Evidence citations      │ │ - Schema validation       │
│ - Strict grounding rules  │ │ - Extra arg rejection     │
│ - Local / Groq fallback   │ │ - Preview-only (dry-run)  │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 🚀 What Actually Works & Implementation Status

| Component | Status | Details |
|---|:---:|---|
| **Temporal Filtering** | ✅ Working | Pre-retrieval boundary excludes future records; masks deleted messages; handles chronological edit trails. |
| **Hybrid Search (BM25 + BGE)** | ✅ Working | BM25 exact match + local FastEmbed (`BAAI/bge-small-en-v1.5`) merged via RRF. Runs 100% locally on CPU without external API keys. |
| **Cross-Encoder Reranker** | ✅ Working | Local reranking via `Xenova/ms-marco-MiniLM-L-6-v2` over top candidates. |
| **Dry-Run Action Parser** | ✅ Working | Translates commands into validated JSON actions. Enforces allowlist (`slack.send_message`, `gmail.send`, `calendar.create_event`, `calendar.update_event`, `reminder.create`, `memory.ask`, `app.open`, `clarify`, `confirm`) and rejects extra/unknown arguments. |
| **Prompt Injection Defense** | ✅ Working | XML tag boundary sandboxing; neutralizes indirect injection instructions embedded in mock messages. |
| **Browser Demo (`index.html`)** | ✅ Working | Client-side dashboard replaying pre-computed benchmark outputs, demonstrating temporal sliding and schema previews. |
| **External Live Service Actions** | 🚫 Out of Scope | Actions are strictly dry-run preview objects; no live OAuth integrations, emails, or Slack API calls are executed. |

### ⚙️ Optional Dependencies & Reduced Functionality

Chronexa includes import guards and fallback logic for optional packages so the pipeline can operate without crashing if external dependencies are missing. 

> [!NOTE]
> The fallback code gracefully handles missing dependencies (for instance, missing `fastembed` bypasses dense vector search and cross-encoder reranking, falling back to pure BM25 search). In the test suite, these fallback paths are verified by simulating unavailable models/cache in unit tests; they do not uninstall the package to verify a clean install end-to-end.

When these packages are omitted or unavailable, the system operates with **explicitly reduced functionality**:

| Omitted Package | Fallback Behavior | Reduced Functionality Impact |
|---|---|---|
| **Without `fastembed`** | Falls back to pure BM25 search. `DenseRetriever` returns empty matches and `CrossEncoderReranker` acts as a passthrough. | **Lexical-only retrieval:** Retrieval runs purely via Okapi BM25. Semantic vector similarity and deep cross-attention reranking are disabled. |
| **Without `dateparser`** | Falls back to explicit ISO regex (`YYYY-MM-DD`) and `datetime.strptime` patterns. | **No relative date boosting:** Relative expressions like *"yesterday"*, *"last week"*, or *"next Monday"* are not resolved for temporal score boosting or calendar chaining. Queries rely strictly on lexical matching. |

---

## 📁 Evaluation Datasets & Runners

Chronexa includes structured evaluation datasets across Development and Holdout splits:
- **Dev Split (`evals/memory_train.jsonl`):** 19 original questions + 76 paraphrased variants used to inspect retrieval rank fusion and temporal boundaries.
- **Holdout Split (`eval/holdout_original.jsonl` & `eval/holdout_paraphrases.jsonl`):** 20 core questions, each with four paraphrases (80 paraphrases, 100 total evaluation runs), providing held-out questions used to evaluate phrasing robustness across diverse user queries.
- **Action Benchmark (`evals/actions_train.jsonl`):** 12 complex workplace commands evaluated for type resolution, argument extraction, and strict schema compliance.

Run the evaluation commands locally without cloud API dependencies:
```bash
# Run memory retrieval & answer evaluation:
./run.sh memory

# Run action schema validation & safety guardrail checks:
./run.sh actions

# Run holdout retrieval evaluation:
./run.sh eval-holdout hybrid_rrf
```

---

## 🛠️ Setup & How to Run

### Prerequisites
- Python 3.10+ (tested through Python 3.14)
- macOS or Linux

### 1. Installation

```bash
git clone https://github.com/Spidey173/Chronexa.git
cd Chronexa

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|:---:|:---:|---|
| `CHRONEXA_OFFLINE` | No | `0` | Set to `1` to force 100% offline local CPU execution (no network calls). |
| `GROQ_API_KEY` | No | `""` | Optional Groq key for generative answer synthesis and LLM-as-a-judge scoring. |
| `OPENAI_API_KEY` | No | `""` | Optional OpenAI key for answer generation. |

### 3. Execution Commands via `run.sh`

The runner script [`run.sh`](run.sh) provides simple commands for all operations:

```bash
# 1. Run the automated test suite (46 unit and integration tests):
./run.sh test

# 2. Run the dry-run action assistant benchmark:
./run.sh actions

# 3. Run the memory retrieval and answer scoring pipeline:
./run.sh memory

# 4. Run the holdout evaluation benchmark:
./run.sh eval-holdout

# 5. Run complete end-to-end verification (test + memory + actions):
./run.sh all
```

### 4. 🔍 "Run This Query Locally" Guide (Ad-Hoc CLI Search)

You can query the pipeline directly from the command line for any custom question and historical cutoff date:

```bash
# Basic query using hybrid BM25 + dense retrieval as of Sep 9:
./run.sh query "When are we launching Route Planner?" "2026-09-09T09:00:00-07:00"

# Same query as of Sep 18 (after subsequent delays were decided):
./run.sh query "When are we launching Route Planner?" "2026-09-18T18:00:00-07:00"

# Query the status of a client contract:
./run.sh query "Did Sarah Patel approve the pricing proposal?" "2026-09-16T15:00:00-07:00"

# Pure lexical BM25 mode:
./run.sh query "What is the NRR metric fix?" "2026-09-18T18:00:00-07:00" "bm25"

# Direct invocation via Python CLI:
python3 src/pipeline.py --query "What blockers did Priya mention for QA?" --as-of "2026-09-11T12:00:00-07:00" --top-k 5
```

Each run displays the strictly visible message pool size, the grounded synthesized answer (with local fallback if offline), cited source records, and the top candidate units with timestamps.

### 5. 🚀 Local Demo Server & Interactive Dashboard (Loopback Prototype)

Chronexa provides a lightweight local demonstration server built on Python's standard library (`http.server`). It is designed specifically for interactive developer testing and portfolio exploration on a trusted local machine—not as a production SaaS web service. For safe local evaluation, it binds strictly to local loopback (`127.0.0.1`), enforces strict CORS origin checks, caps request bodies at 64 KB, and shares pre-warmed retrieval engines.

To connect the browser dashboard to your live local Python search engine:

```bash
# Launch the built-in local demo server (port 8000 default):
./run.sh serve
# or: python3 src/server.py --port 8000
```

1. Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
2. The top banner will immediately update to:  
   `🟢 LIVE LOCAL PIPELINE ACTIVE · FULL CORPUS SEARCH CONNECTED`
   *(An 8-second background heartbeat keeps the UI banner in sync; if the server is stopped, the UI cleanly falls back to pre-indexed replay mode).*
3. **Live Search**: Enter any arbitrary question or adjust the historical date slider. The UI executes a live POST request to `/api/search` on your local Python backend using the pre-warmed hybrid BM25 + dense engine and renders retrieved citations and synthesized answers dynamically.
4. **Live Action Assistant**: Enter natural language commands (e.g. *"Open Figma"*, *"Message Sarah on Slack that the geocoding fix looks good"*, *"Delete all my emails from Marcus"*). The UI posts to `/api/action`, invoking `ActionParser` to produce strictly validated dry-run action JSON with destructive confirmation and ambiguity guards.

*(If opened directly as a static file in the browser or hosted on static edge without the local Python server running, the dashboard automatically and honestly falls back to pre-indexed benchmark replay mode).*

---

## 🧪 Automated Test Suite

Chronexa defines 46 test methods across unit and integration tests covering safety, edge cases, algorithmic correctness, API endpoints, and optional dependency fallbacks:

- [`tests/test_temporal.py`](tests/test_temporal.py) (10 tests): Timezone normalization (naive vs. aware datetimes), temporal boundary cutoffs, edit trail preservation, deletion masking, and **dateparser fallback behavior** (tested by simulating parser unavailability to ensure explicit ISO dates extract cleanly while relative expressions degrade gracefully without crashing).
- [`tests/test_retrieval.py`](tests/test_retrieval.py) (5 tests): BM25 exact identifier matching, synonym retrieval, temporal boundary enforcement (`timestamp <= as_of`), and **fastembed fallback behavior** (tested by simulating model/cache unavailability to verify dense retriever returns empty matches and reranker acts as safe passthrough to preserve pure BM25 ranking).
- [`tests/test_actions.py`](tests/test_actions.py) (10 tests): Strict schema validation, rejection of unauthorized extra arguments, email/datetime format checking, ambiguity clarification, and destructive confirmation.
- [`tests/test_injection.py`](tests/test_injection.py) (2 tests): XML breakout tag escaping and prompt injection directive neutralization.
- [`tests/test_abstention.py`](tests/test_abstention.py) (4 tests): Calibrated abstention on unrecorded facts (e.g. SOC 2 reports, salary figures).
- [`tests/test_server.py`](tests/test_server.py) (15 tests): Local API server integration tests covering health check, live search, live action parsing, input validation bounds (`top_k`, `mode`, `as_of`), malformed JSON handling, CORS headers, payload limits, and loopback security.

To execute and verify the 46 test methods:
```bash
# Recommended via project test runner (executes pytest with verbose output):
./run.sh test

# Or direct pytest:
python3 -m pytest tests/ -v

# Or standard unittest:
python3 -m unittest discover tests -v
```

---

## 👤 Engineering Contributions vs. External Libraries

- **Custom Implementations:**
  - Temporal boundary filter and chronological edit/deletion reconciler (`src/retrieval/temporal.py`, `src/ingestion/loader.py`).
  - Pure Python BM25 indexer with custom tokenization and entity boosting (`src/retrieval/bm25.py`).
  - Reciprocal Rank Fusion (RRF) logic and candidate pool merger (`src/retrieval/ranker.py`).
  - Strict Action Schemas with field-level validators and unauthorized argument rejections (`src/actions/schema.py`).
  - XML sandboxing, prompt injection sanitization, and calibrated abstention logic (`src/generation/prompt.py`, `src/generation/generator.py`).
  - 46-test automated verification suite (`tests/`).
- **External Libraries Used:**
  - `fastembed`: ONNX-based local embedding inference (`BAAI/bge-small-en-v1.5`).
  - `numpy`: Vector array operations and similarity scoring.
  - `dateparser`: Fallback fuzzy date parsing.

---

## ⚠️ Known Limitations & Future Work

1. **Synthetic Corpus Size:** Evaluated on a mock dataset of 1,316 records across 7 connectors. Production enterprise deployments would require streaming connectors and partition pruning for tens of thousands of records.
2. **Vocabulary & Phrasing Gap:** Retrieval variance on colloquial phrasing highlights the opportunity for domain-specific query expansion or fine-tuned embeddings tailored to internal company jargon.
3. **CPU Reranker Latency:** Cross-encoder reranking over 40 candidates on CPU averages ~2.5s per query. In production, this would be accelerated via ONNX INT8 quantization or GPU batching.
4. **Preview-Only Actions:** The action assistant outputs verified JSON payloads with human-in-the-loop confirmation. Live integrations with Google Workspace or Slack APIs would require OAuth token management and webhook listeners.
