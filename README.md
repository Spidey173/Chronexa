# ⏱️ Chronexa: Workplace-Memory Search & Action Assistant

Chronexa is a standalone workplace-memory search engine and safe, dry-run action-planning assistant. It indexes scattered workplace records (Slack messages, Gmail threads, Google Calendar events, meeting transcripts, and voice dictations), searches across historical states with strict point-in-time ("as of") date filtering, and parses natural-language requests into strictly validated preview actions.

---

## 🎯 Core Capabilities

1. **Strict Point-in-Time Filtering:** Pre-filters documents at the ingestion stage (`timestamp <= as_of`). Records created, edited, or deleted after the query cutoff date are physically excluded from retrieval, preventing temporal data leakage.
2. **Hybrid Lexical & Semantic Retrieval:**
   - **Lexical:** Okapi BM25 index over tokenized workplace records for exact identifiers, dates, and names.
   - **Semantic:** FastEmbed dense vectors (`BAAI/bge-small-en-v1.5`) running locally on CPU.
   - **Fusion & Reranking:** Merges rankings via Reciprocal Rank Fusion (RRF) with reciprocal rank blending and date proximity weighting.
3. **Evidence-Grounded Generation & Injection Defense:**
   - Encloses retrieved records within explicit `<evidence_records>` XML boundaries.
   - Neutralizes prompt injection patterns and breakout tags embedded inside untrusted workplace messages.
   - Features calibrated abstention when evidence is absent (e.g., undisclosed compensation, unverified credentials, missing credentials).
4. **Strict Schema-Validated Dry-Run Actions:**
   - Translates natural language into structured actions (`slack.send_message`, `gmail.send`, `calendar.create_event`, `calendar.update_event`, `reminder.create`, `memory.ask`, `app.open`, `clarify`, `confirm`).
   - Validates all generated actions against strict schemas: verifies allowlisted types, required fields, ISO 8601 datetimes, email formatting, and **explicitly rejects unauthorized extra arguments**.
   - Zero side-effects: Operates strictly in preview / dry-run mode without connecting to live external accounts.
5. **Standalone Client-Side Dashboard:**
   - Interactive web interface in `index.html` (deployable on Vercel or openable directly in any modern browser).
   - Renders search timelines, query results, and dry-run action plans with sanitized HTML to prevent XSS injection.

---

## 🏗️ Architecture & Trust Boundaries

```
User Query + Cutoff Date (as_of)
               │
               ▼
┌──────────────────────────────────────────────┐
│ 1. Ingestion & Temporal Filter (loader.py)   │
│    - Parses ISO-8601 & timezone offsets      │
│    - Applies chronologically ordered edits   │
│    - Discards records where timestamp > as_of│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. Hybrid Retrieval (retrieval/)             │
│    ├── Lexical Index (BM25Okapi)             │
│    └── Dense Vectors (FastEmbed BGE CPU)     │
│    └── Reciprocal Rank Fusion (RRF k=60)     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 3. Untrusted Content Sandboxing (prompt.py)  │
│    - Sanitizes XML breakout tags             │
│    - Enforces prompt-injection isolation     │
│    - Calibrated abstention on missing facts  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 4. Grounded Output / Dry-Run Action Parser   │
│    - Grounded Answer Generation (LLM/Local)  │
│    - Schema Validation (src/actions/schema)  │
│    - Preview actions (no side-effects)       │
└──────────────────────────────────────────────┘
```

---

## 📊 Evaluation & Benchmark Results

Chronexa was evaluated across three distinct evaluation datasets:
1. **Memory Dev Split (19 original queries + 76 paraphrased variations = 95 total queries):** Used during active development to calibrate retrieval fusion and temporal filters.
2. **Memory Holdout Split (8 original queries + 32 paraphrased variations = 40 total queries):** Held out completely unobserved during development to test out-of-domain generalization without data leakage or overfitting.
3. **Action Benchmark (12 natural-language workplace commands):** Assesses action type selection, argument extraction, entity resolution, and schema compliance.

### 1. Retrieval Performance

| Split | Queries | Top-5 Recall | Top-10 Recall | Top-20 Recall | MRR |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dev Split (Original)** | 19 | 88.9% | 94.4% | 100.0% | 0.7870 |
| **Dev Split (Paraphrases)** | 76 | 75.0% | 83.3% | 100.0% | 0.6831 |
| **Holdout Split (Original)** | 8 | 42.9% | 71.4% | 100.0% | 0.5315 |
| **Holdout Split (Paraphrases)** | 32 | 35.7% | 78.6% | 100.0% | 0.4422 |
| **Train Set** | 27 | 96.3% | 100.0% | 100.0% | 0.8912 |

> **Evaluation Insight:** On the unobserved Holdout set, Top-10 recall stands at **71.4% (original)** and **78.6% (paraphrases)**. This reflects real-world workplace vocabulary gaps when questions refer to unindexed internal nicknames or topics without prior lexical alignment.

### 2. Answer Generation Accuracy

- **Train Set (27 queries, Deterministic Exact-Match):** **88.9%** (24/27 queries match exact canonical answer strings; remaining 3 questions capture valid intermediate project dates requiring semantic equivalence or LLM judging).
- **Dev Set (with Groq `llama-3.3-70b` / OpenAI `gpt-4o-mini` LLM Judge):**
  - Dev Original Queries: **78.9%** strict accuracy
  - Dev Paraphrased Queries: **65.8%** strict accuracy

### 3. Action Assistant Accuracy & Schema Compliance

- **Command Dataset:** 12 complex workplace commands (e.g. reschedule meetings, send email to multiple stakeholders, post Slack updates, request confirmations for destructive actions).
- **Execution Mode:** 100% Dry-Run / Preview.
- **Schema Compliance:** **12 / 12 (100%)** valid action plans.
- **Safety Enforcement:** Strict type validation (`src/actions/schema.py`) verifies all required fields, enforces email formats, validates ISO 8601 timestamps, and rejects unknown or unauthorized extra arguments with descriptive `ActionValidationError` exceptions.

---

## ⚡ Setup & Installation

### Prerequisites
- Python 3.10+ (tested through Python 3.14)
- Git

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Spidey173/Chronexa.git
cd Chronexa

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|:---:|:---:|---|
| `GROQ_API_KEY` | Optional | `""` | API key for Groq inference (enables Llama-3.3-70B generation and evaluation). |
| `OPENAI_API_KEY` | Optional | `""` | API key for OpenAI inference (alternative LLM provider). |
| `CHRONEXA_OFFLINE` | Optional | `0` | Set to `1` to run retrieval and evaluation completely offline without API calls. |

---

## 🧪 Running Tests & Evaluations

The project includes an execution script `run.sh` supporting multiple operational modes:

### 1. Run Automated Unit Tests
Runs the test suite (temporal edge cases, hybrid retrieval ranking, action schema validation, prompt-injection defense, and abstention):

```bash
./run.sh test
# or: .venv/bin/python -m unittest discover tests -v
```
*(27 tests covering all safety and correctness requirements; executes in ~8 seconds on CPU).*

### 2. Run Benchmark Evaluations (Dev & Holdout Splits)

```bash
# Evaluate both Dev and Holdout splits offline:
./run.sh eval

# Evaluate specifically on Dev or Holdout split:
./run.sh eval-dev
./run.sh eval-holdout
```

### 3. Run Memory Pipeline & Actions Pipeline

```bash
# Run historical memory retrieval & scoring:
./run.sh memory

# Run dry-run action assistant & schema verification:
./run.sh actions

# Run complete end-to-end suite (tests + memory + actions):
./run.sh all
```

---

## 🖥️ Web Dashboard Demo

Chronexa includes a client-side dashboard (`index.html`) featuring:
- Historical point-in-time date slider (demonstrating how query results change as time progresses).
- Keyword + semantic hybrid search tester.
- Dry-run TextOS action planner with structured JSON previews.
- Full client-side HTML escaping on all dynamic data fields.

### Running the Dashboard Locally:
No web server setup is required. Open `index.html` directly in your browser:
```bash
open index.html  # On macOS
# or: xdg-open index.html (Linux) / start index.html (Windows)
```

---

## 📁 Repository Structure

```text
Chronexa/
├── data/                      # Sample workplace records (Slack, Gmail, Calendar, Meetings)
├── src/
│   ├── ingestion/             # Record loading, timezone parsing, and temporal filtering
│   │   ├── loader.py          # Enriches headers, handles naive/aware dates, applies edits
│   │   └── mock_data.py       # Raw source corpus definitions
│   ├── retrieval/             # Search engine core
│   │   ├── bm25.py            # Okapi BM25 implementation
│   │   ├── dense.py           # FastEmbed CPU embedding indexer
│   │   ├── ranker.py          # Reciprocal Rank Fusion (RRF) & reranking
│   │   ├── temporal.py        # Point-in-time cutoff logic and timezone normalization
│   │   └── decomposition.py   # Query sub-goal decomposition
│   ├── generation/            # Evidence-grounded generation
│   │   ├── prompt.py          # XML-sandboxed prompt templates and injection boundaries
│   │   └── generator.py       # LLM generation client, fallback synthesis, and abstention
│   ├── actions/               # TextOS Dry-Run Action Assistant
│   │   ├── schema.py          # Strict Pydantic-style action schemas and field validators
│   │   ├── parser.py          # Natural language command parser with deterministic fallback
│   │   └── runner.py          # Batch runner and schema validator
│   └── pipeline.py            # End-to-end memory retrieval pipeline CLI
├── eval/                      # Evaluation drivers
│   └── run_eval.py            # Multi-split evaluation harness (dev, holdout, offline)
├── eval_harness/              # Ground-truth evaluation scoring utilities
│   ├── records.py             # Ground-truth record loading and visibility logic
│   ├── score_retrieval.py     # Recall@K and MRR computation
│   ├── score_memory.py        # Deterministic match and LLM-judge memory scoring
│   └── score_actions.py       # Action schema compliance and argument validator
├── evals/                     # Benchmark datasets (Train, Dev, Holdout splits)
├── tests/                     # Automated unit test suite
│   ├── test_temporal.py       # Timezone, edit history, and boundary cutoff tests
│   ├── test_retrieval.py      # BM25, dense search, and RRF fusion tests
│   ├── test_actions.py        # Strict schema validation and rejection tests
│   ├── test_injection.py      # Prompt injection and XML boundary tests
│   └── test_abstention.py     # Calibrated abstention tests on missing evidence
├── index.html                 # Standalone web demo dashboard
├── vercel.json                # Vercel deployment configuration with HTTP security headers
├── requirements.txt           # Python dependencies
└── run.sh                     # Automated CLI execution wrapper
```

---

## 🛡️ Security, Privacy & Limitations

- **Untrusted Content:** All stored communications are treated as untrusted data. Records are enclosed in `<evidence_records>` tags, and breakout strings (`</record>`, `</evidence_records>`) are sanitized to prevent indirect prompt injection attacks.
- **Action Safety:** Actions are never executed against live APIs in this repository. All generated actions are output in structured dry-run JSON format for human-in-the-loop inspection.
- **Privacy Boundary:** When configured with `CHRONEXA_OFFLINE=1` (or without cloud API keys), all indexing, embedding, retrieval, and action parsing run 100% locally on CPU without outbound network requests.
- **Known Limitations:**
  - FastEmbed downloads model weights (`BAAI/bge-small-en-v1.5`, ~130MB) on the very first run, caching them in `~/.cache/fastembed/`.
  - Holdout retrieval performance demonstrates that unseen synonyms require additional vocabulary expansion or domain-specific fine-tuning for production deployments.

---

## 👤 Author

**Pruthvi R**  
* **GitHub:** [@Spidey173](https://github.com/Spidey173)  
* **Portfolio:** [pruthvi-17.vercel.app](https://pruthvi-17.vercel.app/)  
* **LinkedIn:** [linkedin.com/in/pruthvir173](https://www.linkedin.com/in/pruthvir173/)
