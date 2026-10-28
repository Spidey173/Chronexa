# ⏱️ Chronexa: Workplace-Memory Search & Safe Action Previews

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-50%20passed-success.svg)](tests/)
[![FastEmbed Local ONNX](https://img.shields.io/badge/embeddings-FastEmbed%20ONNX-orange.svg)](https://github.com/qdrant/fastembed)
[![Point-in-Time Boundaries](https://img.shields.io/badge/temporal%20filter-timestamp%20%3C%3D%20as__of-emerald.svg)](src/retrieval/temporal.py)
[![Dry-Run Action Schema](https://img.shields.io/badge/actions-dry--run%20preview-purple.svg)](src/actions/schema.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Chronexa** is an engineering portfolio project demonstrating a prototype workplace-memory assistant. It searches across mock workplace communications—**Slack messages and threads, Gmail conversations, Google Calendar events, meeting transcripts, and voice dictation notes**—with strict historical point-in-time (`as_of`) date filtering and generates schema-validated dry-run action plans.

> **Project Scope:**  
> This is a portfolio proof-of-concept operating on a local synthetic corpus (~1,300 records across 7 mock data sources). Actions are strictly dry-run JSON previews with human confirmation guardrails; no live external SaaS APIs (OAuth, real email sending, or Slack posting) are executed.

---

## 🔍 Technical Focus Areas

This project explores three practical challenges in workplace search and assistants:

1. **Point-in-Time Temporal Boundaries (`timestamp <= as_of`):**  
   Standard search pipelines index all documents uniformly. When asking what a project's timeline was on September 9, naive search can leak delays announced a week later. Chronexa applies pre-retrieval timestamp filtering (`timestamp <= as_of`), masks deleted messages, and reconciles message edit histories chronologically.

2. **Hybrid Lexical & Semantic Retrieval:**  
   Colloquial questions benefit from semantic similarity, while exact project codes, ticket IDs, and person names require lexical precision. Chronexa combines **passage-structured Okapi BM25** with **local FastEmbed dense vectors (`BAAI/bge-small-en-v1.5`)** fused via **Reciprocal Rank Fusion (RRF, k=60)**.

3. **Dry-Run Action Schema Validation:**  
   Translates user commands into structured JSON actions validated against allowlisted schemas (`slack.send_message`, `gmail.send`, `calendar.create_event`, `calendar.update_event`, `reminder.create`, `app.open`, `memory.ask`, `clarify`, `confirm`). It rejects unknown/extra arguments, asks for clarification on ambiguous names, and requires confirmation for destructive operations.

---

## 🏗️ Architecture & Data Flow

```text
                  User Question / Command + Cutoff Date (as_of)
                                       │
                                       ▼
       ┌────────────────────────────────────────────────────────┐
       │ 1. Point-in-Time Filter (src/ingestion/)               │
       │    - Loads records across 7 workplace mock connectors  │
       │    - Normalizes ISO-8601 & timezone offsets (-07:00)   │
       │    - Discards any record where timestamp > as_of       │
       │    - Applies chronological edit trails and deletions   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ 2. Hybrid Retrieval Engine (src/retrieval/)            │
       │    ├── Lexical Index: Okapi BM25 with Snowball stemmer │
       │    ├── Dense Vectors: Local BGE-small ONNX CPU         │
       │    └── Rank Fusion: Reciprocal Rank Fusion (RRF k=60)  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ 3. Evidence Sandboxing & Formatting (src/generation/)  │
       │    - Wraps retrieved records in <record> XML tags      │
       │    - Sanitizes potential prompt injection directives   │
       │    - Calibrated abstention when records are absent     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │ Grounded Answer Synthesis │ │ Dry-Run Action Preview    │
       │ - Evidence citations      │ │ - Schema validation       │
       │ - Point-in-time answers   │ │ - Unknown arg rejection   │
       │ - Local & API providers   │ │ - Disambiguation & confirm│
       └───────────────────────────┘ └───────────────────────────┘
```

---

## ⚙️ Component Summary

| Component | Status | Implementation Details |
|---|:---:|---|
| **Point-in-Time Boundary** | Implemented | Discards any unit where `timestamp > as_of`. Masks deleted records and reconciles multiline chronological edit trails. (`src/ingestion/loader.py`, `src/retrieval/temporal.py`). |
| **BM25 Lexical Index** | Implemented | Pure Python Okapi BM25 with tokenization caching, Snowball stemming (`py_rust_stemmers`), domain synonym expansion, and ISO timestamp normalization. (`src/retrieval/bm25.py`). |
| **Dense Embeddings** | Implemented | Local FastEmbed ONNX runtime (`BAAI/bge-small-en-v1.5`) with persistent `.npz` vector caching and pre-stacked matrix dot-products. (`src/retrieval/dense.py`). |
| **Hybrid Rank Fusion** | Implemented | Reciprocal Rank Fusion (RRF) merging BM25 and dense results with shared index reuse and conversational meeting turn expansion. (`src/retrieval/hybrid.py`). |
| **Dry-Run Action Parser** | Implemented | Translates arbitrary commands into validated action objects with contextual entity disambiguation. Enforces strict schema allowlists and rejects extra arguments. (`src/actions/`). |
| **Injection Defense** | Implemented | XML document sandboxing, punctuation preservation, and redaction of indirect injection directives. (`src/generation/prompt.py`, `src/generation/generator.py`). |
| **Local Demo Server** | Implemented | Multi-threaded HTTP loopback server (`ThreadedHTTPServer`) with security controls blocking `.env`, `.git`, source files, and raw databases. (`src/server.py`). |
| **Browser Demo (`index.html`)** | Implemented | Standalone client-side dashboard with grounded temporal milestone telemetry and action preview inspector. Connects to the local server or replays pre-computed outputs. |
| **Live External Mutations** | Out of Scope | Actions are strictly dry-run preview plans; no live OAuth integrations, external emails, or Slack API calls are executed. |

---

## 🛠️ Setup & Running Locally

### Prerequisites
- Python 3.10+ (tested through Python 3.14 on macOS and Linux)

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

| Variable | Default | Purpose |
|---|:---:|---|
| `CHRONEXA_OFFLINE` | `0` | Set to `1` to run completely offline on local CPU without any external API calls. |
| `GROQ_API_KEY` | `""` | Optional Groq key for fast cloud LLM answer synthesis (`qwen/qwen3.8-27b`). |
| `OPENAI_API_KEY` | `""` | Optional OpenAI key for GPT-4o-mini generation. |
| `GEMINI_API_KEY` | `""` | Optional Google Gemini key for Gemini Flash models. |
| `ANTHROPIC_API_KEY`| `""` | Optional Anthropic key for Claude 3.5 Haiku. |

---

## 💻 Runner Commands (`run.sh`)

The script [`run.sh`](run.sh) wraps common workflows:

```bash
# 1. Run the automated test suite (50 unit and integration tests via pytest):
./run.sh test

# 2. Launch the local demonstration server (loopback only on 127.0.0.1:8000):
./run.sh serve 8000

# 3. Run a query directly from the terminal with a historical date cutoff:
./run.sh query "When are we launching Route Planner v2?" "2026-09-09T09:00:00-07:00"

# 4. Same question after subsequent delays were announced (sliding cutoff date):
./run.sh query "When are we launching Route Planner v2?" "2026-09-18T18:00:00-07:00"

# 5. Run the dry-run action assistant runner:
./run.sh actions

# 6. Run end-to-end verification (tests + memory search + actions):
./run.sh all
```

---

## 🖥️ Local Demonstration Dashboard

Chronexa includes a client-side interface (`index.html`) backed by a lightweight loopback server (`src/server.py`):

1. Start the server:
   ```bash
   ./run.sh serve 8000
   ```
2. Navigate to **`http://127.0.0.1:8000`** in your browser.
3. The dashboard provides:
   - **Point-in-Time Date Slider:** Adjust the cutoff date to see how search results change as new communications arrive over time.
   - **Search Query Input:** Runs hybrid search against the mock dataset and displays retrieved citations with grounded answers.
   - **Action Assistant Inspector:** Enter natural language commands (e.g., *"Message Sarah on Slack that the geocoding fix looks good"*, *"Delete all my emails from Marcus"*) to inspect schema-validated JSON actions, ambiguity prompts, and confirmation dialogues.

---

## 🧪 Automated Test Suite (50 Tests)

The test suite in [`tests/`](tests/) covers boundary conditions, validation schemas, fallbacks, and security guards:

- [`tests/test_temporal.py`](tests/test_temporal.py) (10 tests): ISO 8601 parsing, timezone offsets, temporal cutoffs, multiline message edit preservation, deletion masking, and dateparser fallbacks.
- [`tests/test_retrieval.py`](tests/test_retrieval.py) (5 tests): BM25 term matching, synonym expansion, token caching, temporal cutoff boundary filtering, and fastembed fallback handling.
- [`tests/test_actions.py`](tests/test_actions.py) (11 tests): Schema enforcement, rejection of unauthorized extra arguments, email validation, contextual ambiguity clarification, generalized arbitrary command fallback parsing, and destructive confirmation.
- [`tests/test_injection.py`](tests/test_injection.py) (2 tests): XML breakout escaping and prompt injection neutralization.
- [`tests/test_abstention.py`](tests/test_abstention.py) (4 tests): Calibrated abstention on unrecorded facts.
- [`tests/test_server.py`](tests/test_server.py) (18 tests): HTTP health check, live search API, action parser API, input bounds, payload caps, CORS headers, loopback security, and blocking of `.env`, `.git`, `.py`, and raw data access.

Run tests directly:
```bash
pytest tests/ -v
```

---

## 📂 Project Organization

```text
Chronexa/
├── data/                      # Synthetic multi-source mock workplace corpus
│   ├── connectors/            # Mock Slack, Gmail, Google Calendar, ChatGPT, Codex
│   └── native/                # Meeting recordings & voice dictation transcripts
├── src/                       # Core system implementation
│   ├── actions/               # Action schemas, validators, entity resolution, parser
│   ├── generation/            # Answer generator and prompt injection defense
│   ├── ingestion/             # Point-in-time loader and contextual chunker
│   ├── retrieval/             # Okapi BM25, FastEmbed dense retriever, hybrid RRF
│   ├── config.py              # Path configuration and environment loading
│   ├── pipeline.py            # CLI query-to-answer pipeline runner
│   └── server.py              # Multi-threaded loopback demonstration server
├── tests/                     # 50 automated unit and integration tests
├── eval/                      # Evaluation datasets and reference test queries
├── eval_harness/              # Scoring scripts and validation utilities
├── index.html                 # Interactive demonstration dashboard
├── run.sh                     # Unified runner script
├── requirements.txt           # Python dependencies
└── vercel.json                # Static edge hosting configuration
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
