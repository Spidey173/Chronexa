# ⏱️ Chronexa: Enterprise Workplace-Memory Engine & Safe Action Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-46%20passed-success.svg)](tests/)
[![FastEmbed Local ONNX](https://img.shields.io/badge/embeddings-FastEmbed%20ONNX-orange.svg)](https://github.com/qdrant/fastembed)
[![Temporal Sandboxing](https://img.shields.io/badge/temporal%20safety-zero%20leakage-emerald.svg)](src/retrieval/temporal.py)
[![Dry-Run Action Schema](https://img.shields.io/badge/TextOS-strict%20allowlist-purple.svg)](src/actions/schema.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Chronexa** is an enterprise workplace-memory search and safe action execution platform. It indexes and searches across scattered corporate communications—**Slack messages and threads, Gmail conversations, Google Calendar events, meeting transcripts, and voice dictation notes**—with strict historical point-in-time (`as_of`) temporal boundaries and translates natural language commands into schema-validated, dry-run action plans.

---

## 🎯 The Three Core Problems Chronexa Solves

Standard Retrieval-Augmented Generation (RAG) architectures and workplace AI assistants fail in corporate environments due to three structural flaws:

1. **Temporal Inconsistency & Future Data Leakage:**  
   Standard vector databases index all documents uniformly. If an executive asks *"What is our scheduled launch date?"* with a context date of September 9, naive RAG retrieves messages from September 16 announcing delays, corrupting historical decision audits. Chronexa enforces **pre-retrieval physical boundaries** (`delivery_time <= as_of`), masks deleted records, and applies chronological edit trails in real time.

2. **The Lexical vs. Semantic Vocabulary Gap:**  
   Keyword search fails on conversational paraphrasing (*"the roadmap slip"* vs. *"geocoding regression"*); pure vector search fails on precise identifiers, employee IDs, customer codes, and exact calendar timestamps. Chronexa fuses **passage-structured Okapi BM25** with **local FastEmbed dense vectors** via **Reciprocal Rank Fusion (RRF, k=60)**.

3. **Unsafe Autonomous Side-Effects:**  
   Autonomous AI agents should never blindly trigger external mutations (sending emails, modifying executive calendar events, deleting message archives). Chronexa's **TextOS Action Assistant** parses commands into strongly-typed, schema-validated dry-run actions, enforcing strict field allowlists, triggering human confirmation for destructive operations, and requesting clarification for ambiguous entities.

---

## 🏗️ System Architecture & Data Flow

```text
                       User Query / Action Command
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 1. Point-in-Time Temporal Sandbox (src/ingestion/)     │
       │    - Multi-connector ingestion across 7 data streams   │
       │    - Strict cutoff: discards any unit where t > as_of  │
       │    - Applies chronological edit history up to as_of    │
       │    - Masks records deleted prior to as_of moment       │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 2. Hybrid Retrieval Engine (src/retrieval/)            │
       │    ├── Lexical Index: Passage-Structured Okapi BM25    │
       │    ├── Dense Vectors: Local BGE-small ONNX CPU         │
       │    ├── Rank Fusion: Reciprocal Rank Fusion (RRF k=60)  │
       │    └── Context Injection: Dynamic meeting dialog turns │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │ 3. Evidence Sandboxing & Defense (src/generation/)     │
       │    - XML tag document sandboxing (<record id="...">)   │
       │    - Indirect prompt injection neutralization          │
       │    - Calibrated abstention for unrecorded data         │
       └────────────────────────────┬───────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │ Grounded Answer Synthesis │ │ TextOS Action Preview     │
       │ - Verified citations      │ │ - Strict schema validator │
       │ - Exact key-term match    │ │ - Disambiguation guard    │
       │ - Local & API providers   │ │ - Destructive confirm     │
       └───────────────────────────┘ └───────────────────────────┘
```

---

## ✨ Key Capabilities & Engineering Highlights

| Capability | Module | Technical Description |
|---|---|---|
| **Zero Future Leakage** | `src/retrieval/temporal.py` | Strict temporal filtering ensures no record delivered after `as_of` enters the retrieval pool. Fully handles timezone offsets (`-07:00`), ISO 8601 strings, and naive fallback conversion. |
| **Sub-Millisecond Vector Search** | `src/retrieval/dense.py` | Pre-stacks cached unit embeddings into a contiguous 2D float32 matrix `(2641, 384)`. Vector search uses BLAS matrix-vector dot products executing in under 1 ms on CPU. |
| **Passage-Structured BM25** | `src/retrieval/bm25.py` | Pure Python Okapi BM25 implementation featuring custom tokenization, Rust-accelerated Snowball stemming (`py_rust_stemmers`), entity boosting, and normalized timestamp indexing. |
| **TextOS Safe Action Assistant** | `src/actions/` | Translates natural language commands into dry-run JSON actions. Validates against strict allowlisted schemas (`slack.send_message`, `gmail.send`, `calendar.create_event`, `calendar.update_event`, `reminder.create`, `app.open`, `memory.ask`, `clarify`, `confirm`) and rejects unauthorized arguments. |
| **Adversarial Injection Defense** | `src/generation/prompt.py` | Defuses adversarial prompt injection embedded in mock emails and Slack messages (e.g., hidden exfiltration directives) via XML boundary isolation and directive scrubbing. |
| **Live Interactive Dashboard** | `index.html` & `src/server.py` | Standalone, dark-mode glassmorphic web dashboard connected to a lightweight loopback API server (`127.0.0.1:8000`) with real-time time-travel date slider and action preview inspector. |

---

## 💻 Tech Stack & Dependencies

- **Core Runtime:** Python 3.10+ (tested through Python 3.14 on macOS and Linux)
- **Dense Embeddings:** `fastembed` (ONNX Runtime CPU, `BAAI/bge-small-en-v1.5`)
- **Array Computing:** `numpy` (BLAS vectorized cosine similarity)
- **Stemming & NLP:** `py_rust_stemmers` (high-performance Snowball stemmer)
- **Temporal Parsing:** `dateparser`, Python standard library `zoneinfo` and `datetime`
- **Testing Framework:** `pytest` (46 unit and integration tests)
- **Local Server:** Standard library `http.server` with custom JSON routing, CORS, and payload security guards

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Virtual Environment

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

| Environment Variable | Default | Purpose |
|---|:---:|---|
| `CHRONEXA_OFFLINE` | `0` | Set to `1` to run 100% locally on CPU without external API network calls. |
| `GROQ_API_KEY` | `""` | Optional Groq key for fast cloud LLM answer synthesis (`qwen/qwen3.8-27b`). |
| `OPENAI_API_KEY` | `""` | Optional OpenAI key for GPT-4o-mini generation. |
| `GEMINI_API_KEY` | `""` | Optional Google Gemini key for Gemini Flash models. |
| `ANTHROPIC_API_KEY`| `""` | Optional Anthropic key for Claude 3.5 Haiku. |

---

## 🛠️ CLI & Runner Commands (`run.sh`)

The unified execution script [`run.sh`](run.sh) manages all tasks:

```bash
# 1. Run the complete automated test suite (46 unit & integration tests):
./run.sh test

# 2. Launch the local live search server:
./run.sh serve 8000

# 3. Execute an ad-hoc query directly from the terminal:
./run.sh query "When are we launching Route Planner v2?" "2026-09-10T12:00:00-07:00"

# 4. Compare point-in-time answers across different cutoff dates:
# On Sept 9 (initial target):
./run.sh query "When are we launching Route Planner v2?" "2026-09-09T09:00:00-07:00"
# On Sept 18 (after launch slipped to Oct 21):
./run.sh query "When are we launching Route Planner v2?" "2026-09-18T18:00:00-07:00"

# 5. Run the TextOS Action Assistant runner:
./run.sh actions

# 6. Run complete end-to-end pipeline verification:
./run.sh all
```

---

## 🖥️ Interactive Web Dashboard & Local Server

Chronexa features an interactive, dark-mode web application:

1. Launch the local Python server:
   ```bash
   ./run.sh serve 8000
   ```
2. Open **`http://127.0.0.1:8000`** in your browser.
3. The dashboard connects to the live backend:
   - **Historical Date Slider:** Slide between Sept 8 and Sept 18 to observe how temporal boundaries dynamically mask future communications.
   - **Live Workplace Search:** Type any query to inspect hybrid BM25 + dense ranking, candidate score breakdowns, and grounded synthesized answers.
   - **TextOS Action Playground:** Test natural language commands (e.g. *"Message Sarah on Slack that the geocoding fix looks good"*, *"Delete all my emails from Marcus"*), preview schema-validated JSON actions, and test ambiguity disambiguation.

---

## 🧪 Automated Test Suite (46 Tests)

Every core invariant is verified by unit and integration tests in [`tests/`](tests/):

- [`tests/test_temporal.py`](tests/test_temporal.py) (10 tests): ISO 8601 parsing, timezone normalization, temporal boundary cutoffs, message edit preservation, deletion masking, and graceful dateparser fallbacks.
- [`tests/test_retrieval.py`](tests/test_retrieval.py) (5 tests): BM25 identifier matching, synonym expansion, zero future leakage validation, and fastembed offline fallback handling.
- [`tests/test_actions.py`](tests/test_actions.py) (10 tests): Strict schema enforcement, rejection of unauthorized extra arguments, email validation, ambiguity clarification, and destructive confirmation.
- [`tests/test_injection.py`](tests/test_injection.py) (2 tests): XML breakout escaping and prompt injection neutralization.
- [`tests/test_abstention.py`](tests/test_abstention.py) (4 tests): Calibrated abstention on unrecorded company data.
- [`tests/test_server.py`](tests/test_server.py) (15 tests): HTTP health check, live search API, action parser API, input validation bounds, payload caps, CORS headers, and loopback security.

Run tests anytime with:
```bash
pytest tests/ -v
```

---

## 📂 Project Organization

```text
Chronexa/
├── data/                      # Multi-source enterprise corpus
│   ├── connectors/            # Slack, Gmail, Google Calendar, ChatGPT, Codex
│   └── native/                # Meeting recordings & voice dictation transcripts
├── src/                       # Core system architecture
│   ├── actions/               # TextOS schema validators, entity resolver, parser
│   ├── generation/            # Grounded answer synthesis & injection defense
│   ├── ingestion/             # Point-in-time loader & contextual chunker
│   ├── retrieval/             # BM25, FastEmbed dense retriever, hybrid RRF
│   ├── config.py              # Path definitions & global environment settings
│   ├── pipeline.py            # Unified query-to-answer CLI pipeline
│   └── server.py              # Lightweight loopback demo HTTP server
├── tests/                     # 46 automated unit and integration tests
├── eval/                      # Evaluation datasets & benchmark runners
├── eval_harness/              # Ground truth scoring harnesses & validation metrics
├── index.html                 # Interactive dark-mode web application
├── run.sh                     # Unified runner script
├── requirements.txt           # Python package dependencies
└── vercel.json                # Static edge hosting configuration
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
