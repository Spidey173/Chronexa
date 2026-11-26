# ⏱️ Chronexa: Workplace-Memory Search & Safe Action Assistant

[![CI Pipeline](https://github.com/Spidey173/Chronexa/actions/workflows/ci.yml/badge.svg)](https://github.com/Spidey173/Chronexa/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Pytest-87%20Passing-brightgreen?style=flat-square&logo=pytest)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Service-009688?style=flat-square&logo=fastapi)](src/api/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

🔗 **[Live Demo (Vercel)](https://chronexa17.vercel.app/)** | ⚡ **[Live API Health](https://chronexa-api.vercel.app/health)** | 📦 **[Repository](https://github.com/Spidey173/Chronexa)**

---

## What Chronexa Does

Chronexa is a **point-in-time workplace memory search engine and human-in-the-loop action assistant**:
- **Zero Future Knowledge Leakage:** Enforces a pre-retrieval temporal boundary (`timestamp <= as_of` and not deleted) before scoring candidates.
- **Hybrid Dual-Stream Retrieval:** Fuses sparse BM25 (Snowball stemming) and dense FastEmbed BGE vectors (disk-backed FAISS index) via Reciprocal Rank Fusion ($k=60$), with optional cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`).
- **Grounded Answer Synthesis & Abstention:** Sanitizes retrieved records against prompt injection, strips stale superseded facts, and abstains cleanly when evidence is absent.
- **Human-in-the-Loop Safe Actions:** Parses natural-language commands into strict, allowlisted JSON proposals (`proposed` -> `approved` -> background execution), gated by JWT Bearer authentication, RBAC (`viewer`, `operator`, `admin`), and an immutable `audit_log`.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **API & Auth** | FastAPI, Uvicorn, Pydantic v2 (`pydantic-settings`), `python-jose` (JWT), RBAC |
| **Database** | SQLAlchemy 2, Alembic migrations, SQLite (local/dev), PostgreSQL (`psycopg2-binary`) |
| **Retrieval & ML** | `fastembed` (ONNX BGE + MiniLM cross-encoder), `faiss-cpu`, `numpy`, `py_rust_stemmers`, `dateparser` |
| **LLM Providers** | Unified `LLMClient` supporting Groq, OpenAI, Anthropic, Google Gemini, Claude CLI, plus deterministic offline fallback |
| **Frontend** | Vanilla HTML/CSS/JS (`index.html`, `styles.css`, `app.js`, `data.js`) |
| **Tooling & Deploy** | `pytest`, `hypothesis`, `respx`, `ruff`, `black`, `pre-commit`, Docker / Docker Compose, Vercel |

---

## Architecture

```mermaid
flowchart TD
    UI["Frontend UI (index.html / app.js)"] -->|HTTP + JWT Bearer| API["FastAPI Service (src/api/)"]
    API -->|Middleware: CORS, Rate Limit, Size Cap| Routes["Routes (src/api/routes.py)"]

    Routes -->|/query, /api/search| Temporal["Temporal Gate (timestamp <= as_of)"]
    Temporal --> Hybrid["Hybrid Retriever (BM25 + FAISS RRF + Reranker)"]
    Hybrid --> Gen["Answer Generator (src/generation/)"]
    Gen --> LLM["Shared LLMClient (src/core/llm_client.py)"]

    Routes -->|/actions/propose| Parser["Action Parser & Schema Validator (src/actions/)"]
    Parser --> LLM
    Parser --> DB[("SQLAlchemy DB (users, action_proposals, audit_log)")]

    Routes -->|/actions/:id/approve (Operator+ RBAC)| DB
    Routes -->|Decoupled BackgroundTask| Exec["Action Executor & Adapters (Slack, Gmail, Calendar)"]
```

---

## Folder Structure

```text
Chronexa/
├── api/
│   └── index.py                  # Vercel serverless ASGI entrypoint
├── alembic/                      # Database migration scripts
├── data/
│   ├── connectors/               # Slack, Gmail, Calendar, ChatGPT, Codex data
│   ├── embeddings/               # Prebuilt FAISS index & NPZ vectors
│   └── native/                   # Meetings & dictation records
├── eval/                         # Benchmark runners & dev/holdout splits
├── eval_harness/                 # Scoring, LLM judge, stats & record loaders
├── evals/                        # Training evaluation datasets
├── examples/                     # Example command & answer JSONL payloads
├── scripts/
│   └── deploy_backend.sh         # Vercel backend deployment script
├── src/
│   ├── core/                     # config.py, auth.py, llm_client.py
│   ├── db/                       # models.py, session.py
│   ├── api/                      # app.py, deps.py, routes.py, schemas.py
│   ├── ingestion/                # chunker.py, loader.py
│   ├── retrieval/                # bm25.py, decomposition.py, dense.py, hybrid.py, reranker.py, temporal.py, vector_store.py
│   ├── generation/               # generator.py, prompt.py
│   ├── actions/                  # adapters.py, entities.py, executor.py, parser.py, runner.py, schema.py
│   ├── pipeline.py               # CLI for temporal retrieval & generation
│   └── server.py                 # Local Uvicorn CLI entrypoint
├── tests/                        # Unit, property-based (Hypothesis), and API tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml                # pytest, ruff, and black configuration
├── .pre-commit-config.yaml
├── requirements.txt
├── requirements-dev.txt
├── run.sh
└── vercel.json
```

---

## Setup & Environment Variables

### 1. Local Python Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional: tests, ruff, black
cp .env.example .env
```

### 2. Key Environment Variables (`.env.example`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Server bind host and port |
| `DATABASE_URL` | `sqlite:///./chronexa.db` | SQLAlchemy DB URI (SQLite or PostgreSQL) |
| `JWT_SECRET_KEY` | *(required in prod)* | Secret key for signing JWT Bearer tokens |
| `JWT_ALGORITHM` / `JWT_EXPIRE_MINUTES` | `HS256` / `1440` | Token signing algorithm and expiry |
| `INTEGRATION_MODE` | `mock` | `mock` (dry-run) or `live` (Slack/Gmail/Calendar APIs) |
| `CHRONEXA_OFFLINE` | `0` | Set `1` to disable external LLM calls (deterministic fallback) |
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | *(optional)* | LLM provider credentials |

---

## How to Run

* **Native Python (or Makefile):**
  ```bash
  make run
  # or: ./run.sh serve
  ```
* **Docker Compose (FastAPI + PostgreSQL 16):**
  ```bash
  docker compose up --build
  docker compose exec api alembic upgrade head
  ```
* **Interactive API Docs (Swagger / OpenAPI):**
  Visit [http://localhost:8000/docs](http://localhost:8000/docs) once the server is running (health check at [http://localhost:8000/health](http://localhost:8000/health)).

---

## How to Test & Lint

```bash
make test          # Run full pytest suite (87 tests)
make lint          # Run ruff checks
make format        # Check formatting with black
./run.sh eval      # Run offline retrieval & action benchmarks
```

---

## Retrieval Benchmark Results

Evaluated on synthetic workplace memory suites (1,316 corpus units, `n=36` dev / `n=100` unseen holdout):

| Metric | Dev Set | Holdout (`n=100`) | Naive Dense Baseline |
| :--- | :---: | :---: | :---: |
| **Top-10 Retrieval Accuracy** | **94.4%** | **86.8%** | 68.2% |
| **Recall@5** | **86.1%** | **75.0%** | 54.2% |
| **Mean Reciprocal Rank (MRR)** | **0.742** | **0.683** | 0.538 |
| **Temporal Future Leakage** | **0.0%** | **0.0%** | 18.4% |
| **Action Schema Accuracy** | **12 / 12 (100%)** | **12 / 12 (100%)** | N/A |

---

## License

MIT — see [LICENSE](LICENSE).
