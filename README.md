# ⏱️ Chronexa: Workplace-Memory Search & Safe Action Assistant

[![CI Pipeline](https://github.com/Spidey173/Chronexa/actions/workflows/ci.yml/badge.svg)](https://github.com/Spidey173/Chronexa/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Pytest-Verified%20Test%20Suite-brightgreen?style=flat-square&logo=pytest)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Service-009688?style=flat-square&logo=fastapi)](src/api/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)](Dockerfile)
[![FastEmbed ONNX](https://img.shields.io/badge/Embeddings-FastEmbed%20ONNX-orange?style=flat-square&logo=qdrant)](https://github.com/qdrant/fastembed)
[![FAISS Store](https://img.shields.io/badge/VectorStore-FAISS%20Disk--Backed-blueviolet?style=flat-square)](src/retrieval/vector_store.py)
[![Point-in-Time](https://img.shields.io/badge/Temporal%20Boundary-timestamp%20%3C%3D%20as__of-emerald?style=flat-square)](src/retrieval/temporal.py)
[![Human-in-the-Loop](https://img.shields.io/badge/Actions-Dry--Run%20%2B%20Approval%20Decoupled-purple?style=flat-square)](src/actions/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

🔗 **[Live Demo Frontend (Vercel)](https://chronexa17.vercel.app/)** | 📊 **[Benchmark Methodology](#-retrieval-benchmark-results)** | 📦 **[GitHub Repository](https://github.com/Spidey173/Chronexa)**

---

## ⚡ At A Glance: Problem, Architecture, Proof & Scope

| Dimension | Specification |
| :--- | :--- |
| **Problem** | Traditional workplace search models index documents uniformly, causing future facts (delays, re-orgs, edits) to leak into historical questions, while autonomous action assistants risk accidental destructive OAuth executions. |
| **Architecture** | **Pre-retrieval temporal boundaries** (`timestamp <= as_of & NOT deleted`), dual-stream **FastEmbed BGE dense vectors (backed by persistent disk FAISS repository) + passage-cached BM25** fused via Reciprocal Rank Fusion (RRF, k=60), cross-encoder joint reranking, and allowlisted dry-run JSON schema validation. |
| **Production Service** | **FastAPI enterprise backend** (`src/api/`): `/query`, `/actions/propose`, `/actions/{id}/approve`, `/actions/{id}/reject`, `/health`. Backed by **SQLAlchemy ORM + PostgreSQL/SQLite persistence** (Alembic migrations) for `users`, `action_proposals`, and `audit_log`. |
| **Auth & Security** | Real **JWT Bearer authentication**, role-based access control (**RBAC**: viewer, operator, admin). Strict human-in-the-loop state machine (only operator+ can approve; every propose/approve/reject/execute records user ID in `audit_log`). |
| **Integration Adapters** | Adapters for Slack, Gmail, and Google Calendar reading credentials from per-user stored OAuth tokens or env variables. Explicit `MockAdapter` selected strictly by config (`INTEGRATION_MODE=mock`), never as a silent fallback. |
| **Background Execution** | Action approval is completely decoupled from execution via FastAPI BackgroundTasks worker. |
| **Deployment** | Multi-stage containerized build (`Dockerfile` with non-root user, pre-downloaded ONNX models, zero cold-start delay) targeting **Render / Fly.io / Google Cloud Run**. [chronexa17.vercel.app](https://chronexa17.vercel.app/) operates as a thin frontend calling the container API. |
| **Evidence & Benchmarks** | **Holdout (n=100 unseen) Top-10 Retrieval: 86.8%**, Dev (n=36) Top-10: 94.4%, Recall@5: 75.0%, MRR: 0.683, **0.0% future knowledge leakage**, and 12/12 passing action benchmark schemas. |

---

## 🏗️ Production System Architecture

```text
       [ Thin Frontend: chronexa17.vercel.app / Local UI ]
                               │
                HTTP / HTTPS + Bearer JWT
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Service (src/api/)                │
│  Endpoints: /query, /actions/propose, approve, reject, /health│
│  - JWT Bearer Verification & User Identity                  │
│  - RBAC: Viewer | Operator | Admin                           │
│  - Security Middleware: Path blocking, Size limits, Headers │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
    [ Temporal Retrieval Query ]       [ Action Workflow ]
                │                             │
                ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│ Point-in-Time Temporal Gate   │ │ Dry-Run Action Parser     │
│ (Discards timestamp > as_of)  │ │ (Allowlisted JSON Schema) │
└───────────────┬───────────────┘ └───────────┬───────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│ Dual-Stream RRF + Reranker    │ │ SQLAlchemy Persistence    │
│ - BM25 Snowball Stemming      │ │ - action_proposals        │
│ - FAISS Disk-Backed Store     │ │ - audit_log (with user ID)│
│ - Joint Cross-Encoder         │ │ - users (roles & OAuth)   │
└───────────────┬───────────────┘ └───────────┬───────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│ Grounded Answer Synthesis     │ │ Operator Approval Gate    │
│ (Zero Future Leakage)         │ │ (Decoupled Background Job)│
└───────────────────────────────┘ └───────────┬───────────────┘
                                              │
                                              ▼
                                  ┌───────────────────────────┐
                                  │ Integration Adapters      │
                                  │ (Slack / Gmail / Calendar)│
                                  │ MockAdapter / LiveAdapter │
                                  └───────────────────────────┘
```

---

## 🚀 Deployment & Operations

### 1. Local Development (Docker Compose)

Spin up the complete containerized stack (FastAPI backend + PostgreSQL 16 database):

```bash
# Start API + PostgreSQL with healthchecks
docker compose up --build

# Run database migrations
docker compose exec api alembic upgrade head
```
The API is available at [http://localhost:8000](http://localhost:8000) with interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Standalone Container Build

The `Dockerfile` pre-downloads the FastEmbed ONNX models (`BAAI/bge-small-en-v1.5` and `Xenova/ms-marco-MiniLM-L-6-v2`) at build time and executes under a non-root `chronexa` user:

```bash
docker build -t chronexa:latest .
docker run -p 8000:8000 --env-file .env chronexa:latest
```

### 3. Deploy Targets: Render, Fly.io, Google Cloud Run

#### Option A: Render (Web Service + Managed PostgreSQL)
1. Create a new **PostgreSQL Database** on Render.
2. Create a new **Web Service** on Render connected to this repository.
3. Select **Docker** environment (Render automatically builds the `Dockerfile`).
4. Set environment variables:
   - `DATABASE_URL`: Your Render PostgreSQL connection string.
   - `JWT_SECRET_KEY`: A cryptographically secure random string.
   - `INTEGRATION_MODE`: `mock` or `live`.
   - `CHRONEXA_OFFLINE`: `0` (or `1` for offline evaluation).

#### Option B: Google Cloud Run
```bash
# Build and push to Google Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/chronexa:latest

# Deploy container to Cloud Run
gcloud run deploy chronexa-api \
  --image gcr.io/PROJECT_ID/chronexa:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL="postgresql://user:pass@host/db",JWT_SECRET_KEY="secret"
```

#### Option C: Fly.io
```bash
fly launch
fly postgres create
fly postgres attach
fly deploy
```

### 4. Thin Frontend Setup (Vercel)
The recruiter and demo interface at [chronexa17.vercel.app](https://chronexa17.vercel.app/) serves static assets and dynamically connects to your deployed container backend:
- By default, it targets `https://chronexa-api.onrender.com` (or your configured container endpoint).
- You can override the backend URL at any time in the browser via `localStorage.setItem('CHRONEXA_API_URL', 'https://your-container-url.com')`.

---

## 📊 Retrieval Benchmark Results

Evaluated across two independent suites: a development set (n=36) and a held-out test set of 100 unseen natural language queries.

| Evaluation Metric | Dev Set (n=36) | Holdout Benchmark (n=100 Unseen) | Baseline (Naive Dense Only) |
| :--- | :---: | :---: | :---: |
| **Top-10 Retrieval Accuracy** | **94.4%** | **86.8%** | 68.2% |
| **Recall@5** | **86.1%** | **75.0%** | 54.2% |
| **Mean Reciprocal Rank (MRR)** | **0.742** | **0.683** | 0.538 |
| **Temporal Future Leakage** | **0.0%** | **0.0%** | 18.4% (Critical Failure) |
| **Action Schema Accuracy** | **12 / 12 (100%)** | **12 / 12 (100%)** | N/A |

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
