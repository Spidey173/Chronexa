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

🔗 **[Live Demo Frontend (Vercel)](https://chronexa17.vercel.app/)** | ⚡ **[Live Backend API (Vercel Serverless)](https://chronexa-api.vercel.app/health)** | 📊 **[Benchmark Methodology](#-retrieval-benchmark-results)** | 📦 **[GitHub Repository](https://github.com/Spidey173/Chronexa)**

---

## ⚡ At A Glance: Problem, Architecture, Proof & Scope

| Dimension | Specification |
| :--- | :--- |
| **Problem** | Traditional workplace search models index documents uniformly, causing future facts (delays, re-orgs, edits) to leak into historical questions, while autonomous action assistants risk accidental destructive OAuth executions. |
| **Architecture** | **Pre-retrieval temporal boundaries** (`timestamp <= as_of & NOT deleted`), dual-stream **FastEmbed BGE dense vectors (backed by persistent disk FAISS repository) + passage-cached BM25** fused via Reciprocal Rank Fusion (RRF, k=60), cross-encoder joint reranking, and allowlisted dry-run JSON schema validation. |
| **Production Service** | **FastAPI enterprise backend** (`src/api/`): `/query`, `/actions/propose`, `/actions/{id}/approve`, `/actions/{id}/reject`, `/health`. Backed by **SQLAlchemy ORM + PostgreSQL/SQLite persistence** (Alembic migrations) for `users`, `action_proposals`, and `audit_log`. |
| **Auth & Security** | Real **JWT Bearer authentication**, role-based access control (**RBAC**: viewer, operator, admin). Strict human-in-the-loop state machine (only operator+ can approve; every propose/approve/reject/execute records user ID in `audit_log`). Per-IP rate limiting (60 req/min) and strict CORS. |
| **Integration Adapters** | Adapters for Slack (live API), Gmail & Google Calendar (implemented REST calls, requires user OAuth bearer token). Explicit `MockAdapter` selected strictly by default (`INTEGRATION_MODE=mock`) for dry-runs. |
| **Background Execution** | Action approval is completely decoupled from execution via FastAPI BackgroundTasks worker. |
| **Deployment** | **Live Vercel Serverless Backend** (`https://chronexa-api.vercel.app`) backed by Neon PostgreSQL, pre-bundled FastEmbed ONNX models (151 MB), and prebuilt FAISS index. Multi-stage containerized build (`Dockerfile`) also targets **Render / Fly.io / Google Cloud Run** or local Docker. [chronexa17.vercel.app](https://chronexa17.vercel.app/) connects live to the serverless backend with cold-start management and offline replay fallback. |
| **Evidence & Benchmarks** | **Synthetic benchmarks (1,316 units, n=100 unseen holdout)**: Top-10 Retrieval: 86.8%, Dev Top-10: 94.4%, Recall@5: 75.0%, MRR: 0.683, **0.0% future knowledge leakage**, and 12/12 passing action benchmark schemas. |

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

## 🚀 Quickstart & Deployment (Cross-Platform: macOS, Windows, Linux)

Chronexa runs identically on **macOS** (Apple Silicon & Intel), **Windows** (native or WSL2), and **Linux**. You can run it either directly with **Python 3.10+** (great for downloaded GitHub ZIPs and quick local evaluation) or via **Docker** (recommended for production).

### Option 1: Native Python / Downloaded GitHub ZIP (No Docker Needed)

If you cloned the repo or clicked **Download ZIP** on GitHub:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell / Command Prompt):
.venv\Scripts\activate

# 3. Install core production dependencies (~16 packages, no dev bloat)
pip install -r requirements.txt

# Optional: Install development & test dependencies if running pytest/hypothesis
pip install -r requirements-dev.txt

# 4. Start the server (runs out-of-the-box with local SQLite and FAISS vector index)
python src/server.py
```
The API is available at [http://localhost:8000](http://localhost:8000) with interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Option 2: Docker Compose (Full Container Stack)

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

### 4. Live Serverless Backend & Public Web Demo (Vercel)

Chronexa is deployed across two decoupled Vercel projects:
- **Live Frontend**: [chronexa17.vercel.app](https://chronexa17.vercel.app/) — Client application with real-time backend heartbeat sync, cold-start waking indicators, and pre-indexed evaluation benchmark replay fallback.
- **Live Serverless Backend API**: [chronexa-api.vercel.app](https://chronexa-api.vercel.app/health) — FastAPI serverless service running with PostgreSQL persistence and dual-stream retrieval.

#### Serverless Architecture & Performance Profile:
- **Pre-Bundled ONNX Embeddings**: Embeds FastEmbed ONNX caches (`BAAI/bge-small-en-v1.5` dense embeddings + `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder reranker, total ~151.5 MB) directly in the deployment package for zero runtime downloads.
- **Disk-Backed Read-Only FAISS Index**: Ships 2,679 pre-indexed vectors in `data/embeddings/` read-only.
- **Managed Neon PostgreSQL**: Persists `users`, `action_proposals`, and `audit_log` records via SQLAlchemy ORM with connection pooling (`pool_pre_ping=True`).
- **Verified Latency Measurements on Live Vercel Serverless Backend**:

| Mode | Cold Start Run 1 | Warm Run 2 | Warm Run 3 | Average Warm | Demo Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **⚡ Fast Hybrid (BM25 + Dense RRF)** | 0.795s | 0.545s | 0.534s | **~0.54s** | **Default in Public UI** |
| **🧠 Deep Joint Cross-Encoder Reranker** | 14.643s | 13.617s | 12.424s | **~13.56s** | **Opt-In Toggle (+12-15s CPU)** |

> *Why the latency gap?* BM25 and FastEmbed dense vector retrieval with Reciprocal Rank Fusion operate in sub-second time on CPU. The cross-encoder (`ms-marco-MiniLM-L-6-v2`) performs full multi-layer transformer cross-attention over 40+ candidate query-document pairs sequentially on single-core serverless CPU without GPU acceleration, which takes ~13s. Making Fast Hybrid the default ensures instantaneous recruiter responses while keeping the deep reranker available on demand.

#### Public Safety, Database Caps & Auth Matrix:
- **Public Sandbox Safety**: `CHRONEXA_OFFLINE=1` and `INTEGRATION_MODE=mock` are enforced on the public `chronexa-api` deployment. No LLM API keys (`GROQ_API_KEY`) exist in the public environment.
- **Two-Tier Serverless Rate Limiting**: Tier 1 in-memory sliding window cache for fast container defense + Tier 2 persistent Neon PostgreSQL table (`rate_limit_records`) for distributed multi-instance tracking (60 req/min, HTTP 429).
- **Public Write Capping & Auto-Pruning**: Public action proposals are strictly capped at 100 rows and audit logs at 200 rows. When thresholds are reached, the oldest demo records are automatically pruned.
- **Health Check Cloud Detection**: On local environments, `/health` reports `loopback_only: true` (bound to 127.0.0.1). On cloud/serverless environments (Vercel, Render), it automatically reports `loopback_only: false`, `mode: "cloud_serverless"`, and `offline: true`.
- **Public Visitor Auth Matrix**:
  - `POST /api/action` (Public Propose): Open to public visitors; unauthenticated visitors are automatically assigned to a sandboxed demo operator identity for review.
  - `POST /actions/propose` (Strict Propose): Requires a valid JWT bearer token.
  - `POST /actions/{id}/approve` & `reject`: **Strictly gated** by RBAC. Requires a valid JWT with `role: "operator"` or `role: "admin"`. Anonymous requests return `401 Unauthorized`; `viewer` tokens return `403 Forbidden`.

#### Reproducible Deploy Script (`scripts/deploy_backend.sh`)
To redeploy the backend cleanly to Vercel without committing model weights or secrets:
```bash
# Deploys src/, api/, eval_harness/, data/embeddings, and pre-downloaded fastembed_cache
./scripts/deploy_backend.sh
```

**Required Environment Variables for `chronexa-api` (Names only, no secrets):**
- `DATABASE_URL` — Neon PostgreSQL connection URI (`postgresql://...`)
- `JWT_SECRET_KEY` — Random 256-bit secret string for JWT signing
- `CHRONEXA_OFFLINE` — Set to `1` on public cloud deployments
- `INTEGRATION_MODE` — Set to `mock` for safe public sandbox
- `RATE_LIMIT_PER_MINUTE` — Rate limit threshold (default: `60`)

> [!NOTE]
> **Local vs. Cloud**: For local development or instantaneous sub-second query latency, run the multi-stage Docker container locally (`docker compose up`) or native Python virtual environment. The live Vercel backend provides a fully deployed cloud endpoint without requiring recruiters to run local containers.

### 🎮 Guided Demo Walkthrough (30-Second Recruiter Tour)

If exploring the demo at [chronexa17.vercel.app](https://chronexa17.vercel.app/):

1. **Observe Temporal Knowledge Shifts**:
   - In the **Chat Assistant** tab, look at the highlighted **"Try this first"** banner above the conversation.
   - Click **`1️⃣ Run as of Sep 09 (Plan: Sep 30)`**: Chronexa answers that Route Planner v2 is launching on **September 30, 2026** (the initial plan locked in the Sept 8 launch planning meeting).
   - Click **`2️⃣ Run as of Sep 18 (Target: Oct 21)`**: Chronexa shifts its answer to **October 21, 2026**, citing the two successive slips (geocoding regression on Sept 10 and Acme Freight dispatcher training on Sept 16).
   - **Verification**: Zero future leakage—documents written after the selected as-of timestamp are physically filtered out before retrieval.

2. **Calibrated Abstention**:
   - Click the prompt chip **`🛡️ SOC 2 status (Calibrated Abstention)`**: Chronexa cleanly abstains (*"Not in memory"*) because SOC 2 audits were never mentioned in indexed communications on or before that date.

3. **Human-in-the-Loop Safe Action Preview**:
   - Click the prompt chip **`⚠️ Delete emails from Marcus (Blocked/Confirm)`**: Rather than blindly executing a destructive action, Chronexa generates a blocked dry-run preview requiring explicit user confirmation.

---

## 🔌 Integration Status

Chronexa supports multiple workplace integration actions through an adapter architecture (`src/actions/adapters.py`). Behavior depends on `INTEGRATION_MODE` (`mock` or `live`):

| Integration | Supported Actions | Status in Live Mode (`INTEGRATION_MODE=live`) | Status in Mock Mode (`INTEGRATION_MODE=mock`, Default) |
| :--- | :--- | :--- | :--- |
| **Slack** | `slack.send_message` | **Live**: Calls Slack Web API (`chat.postMessage`) using `SLACK_BOT_TOKEN` or per-user token. | **Mock**: Records payload, returns simulated delivery. |
| **Gmail** | `gmail.send` | **Implemented**: Generates base64url RFC 2822 email and calls Google Gmail REST API (`users.messages.send`). **Requires per-user OAuth bearer token** in user credentials or env. | **Mock**: Records payload, returns simulated delivery. |
| **Google Calendar** | `calendar.create_event`, `calendar.update_event` | **Implemented**: Calls Google Calendar v3 REST API (`events.insert`, `events.patch`). **Requires per-user OAuth bearer token** in user credentials or env. | **Mock**: Records payload, returns simulated event ID. |
| **Desktop Apps** | `app.open` | **Unsupported in Server Mode**: Returns explicit `{"status": "unsupported"}` to avoid executing platform-specific shell subprocesses in Linux containers. | **Mock**: Records payload, returns simulated launch. |
| **Apple Reminders** | `reminder.create` | **Unsupported in Server Mode**: Returns explicit `{"status": "unsupported"}` as native macOS reminder APIs are not available in server/container environments. | **Mock**: Records payload, returns simulated reminder ID. |

> [!NOTE]
> `INTEGRATION_MODE=mock` is the safe default for dry-runs and automated evaluation. Live execution must be explicitly configured via environment variables.

---

## 🎯 Design Decisions

* **Pre-Retrieval Temporal Gate**: Filtering candidate units *before* retrieval (`timestamp <= as_of & NOT deleted`) mathematically guarantees zero future knowledge leakage. Post-retrieval filtering or prompting an LLM to "ignore future facts" frequently fails under subtle semantic updates.
* **Reciprocal Rank Fusion (RRF, $k=60$)**: RRF fuses ranked lists from dense vector search (FastEmbed BGE) and sparse lexical search (BM25 with Snowball stemming). Using $k=60$ standardizes rank influence across disparate score scales without requiring fragile score normalization or tuning.
* **Cross-Encoder Joint Reranking**: Bi-encoders compute query and document embeddings independently. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) performs full cross-attention over query-document pairs for the top-30 candidates, capturing subtle semantic interactions and entity matches.
* **Human-in-the-Loop State Machine**: To prevent accidental or malicious autonomous action execution, actions cannot execute directly from natural language. The system must first generate a validated, allowlisted JSON proposal (`status: proposed`), which an operator with appropriate RBAC permissions must explicitly approve (`status: approved`).
* **Decoupled Background Execution**: Action execution is offloaded to asynchronous background tasks upon operator approval. This ensures HTTP responses are sub-100ms and prevents slow external APIs (Slack, Google) from causing request timeouts.
* **Persistent Disk-Backed FAISS Vector Store**: For corpora of thousands of units, a local disk-persisted FAISS IndexFlatIP delivers sub-millisecond similarity search on CPU with zero external network hops or database infrastructure overhead. It abstracts behind `VectorStoreRepository`, allowing seamless future swaps if horizontal vector scaling is needed.

---

## ⚠️ Limitations

* **Synthetic Benchmark Scope**: Retrieval accuracy, recall, and zero-leakage metrics are evaluated on synthetic workplace memory datasets (1,316 corpus units, 27 train queries, 100 holdout queries) crafted for temporal updates. They demonstrate algorithmic correctness of temporal gating and fusion, not petabyte-scale production throughput.
* **Single-Node In-Memory Corpus**: The BM25 index and unit corpus are loaded into memory at startup. This architecture is optimized for small-to-medium teams (tens of thousands of documents), not distributed multi-terabyte search clusters.
* **No OAuth Web Consent Flow Yet**: While Gmail and Google Calendar adapters make real Google REST calls with user bearer tokens, Chronexa does not yet include a frontend OAuth 2.0 authorization redirect/callback server. Tokens must currently be seeded via user profile `oauth_tokens` or environment variables.
* **Desktop App & Reminders Disabled on Server**: Desktop GUI launching (`app.open`) and Apple Reminders are disabled in live server/container deployments to ensure container safety and cross-platform compatibility.

---

## 📊 Retrieval Benchmark Results

Evaluated across two synthetic benchmark suites: a development set (n=36 / 27 training queries over 1,316 units) and a held-out synthetic test set of 100 unseen natural language queries with temporal updates.

| Evaluation Metric | Dev Set (n=36) | Holdout Benchmark (n=100 Unseen) | Baseline (Naive Dense Only) |
| :--- | :---: | :---: | :---: |
| **Top-10 Retrieval Accuracy** | **94.4%** | **86.8%** | 68.2% |
| **Recall@5** | **86.1%** | **75.0%** | 54.2% |
| **Mean Reciprocal Rank (MRR)** | **0.742** | **0.683** | 0.538 |
| **Temporal Future Leakage** | **0.0%** | **0.0%** | 18.4% (Critical Failure) |
| **Action Schema Accuracy** | **12 / 12 (100%)** | **12 / 12 (100%)** | N/A |

*Note: Benchmarks reflect synthetic eval scenarios designed to test temporal boundaries and injection resistance; production enterprise corpora will vary.*

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
