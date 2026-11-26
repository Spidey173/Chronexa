# Deployment & Configuration Guide

## Docker Compose (FastAPI + PostgreSQL 16)

```bash
docker compose up --build
```

The API listens on `http://localhost:8001` by default (`API_PORT` in `docker-compose.yml`), and PostgreSQL 16 listens on `5433` (`POSTGRES_PORT`).

## Standalone Docker Container

```bash
docker build -t chronexa:latest .
docker run -p 8000:8000 --env-file .env chronexa:latest
```

## Vercel Serverless Deployment

`scripts/deploy_backend.sh` stages `src/`, `api/index.py`, `eval_harness/`, `data/`, and `fastembed_cache/` for deployment to `chronexa-api` on Vercel with `CHRONEXA_OFFLINE=1`, `INTEGRATION_MODE=mock`, and `DEMO_MODE=1`.

## All Environment Variables (`src/core/config.py`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Server bind address and port |
| `DATABASE_URL` | `sqlite:///./chronexa.db` | SQLite or PostgreSQL connection URI |
| `DEMO_MODE` | `False` | Enables public demo capping behavior |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP sliding window limit |
| `CORS_ALLOWED_ORIGINS` | `None` | Comma-separated extra CORS origins |
| `FASTEMBED_CACHE_DIR` | `None` | Override FastEmbed ONNX model cache path |
| `JWT_SECRET_KEY` | Dev fallback | Signing secret for JWT Bearer tokens |
| `JWT_ALGORITHM` / `JWT_EXPIRE_MINUTES` | `HS256` / `1440` | JWT algorithm and expiry |
| `CHRONEXA_OFFLINE` | `0` | `1` disables external LLM API calls |
| `INTEGRATION_MODE` | `mock` | `mock` or `live` |
| `GROQ_API_KEY` / `GROQ_MODEL` | `None` / `qwen/qwen3.8-27b` | Groq credentials and model |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | `None` / `gpt-4o-mini` / OpenAI URL | OpenAI-compatible config |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | `None` / `claude-3-5-haiku-20241022` | Anthropic config |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | `None` / `gemini-flash-latest` | Google Gemini config |
| `CLAUDE_MODEL` | `haiku` | Claude CLI fallback model |
| `SLACK_BOT_TOKEN` / `GMAIL_CREDENTIALS` / `CALENDAR_CREDENTIALS` | `None` | Fallback integration credentials |

## Developer Commands (`Makefile` & `run.sh`)

```bash
make run      # Start local FastAPI server
make test     # Run pytest suite
make lint     # Run ruff checks
make format   # Check formatting with black
./run.sh eval # Run offline retrieval & action benchmarks
```

