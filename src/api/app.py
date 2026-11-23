"""FastAPI Application factory and lifespan configuration."""
import time
from collections import defaultdict
from typing import Dict, List
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import settings, ROOT_DIR
from src.api.routes import router
from src.api.deps import EngineState
from src.db.session import engine, Base

MAX_REQUEST_BODY_BYTES = 64 * 1024

# In-memory per-IP sliding window rate limiter
_RATE_LIMIT_PER_MINUTE = getattr(settings, "RATE_LIMIT_PER_MINUTE", 60)
_RATE_LIMIT_WINDOW_SEC = 60.0
_ip_request_timestamps: Dict[str, List[float]] = defaultdict(list)


def _is_rate_limited(client_ip: str) -> bool:
    """Two-tier sliding window rate limiter:
    1. Tier 1: Fast in-memory per-instance sliding window cache.
    2. Tier 2: Neon PostgreSQL persistent table for distributed multi-instance serverless.
    If database is unavailable or SQLite in-memory, gracefully relies on Tier 1.
    """
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW_SEC

    # Tier 1: In-memory sliding window check
    timestamps = [t for t in _ip_request_timestamps[client_ip] if t > cutoff]
    if len(timestamps) >= _RATE_LIMIT_PER_MINUTE:
        _ip_request_timestamps[client_ip] = timestamps
        return True

    # Tier 2: Persistent database-backed check for distributed serverless functions
    if (
        getattr(settings, "DATABASE_URL", None)
        and "postgresql" in settings.DATABASE_URL
        and not getattr(settings, "DEMO_MODE", False)
    ):
        try:
            from src.db.session import SessionLocal
            from src.db.models import RateLimitRecord
            db = SessionLocal()
            try:
                db_count = db.query(RateLimitRecord).filter(
                    RateLimitRecord.client_ip == client_ip,
                    RateLimitRecord.timestamp > cutoff
                ).count()
                if db_count >= _RATE_LIMIT_PER_MINUTE:
                    return True
                # Record event in database
                db.add(RateLimitRecord(client_ip=client_ip, timestamp=now))
                # Prune old records periodically
                if len(timestamps) % 10 == 0:
                    db.query(RateLimitRecord).filter(
                        RateLimitRecord.timestamp < cutoff - 60
                    ).delete(synchronize_session=False)
                db.commit()
            finally:
                db.close()
        except Exception:
            # Fall back to in-memory decision on transient DB connectivity errors
            pass

    timestamps.append(now)
    _ip_request_timestamps[client_ip] = timestamps
    return False


def reset_rate_limits():
    """Helper for testing to reset rate limit states."""
    _ip_request_timestamps.clear()
    if (
        getattr(settings, "DATABASE_URL", None)
        and "postgresql" in settings.DATABASE_URL
        and not getattr(settings, "DEMO_MODE", False)
    ):
        try:
            from src.db.session import SessionLocal
            from src.db.models import RateLimitRecord
            db = SessionLocal()
            try:
                db.query(RateLimitRecord).delete()
                db.commit()
            finally:
                db.close()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables (for sqlite / test runs if migrations not run)
    Base.metadata.create_all(bind=engine)
    # Pre-warm AI engines
    EngineState.get_instance(pre_warm=False)
    yield


app = FastAPI(
    title="Chronexa API",
    description="Workplace Memory System & Safe Action Assistant API with Point-in-Time Retrieval and Human-in-the-Loop Dry-Runs",
    version="2.0.0",
    lifespan=lifespan
)

# Custom CORS & Security Middleware
@app.middleware("http")
async def security_and_limits_middleware(request: Request, call_next):
    # 1. Block access to hidden files (.env, .git) and internal source/data directories
    path = request.url.path
    if (
        "/.env" in path
        or "/.git" in path
        or path.startswith(("/src/", "/tests/", "/eval/", "/data/"))
        or any(part.startswith(".") for part in Path(path).parts)
    ):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Forbidden: Access to hidden files, system configuration, and internal source code is forbidden.",
                "detail": "Forbidden: Access denied",
                "status": "error"
            }
        )

    # 2. Per-IP Rate Limiting (60 requests/minute)
    client_ip = request.client.host if request.client else "127.0.0.1"
    if _is_rate_limited(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Too Many Requests: Rate limit exceeded (60 req/min). Please try again shortly.",
                "detail": "Rate limit exceeded",
                "status": "error"
            }
        )

    # 3. Enforce maximum request body size (64 KB)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": f"Payload Too Large: Maximum allowed body is {MAX_REQUEST_BODY_BYTES} bytes.",
                        "detail": f"Payload Too Large: Maximum allowed body is {MAX_REQUEST_BODY_BYTES} bytes.",
                        "status": "error"
                    }
                )
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid Content-Length header", "status": "error"}
            )

    response = await call_next(request)

    # 3. Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"

    return response


# CORS configuration: Restricted to official frontend and localhost for development
allowed_origins = [
    "https://chronexa17.vercel.app",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
]
if getattr(settings, "CORS_ALLOWED_ORIGINS", None):
    extra = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
    allowed_origins.extend(extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Standardized error response handlers with both "error" and "detail"
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "detail": str(exc.detail),
            "status": "error"
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first_err = exc.errors()[0] if exc.errors() else {"msg": "Validation error"}
    err_msg = f"{first_err.get('loc', ['parameter'])[-1]}: {first_err.get('msg', 'invalid')}"
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": err_msg,
            "detail": exc.errors(),
            "status": "error"
        }
    )


# Include API routes
app.include_router(router)


# Root route: serve index.html if it exists, or API info
@app.get("/", include_in_schema=False)
def serve_root():
    index_path = ROOT_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "Chronexa API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/styles.css", include_in_schema=False)
def serve_styles():
    p = ROOT_DIR / "styles.css"
    if p.exists():
        return FileResponse(str(p), media_type="text/css")
    raise StarletteHTTPException(status_code=404, detail="styles.css not found")


@app.get("/data.js", include_in_schema=False)
def serve_data():
    p = ROOT_DIR / "data.js"
    if p.exists():
        return FileResponse(str(p), media_type="application/javascript")
    raise StarletteHTTPException(status_code=404, detail="data.js not found")


@app.get("/app.js", include_in_schema=False)
def serve_app_js():
    p = ROOT_DIR / "app.js"
    if p.exists():
        return FileResponse(str(p), media_type="application/javascript")
    raise StarletteHTTPException(status_code=404, detail="app.js not found")
