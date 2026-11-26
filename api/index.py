"""Vercel Serverless entrypoint exposing FastAPI app."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bundle path overrides for serverless execution
fastembed_cache = ROOT / "fastembed_cache"
if fastembed_cache.exists():
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(fastembed_cache))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CHRONEXA_OFFLINE", "1")
os.environ.setdefault("INTEGRATION_MODE", "mock")
os.environ.setdefault("DEMO_MODE", "1")

from src.api.app import app  # noqa: E402

__all__ = ["app"]

