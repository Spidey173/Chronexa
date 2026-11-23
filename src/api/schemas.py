"""Pydantic request and response schemas for Chronexa FastAPI endpoints."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User search query or question")
    as_of: str = Field(default="2026-09-18T18:00:00-07:00", description="ISO-8601 temporal point-in-time timestamp")
    mode: str = Field(default="hybrid", description="Retrieval mode: 'hybrid' or 'bm25'")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to retrieve")
    rerank: bool = Field(default=False, description="Enable heavy cross-encoder reranking (slower on CPU)")


class QueryResultItem(BaseModel):
    unit_id: str
    channel: str
    sender: str
    time: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    as_of: str
    mode: str
    answer: str
    results: List[Dict[str, Any]]
    sources: List[str] = []
    count: int
    status: str = "success"


class ActionProposeRequest(BaseModel):
    command: str = Field(..., min_length=1, description="Natural language action command")
    as_of: str = Field(default="2026-09-18T18:00:00-07:00", description="ISO-8601 temporal context timestamp")


class ActionProposalResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    command: str
    as_of: str
    status: str  # proposed, approved, rejected, executed, failed
    predicted: Any
    created_at: datetime
    updated_at: Optional[datetime] = None
    execution_result: Optional[Any] = None


class ActionApproveResponse(BaseModel):
    id: str
    status: str
    message: str


class ActionRejectResponse(BaseModel):
    id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str = "live_local_engine"
    database: str = "connected"
    vector_store_count: int
    corpus_size: int = 1316
    offline: bool = False
    loopback_only: bool = True
    version: str = "2.0.0"
    rate_limiter: Optional[str] = None
