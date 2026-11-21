"""FastAPI route definitions for Chronexa: /query, /actions/propose, approve, reject, /health."""
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Query
from sqlalchemy.orm import Session

from src.db.session import get_db, SessionLocal
from src.db.models import User, ActionProposal, AuditLog
from src.api.schemas import (
    QueryRequest, QueryResponse,
    ActionProposeRequest, ActionProposalResponse,
    ActionApproveResponse, ActionRejectResponse,
    HealthResponse
)
from src.api.deps import (
    get_engine, EngineState,
    get_current_user, get_optional_current_user,
    require_role
)
from src.actions.schema import ActionValidationError
from src.actions.executor import ActionExecutor
from src.pipeline import search_single
from src.retrieval.temporal import parse_iso_datetime
from src.auth import create_access_token, get_password_hash, verify_password

router = APIRouter()


# ---------------------------------------------------------------------------
# Background Execution Worker
# ---------------------------------------------------------------------------
def execute_proposal_background(proposal_id: str, approver_id: Optional[str] = None):
    """Decoupled background execution of approved action proposals."""
    db: Session = SessionLocal()
    try:
        proposal = db.query(ActionProposal).filter(ActionProposal.id == proposal_id).first()
        if not proposal:
            return

        # Fetch user credentials if available
        user_creds = None
        if proposal.user_id:
            user = db.query(User).filter(User.id == proposal.user_id).first()
            if user and user.oauth_tokens:
                user_creds = user.oauth_tokens

        executor = ActionExecutor()
        try:
            results = executor.execute(proposal.payload, user_credentials=user_creds)
            proposal.status = "executed"
            proposal.execution_result = results
        except Exception as e:
            proposal.status = "failed"
            proposal.execution_result = {"error": str(e)}

        # Write execution to audit log
        audit = AuditLog(
            user_id=approver_id or proposal.user_id,
            action="execute",
            resource_type="action_proposal",
            resource_id=proposal.id,
            details={
                "status": proposal.status,
                "result": proposal.execution_result
            }
        )
        db.add(audit)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[execute_proposal_background] Error executing proposal {proposal_id}: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["System"])
@router.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def health_check(engine: EngineState = Depends(get_engine)):
    vec_count = 0
    if hasattr(engine.retriever, "dense") and engine.retriever.dense is not None:
        if hasattr(engine.retriever.dense, "vector_store") and engine.retriever.dense.vector_store is not None:
            vec_count = engine.retriever.dense.vector_store.count()
        elif hasattr(engine.retriever.dense, "embedding_cache"):
            vec_count = len(engine.retriever.dense.embedding_cache)

    return HealthResponse(
        status="ok",
        mode="live_local_engine",
        database="connected",
        vector_store_count=vec_count,
        corpus_size=1316,
        offline=False,
        loopback_only=True,
        version="2.0.0"
    )


# ---------------------------------------------------------------------------
# Query Endpoints (/query & /api/search)
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse, tags=["Retrieval"])
@router.post("/api/search", response_model=QueryResponse, include_in_schema=False)
def search_query(
    req: QueryRequest,
    engine: EngineState = Depends(get_engine),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty and must be a string")

    if len(req.query) > 2000:
        raise HTTPException(status_code=400, detail="Query exceeds maximum allowed length of 2000 characters")

    try:
        parse_iso_datetime(req.as_of)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid as_of timestamp: {str(e)}")

    if req.mode not in ("hybrid", "bm25"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{req.mode}': must be one of ['bm25', 'hybrid']")

    if not (1 <= req.top_k <= 50):
        raise HTTPException(status_code=400, detail=f"Invalid top_k '{req.top_k}': must be an integer between 1 and 50")

    res = search_single(
        query=req.query,
        as_of=req.as_of,
        mode=req.mode,
        top_k=req.top_k,
        generate_answer=True,
        retriever=engine.retriever,
        generator=engine.generator
    )

    formatted_results = []
    for r in res["results"]:
        formatted_results.append({
            "unit_id": r["id"],
            "id": r["id"],
            "channel": r.get("record", ""),
            "record": r.get("record", ""),
            "sender": r.get("speaker", ""),
            "speaker": r.get("speaker", ""),
            "time": r["time"],
            "text": r["text"],
            "score": round(float(r["score"]), 4)
        })

    return QueryResponse(
        query=res["query"],
        as_of=res["as_of"],
        mode=req.mode,
        answer=res["answer"],
        results=formatted_results,
        sources=res.get("sources", []),
        count=len(formatted_results),
        status="success"
    )


@router.get("/query", response_model=QueryResponse, tags=["Retrieval"])
@router.get("/api/search", response_model=QueryResponse, include_in_schema=False)
def search_query_get(
    q: str = Query(..., min_length=1, max_length=2000),
    as_of: str = Query("2026-09-18T18:00:00-07:00"),
    mode: str = Query("hybrid"),
    top_k: int = Query(10, ge=1, le=50),
    engine: EngineState = Depends(get_engine),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    req = QueryRequest(query=q, as_of=as_of, mode=mode, top_k=top_k)
    return search_query(req, engine, current_user)


# ---------------------------------------------------------------------------
# Action Endpoints (/actions/propose, /actions/{id}/approve, /actions/{id}/reject)
# ---------------------------------------------------------------------------
@router.post("/actions/propose", response_model=ActionProposalResponse, tags=["Actions"])
@router.post("/api/action", include_in_schema=False)
def propose_action(
    req: ActionProposeRequest,
    db: Session = Depends(get_db),
    engine: EngineState = Depends(get_engine),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    if not req.command or not req.command.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'command' parameter (must be non-empty string)")

    try:
        parse_iso_datetime(req.as_of)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid as_of timestamp: {str(e)}")

    try:
        actions = engine.action_parser.parse(command=req.command, as_of=req.as_of)
    except ActionValidationError as e:
        raise HTTPException(status_code=400, detail=f"Action validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing action command: {str(e)}")

    proposal = ActionProposal(
        id=str(uuid.uuid4()),
        user_id=current_user.id if current_user else None,
        command=req.command,
        as_of=req.as_of,
        status="proposed",
        payload=actions
    )
    db.add(proposal)

    # Write audit log
    audit = AuditLog(
        user_id=current_user.id if current_user else None,
        action="propose",
        resource_type="action_proposal",
        resource_id=proposal.id,
        details={"command": req.command, "actions_count": len(actions)}
    )
    db.add(audit)
    db.commit()
    db.refresh(proposal)

    return ActionProposalResponse(
        id=proposal.id,
        user_id=proposal.user_id,
        command=proposal.command,
        as_of=proposal.as_of,
        status=proposal.status,
        predicted=proposal.payload,
        created_at=proposal.created_at,
        updated_at=proposal.updated_at
    )


@router.post("/actions/{proposal_id}/approve", response_model=ActionApproveResponse, tags=["Actions"])
def approve_action(
    proposal_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("operator"))
):
    proposal = db.query(ActionProposal).filter(ActionProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Action proposal '{proposal_id}' not found")

    # State Machine: Proposal can NEVER execute without approval, can't be approved twice, rejected can never execute
    if proposal.status == "approved" or proposal.status == "executed":
        raise HTTPException(status_code=400, detail=f"Invalid state: proposal '{proposal_id}' has already been {proposal.status}")

    if proposal.status == "rejected":
        raise HTTPException(status_code=400, detail=f"Invalid state: rejected proposal '{proposal_id}' cannot be approved or executed")

    if proposal.status != "proposed":
        raise HTTPException(status_code=400, detail=f"Invalid proposal state '{proposal.status}' for approval")

    proposal.status = "approved"

    # Write approval to audit log with operator user id
    audit = AuditLog(
        user_id=current_user.id,
        action="approve",
        resource_type="action_proposal",
        resource_id=proposal.id,
        details={"command": proposal.command}
    )
    db.add(audit)
    db.commit()

    # Schedule decoupled background execution
    background_tasks.add_task(execute_proposal_background, proposal.id, current_user.id)

    return ActionApproveResponse(
        id=proposal.id,
        status="approved",
        message="Action proposal approved successfully. Execution enqueued in background."
    )


@router.post("/actions/{proposal_id}/reject", response_model=ActionRejectResponse, tags=["Actions"])
def reject_action(
    proposal_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    proposal = db.query(ActionProposal).filter(ActionProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Action proposal '{proposal_id}' not found")

    # State Machine: Once executed or approved, cannot reject; rejected cannot be re-rejected
    if proposal.status == "executed":
        raise HTTPException(status_code=400, detail="Cannot reject an already executed proposal")

    if proposal.status == "rejected":
        raise HTTPException(status_code=400, detail="Proposal is already rejected")

    proposal.status = "rejected"

    # Write rejection to audit log
    audit = AuditLog(
        user_id=current_user.id,
        action="reject",
        resource_type="action_proposal",
        resource_id=proposal.id,
        details={"command": proposal.command}
    )
    db.add(audit)
    db.commit()

    return ActionRejectResponse(
        id=proposal.id,
        status="rejected",
        message="Action proposal rejected successfully."
    )


@router.get("/actions/{proposal_id}", response_model=ActionProposalResponse, tags=["Actions"])
def get_action_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    proposal = db.query(ActionProposal).filter(ActionProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=404, detail=f"Action proposal '{proposal_id}' not found")

    return ActionProposalResponse(
        id=proposal.id,
        user_id=proposal.user_id,
        command=proposal.command,
        as_of=proposal.as_of,
        status=proposal.status,
        predicted=proposal.payload,
        created_at=proposal.created_at,
        updated_at=proposal.updated_at,
        execution_result=proposal.execution_result
    )


# ---------------------------------------------------------------------------
# Auth / Token Endpoints
# ---------------------------------------------------------------------------
from pydantic import BaseModel

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "viewer"  # viewer, operator, admin
    oauth_tokens: Optional[Dict[str, Any]] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str

@router.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"User with email '{req.email}' already exists")

    if req.role not in ("viewer", "operator", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role: must be viewer, operator, or admin")

    hashed = get_password_hash(req.password)
    user = User(
        id=str(uuid.uuid4()),
        email=req.email,
        hashed_password=hashed,
        role=req.role,
        oauth_tokens=req.oauth_tokens
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer", role=user.role, user_id=user.id)


@router.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
def login_for_access_token(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.hashed_password or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})
    return TokenResponse(access_token=token, token_type="bearer", role=user.role, user_id=user.id)
