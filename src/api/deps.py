"""FastAPI dependencies: Database sessions, pre-warmed AI engines, JWT auth and RBAC."""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.core.config import DATA_DIR
from src.db.session import get_db
from src.db.models import User
from src.core.auth import decode_access_token, has_sufficient_role
from src.ingestion.loader import _get_raw_records
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import AnswerGenerator
from src.actions.parser import ActionParser

security = HTTPBearer(auto_error=False)


class EngineState:
    """Pre-warmed AI engines shared across requests."""
    _instance: Optional["EngineState"] = None

    def __init__(self, pre_warm: bool = True):
        raw_units, _, _ = _get_raw_records(DATA_DIR)
        self.retriever = HybridRetriever()
        if pre_warm and hasattr(self.retriever, "dense") and self.retriever.dense is not None:
            self.retriever.dense.warm_cache(raw_units)
        self.generator = AnswerGenerator()
        self.action_parser = ActionParser(retriever=self.retriever)

    @classmethod
    def get_instance(cls, pre_warm: bool = False) -> "EngineState":
        if cls._instance is None:
            cls._instance = cls(pre_warm=pre_warm)
        return cls._instance


def get_engine() -> EngineState:
    return EngineState.get_instance(pre_warm=False)


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Verifies JWT Bearer token and returns User identity on every authenticated request."""
    if auth is None or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(auth.credentials)
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload missing user identifier ('sub')",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        # Check by email if sub was email
        user = db.query(User).filter(User.email == user_id).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with token not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_optional_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Extracts user identity if Bearer token is provided, else returns None."""
    if auth is None or not auth.credentials:
        return None
    try:
        payload = decode_access_token(auth.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            user = db.query(User).filter(User.email == user_id).first()
        return user
    except Exception:
        return None


def require_role(required_role: str):
    """RBAC dependency ensuring the authenticated user has at least the required role."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not has_sufficient_role(current_user.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Action requires role '{required_role}' or higher (current role: '{current_user.role}').",
            )
        return current_user
    return role_checker
