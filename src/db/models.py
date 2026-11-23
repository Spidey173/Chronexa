"""SQLAlchemy ORM models for Chronexa: users, action proposals, and audit logs."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, ForeignKey, Float
from sqlalchemy.orm import relationship
from src.db.session import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)
    role = Column(String(32), default="viewer", nullable=False)  # viewer, operator, admin
    oauth_tokens = Column(JSON, nullable=True)  # per-user integration credentials (slack, gmail, calendar)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    proposals = relationship("ActionProposal", back_populates="user", cascade="all, delete-orphan")


class ActionProposal(Base):
    __tablename__ = "action_proposals"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    command = Column(Text, nullable=False)
    as_of = Column(String(64), nullable=False)
    # status transitions: proposed -> approved -> executed / failed OR proposed -> rejected
    status = Column(String(32), default="proposed", index=True, nullable=False)
    payload = Column(JSON, nullable=False)  # parsed action list or dict
    execution_result = Column(JSON, nullable=True)  # output after background execution
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="proposals")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), nullable=True, index=True)
    action = Column(String(64), nullable=False, index=True)  # e.g., propose, approve, reject, execute
    resource_type = Column(String(64), default="action_proposal", nullable=False)
    resource_id = Column(String(64), nullable=True, index=True)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class RateLimitRecord(Base):
    __tablename__ = "rate_limit_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_ip = Column(String(45), index=True, nullable=False)
    timestamp = Column(Float, index=True, nullable=False)

