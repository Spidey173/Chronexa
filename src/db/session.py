"""Database session and connection management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from src.config import settings

db_url = settings.DATABASE_URL or "sqlite:///./chronexa.db"

if db_url.startswith("postgresql"):
    # PostgreSQL connection (e.g. Neon serverless / Render)
    try:
        import psycopg2  # noqa: F401
        if not db_url.startswith("postgresql+psycopg2://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    except ImportError:
        pass

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
    )
elif settings.DEMO_MODE or db_url == "sqlite:///:memory:":
    # Read-only serverless DEMO_MODE fallback: in-memory state shared across warm instance
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args=connect_args,
        poolclass=StaticPool,
    )
else:
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=False,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
