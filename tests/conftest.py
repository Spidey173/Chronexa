"""Shared pytest fixtures for Chronexa API tests."""
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.deps import EngineState


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with engines initialized in deterministic offline mode."""
    state = EngineState.get_instance(pre_warm=False)
    if state.action_parser is not None:
        state.action_parser.provider = "none"
    if state.generator is not None:
        state.generator.provider = "none"
    with TestClient(app) as c:
        yield c
