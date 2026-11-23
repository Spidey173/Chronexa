"""Integration tests for Chronexa API server and security guardrails using pytest and TestClient.

Tests cover:
- Health check endpoints (/health and /api/health)
- Search query endpoints (/query and /api/search) via GET and POST
- Action proposal endpoints (/actions/propose and /api/action) with schema validation
- Input validation (empty query, invalid mode, invalid top_k, malformed date)
- Security enforcement (security headers, CORS, payload size limits)
- File access restrictions (blocking .env, .git, and internal source code)
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app, MAX_REQUEST_BODY_BYTES
from src.api.deps import EngineState


@pytest.fixture(scope="module")
def client():
    # Initialize engines in offline mode for fast deterministic testing
    state = EngineState.get_instance(pre_warm=False)
    if state.action_parser is not None:
        state.action_parser.provider = "none"
    if state.generator is not None:
        state.generator.provider = "none"
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body.get("status") == "ok"
    assert body.get("mode") == "live_local_engine"
    assert body.get("corpus_size") == 1316
    assert body.get("loopback_only") is True

    # Check root health endpoint
    res2 = client.get("/health")
    assert res2.status_code == 200
    assert res2.json().get("status") == "ok"


def test_search_post_valid_hybrid(client):
    res = client.post("/api/search", json={
        "query": "When is Route Planner v2 launching?",
        "as_of": "2026-09-18T18:00:00-07:00",
        "mode": "hybrid",
        "top_k": 5
    })
    assert res.status_code == 200
    body = res.json()
    assert "results" in body
    assert "answer" in body
    assert "sources" in body
    assert len(body["results"]) == 5


def test_search_get_query_params(client):
    res = client.get("/api/search?q=geocoding+regression&mode=bm25&top_k=3")
    assert res.status_code == 200
    body = res.json()
    assert "results" in body
    assert len(body["results"]) > 0
    assert len(body["results"]) <= 3


def test_search_rejects_empty_query(client):
    res = client.post("/api/search", json={
        "query": "   ",
        "as_of": "2026-09-18T18:00:00-07:00"
    })
    assert res.status_code == 400
    body = res.json()
    assert "error" in body
    assert "empty" in body["error"].lower()


def test_search_rejects_invalid_mode(client):
    res = client.post("/api/search", json={
        "query": "launch date",
        "mode": "unsupported_mode"
    })
    assert res.status_code == 400
    body = res.json()
    assert "error" in body
    assert "invalid mode" in body["error"].lower()


def test_search_rejects_out_of_bounds_top_k(client):
    res = client.post("/api/search", json={
        "query": "launch date",
        "top_k": 0
    })
    assert res.status_code == 400

    res2 = client.post("/api/search", json={
        "query": "launch date",
        "top_k": 100
    })
    assert res2.status_code == 400


def test_search_rejects_malformed_as_of(client):
    res = client.post("/api/search", json={
        "query": "launch date",
        "as_of": "not-a-real-date"
    })
    assert res.status_code == 400
    body = res.json()
    assert "as_of" in body.get("error", "").lower()


def test_action_parsing_app_open(client):
    res = client.post("/api/action", json={
        "command": "Open Figma",
        "as_of": "2026-09-18T18:00:00-07:00"
    })
    assert res.status_code == 200
    body = res.json()
    assert body.get("command") == "Open Figma"
    predicted = body.get("predicted", [])
    assert len(predicted) == 1
    assert predicted[0]["type"] == "app.open"
    assert predicted[0]["args"]["app"] == "Figma"


def test_action_parsing_destructive_confirm(client):
    res = client.post("/api/action", json={
        "command": "Delete all my emails from Marcus",
        "as_of": "2026-09-18T18:00:00-07:00"
    })
    assert res.status_code == 200
    body = res.json()
    predicted = body.get("predicted", [])
    assert len(predicted) == 1
    assert predicted[0]["type"] == "confirm"


def test_action_parsing_ambiguity_clarify(client):
    res = client.post("/api/action", json={
        "command": "Message Sarah about the pricing proposal",
        "as_of": "2026-09-18T18:00:00-07:00"
    })
    assert res.status_code == 200
    body = res.json()
    predicted = body.get("predicted", [])
    assert len(predicted) == 1
    assert predicted[0]["type"] == "clarify"


def test_action_rejects_missing_command(client):
    res = client.post("/api/action", json={
        "command": ""
    })
    assert res.status_code == 400
    body = res.json()
    assert "error" in body


def test_action_rejects_malformed_as_of(client):
    res = client.post("/api/action", json={
        "command": "Open Figma",
        "as_of": "2026-99-99T99:99:99"
    })
    assert res.status_code == 400
    body = res.json()
    assert "error" in body


def test_malformed_json_body(client):
    res = client.post("/api/search", content=b"not-json{{{", headers={"Content-Type": "application/json"})
    assert res.status_code == 400
    body = res.json()
    assert "error" in body


def test_payload_too_large_rejected(client):
    large_payload = b"x" * (MAX_REQUEST_BODY_BYTES + 1024)
    res = client.post("/api/search", content=large_payload, headers={"Content-Type": "application/json"})
    assert res.status_code == 413


def test_security_headers_and_cors(client):
    res = client.get("/api/health", headers={"Origin": "http://localhost:8000"})
    assert res.status_code == 200
    assert res.headers.get("x-content-type-options") == "nosniff"
    assert res.headers.get("x-frame-options") == "DENY"


def test_security_blocks_env_file(client):
    res = client.get("/.env")
    assert res.status_code == 403
    body = res.json()
    assert "forbidden" in body.get("error", "").lower()


def test_security_blocks_git_directory(client):
    res = client.get("/.git/config")
    assert res.status_code == 403
    body = res.json()
    assert "forbidden" in body.get("error", "").lower()


def test_security_blocks_python_sources(client):
    res = client.get("/src/config.py")
    assert res.status_code == 403
    body = res.json()
    assert "forbidden" in body.get("error", "").lower()


def test_serve_frontend_static_assets(client):
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")

    res_css = client.get("/styles.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers.get("content-type", "")

    res_data = client.get("/data.js")
    assert res_data.status_code == 200
    assert "javascript" in res_data.headers.get("content-type", "")

    res_app = client.get("/app.js")
    assert res_app.status_code == 200
    assert "javascript" in res_app.headers.get("content-type", "")


def test_frontend_honest_content_and_claims(client):
    res_root = client.get("/")
    html_content = res_root.text

    # No inflated claims
    assert "mathematical zero future leakage" not in html_content.lower()
    assert "cosine > 0.97" not in html_content

    # Grounded benchmark wording and guided walkthrough
    assert "BENCHMARK REPLAY MODE" in html_content
    assert "Under the Hood" in html_content
    assert "Try this first" in html_content
    assert "btnTrySep9" in html_content
    assert "btnTrySep18" in html_content
    assert "Cross-Encoder Joint Reranking" in html_content
    assert "https://www.linkedin.com/in/pruthvir173/" in html_content


def test_frontend_api_origin_security(client):
    res_app = client.get("/app.js")
    js_content = res_app.text

    # Origin security and allowlist functions
    assert "isAllowedApiOrigin" in js_content
    assert "sanitizeApiUrl" in js_content
    assert "getAuthHeaders" in js_content
    assert "ALLOWED_API_HOSTS" in js_content
    assert "localhost" in js_content
    assert "127.0.0.1" in js_content

    # Ensure unverified external subdomains are not in trusted allowlist
    assert "chronexa-api.onrender.com" not in js_content
    assert "chronexa-api.fly.dev" not in js_content
