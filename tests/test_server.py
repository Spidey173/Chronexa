"""Unit and integration tests for Chronexa local API server and security guardrails.

Tests cover:
- Health check endpoint (/api/health)
- Search endpoint (/api/search) via GET and POST
- Action parsing endpoint (/api/action) with schema validation
- Input validation (empty query, invalid mode, invalid top_k, malformed date)
- Security enforcement (loopback verification, CORS headers, payload size limits)
- Static file access restrictions (blocking .env, .git, and source code)
"""
import io
import json
import re
import unittest
from src.server import ChronexaRequestHandler, MAX_REQUEST_BODY_BYTES


class MockSocket:
    """In-memory mock transport for deterministic, zero-network HTTP testing."""
    def __init__(self, data: bytes):
        self.rfile = io.BytesIO(data)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.rfile if "b" in mode and "r" in mode else self.wfile

    def sendall(self, data):
        self.wfile.write(data)


class DummyServer:
    server_address = ("127.0.0.1", 8000)


class TestChronexaServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize engines once for fast, fully offline tests
        ChronexaRequestHandler.initialize_engines(pre_warm=False)
        if ChronexaRequestHandler.action_parser is not None:
            ChronexaRequestHandler.action_parser.provider = "none"
        if ChronexaRequestHandler.generator is not None:
            ChronexaRequestHandler.generator.provider = "none"

    def _execute(self, method: str, path: str, body_bytes: bytes = b"", origin: str = None, headers: dict = None) -> tuple:
        hdrs = dict(headers or {})
        if origin:
            hdrs["Origin"] = origin
        req_lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1"]
        for k, v in hdrs.items():
            req_lines.append(f"{k}: {v}")
        if body_bytes and "Content-Length" not in hdrs:
            req_lines.append(f"Content-Length: {len(body_bytes)}")
        raw_req = "\r\n".join(req_lines).encode("utf-8") + b"\r\n\r\n" + body_bytes

        sock = MockSocket(raw_req)
        ChronexaRequestHandler(sock, ("127.0.0.1", 54321), DummyServer())
        raw_res = sock.wfile.getvalue()

        header_part, _, body_part = raw_res.partition(b"\r\n\r\n")
        lines = header_part.decode("utf-8", errors="replace").split("\r\n")
        status_line = lines[0] if lines else ""
        m = re.match(r"^HTTP/\d\.\d\s+(\d+)", status_line)
        status_code = int(m.group(1)) if m else 500

        resp_headers = {}
        for line in lines[1:]:
            if ": " in line:
                hk, hv = line.split(": ", 1)
                resp_headers[hk.lower()] = hv

        try:
            res_body = json.loads(body_part.decode("utf-8")) if body_part else {}
        except Exception:
            res_body = body_part.decode("utf-8", errors="replace")

        return status_code, res_body, resp_headers

    def _post(self, path: str, data: dict, origin: str = None) -> tuple:
        """Helper to send a POST request with JSON payload."""
        body = json.dumps(data).encode("utf-8")
        return self._execute("POST", path, body, origin=origin, headers={"Content-Type": "application/json"})

    def _post_raw(self, path: str, raw_bytes: bytes, content_type: str = "application/json", origin: str = None) -> tuple:
        return self._execute("POST", path, raw_bytes, origin=origin, headers={"Content-Type": content_type})

    def _get(self, path: str, origin: str = None) -> tuple:
        """Helper to send a GET request."""
        return self._execute("GET", path, origin=origin)

    def test_health_endpoint(self):
        status, body, headers = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "ok")
        self.assertEqual(body.get("mode"), "live_local_engine")
        self.assertTrue(body.get("loopback_only"))
        self.assertEqual(body.get("corpus_size"), 1316)

    def test_search_post_valid_hybrid(self):
        status, body, _ = self._post("/api/search", {
            "query": "When is Route Planner v2 launching?",
            "as_of": "2026-09-18T18:00:00-07:00",
            "mode": "hybrid",
            "top_k": 5
        })
        self.assertEqual(status, 200)
        self.assertIn("results", body)
        self.assertIn("answer", body)
        self.assertIn("sources", body)
        self.assertEqual(len(body["results"]), 5)

    def test_search_get_query_params(self):
        status, body, _ = self._get("/api/search?q=geocoding+regression&mode=bm25&top_k=3")
        self.assertEqual(status, 200)
        self.assertIn("results", body)
        self.assertTrue(len(body["results"]) > 0)
        self.assertLessEqual(len(body["results"]), 3)

    def test_search_rejects_empty_query(self):
        status, body, _ = self._post("/api/search", {
            "query": "   ",
            "as_of": "2026-09-18T18:00:00-07:00"
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertIn("empty", body["error"].lower())

    def test_search_rejects_invalid_mode(self):
        status, body, _ = self._post("/api/search", {
            "query": "launch date",
            "mode": "unsupported_mode"
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertIn("invalid mode", body["error"].lower())

    def test_search_rejects_out_of_bounds_top_k(self):
        status, body, _ = self._post("/api/search", {
            "query": "launch date",
            "top_k": 0
        })
        self.assertEqual(status, 400)

        status2, body2, _ = self._post("/api/search", {
            "query": "launch date",
            "top_k": 100
        })
        self.assertEqual(status2, 400)

    def test_search_rejects_malformed_as_of(self):
        status, body, _ = self._post("/api/search", {
            "query": "launch date",
            "as_of": "not-a-real-date"
        })
        self.assertEqual(status, 400)
        self.assertIn("as_of", body.get("error", "").lower())

    def test_action_parsing_app_open(self):
        status, body, _ = self._post("/api/action", {
            "command": "Open Figma",
            "as_of": "2026-09-18T18:00:00-07:00"
        })
        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "success")
        self.assertEqual(body.get("command"), "Open Figma")
        predicted = body.get("predicted", [])
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0]["type"], "app.open")
        self.assertEqual(predicted[0]["args"]["app"], "Figma")

    def test_action_parsing_destructive_confirm(self):
        status, body, _ = self._post("/api/action", {
            "command": "Delete all my emails from Marcus",
            "as_of": "2026-09-18T18:00:00-07:00"
        })
        self.assertEqual(status, 200)
        predicted = body.get("predicted", [])
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0]["type"], "confirm")

    def test_action_parsing_ambiguity_clarify(self):
        status, body, _ = self._post("/api/action", {
            "command": "Message Sarah about the pricing proposal",
            "as_of": "2026-09-18T18:00:00-07:00"
        })
        self.assertEqual(status, 200)
        predicted = body.get("predicted", [])
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0]["type"], "clarify")

    def test_action_rejects_missing_command(self):
        status, body, _ = self._post("/api/action", {
            "command": ""
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_action_rejects_malformed_as_of(self):
        status, body, _ = self._post("/api/action", {
            "command": "Open Figma",
            "as_of": "2026-99-99T99:99:99"
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_malformed_json_body(self):
        status, body, _ = self._post_raw("/api/search", b"not-json{{{")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_payload_too_large_rejected(self):
        large_payload = b"x" * (MAX_REQUEST_BODY_BYTES + 1024)
        status, body, _ = self._post_raw("/api/search", large_payload)
        self.assertEqual(status, 413)

    def test_security_headers_and_cors(self):
        status, body, headers = self._get("/api/health", origin="http://localhost:8000")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://localhost:8000")
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")

    def test_security_blocks_env_file(self):
        status, body, _ = self._get("/.env")
        self.assertEqual(status, 403)
        self.assertIn("forbidden", body.get("error", "").lower())

    def test_security_blocks_git_directory(self):
        status, body, _ = self._get("/.git/config")
        self.assertEqual(status, 403)
        self.assertIn("forbidden", body.get("error", "").lower())

    def test_security_blocks_python_sources(self):
        status, body, _ = self._get("/src/config.py")
        self.assertEqual(status, 403)
        self.assertIn("forbidden", body.get("error", "").lower())


if __name__ == "__main__":
    unittest.main()
