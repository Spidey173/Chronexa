"""Unit and integration tests for Chronexa local API server and security guardrails.

Tests cover:
- Health check endpoint (/api/health)
- Search endpoint (/api/search) via GET and POST
- Action parsing endpoint (/api/action) with schema validation
- Input validation (empty query, invalid mode, invalid top_k, malformed date)
- Security enforcement (loopback verification, CORS headers, payload size limits)
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from src.server import ChronexaRequestHandler, MAX_REQUEST_BODY_BYTES


class TestChronexaServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize engines once for fast, fully offline tests
        ChronexaRequestHandler.initialize_engines(pre_warm=False)
        if ChronexaRequestHandler.action_parser is not None:
            ChronexaRequestHandler.action_parser.provider = "none"
        if ChronexaRequestHandler.generator is not None:
            ChronexaRequestHandler.generator.provider = "none"
        # Bind to port 0 to get an ephemeral open port on loopback
        cls.server = HTTPServer(("127.0.0.1", 0), ChronexaRequestHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        # Run server in background daemon thread
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, path: str, data: dict, origin: str = None) -> tuple:
        """Helper to send a POST request with JSON payload."""
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if origin:
            req.add_header("Origin", origin)
        try:
            with urllib.request.urlopen(req) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))
                return resp.status, res_body, resp.headers
        except urllib.error.HTTPError as e:
            try:
                res_body = json.loads(e.read().decode("utf-8")) if e.fp else {}
            finally:
                e.close()
            return e.code, res_body, e.headers

    def _get(self, path: str, origin: str = None) -> tuple:
        """Helper to send a GET request."""
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        if origin:
            req.add_header("Origin", origin)
        try:
            with urllib.request.urlopen(req) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))
                return resp.status, res_body, resp.headers
        except urllib.error.HTTPError as e:
            try:
                res_body = json.loads(e.read().decode("utf-8")) if e.fp else {}
            finally:
                e.close()
            return e.code, res_body, e.headers

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
        url = f"{self.base_url}/api/search"
        req = urllib.request.Request(url, data=b"not-json{{{", method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                self.fail("Expected 400 for malformed JSON")
        except urllib.error.HTTPError as e:
            try:
                self.assertEqual(e.code, 400)
            finally:
                e.close()

    def test_payload_too_large_rejected(self):
        url = f"{self.base_url}/api/search"
        large_payload = b"x" * (MAX_REQUEST_BODY_BYTES + 1024)
        req = urllib.request.Request(url, data=large_payload, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                self.fail("Expected 413 for oversized payload")
        except urllib.error.HTTPError as e:
            try:
                self.assertEqual(e.code, 413)
            finally:
                e.close()

    def test_security_headers_and_cors(self):
        status, body, headers = self._get("/api/health", origin="http://localhost:8000")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:8000")
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")


if __name__ == "__main__":
    unittest.main()
