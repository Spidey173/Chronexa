"""Lightweight local development & demo server for Chronexa workplace search prototype.

Designed strictly as a local evaluation tool and portfolio prototype server:
  - Binds strictly to local loopback (127.0.0.1) for safe developer evaluation
  - Strict CORS origin validation (allows only localhost / 127.0.0.1 / local file://)
  - Strict request size limit (64 KB max payload to prevent resource exhaustion)
  - Pure Python standard library (http.server) with zero production web server claims
"""
import argparse
import json
import os
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR
from src.pipeline import search_single
from src.actions.parser import ActionParser
from src.actions.schema import ActionValidationError
from src.generation.generator import AnswerGenerator
from src.ingestion.loader import _get_raw_records
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.temporal import parse_iso_datetime

MAX_REQUEST_BODY_BYTES = 64 * 1024  # 64 KB limit
ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
ALLOWED_MODES = {"hybrid", "bm25"}


class ChronexaRequestHandler(SimpleHTTPRequestHandler):
    """Handles static web assets and dynamic Chronexa search & action API routes securely."""

    # Shared pre-warmed engines to avoid per-request recreation
    retriever: Optional[HybridRetriever] = None
    generator: Optional[AnswerGenerator] = None
    action_parser: Optional[ActionParser] = None

    @classmethod
    def initialize_engines(cls, pre_warm: bool = True):
        """Initializes shared engines once for the server process."""
        if cls.retriever is None:
            raw_units, _, _ = _get_raw_records(DATA_DIR)
            cls.retriever = HybridRetriever()
            if pre_warm and hasattr(cls.retriever, "dense") and cls.retriever.dense is not None:
                cls.retriever.dense.warm_cache(raw_units)
            cls.generator = AnswerGenerator()
            cls.action_parser = ActionParser(retriever=cls.retriever)

    def __init__(self, *args, **kwargs):
        # Serve static files strictly from repository root
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def _verify_local_client(self) -> bool:
        """Restricts all incoming connections strictly to the loopback interface."""
        client_ip = self.client_address[0]
        if client_ip not in ALLOWED_LOCAL_HOSTS and not client_ip.startswith("127."):
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Forbidden: Chronexa local server only accepts trusted loopback connections (127.0.0.1)."
            }).encode("utf-8"))
            return False
        return True

    def _get_allowed_origin(self) -> Optional[str]:
        """Validates incoming Origin against trusted local origins only."""
        origin = self.headers.get("Origin")
        if not origin:
            return None
        # Allow local development origins: localhost, 127.0.0.1, and null (local file protocol)
        if (
            origin == "null"
            or origin == "http://localhost"
            or origin.startswith("http://localhost:")
            or origin == "http://127.0.0.1"
            or origin.startswith("http://127.0.0.1:")
        ):
            return origin
        return None

    def _set_cors_headers(self):
        allowed = self._get_allowed_origin()
        if allowed:
            self.send_header("Access-Control-Allow-Origin", allowed)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")

    def _send_json_error(self, status: int, message: str):
        """Sends a structured JSON error response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"error": message, "status": "error"}).encode("utf-8"))

    def do_OPTIONS(self):
        if not self._verify_local_client():
            return
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if not self._verify_local_client():
            return

        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            resp = {
                "status": "ok",
                "mode": "live_local_engine",
                "host": self.server.server_address[0],
                "port": self.server.server_address[1],
                "corpus_size": 1316,
                "offline": os.environ.get("CHRONEXA_OFFLINE") == "1",
                "loopback_only": True
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/search":
            # Support GET search with validated query parameters
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", params.get("query", [""]))[0]
            as_of = params.get("as_of", ["2026-09-18T18:00:00-07:00"])[0]
            mode = params.get("mode", ["hybrid"])[0]
            top_k_raw = params.get("top_k", ["10"])[0]

            # Validate top_k
            try:
                top_k = int(top_k_raw)
            except (ValueError, TypeError):
                self._send_json_error(400, "Invalid top_k: must be an integer between 1 and 50")
                return

            self._handle_search(query, as_of, mode, top_k)
            return

        # Default static file serving (e.g. index.html)
        return super().do_GET()

    def do_POST(self):
        if not self._verify_local_client():
            return

        parsed = urllib.parse.urlparse(self.path)

        # Enforce request payload size limit to prevent memory exhaustion
        content_length_header = self.headers.get("Content-Length")
        if not content_length_header:
            content_length = 0
        else:
            try:
                content_length = int(content_length_header)
            except ValueError:
                self._send_json_error(400, "Invalid Content-Length header")
                return

        if content_length > MAX_REQUEST_BODY_BYTES:
            self.send_response(413)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Payload Too Large: Maximum allowed body is {MAX_REQUEST_BODY_BYTES} bytes."
            }).encode("utf-8"))
            return

        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception as e:
            self._send_json_error(400, f"Malformed JSON in request body: {str(e)}")
            return

        if not isinstance(body, dict):
            self._send_json_error(400, "Request body must be a JSON object")
            return

        if parsed.path == "/api/search":
            query = body.get("query")
            as_of = body.get("as_of", "2026-09-18T18:00:00-07:00")
            mode = body.get("mode", "hybrid")
            top_k_raw = body.get("top_k", 10)

            try:
                top_k = int(top_k_raw)
            except (ValueError, TypeError):
                self._send_json_error(400, "Invalid top_k: must be an integer between 1 and 50")
                return

            self._handle_search(query, as_of, mode, top_k)
            return

        elif parsed.path == "/api/action":
            command = body.get("command")
            as_of = body.get("as_of", "2026-09-18T18:00:00-07:00")

            if not isinstance(command, str) or not command.strip():
                self._send_json_error(400, "Missing or invalid 'command' parameter (must be non-empty string)")
                return

            if not isinstance(as_of, str):
                self._send_json_error(400, "Invalid 'as_of' parameter: must be an ISO-8601 string")
                return

            try:
                parse_iso_datetime(as_of)
            except Exception as e:
                self._send_json_error(400, f"Invalid as_of timestamp: {str(e)}")
                return

            # Ensure shared action parser is ready
            self.initialize_engines(pre_warm=False)

            try:
                actions = self.action_parser.parse(command=command, as_of=as_of)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "command": command,
                    "as_of": as_of,
                    "predicted": actions,
                    "status": "success"
                }).encode("utf-8"))
            except ActionValidationError as e:
                self._send_json_error(400, f"Action validation error: {str(e)}")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Server error processing action: {str(e)}",
                    "status": "error"
                }).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def _handle_search(self, query: Any, as_of: Any, mode: str, top_k: int):
        # Validate query
        if not isinstance(query, str) or not query.strip():
            self._send_json_error(400, "Query cannot be empty and must be a string")
            return

        if len(query) > 2000:
            self._send_json_error(400, "Query exceeds maximum allowed length of 2000 characters")
            return

        # Validate mode
        if mode not in ALLOWED_MODES:
            self._send_json_error(400, f"Invalid mode '{mode}': must be one of {sorted(list(ALLOWED_MODES))}")
            return

        # Validate top_k bounds
        if not (1 <= top_k <= 50):
            self._send_json_error(400, f"Invalid top_k '{top_k}': must be an integer between 1 and 50")
            return

        # Validate as_of date format
        if not isinstance(as_of, str):
            self._send_json_error(400, "Invalid as_of parameter: must be an ISO-8601 string")
            return

        try:
            parse_iso_datetime(as_of)
        except Exception as e:
            self._send_json_error(400, f"Invalid as_of timestamp: {str(e)}")
            return

        # Ensure engines are initialized
        self.initialize_engines(pre_warm=False)

        try:
            result = search_single(
                query=query.strip(),
                as_of=as_of,
                data_dir=DATA_DIR,
                mode=mode,
                generate_answer=True,
                top_k=top_k,
                retriever=self.retriever,
                generator=self.generator
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Search execution failed: {str(e)}"}).encode("utf-8"))


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Starts the Chronexa local demo server restricted strictly to trusted loopback."""
    print(f"=== Chronexa Local Demo Server (Loopback Prototype) ===")
    print(f"Loopback Interface: {host} (refuses non-local network traffic)")
    print(f"Notice: Local development demo server for interactive testing and portfolio review.")
    print(f"Max Request Payload: {MAX_REQUEST_BODY_BYTES // 1024} KB")
    print(f"Pre-warming shared embedding cache & engines...")
    ChronexaRequestHandler.initialize_engines(pre_warm=True)
    raw_units, _, _ = _get_raw_records(DATA_DIR)
    print(f"Corpus index and shared engines ready ({len(raw_units)} units).")

    server_address = (host, port)
    httpd = HTTPServer(server_address, ChronexaRequestHandler)
    print(f"\n🚀 Chronexa local API server running at: http://{host}:{port}/")
    print(f"   Dashboard UI: http://{host}:{port}/")
    print(f"   Health Check: http://{host}:{port}/api/health")
    print(f"   Search Route: POST http://{host}:{port}/api/search")
    print(f"   Action Route: POST http://{host}:{port}/api/action")
    print(f"\nPress Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronexa secure local server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (defaults to 127.0.0.1 loopback only)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("port_pos", nargs="?", type=int, default=None, help="Port (optional positional alternative)")
    args = parser.parse_args()
    selected_port = args.port_pos if args.port_pos is not None else args.port
    run_server(host=args.host, port=selected_port)
