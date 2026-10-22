"""Lightweight local HTTP server for Chronexa interactive search & action demo.

Uses Python standard library (http.server) with zero external server dependencies.
Provides:
  - GET  /api/health      -> Server status & engine metadata
  - POST /api/search      -> Live hybrid search + grounded answer synthesis for arbitrary queries
  - POST /api/action      -> Live TextOS dry-run action parser
  - GET  /                -> Serves index.html UI
"""
import json
import os
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR
from src.pipeline import search_single
from src.actions.parser import ActionParser
from src.ingestion.loader import _get_raw_records
from src.retrieval.hybrid import HybridRetriever


class ChronexaRequestHandler(SimpleHTTPRequestHandler):
    """Handles static web assets and dynamic Chronexa search API routes."""

    def __init__(self, *args, **kwargs):
        # Serve static files from the repository root directory
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            resp = {
                "status": "ok",
                "mode": "live_local_engine",
                "corpus_size": 1316,
                "offline": os.environ.get("CHRONEXA_OFFLINE") == "1"
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif parsed.path == "/api/search":
            # Support GET search with query parameters
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", params.get("query", [""]))[0]
            as_of = params.get("as_of", ["2026-09-18T18:00:00-07:00"])[0]
            mode = params.get("mode", ["hybrid"])[0]
            top_k = int(params.get("top_k", [10])[0])

            self._handle_search(query, as_of, mode, top_k)
            return

        # Default static file serving (e.g. index.html)
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if parsed.path == "/api/search":
            query = body.get("query", "")
            as_of = body.get("as_of", "2026-09-18T18:00:00-07:00")
            mode = body.get("mode", "hybrid")
            top_k = int(body.get("top_k", 10))
            self._handle_search(query, as_of, mode, top_k)
            return

        elif parsed.path == "/api/action":
            command = body.get("command", "")
            as_of = body.get("as_of", "2026-09-18T18:00:00-07:00")

            if not command:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing 'command' parameter"}).encode("utf-8"))
                return

            try:
                parser = ActionParser(as_of=as_of)
                actions = parser.parse(command)
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
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def _handle_search(self, query: str, as_of: str, mode: str, top_k: int):
        if not query.strip():
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Query cannot be empty"}).encode("utf-8"))
            return

        try:
            result = search_single(
                query=query,
                as_of=as_of,
                data_dir=DATA_DIR,
                mode=mode,
                generate_answer=True,
                top_k=top_k
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
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def run_server(port: int = 8000):
    print(f"=== Chronexa Local Engine Server ===")
    print(f"Pre-warming embedding cache...")
    raw_units, _, _ = _get_raw_records(DATA_DIR)
    retriever = HybridRetriever()
    if hasattr(retriever, "dense") and retriever.dense is not None:
        retriever.dense.warm_cache(raw_units)
    print(f"Corpus index ready ({len(raw_units)} units).")

    server_address = ("", port)
    httpd = HTTPServer(server_address, ChronexaRequestHandler)
    print(f"\n🚀 Chronexa local API server running at: http://localhost:{port}/")
    print(f"   Dashboard UI: http://localhost:{port}/")
    print(f"   Health Check: http://localhost:{port}/api/health")
    print(f"   Search Route: POST http://localhost:{port}/api/search")
    print(f"   Action Route: POST http://localhost:{port}/api/action")
    print(f"\nPress Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
