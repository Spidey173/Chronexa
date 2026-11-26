"""Chronexa Production FastAPI Server.

Serves endpoints:
- /health: System health and diagnostic status
- /query: Point-in-time temporal search and answer generation
- /actions/propose: Dry-run schema validation and proposal persistence
- /actions/{id}/approve: Operator approval & background execution decoupling
- /actions/{id}/reject: Proposal rejection
- /actions/{id}: Proposal details and execution status
"""
import argparse
import uvicorn
from src.config import settings


def run_server(host: str = None, port: int = None):
    h = host or settings.HOST
    p = port or settings.PORT
    print(f"\n🚀 Chronexa Production FastAPI Server running at: http://{h}:{p}/")
    print(f"📖 OpenAPI Swagger documentation available at: http://{h}:{p}/docs\n")
    uvicorn.run("src.api.app:app", host=h, port=p, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronexa Production API Server")
    parser.add_argument("--host", default=settings.HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind to")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
