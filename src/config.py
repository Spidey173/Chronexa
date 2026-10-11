"""Global configuration and path management."""
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EVALS_DIR = ROOT_DIR / "evals"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Ensure output directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
