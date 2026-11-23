#!/usr/bin/env bash
set -euo pipefail

# scripts/deploy_backend.sh
# Reproducible deployment script for Chronexa FastAPI Backend to Vercel (chronexa-api)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${ROOT_DIR}/.deploy_tmp"

echo "=== Deploying Chronexa Backend to Vercel (chronexa-api) ==="

# Check Vercel CLI
if ! command -v vercel &> /dev/null; then
  echo "Error: Vercel CLI is not installed. Install via: npm i -g vercel"
  exit 1
fi

# Ensure clean deploy directory
rm -rf "${DEPLOY_DIR}"
mkdir -p "${DEPLOY_DIR}"

cleanup() {
  echo "Cleaning up temporary deployment directory..."
  rm -rf "${DEPLOY_DIR}"
}
trap cleanup EXIT

echo "1. Staging source and application files..."
cp -R "${ROOT_DIR}/src" "${DEPLOY_DIR}/src"
cp -R "${ROOT_DIR}/api" "${DEPLOY_DIR}/api"
cp -R "${ROOT_DIR}/eval_harness" "${DEPLOY_DIR}/eval_harness"

echo "2. Staging corpus and prebuilt FAISS embeddings..."
mkdir -p "${DEPLOY_DIR}/data"
cp -R "${ROOT_DIR}/data/connectors" "${DEPLOY_DIR}/data/connectors"
cp -R "${ROOT_DIR}/data/native" "${DEPLOY_DIR}/data/native"
cp -R "${ROOT_DIR}/data/embeddings" "${DEPLOY_DIR}/data/embeddings"

echo "3. Staging FastEmbed ONNX model caches (if present)..."
if [ -d "${ROOT_DIR}/fastembed_cache" ]; then
  cp -R "${ROOT_DIR}/fastembed_cache" "${DEPLOY_DIR}/fastembed_cache"
elif [ -d "${ROOT_DIR}/.cache/fastembed" ]; then
  mkdir -p "${DEPLOY_DIR}/fastembed_cache"
  cp -R "${ROOT_DIR}/.cache/fastembed/"* "${DEPLOY_DIR}/fastembed_cache/"
fi

echo "4. Staging configuration and manifests..."
cp "${ROOT_DIR}/requirements.txt" "${DEPLOY_DIR}/requirements.txt"

# Create serverless vercel.json for chronexa-api
cat << 'EOF' > "${DEPLOY_DIR}/vercel.json"
{
  "version": 2,
  "framework": null,
  "installCommand": "pip install -r requirements.txt",
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python",
      "config": {
        "maxDuration": 60,
        "memory": 1024
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
EOF

cat << 'EOF' > "${DEPLOY_DIR}/.vercelignore"
.venv
__pycache__
*.pyc
.git
.env
.env.*
tests/
eval/
evals/
outputs/
EOF

echo "5. Linking project chronexa-api..."
(cd "${DEPLOY_DIR}" && vercel link --project chronexa-api --yes)

echo "6. Running production deployment..."
(cd "${DEPLOY_DIR}" && vercel --prod --yes)

echo "=== Deployment to chronexa-api successfully finished! ==="
