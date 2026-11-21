# Chronexa Containerized Production Service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/eval_harness:/app \
    CHRONEXA_OFFLINE=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and application directory
RUN useradd -m -u 1000 -s /bin/bash chronexa
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download FastEmbed ONNX models at build time so there is zero cold-start delay
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); \
print('FastEmbed ONNX models pre-downloaded successfully.')"

# Copy application source code and data assets
COPY . .

# Set proper ownership for non-root user
RUN chown -R chronexa:chronexa /app /home/chronexa

# Switch to non-root user
USER chronexa

# Expose default HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production uvicorn server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
