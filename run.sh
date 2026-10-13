#!/usr/bin/env bash
# Chronexa Take-Home Execution Script
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

# 1. Setup Virtualenv if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
fi

PYTHON=".venv/bin/python"
export PYTHONPATH="eval_harness:."

# 2. Check environment variables
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Notice: .env not found. Copying .env.example -> .env"
    cp .env.example .env
fi

MODE="${1:-all}"

case "$MODE" in
    memory)
        echo "=== Running Chronexa Memory Pipeline ==="
        $PYTHON src/pipeline.py --mode hybrid --output outputs/memory_train_answers.jsonl
        echo "=== Scoring Retrieval ==="
        python3 eval_harness/score_retrieval.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl
        echo "=== Scoring Memory Answers ==="
        python3 eval_harness/score_memory.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl --judge none
        ;;
    actions)
        echo "=== Running Chronexa Action Assistant ==="
        $PYTHON src/actions/runner.py --commands evals/actions_train.jsonl --output outputs/actions_train_predictions.jsonl
        echo "=== Scoring Actions ==="
        python3 eval_harness/score_actions.py --gold evals/actions_train.jsonl --predictions outputs/actions_train_predictions.jsonl
        ;;
    all|*)
        echo "=== Running Chronexa Memory Pipeline ==="
        $PYTHON src/pipeline.py --mode hybrid --output outputs/memory_train_answers.jsonl
        echo ""
        echo "=== Running Chronexa Action Assistant ==="
        $PYTHON src/actions/runner.py --commands evals/actions_train.jsonl --output outputs/actions_train_predictions.jsonl
        echo ""
        echo "=========================================="
        echo "           EVALUATION RESULTS             "
        echo "=========================================="
        echo "--- 1. Retrieval Quality ---"
        python3 eval_harness/score_retrieval.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl --quiet
        echo ""
        echo "--- 2. Memory Answers (Deterministic) ---"
        python3 eval_harness/score_memory.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl --judge none
        echo ""
        echo "--- 3. Action Assistant (Bonus) ---"
        python3 eval_harness/score_actions.py --gold evals/actions_train.jsonl --predictions outputs/actions_train_predictions.jsonl
        ;;
esac
