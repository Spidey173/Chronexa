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
    test)
        echo "=== Running Chronexa Automated Test Suite ==="
        $PYTHON -m unittest discover tests -v
        ;;
    eval)
        echo "=== Running Chronexa Benchmark Evaluation (Dev + Holdout) ==="
        $PYTHON eval/run_eval.py --split both --offline --config "${2:-hybrid_rrf_rerank}"
        ;;
    eval-dev)
        echo "=== Running Chronexa Benchmark Evaluation (Dev Split) ==="
        $PYTHON eval/run_eval.py --split dev --offline --config "${2:-hybrid_rrf_rerank}"
        ;;
    eval-holdout)
        echo "=== Running Chronexa Benchmark Evaluation (Holdout Split) ==="
        $PYTHON eval/run_eval.py --split holdout --offline --config "${2:-hybrid_rrf_rerank}"
        ;;
    memory)
        echo "=== Running Chronexa Memory Pipeline ==="
        $PYTHON src/pipeline.py --mode hybrid --output outputs/memory_train_answers.jsonl
        echo "=== Scoring Retrieval ==="
        $PYTHON eval_harness/score_retrieval.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl
        echo "=== Scoring Memory Answers ==="
        $PYTHON eval_harness/score_memory.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl --judge none
        ;;
    actions)
        echo "=== Running Chronexa Action Assistant ==="
        $PYTHON src/actions/runner.py --commands evals/actions_train.jsonl --output outputs/actions_train_predictions.jsonl
        echo "=== Scoring Actions ==="
        $PYTHON eval_harness/score_actions.py --gold evals/actions_train.jsonl --predictions outputs/actions_train_predictions.jsonl
        ;;
    all|*)
        echo "=== Running Chronexa Automated Test Suite ==="
        $PYTHON -m unittest discover tests
        echo ""
        echo "=== Running Chronexa Memory Pipeline ==="
        $PYTHON src/pipeline.py --mode hybrid --output outputs/memory_train_answers.jsonl
        echo ""
        echo "=== Running Chronexa Action Assistant ==="
        $PYTHON src/actions/runner.py --commands evals/actions_train.jsonl --output outputs/actions_train_predictions.jsonl
        echo ""
        echo "=========================================="
        echo "           EVALUATION RESULTS             "
        echo "=========================================="
        echo "--- 1. Retrieval Quality (Train Set) ---"
        $PYTHON eval_harness/score_retrieval.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl --quiet
        echo ""
        echo "--- 2. Memory Answers (Deterministic Match) ---"
        $PYTHON eval_harness/score_memory.py --gold evals/memory_train.jsonl --answers outputs/memory_train_answers.jsonl --judge none
        echo ""
        echo "--- 3. Action Assistant (Strict Schema Validation) ---"
        $PYTHON eval_harness/score_actions.py --gold evals/actions_train.jsonl --predictions outputs/actions_train_predictions.jsonl
        ;;
esac
