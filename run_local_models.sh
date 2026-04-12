#!/bin/bash
# Run v3 pipeline with local Ollama models sequentially
# Usage: ./run_local_models.sh
set -e

cd "$(dirname "$0")"

export LANGFUSE_ENABLED=false
export LLM_MAX_TOKENS=8192

echo "=========================================="
echo "  Starting local model benchmark runs"
echo "=========================================="

# --- Run 1: gemma4:26b ---
echo ""
echo "[$(date)] Starting gemma4:26b run..."
python3 -m src.main \
  --dataset industryOR \
  --llm-provider ollama \
  --llm-model gemma4:26b \
  --output results_v3_gemma4 \
  --parallel-problems 1 \
  --sequential \
  --log-level INFO \
  2>&1 | tee results_v3_gemma4_run.log

echo "[$(date)] gemma4:26b run complete."
echo ""

# --- Run 2: llama3:8b ---
echo "[$(date)] Starting llama3:8b run..."
python3 -m src.main \
  --dataset industryOR \
  --llm-provider ollama \
  --llm-model llama3:8b \
  --output results_v3_llama3 \
  --parallel-problems 1 \
  --sequential \
  --log-level INFO \
  2>&1 | tee results_v3_llama3_run.log

echo "[$(date)] llama3:8b run complete."
echo ""
echo "=========================================="
echo "  All runs complete!"
echo "=========================================="
