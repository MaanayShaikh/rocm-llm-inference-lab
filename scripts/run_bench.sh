#!/usr/bin/env bash
# Sweep concurrency levels and write results/*.md and results/*.csv
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
python scripts/bench_serving.py \
  --base-url "http://localhost:${PORT:-8000}" \
  --model "${MODEL:-Qwen/Qwen3-8B}" \
  --concurrency 1 4 8 16 32 \
  --rounds 4 --input-words 256 --output-tokens 256 \
  --out "results/bench_$(date +%Y%m%d_%H%M)"
