#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server on an AMD Instinct GPU using the prebuilt ROCm vLLM image.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

MODEL="${MODEL:-Qwen/Qwen3-8B}"
IMAGE="${VLLM_IMAGE:-rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103}"
PORT="${PORT:-8000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PWD/hf_cache}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.9}"
mkdir -p "$HF_CACHE_DIR"

docker rm -f vllm-rocm >/dev/null 2>&1 || true
docker run -d --name vllm-rocm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --cap-add=SYS_PTRACE \
  -p "${PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL"

echo "Started vllm-rocm. Follow logs with: docker logs -f vllm-rocm"
echo "Ready when this returns JSON:  curl http://localhost:${PORT}/v1/models"
