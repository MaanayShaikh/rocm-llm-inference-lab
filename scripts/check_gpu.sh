#!/usr/bin/env bash
# Record the hardware/software environment for the results table.
set -u
echo "== rocm-smi =="; rocm-smi --showproductname --showmeminfo vram 2>/dev/null || echo "rocm-smi not found"
echo; echo "== kernel =="; uname -r
echo; echo "== docker image in use =="; docker inspect vllm-rocm --format '{{.Config.Image}}' 2>/dev/null || echo "container not running"
echo; echo "== vllm version in container =="; docker exec vllm-rocm vllm --version 2>/dev/null || echo "n/a"
