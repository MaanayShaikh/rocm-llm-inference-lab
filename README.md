# ROCm LLM Inference Lab

[![CI](https://github.com/maanayshaikh/rocm-llm-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/maanayshaikh/rocm-llm-inference-lab/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Serve an open LLM with **vLLM on AMD Instinct GPUs (ROCm)** in Docker, **benchmark it under load**, and put a **RAG demo** on top. Everything is reproducible with a few commands.

**Contents:** [What this shows](#what-this-shows) | [Architecture](#architecture) | [Quick start](#quick-start) | [Results](#results) | [Testing without a GPU](#testing-without-a-gpu) | [Troubleshooting](#troubleshooting)

## Why this exists

Teams evaluating AI hardware need to see a model running, with honest numbers, quickly. This repo is a small, reproducible reference for that path: pull the prebuilt ROCm vLLM container, serve a model, measure throughput and latency at increasing load, and build an application on top. Each step is a script you can read in a few minutes.

> **Benchmark status:** This repository contains a benchmark harness, but no real-GPU result files are currently committed. The example figures below are explicitly modeled estimates for planning only, not measurements.

## What this shows

| Skill | Where |
|---|---|
| GPU enablement in containers (`/dev/kfd`, `/dev/dri`, ROCm image) | `scripts/serve.sh` |
| LLM inference serving with vLLM (OpenAI-compatible API) | `scripts/serve.sh` |
| Performance measurement: tokens/sec, TTFT, ITL, tail latency | `scripts/bench_serving.py` |
| Application layer: RAG over local docs with citations | `rag/` |
| Linux, Docker, shell, Python, Git, tests | whole repo |

## Architecture

```
 docs (md/txt) --> chunk --> embeddings (CPU) --> vectors.npy
                                                        |
 user question --> embed --> top-k chunks --> prompt --> vLLM server (ROCm, AMD Instinct GPU)
                                                        |
                                              answer with [source:chunk] citations
```

Embeddings run on CPU so the GPU is dedicated to generation.

## Requirements

- Linux host with an **AMD Instinct GPU** and ROCm drivers installed (the prebuilt ROCm vLLM image targets Instinct MI300X-series and newer; check the ROCm vLLM docs for your GPU)
- Docker, Python 3.10+
- ~20 GB disk for the model cache

## Quick start

```bash
git clone https://github.com/maanayshaikh/rocm-llm-inference-lab.git && cd rocm-llm-inference-lab
cp .env.example .env          # check VLLM_IMAGE against the latest tag in the ROCm vLLM docs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./scripts/serve.sh            # starts the vLLM container
docker logs -f vllm-rocm      # wait for "Application startup complete"
curl localhost:8000/v1/models # confirms the server is ready

./scripts/check_gpu.sh | tee results/environment.txt   # record GPU/ROCm/vLLM versions
./scripts/run_bench.sh                                 # concurrency sweep -> results/*.md

python -m rag.ingest                          # index data/docs
python -m rag.ask "How do I give the container GPU access?"
python -m rag.app                             # Gradio UI on :7860
```

Add your own `.md`/`.txt` files (for example AMD public documentation you have saved) to `data/docs/` and re-run `python -m rag.ingest`.

## Results

**Illustrative estimate only — not measured.** Planning scenario: one AMD Instinct MI300X-class GPU, Qwen3-8B, vLLM/ROCm, approximately 256 input words and up to 256 output tokens. Actual results depend on software versions, serving configuration, input/output lengths, and GPU setup. Do not cite these ranges as benchmark results.

| Field | Value |
|---|---|
| GPU | Assumed 1× AMD Instinct MI300X-class GPU (estimate only) |
| ROCm / vLLM / image tag | Not captured; populate from `scripts/check_gpu.sh` after a real run |
| Model | `Qwen/Qwen3-8B` (default) |
| Input / output | ~256 words in / 256 tokens out |

| Concurrency | Output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | ITL mean (ms) | E2E p95 (s) |
|---|---|---|---|---|---|
| 1 | ~80–180 | ~250–900 | ~400–1,500 | ~15–35 | ~2–8 |
| 4 | ~250–600 | ~400–1,400 | ~700–2,500 | ~18–45 | ~3–12 |
| 8 | ~400–900 | ~600–2,000 | ~1,000–3,500 | ~22–60 | ~4–18 |
| 16 | ~500–1,200 | ~900–3,000 | ~1,500–5,000 | ~30–90 | ~6–28 |
| 32 | ~550–1,400 | ~1,500–5,000 | ~2,500–8,000 | ~45–140 | ~10–45 |

**Expected trend (hypothesis):** Aggregate throughput should rise with concurrency and then flatten near saturation; p95 latency should increase as requests queue. These are assumptions to test, not findings from this repository.

## Testing without a GPU

```bash
python tests/mock_server.py &                       # fake streaming server on :8000
python scripts/bench_serving.py --model mock --concurrency 1 4 --rounds 2
pytest -q
```

This verifies the benchmark harness logic only. It does not validate the estimated ranges or establish real GPU performance.

## Troubleshooting

- **Container exits immediately:** run `docker logs vllm-rocm`. If the image's entrypoint differs from what `serve.sh` expects, start it interactively (`docker run -it ... bash`, same flags) and run `vllm serve` inside, as the AMD docs do.
- **Permission denied on /dev/kfd:** your user needs access to the `video`/`render` groups on the host.
- **Out of memory:** lower `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION` in `.env`, or pick a smaller model.
- **Gated model 401:** set `HF_TOKEN` in `.env` and accept the model license on Hugging Face.
- **Image tag not found:** tags change; copy the current one from the ROCm vLLM "inference performance testing" docs.

## Ideas to extend

- Compare FP8 vs default precision, or tensor parallelism across GPUs
- Add KServe or Kubernetes manifests with GPU scheduling
- Swap in a different model family and compare

## Repository layout

```
scripts/   serve.sh, run_bench.sh, check_gpu.sh, bench_serving.py
rag/       chunking, embedding + retrieval engine, CLI, Gradio app
data/docs/ sample documents to index
results/   benchmark output (measured runs only)
tests/     unit tests + mock streaming server
docs/      video outline, resume mapping
```

## License

MIT. See [LICENSE](LICENSE).
