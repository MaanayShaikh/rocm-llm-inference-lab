# ROCm LLM Inference Lab

A containerized LLM inference and retrieval-augmented generation (RAG) application using vLLM and AMD ROCm.

The project provides an inference serving workflow, a Python benchmarking client, and a document-grounded question-answering application.

## Architecture

```text
                  AMD Instinct GPU
                         |
                  AMD ROCm Runtime
                         |
                  Docker Container
                         |
                    vLLM Server
                  OpenAI-compatible API
                         |
              +----------+----------+
              |                     |
        Benchmark Client        RAG Application
        (Python / aiohttp)      (Python / Gradio)
                                      |
                               Document Ingestion
                                      |
                              Text Chunking
                                      |
                              CPU Embeddings
                                      |
                              Vector Retrieval
                                      |
                              Context Construction
                                      |
                                LLM Generation
                                      |
                              Cited Responses
```

## Features

- Containerized vLLM inference using AMD ROCm.
- AMD GPU device passthrough and runtime configuration.
- OpenAI-compatible inference API.
- Concurrent request benchmarking.
- Streaming token and latency measurements.
- Document ingestion and paragraph-aware chunking.
- Semantic retrieval using sentence-transformers.
- Context-grounded LLM responses with source citations.
- Command-line and Gradio interfaces.
- Automated tests and GitHub Actions CI.

## Technology Stack

| Component | Technology |
|---|---|
| Inference | vLLM |
| GPU Runtime | AMD ROCm |
| Model | Qwen3-8B |
| Containerization | Docker |
| Backend | Python |
| Embeddings | Sentence Transformers |
| Vector Search | NumPy |
| API Client | OpenAI Python SDK |
| Benchmarking | asyncio, aiohttp |
| Interface | Gradio |
| Testing | pytest |
| CI | GitHub Actions |

## Requirements

- Linux host with a compatible AMD Instinct GPU.
- AMD ROCm drivers.
- Docker.
- Python 3.10 or newer.
- Hugging Face access to the selected model.

## Setup

```bash
git clone https://github.com/MaanayShaikh/rocm-llm-inference-lab.git
cd rocm-llm-inference-lab

cp .env.example .env

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Configure the model and ROCm vLLM Docker image in `.env`.

## Run Inference

Start the inference server:

```bash
./scripts/serve.sh
```

Check server logs:

```bash
docker logs -f vllm-rocm
```

Verify the API:

```bash
curl http://localhost:8000/v1/models
```

## Benchmark Inference

Run the concurrency benchmark:

```bash
./scripts/check_gpu.sh
./scripts/run_bench.sh
```

The benchmark client measures:

- Aggregate output tokens per second.
- Time to first token (TTFT).
- Inter-token latency (ITL).
- End-to-end request latency.
- p50 and p95 latency across concurrency levels.

Benchmark output is generated locally under `results/`. Performance depends on the GPU, model, software versions and serving configuration.

## Run the RAG Application

Index the documents:

```bash
python -m rag.ingest
```

Ask a question through the CLI:

```bash
python -m rag.ask "How do I configure GPU access?"
```

Launch the Gradio interface:

```bash
python -m rag.app
```

The application retrieves relevant document chunks, constructs a context-aware prompt, and generates an answer with source references.

Additional Markdown and text documents can be placed in `data/docs/`.

## Testing

Run the test suite:

```bash
pytest -q
```

Run the benchmark client against the mock server:

```bash
python tests/mock_server.py
```

In another terminal:

```bash
python scripts/bench_serving.py \
  --model mock \
  --concurrency 1 4 \
  --rounds 2
```

The mock server validates the benchmark client, not real GPU performance.

## Project Structure

```text
rocm-llm-inference-lab/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── docs/
├── rag/
│   ├── app.py
│   ├── ask.py
│   ├── chunking.py
│   ├── engine.py
│   └── ingest.py
├── scripts/
│   ├── bench_serving.py
│   ├── check_gpu.sh
│   ├── run_bench.sh
│   └── serve.sh
├── tests/
│   ├── mock_server.py
│   └── test_chunking.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## License

MIT
