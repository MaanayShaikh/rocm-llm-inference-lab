# Resume mapping (be transparent about test status)

The repository currently contains a benchmark harness and a RAG application, but no committed real-GPU benchmark output. Until you run it on AMD Instinct hardware, use a project bullet that describes the implementation rather than claiming measured performance:

> Built a reproducible vLLM + AMD ROCm inference workflow with Docker, a Python concurrency benchmark harness for throughput and latency, and a document-grounded RAG demo; documented an MI300X-class performance estimate as a hypothesis pending hardware validation.

After a real run, replace the estimate with the measured `output_tok_per_s` at the stated concurrency, and commit the generated CSV/Markdown plus `results/environment.txt`. Do not say the project was hardware-benchmarked unless that run actually happened.
