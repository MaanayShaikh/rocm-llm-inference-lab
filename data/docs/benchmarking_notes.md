# Benchmarking LLM serving

Serving benchmarks report different things than offline throughput tests. For an online server the main measurements are output tokens per second, time to first token (TTFT), inter-token latency (ITL) and end-to-end latency.

## Concurrency

Throughput usually rises as concurrency increases because vLLM batches requests continuously, but latency also rises once the GPU is saturated. Sweeping concurrency, for example 1, 4, 8, 16 and 32 simultaneous requests, shows where the throughput curve flattens and tail latency grows.

## Fair measurements

Use unique prompts so prefix caching does not inflate results, run a warm-up request before timing so graph capture and compilation are excluded, and record the GPU model, ROCm version, vLLM version and Docker image tag next to every result. AMD also recommends validating system settings, such as NUMA auto-balancing, before benchmarking.
