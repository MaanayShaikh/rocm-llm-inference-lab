# 7-minute walkthrough video outline

1. (0:00) Goal: what this repo does and who it helps (30s)
2. (0:30) Show the hardware: run `check_gpu.sh`, point out GPU, ROCm, vLLM version (45s)
3. (1:15) Explain `serve.sh`: why each docker flag exists (/dev/kfd, /dev/dri, shm, video group) (90s)
4. (2:45) Start the server, show logs, curl `/v1/models` (45s)
5. (3:30) Run the benchmark sweep, explain TTFT vs throughput vs tail latency (90s)
6. (5:00) Show your results table and where throughput flattens (60s)
7. (6:00) RAG demo: ask 2 questions, show citations, one "I don't know" case (60s)
8. (7:00) What you would tune next, and what you learned
