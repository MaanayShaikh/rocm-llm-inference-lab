# Benchmark artifacts

Commit generated `bench_*.md` and `bench_*.csv` here only when they are produced by running the harness against the actual inference server. Capture the environment with `scripts/check_gpu.sh` and save it as `environment.txt`.

The README includes an explicitly labeled illustrative estimate for planning. It is not a measured result and must not be copied into benchmark artifacts or presented as validated GPU performance. CI's mock-server run checks harness behavior only.
