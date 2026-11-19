# Deploying vLLM on AMD Instinct GPUs with ROCm

AMD publishes a prebuilt ROCm vLLM Docker image for validating LLM inference on AMD Instinct data center GPUs. Using the prebuilt image avoids building vLLM, PyTorch and ROCm libraries from source.

## Container flags

The container needs access to the GPU device nodes. Pass `--device=/dev/kfd` and `--device=/dev/dri` and add the `video` group with `--group-add video`. Shared memory should be raised, for example `--shm-size 16G`. The AMD documentation also uses `--cap-add=SYS_PTRACE` and relaxes seccomp and AppArmor with `--security-opt seccomp=unconfined` and `--security-opt apparmor=unconfined`.

## Serving a model

Inside the container, `vllm serve <model>` starts an OpenAI-compatible HTTP server. Clients can call `/v1/models` to check readiness and `/v1/chat/completions` to generate text. Useful options include `--max-model-len` to bound context length and `--gpu-memory-utilization` to control how much GPU memory vLLM may use for weights and KV cache.

## Gated models

Some Hugging Face models, such as Llama, require accepting a license and passing a Hugging Face token to the container as the `HF_TOKEN` environment variable. Open-weight models that are not gated work without a token.
