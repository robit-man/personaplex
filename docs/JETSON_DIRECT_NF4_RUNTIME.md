# Jetson Direct NF4 Runtime

`cudabenchmarktest/personaplex-7b-nf4` is executed from its packed `model-nf4.safetensors` artifact. The deployment scripts do not create or accept a `model-bf16.safetensors` cache.

The runtime is vendored at `personaplex-setup/moshi`, includes the reviewed Moshirag streaming-sum patch, and is installed into the Jetson virtual environment during setup. `personaplex_nf4` intercepts the Moshi loader before the server starts, keeps 4-bit weight bytes plus group scales resident on CUDA, and supplies CUDA kernels for linear projections and embeddings. CPU offload and CPU inference intentionally fail.

## Deploy

```bash
./scripts/setup_jetson_nf4.sh
PERSONAPLEX_NF4_DTYPE=fp16 ./scripts/deploy_nf4_cloudflared.sh
```

`fp16` is the Jetson default. Set `PERSONAPLEX_NF4_DTYPE=bf16` only for an explicit quality comparison. `PERSONAPLEX_EXTRA_ARGS=--half` is not used because the upstream server does not expose that flag; the direct loader owns activation dtype selection.

Setup builds the kernel once with `TORCH_CUDA_ARCH_LIST=8.7` and `MAX_JOBS=1`, retaining the result under `.cache/personaplex/nf4-kernel`. At server start the checkpoint header, direct runtime, and CUDA availability are checked before the port is opened.

## Invariants

- `start_nf4_server.sh` fails when the vendored runtime is missing.
- A packed-NF4 checkpoint is required and validated by metadata.
- No script calls `dequant_nf4_to_bf16.py` or accepts a BF16 cache.
- All 4-bit weights and scales remain on the CUDA device; only tiny safetensors shape metadata crosses host memory during load.
- The custom attention path preserves Moshi's streaming cache, rotary positions, step-dependent projection weights, and full-duplex server loop.

This removes the silent BF16 fallback. Performance tuning is a separate step: establish a baseline with the direct kernel, fixed Jetson clocks, and one active stream before evaluating CUDA graph capture or a fused Tensor Core replacement.
