# Parallel synthesis topology

`/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env` is the only local source of endpoint, GPU, and batch-routing truth. Version 2 defines three isolated synthesis lanes:

| Lane | GPU | Ornith control | ChatML proxy | Chatterbox Turbo |
| --- | --- | --- | --- | --- |
| 0 | 0 | 12085 | 12086 | 17503 |
| 1 | 1 | 12083 | 12084 | 17504 |
| 2 | 2 | 12087 | 12088 | 17502 |

Each lane uses the same `robit/ornith:35b` model at the configured context length and a local CUDA-only Voicebox Chatterbox Turbo service. The producer and certifier select their semantic proxy by lane. The producer also selects its Voicebox endpoint by lane. No endpoint is inferred from a default port.

The `personaplex-runtime-status.sh` tool must report all six health bindings before a synthesis run is treated as ready. The batch and ASR gates are also versioned in the same runtime contract, so throughput adjustments cannot silently change data-admission policy.
