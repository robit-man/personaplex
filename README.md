# PersonaPlex semantic-control research fork

This repository is a control-plane and evaluation companion for
[NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex). It is not a
standalone PersonaPlex runtime and it does not claim that an external LLM can
change audio that PersonaPlex has already generated.

## Current status

The repository previously described a `--hybrid` server mode and a full
modified PersonaPlex runtime. Those files are not present in this repository.
The historical "hybrid" helper sends a PersonaPlex sentence to Ollama and
returns an overlay text response; it does not feed that response back to the
speech model. It must not be used to make semantic-control or spoken-content
claims.

The historical 3,000-sample distillation artifact records teacher-token loss.
It is a BF16 PyTorch state dict, not an NF4 inference artifact, and has not
been shown to improve grounding, role adherence, ASR accuracy, voice quality,
or full-duplex latency. It is therefore experimental only.

## Architecture

Production use has three independent planes:

1. **Audio plane**: PSTN media is bridged to PersonaPlex for low-latency,
   natural full-duplex audio.
2. **Semantic plane**: streaming caller ASR, an authoritative text LLM,
   tool results, and per-call state generate a versioned semantic plan.
3. **Render and arbitration plane**: a controller applies a plan only at a
   confirmed caller turn boundary. Strict requests use deterministic streaming
   TTS for canonical wording. Expressive requests may guide PersonaPlex, but
   are ASR-checked and fall back to strict rendering on semantic drift.

`personaplex_control` contains the dependency-free wire contracts and turn
state machine used by the server adapter. See:

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Evaluation](docs/EVALUATION.md)
- [Hugging Face card replacements](docs/huggingface)

## Bootstrap an upstream runtime

Use a pinned NVIDIA PersonaPlex revision and retain its model-license terms.
Do not assume this repository contains the upstream server. The server adapter
must be installed into that pinned runtime and tested against the actual
WebSocket protocol before it is deployed.

```bash
git clone https://github.com/NVIDIA/personaplex.git upstream-personaplex
git -C upstream-personaplex checkout <reviewed-upstream-revision>
```

The extension must implement the `control.update` / `control.ack` contract in
`docs/ARCHITECTURE.md`; a command-line `--hybrid` flag alone is not a semantic
control implementation.

## Model releases

No new model weights are released by this revision. A weight release requires:

- licensed and consented multi-channel conversational audio;
- reproducible data manifests and base-model revision;
- an adapter or fine-tune that conditions on the semantic plan before audio
  generation;
- ASR-grounded semantic, safety, latency, and full-duplex evaluations;
- a model card that reports failures as well as aggregate metrics.

See [Training](docs/TRAINING.md) for the required pipeline.

## License

Code added in this repository is MIT unless a file says otherwise. PersonaPlex
code, model weights, voice assets, and derivative releases remain subject to
their respective upstream licenses and consent requirements.
