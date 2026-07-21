# PersonaPlex Semantic Control v4

Status: authoritative redesign and execution source of truth.

This directory defines the next PersonaPlex control implementation. It replaces
the static semantic-prefix experiment as the promotion path. The completed
control-v3 run remains a useful frozen baseline, but it is not a semantically
controllable release candidate.

## Read order

1. `RELIABILITY_CONTRACT.md` defines what "95% reliable" means and what cannot
   be averaged away.
2. `ARCHITECTURE.md` defines the model, protocol, scheduler, and cancellation
   path.
3. `DATASET_CONTRACT.md` defines causal paired data, the synthesis cascade, and
   admission requirements.
4. `TRAINING_CONTRACT.md` defines staged optimization and checkpoint selection.
5. `EVALUATION_CONTRACT.md` defines teacher-forced, generated-audio, duplex, and
   live-equivalent tests.
6. `RESEARCH_SYNTHESIS.md` records the primary-source basis and the limits of
   each source.
7. `TRACEABILITY.md` maps requirements to implementation and evidence.
8. `TODO.md` is the execution ledger. A checked item needs an artifact path or a
   test result, not an assertion.
9. `EMPIRICAL_FINDINGS.md` records local failures, root fixes, measured resource
   behavior, and the current boundary between implementation and validation.

## Baseline finding

The control-v3 adapter completed 12,000 updates over 3,593 certified native
targets. Its held-out control loss improved and shuffled-plan sensitivity became
positive, but ablation of mutable text context changed loss by only about
`0.0009`. That is evidence of coarse control-token recognition, not reliable
fact, tool-result, correction, or revision following.

The root causes are architectural and objective-level:

- The adapter learns a fresh random 32k by 4096 token embedding from a small
  corpus instead of reusing PersonaPlex's pretrained text embedding.
- Most free text is atomized with underscores, damaging ordinary lexical
  semantics before tokenization.
- A fixed prefix is prefetched into the temporal cache, while MoshiRAG's
  evidence-backed design injects compressed semantic representations alongside
  real temporal input frames.
- Training uses only matched teacher-forced likelihood. It never requires the
  same audible context to choose differently under two valid control revisions.
- The corpus contains causal pair lineage, but the trainer ignores it.
- Checkpoint evaluation does not generate speech, decode audio, transcribe it,
  or judge whether the active frame actually changed the spoken decision.

## Existing causal asset

The certified V8 source corpus contains 439 complete two-branch groups. Every
complete group uses a replayed pre-pivot prefix with identical utterance audio
hashes, transcript text, and timing across branches. Quality quarantine leaves
396 usable causal pivot pairs in the native manifest: 313 train, 46 validation,
and 37 test, with group-isolated splits.

This material is sufficient for a v4 smoke and ablation run. The next synthesis
batch expands held-out axes and difficult interactions; it does not discard the
already certified corpus.

## Chosen architecture

The v4 condition is a field-aware, learned temporal control stream:

```text
typed control frame
  -> deterministic field segmentation
  -> frozen PersonaPlex text embeddings
  + field/type/source/revision embeddings
  -> compact trainable control encoder
  -> compressed T x 4096 gated control stream
  -> streaming-sum injection on native 12.5 Hz temporal frames
  -> PersonaPlex text and speech-token generation
```

This is a trained input to speech-token generation. It is not metadata, a
sidecar prompt, or an external LLM writing each spoken line.

## Completion rule

Completion requires a provenance-bound candidate that passes the reliability
contract on generated audio and a live-equivalent full-duplex path. A protocol
acknowledgement, low teacher-forced loss, or one convincing sample is not
completion. If the 95% gate is not met, the run remains open and failure slices
drive another data/training iteration.
