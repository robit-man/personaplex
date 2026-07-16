# Semantic-Control Convergence Plan

Status: execution baseline. This plan governs the Nemotron semantic plane,
PersonaPlex audio plane, synthetic corpus, native training, and promotion gates.
It supersedes any implication that a prompt, a webhook, a text renderer, or an
untrained metadata field constitutes semantic control.

## 1. Target system

```text
Twilio caller media
  -> codec bridge + streaming ASR + turn observations
  -> per-call immutable event log and state tree
  -> custom Nemotron control compiler + tools/policy agents
  -> control.update revision N
  -> cached learned PersonaPlex control prefix at next agent boundary
  -> full-duplex PersonaPlex generation
  -> Twilio outbound media, generation-tagged and cancellable

late tool/reference result
  -> evidence.update revision N+1
  -> invalidate queued stale generation
  -> next agent boundary, or future trained semantic boundary only
```

The Nemotron service is the decision-maker. PersonaPlex is the low-latency
conversational body. The strict renderer is a separate route for obligations
that require exact wording. No component may pretend that expressive
speech-to-speech generation is verbatim deterministic.

## 2. Non-negotiable invariants

1. The audio plane receives only typed, bounded, versioned state. It never
   receives target response text, a natural-language system prompt, or an
   unbounded raw tool payload.
2. `control_revision`, `context_hash`, generation snapshot, adapter revision,
   and cancellation token are immutable for one generation.
3. A new revision invalidates unsent audio. It never rewrites audio already sent
   to Twilio.
4. Policy-sensitive speech requires a current acknowledged control frame. When
   no valid frame is available, the audio model waits or uses only an explicitly
   permitted safe backchannel.
5. Semantic correctness and terminal intent are adjudicated by independent model
   inference, not regexes, phrase lists, or template matching. Structural checks
   may measure codec integrity, timing, hash consistency, and schema validity.
6. All model inference and training runs on CUDA devices `0`, `1`, or `2` only.
   GPU `3` is excluded. CPU may only perform ordinary file and process work.
7. All reference voices require recorded consent or a compatible public-domain
   provenance record. Audio, state, and tool evidence are sensitive artifacts.

## 3. Control protocol

### 3.1 `control.update`

The Nemotron state compiler issues a compact control frame before each
policy-sensitive agent turn:

```json
{
  "type": "control.update",
  "protocolVersion": 2,
  "callId": "CA...",
  "revision": 42,
  "effectiveFrom": "next_agent_turn",
  "contextHash": "sha256:...",
  "expiresAtUnixMs": 0,
  "frame": {
    "intent": "resolve_delivery_issue",
    "knownFacts": ["replacement shipped", "carrier scan pending"],
    "callerPosture": "skeptical",
    "nextGoal": "acknowledge delay and offer escalation options",
    "constraints": ["do_not_invent_delivery_date", "do_not_repeat_greeting"],
    "toolResultRefs": ["shipment:replacement-queued"],
    "style": {"warmth": 0.7, "assertiveness": 0.35, "brevity": 0.55},
    "endCallAuthorized": false
  }
}
```

The frame must be serialized deterministically and encoded once on GPU into a
small learned control representation. The frozen-base initial adapter injects
virtual prefix embeddings or selected per-layer K/V prefixes through a learned
gate. The scheduler snapshots this cached representation at the next agent
boundary; it does not recompute it per media packet.

### 3.2 `evidence.update`

`evidence.update` is a distinct contract for a late tool or reference result:

```json
{
  "type": "evidence.update",
  "protocolVersion": 2,
  "callId": "CA...",
  "revision": 43,
  "supportsControlRevision": 42,
  "contextHash": "sha256:...",
  "expiresAtUnixMs": 0,
  "evidenceId": "shipment:carrier-scan",
  "provenance": {"sourceKind": "tool", "sourceRef": "shipment:carrier-scan"},
  "allowedClaims": ["carrier has not recorded an acceptance scan"],
  "availability": "ready"
}
```

The initial release must validate, persist, acknowledge, expire, and cancel by
this contract but keep learned evidence injection disabled until the encoder has
passed native CUDA evaluation. A late result cannot elevate permission beyond the
next valid `control.update` and cannot mutate an active committed utterance.

The evidence-encoder baseline is the locally inspected MoshiRAG conditioner:
frozen ARC4 text compression, followed by a trainable `3072 -> 2048 -> 4096`
bridge and a streaming-sum fuser at 12.5 Hz. Apply 20% evidence dropout during
training. The initial Voryn scheduler injects this representation only after a
fresh next-turn snapshot; a later semantic-boundary mode requires its own
certified timing and policy evaluation.

### 3.3 Required acknowledgements

Every update has one terminal status: `queued`, `applied`, `superseded`,
`rejected`, `expired`, `context_mismatch`, `prefix_build_failed`,
`evidence_deferred`, `evidence_applied`, or `safe_fallback`. An `applied` status
includes the call, revision, context hash, generation id, frame/evidence hash,
adapter version, and measured GPU encode/prefill latency.

## 4. MoshiRAG reconciliation

MoshiRAG is the direct engineering reference for the evidence lane: it uses a
full-duplex Moshi front end, a separate streaming ASR, an asynchronous text
backend, and a learned reference conditioner whose representations are injected
into the model stream. It demonstrates that delayed semantic information can be
causal without forcing a conventional ASR-LLM-TTS call loop.

Voryn adopts its separation, but not its weaker application semantics:

| MoshiRAG mechanism | Voryn implementation rule |
| --- | --- |
| model emits retrieval trigger | Nemotron/tool policy issues an explicit typed revision |
| backend returns free reference text | evidence has provenance, allowed claims, TTL, and a supporting control revision |
| reference may alter later response content | policy-sensitive effect waits for a trained semantic boundary or next turn |
| pre-RAG filler maintains flow | no generic filler policy; any backchannel must be trained, current, cancellable, and permitted |
| retrieval may swap backend | Nemotron and tools remain independently upgradeable behind the typed contract |

Official reference: [MoshiRAG paper](https://arxiv.org/html/2604.12928),
[MoshiRAG implementation](https://github.com/kyutai-labs/moshi-rag), and
[MoshiRAG BF16 weights](https://huggingface.co/kyutai/moshika-rag-pytorch-bf16).

## 5. Miso One boundary

Miso TTS 8B is an expressive half-duplex renderer with audio context, a 7.7B
temporal backbone, and a 300M depth decoder over 32 Mimi RVQ codebooks. Its
audio-context and rich-prosody handling are useful offline evaluation signals.
It cannot listen while speaking or model turn taking, so it does not replace the
PersonaPlex audio plane or Nemotron control plane.

Miso is not a certified generator until a CUDA-only A/B evaluation against
Chatterbox Turbo shows better perceptual naturalness without worse Whisper WER,
word alignment, telephony-codec behavior, watermark incompatibility, or
provenance/license issues. Miso's public repository has inference code and
weights but no reproducible training or finetuning recipe, so it cannot define
the PersonaPlex training curriculum.

Official reference: [MisoTTS architecture](https://www.misolabs.ai/blog/miso-tts-8b),
[Miso weights](https://huggingface.co/MisoLabs/MisoTTS), and
[Miso license](https://github.com/MisoLabsAI/MisoTTS/blob/main/LICENSE).

## 6. Corpus and Nemotron curriculum

### 6.1 PersonaPlex adapter records

Each supervised agent target requires:

```text
prior duplex audio/code context
+ control frame available before target turn
+ optional evidence availability timeline
+ agent-only target text/audio code labels
+ revision, cancellation, overlap, and recovery metadata
```

Caller audio is context only. Agent targets are cropped at the actual audible
barge-in cutoff. The control/evidence input never contains the target transcript
or a paraphrase designed to leak its wording.

Coverage is mandatory across cooperation, conditional compliance, skepticism,
resistance, refusal, uncertainty, verification, correction, escalation, handoff,
repair, policy alternatives, casual conversation, natural endings, interruption,
and recovery. Openings and closings must be diversified at corpus level.

### 6.2 Counterfactual matrix

For every selected base caller context, produce at least one independently
rendered counterfactual where exactly one causal field changes:

| Changed field | Required target difference |
| --- | --- |
| tool result | claims only facts now authorized |
| policy constraint | declines or offers an allowed alternative |
| caller posture | stance changes while facts remain stable |
| next goal | response advances a different valid action |
| evidence availability | does not claim unavailable/expired fact |
| barge-in revision | cancels old trajectory and repairs naturally |
| end-call authorization | model-driven natural close plus correct private action |

The synthesizer uses independent model inference to generate and adjudicate all
semantic behavior. Whisper ASR, alignment, codec validation, and typed state
consistency certify render fidelity; they never replace semantic adjudication.

### 6.3 Nemotron training data

The same event log becomes a separate Nemotron SFT/control-compiler corpus:

```text
prior state + ASR observations + tool/policy events
-> next typed state patch + control frame + evidence decision
```

Nemotron labels never include the PersonaPlex target utterance. Train it to
produce valid bounded frames, select `wait`/safe-backchannel/strict-render/end
actions, preserve facts and uncertainty, and reject stale evidence. Evaluate it
on held-out counterfactual trajectories before connecting it to the audio plane.

## 7. Training progression

1. **Certification**: produce sufficient V6+ certified conversations with
   non-empty conversation-isolated train, validation, and test splits.
2. **Nemotron compiler SFT**: train/evaluate typed state and control decisions;
   no audio model involved.
3. **Prefix SFT**: freeze PersonaPlex base; train only control encoder plus
   gated prefix/K/V adapter with agent-only native audio/token loss and control
   dropout.
4. **Evidence SFT**: add the separate evidence encoder only after the prefix
   adapter passes held-out semantic adherence and freshness evaluation.
5. **Limited adaptation**: consider upper-layer LoRA only if frozen adapters
   demonstrably cap semantic adherence without reducing quality or latency.
6. **Interactivity alignment**: constrained RL or preference stage for pause,
   turn taking, backchanneling, and interruption, jointly scored with independent
   semantic/policy adherence. Never optimize timing alone.
7. **Twilio evaluation**: run live-equivalent media, barge-in, stale queue,
   codec, and safe-fallback scenarios before any production selection.

All training and inference run on CUDA `0,1,2`, observing external GPU load.
Do not train on a pilot with no validation/test split, and do not promote a model
solely on training loss.

## 8. Evaluation and promotion

Every checkpoint must pass all dimensions below on a held-out, lineage-locked
suite:

| Dimension | Required proof |
| --- | --- |
| semantic control | independent model adjudication and counterfactual sensitivity |
| factuality | correct use of allowed evidence; no claim from missing/expired evidence |
| freshness | stale revision never reaches emitted media |
| interruption | actual cutoff and grounded recovery after barge-in |
| timing | first audio, meaningful-content delay, stop latency, jitter tolerance |
| rendering | Whisper WER, word timing, codec integrity, clipping, voice preservation |
| dialogue | naturalness, no repeated templates, completion without looping goodbye |
| strictness | exact-language cases route to strict renderer only |
| isolation | concurrent calls cannot leak state, prefix, tool data, or voice context |

Promotion fails on any policy/factuality regression, stale-media emission,
unexplained semantic-control drop, or unreproducible run card.

## 9. Completion definition

The program is complete only when a licensed trained adapter and control
compiler have passed the held-out suite, a controlled server has demonstrated
acknowledged causal conditioning on CUDA, and an emulated Twilio call has
exercised tool evidence, state revision, interruption, recovery, natural
termination, telemetry, and safe fallback without stale or unsupported speech.
