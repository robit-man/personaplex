# Moshi Research and Control-Plane Reconciliation

Status: ground truth for the Voryn PersonaPlex control-adapter programme.
Research date: 2026-07-15.

## Decision

Keep the trainable, typed Voryn control-prefix adapter as the primary semantic
control mechanism. Extend it with a separately typed, delayed-evidence
conditioning lane inspired by MoshiRAG. Do not replace it with a system prompt,
an external text-to-speech renderer, or a sidecar that writes the next utterance.

MoshiRAG is the primary external implementation reference because it provides a
working full-duplex model, asynchronous semantic work, a learned text
conditioner, and a production-oriented split between front end and back end.
Miso One is an expressive-rendering reference only. It must not displace
PersonaPlex on the live audio plane and must not become the semantic control
mechanism.

The two conditioning lanes have different contracts:

| Lane | Source | When it is applied | Allowed effect |
| --- | --- | --- | --- |
| `control_prefix` | typed `control.update` state frame | next agent-turn snapshot only | intent, constraints, known facts, style, end-call authorization, and next goal |
| `evidence_stream` | typed, provenance-carrying tool/reference result | a later explicitly trained semantic boundary, or the next agent turn | grounded detail that became available after the control snapshot |

`control_prefix` is mandatory for policy-sensitive speech. `evidence_stream` is
optional and never authorizes an action absent from the acknowledged control
frame. Neither lane carries the target transcript or canonical target wording.

## What "Moshi One" corresponds to

No official Kyutai artifact named `Moshi One` was found. The local server uses
the original `kyutai/moshiko-pytorch-bf16` Moshiko release, not a newer model:
`model.safetensors` is 15.4 GB and the bundle also contains the 385 MB Mimi
codec tokenizer plus the SentencePiece text tokenizer. Moshiko is one of the
two original fixed synthetic voices, alongside Moshika. The official model card
and repository describe it as the 2024 Moshi family, released under CC-BY 4.0.

Relevant newer official releases are:

| Artifact | What it contributes | Weight/access status | Adoption decision |
| --- | --- | --- | --- |
| `kyutai/moshika-rag-pytorch-bf16` | asynchronous external-reference conditioning in a full-duplex Moshi | public BF16 MoshiRAG checkpoint; CC-BY 4.0 | use as a design and evaluation reference, not as a drop-in replacement for PersonaPlex voice control |
| `kyutai/personaplex-rl-seamless` | RL-aligned pause handling, turn taking, backchannels, and interruption recovery | 16.7 GB gated checkpoint; CC-BY-NC 4.0 plus NVIDIA Open Model License | do not download or deploy without accepted access terms and a commercial-use review |
| `kyutai/moshika-rl-seamless` | same interactivity-alignment research applied to Moshi | public model card; license must be checked with the artifact | research/evaluation reference only |

MoshiRAG and the RL PersonaPlex checkpoint are the meaningful recent insights;
neither makes the original Moshiko server a mutable semantic-control system.

## Architecture evidence

### Miso One / MisoTTS

Miso One is Miso Labs' `MisoTTS` 8B text-to-dialogue renderer. Its public
`MisoLabs/MisoTTS` F32 safetensors checkpoint is 32.8 GB and is released under
a modified MIT license. The modification requires prominent "Miso Labs" UI
attribution only for products or services exceeding 50 million monthly active
users or 10 million USD monthly revenue. The published inference stack defaults
to BF16 on CUDA and also downloads the Mimi codec, Llama tokenizer, and
SilentCipher watermarking component.

Architecturally, Miso has a 7.7B temporal Llama-style backbone plus a 300M
depth decoder. It represents one audio frame through 32 RVQ codebooks of 2,048
entries. The backbone receives interleaved text/audio embeddings and predicts
the first codebook; the depth decoder autoregressively predicts the remaining
31 codebooks. Optional prompt audio plus its transcript conditions voice
continuation. These are useful renderer insights: preserve audio context,
represent rich prosody with multi-codebook acoustic detail, and keep an explicit
voice-conditioning path distinct from semantic control.

The critical limitation is explicit in Miso's own release: it models individual
turns and produces half-duplex audio. It neither listens while speaking nor
models turn-taking. It therefore cannot replace PersonaPlex/Moshi for Twilio
streaming, barge-in, or mutable semantic state. The public project contains
weights and inference code, but no released training corpus, optimizer recipe,
finetuning pipeline, or duplex training protocol. Do not distill it or use its
watermarked output in certified PersonaPlex training without a separate license,
watermark, provenance, and ASR/timing audit.

Miso is an optional offline renderer-evaluation arm only. Keep Chatterbox Turbo
as the certified corpus renderer unless a controlled, CUDA-only comparison shows
that Miso improves naturalness and timing without harming transcript faithfulness
or license/provenance compliance.

Source: [Miso architecture announcement](https://www.misolabs.ai/blog/miso-tts-8b), [Miso weights](https://huggingface.co/MisoLabs/MisoTTS), [Miso inference repository](https://github.com/MisoLabsAI/MisoTTS), [Miso license](https://github.com/MisoLabsAI/MisoTTS/blob/main/LICENSE).

### Moshi

Moshi is a 7B full-duplex speech-text model with a streaming temporal
Transformer operating at 12.5 Hz and a depth Transformer that predicts audio
codebooks. It jointly models caller audio, agent audio, and time-aligned agent
text. This makes it the correct family for natural overlap, interruption, and
backchannels, but the base model alone does not provide a mutable application
state interface.

Source: [Moshi paper](https://arxiv.org/html/2410.00037), [Moshi repository](https://github.com/kyutai-labs/moshi), [Moshiko weights](https://huggingface.co/kyutai/moshiko-pytorch-bf16/tree/main).

### PersonaPlex

Official PersonaPlex adds a *static* Hybrid System Prompt: text role tokens are
forced on the agent-text channel while the agent-audio channel is silent; an
audio voice example is then supplied on the agent-audio channel. The prompt is
prefilled before dialogue and its loss is masked during training. This proves
that a trained text/audio prefix can condition a duplex model, but it is not a
per-call revision protocol and it cannot safely absorb mid-call tool outcomes.

The released checkpoint adds 1,217 hours of Fisher conversations for
backchannels, expressions, and emotional response, and uses Chatterbox for the
released synthetic dialogue set. This validates the current decision to keep
Chatterbox Turbo and real duplex timing in the Voryn corpus rather than relying
on isolated, turn-by-turn TTS alone.

Source: [PersonaPlex paper](https://arxiv.org/html/2602.06053), [official PersonaPlex repository](https://github.com/NVIDIA/personaplex).

### MoshiRAG

MoshiRAG is the most relevant external control-plane precedent. It separates a
fast full-duplex front end from a slower text back end. A learned retrieval
trigger starts asynchronous work, then a reference encoder turns returned text
into embeddings which are projected and summed into the temporal-Transformer
input over streaming steps. The system is deliberately modular: Moshi continues
speaking while retrieval runs, and the backend can be swapped without retraining
the retrieval service.

Its timing lesson is important: the research targets retrieval completion within
two seconds and exploits the interval between response onset and the first
informative word. Its data construction uses line-by-line multi-turn scripts,
separates user/reference/agent information to prevent leakage, aligns injection
before the grounded body of the response, randomizes injected delay, and applies
reference dropout. This is a demonstrated learned conditioning input, not
prompt concatenation.

We intentionally differ on one point: Voryn must not change policy-sensitive
guidance halfway through a committed utterance. Therefore a new
`control.update` revision invalidates the current generation and applies at the
next agent turn. A delayed `evidence_stream` may only condition an explicitly
trained semantic boundary whose earlier audio is non-committal and whose
generation is cancellable. Until that boundary is trained and evaluated, all
late evidence waits for the next agent turn.

Source: [MoshiRAG paper](https://arxiv.org/html/2604.12928), [MoshiRAG reference implementation](https://github.com/kyutai-labs/moshi-rag).

#### MoshiRAG artifacts and adoption boundary

The official `kyutai/moshika-rag-pytorch-bf16` checkpoint is an 8B BF16,
CC-BY 4.0 model. The public repository includes the main full-duplex Moshi
server, a separate `reference_with_time` conditioner service, streaming ASR
integration, and an OpenAI-compatible retrieval-back-end contract. The model
card is explicit that the checkpoint contains only the front end: an ARC encoder
and streaming ASR are additional dependencies, and retrieval safety remains the
application's responsibility.

Use MoshiRAG as the direct engineering baseline for `evidence_stream`:
asynchronous text service -> compact text/reference encoder -> causal learned
conditioning path -> later grounded speech. Retain Voryn's stricter revision
snapshot and cancellation protocol: a reference may not authorize a new claim,
invalidate a policy, or mutate already committed audio. The model's pre-RAG
acknowledgement behavior is not a license to emit generic filler; it must be
learned from certified examples and remain cancellable.

Source: [MoshiRAG model card](https://huggingface.co/kyutai/moshika-rag-pytorch-bf16), [MoshiRAG server/conditioner instructions](https://github.com/kyutai-labs/moshi-rag).

#### Locally inspected MoshiRAG conditioner contract

The public BF16 artifact is staged at
`/srv/voxrn_cache/moshi-rag/kyutai-moshika-rag-pytorch-bf16`. Its inspected
configuration pins the actual conditioner shape and timing contract:

| Property | Released value | Voryn implication |
| --- | --- | --- |
| main temporal width | `4096` | evidence bridge must emit the PersonaPlex temporal width, not text logits |
| temporal rate | `12.5 Hz` | evidence is a small streaming sequence, not a single untyped vector |
| text compressor | frozen `kyutai/ARC4_Encoder_Llama` | use a frozen compact reference encoder for the first evidence stage |
| compression | `compress_rates: [-4]` | compact long tool/reference text before temporal injection |
| bridge | `3072 -> 2048 -> 4096` | train bridge/adapter before considering base-model adaptation |
| fuser | `streaming_sum: [reference_with_time]` | evidence enters as learned causal temporal conditioning |
| reference dropout | `0.2` | apply evidence dropout so sparse/late evidence does not destabilize dialogue |
| simulated timing | `start_delay: 1.0s`, `end_gap: 1.0s`, random branch `0.2` | train availability/timing variance, but enforce Voryn's next-turn policy boundary initially |

The artifact contains `model.safetensors` (15,409,202,784 bytes), Mimi codec
weights, SentencePiece tokenizer, and `config.json`. It has not been loaded or
promoted into a live call path. This contract is a reference for the future
evidence encoder, not permission to claim mid-utterance policy updates are safe.

### Interactivity RL

Kyutai's latest alignment work treats pause handling, smooth turn-taking,
backchanneling, and user interruption as separate training axes. It extracts
short duplex segments from the 4,000-hour Seamless Interaction corpus, uses
axis-specific rewards with GRPO, and adds an ASR plus LLM content-quality reward
so latency optimization does not degrade semantic relevance. The released
PersonaPlex RL derivative is trained from the NVIDIA PersonaPlex base and is
gated under non-commercial and NVIDIA license terms.

This is evidence for a *second*, post-SFT timing-alignment stage. It is not a
substitute for the semantic control adapter: timing rewards cannot teach tool
facts, state revisions, or policy constraints. It also identifies a real risk:
interactivity-only optimization can degrade safety and semantic quality.

Source: [interactivity-alignment paper](https://arxiv.org/html/2606.11167), [PersonaPlex RL checkpoint card](https://huggingface.co/kyutai/personaplex-rl-seamless).

## Required Voryn design

```text
caller duplex audio
  -> streaming ASR + turn state
  -> versioned call-state tree
  -> semantic LLM / tools
  -> control.update(revision N, typed compact frame)
  -> GPU-cached ControlFrameEncoder -> control_prefix / per-layer K-V adapter
  -> immutable generation snapshot at next agent turn
  -> PersonaPlex audio/text token generation
  -> Twilio media stream

tool/reference result arriving after snapshot
  -> evidence.update(revision N+1, provenance, expiry, allowed claims)
  -> invalidate stale queued generation
  -> next agent-turn `control_prefix`, or future trained semantic boundary
```

Every update must contain the call id, monotonically increasing revision,
effective boundary, state hash, expiration, and an acknowledgement target.
Generation output must be tagged with the acknowledged revision and generation
id. Barge-in immediately drops queued output for that id; it never rewrites
audio already delivered to Twilio.

The compact frame remains declarative. It carries intent, known facts, caller
posture, next goal, constraints, style, allowed tool-result references, and
terminal authorization. Raw long documents do not belong in this lane.

The delayed evidence frame is separate and contains only a compact encoded
reference plus provenance, allowed claims, TTL, and the revision it supports.
The base model is frozen for the initial stage. Train a control encoder,
gated virtual-prefix or per-layer K/V adapter, and the optional evidence encoder;
keep control dropout so sparse state does not destroy duplex behavior.

## Corpus changes

The certified synthetic corpus must add these aligned examples before native
adapter training:

1. Static control examples with no evidence update, including a control-dropout
   counterpart.
2. State-revision counterfactuals: same audible caller history, different
   permitted facts, tool result, policy constraint, or caller posture, producing
   materially different permissible agent content.
3. Delayed-evidence examples: the exact same control frame with either a timely
   evidence item, delayed item, expired item, failed item, or no item. The
   agent target must not claim an unavailable fact.
4. Cancellation examples: caller barge-in after partial agent speech, a new
   frame revision, and a grounded recovery on the next agent turn.
5. Realistic cooperation, skepticism, conditional acceptance, refusal,
   correction, escalation, handoff, repair, casual discussion, and natural
   endings, without template introductions, placeholder speech, or forced
   closing phrases.
6. Voice-diverse Chatterbox Turbo duplex renderings with target-only loss,
   Whisper transcription/timing verification, codec checks, and explicit
   provenance for every voice reference.

No regex or phrase heuristic may decide semantic correctness, terminal intent,
or training-data admission. Independent model inference adjudicates semantic
and control adherence. Acoustic timing, codec integrity, word alignment, and
typed-state consistency remain measurable structural checks, not content
shortcuts.

## Training and evaluation sequence

1. Build a certified train/validation/test corpus with conversation-level split
   isolation, revision/evidence counterfactual coverage, and no target-text
   leakage.
2. Freeze the PersonaPlex base. Train only `ControlFrameEncoder` plus gated
   prefix/K-V adapter with agent-text/audio loss, control dropout, and
   target-only supervision.
3. Add the evidence encoder only after the prefix adapter clears held-out
   revision and factuality tests. Do not train late evidence as a hidden prompt.
4. Run real CUDA-only streaming evaluation: semantic adherence, factual/tool
   incorporation, stale-revision rejection, cancellation latency, first audio
   latency, Whisper ASR faithfulness, word timing, codec quality, and speaker
   preservation.
5. Only then run a constrained interactivity RL stage. Optimize the four
   duplex axes while retaining an independent model-based semantic, policy, and
   control-adherence reward. Stop the stage on safety or factuality regression.
6. Evaluate static duplex inputs and live multi-turn examiner calls separately.
   Neither replaces an end-to-end Twilio barge-in/cancellation test.

## Immediate implementation backlog

- [ ] Add `evidence.update` to the V2 control protocol with provenance, TTL,
  supported revision, acknowledgement, and cancellation semantics.
- [ ] Implement a separately trainable evidence encoder and keep it disabled by
  default until its causal path passes a native CUDA harness.
- [ ] Extend exporter/precodec manifests with evidence timing, availability, and
  counterfactual-pair identifiers.
- [ ] Add model-inference certification prompts for unavailable/expired
  evidence and stale-revision misuse.
- [ ] Add held-out evaluation suites for static control, state revision,
  delayed evidence, barge-in recovery, and end-call authorization.
- [ ] Run the current prefix-only adapter only after sufficient V6 certified
  data produces non-empty train, validation, and test splits.
- [ ] Keep the gated Kyutai RL weight out of deployment until license acceptance
  and a documented non-commercial/commercial decision.
