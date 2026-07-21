# Semantic Control v5 Primary Sources

Status: evidence map, not a claim of local convergence
Research cutoff inherited from the checked-in synthesis: 2026-07-19

## 1. Evidence rule

Model-specific architecture claims must cite a primary paper, official project
page, official repository, or inspected pinned source. Local implementation
claims must point to repository code or immutable artifacts. Design decisions
must be labeled as local decisions when the cited work does not directly test
the same typed-control objective.

Process IDs, checkpoint counts, process state, and GPU samples are timestamped
operational observations. They are neither research evidence nor durable ground
truth. Stage-completion claims must bind canonical artifacts, expected
cardinality, manifests, and hashes; live operational claims must be refreshed
from the host at the time they are made.

The detailed checked-in literature synthesis remains:

- [`../semantic_control_v4/RESEARCH_SYNTHESIS.md`](../semantic_control_v4/RESEARCH_SYNTHESIS.md)
- [`../../.aiwg/research/reports/semantic-control-v4-literature-synthesis.md`](../../.aiwg/research/reports/semantic-control-v4-literature-synthesis.md)
- [`../../.aiwg/research/quality-assessments/semantic-control-v4-grade.md`](../../.aiwg/research/quality-assessments/semantic-control-v4-grade.md)

## 2. Local immutable scenario evidence

The current scenario-stage decision is established by this local immutable
report, not by an external publication:

```text
/srv/voxrn_cache/personaplex/training/cascade-v5-pilot-20260720/scenario_stage_rejection.v1.json
reportId: sha256:c02f53487d795b213ad87078f1ea133f912b149bbdbc7886a7d20af4dc9755c1
```

The independent Qwen v4 clustered-findings audit rejected `918/1000` initial
scenarios, or `91.8%`, primarily for semantic mode collapse. This establishes
that the named local corpus is quarantined and not training eligible. It does
not establish a general quality claim about Qwen, PersonaPlex, or the replacement
blueprint architecture. The 55 blind repair candidates were discarded and are
not evidence of recovered quality.

## 3. Structured authentic planning

### Ollama structured outputs

Primary source: [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)

The official documentation states that callers may provide a JSON Schema in
the `format` field and that structured outputs are available through the
OpenAI-compatible API using `response_format`. It also recommends validating the
returned object against the same schema.

Local consequence: v5 sends strict JSON Schema through `response_format`, then
parses and validates JSON. The source does not justify regex recovery or
semantic field invention, so v5 prohibits both.

Compatibility boundary: the official documentation supports schema-constrained
output but does not establish that every JSON Schema keyword is implemented by
the active model/proxy path. Local review found `prefixItems` unsupported, so the
compact fan-out is being redesigned around ten named, required candidate-ID
object properties. Local review also found that a roughly 12,000-token proposed
response cannot satisfy the roughly 4,000-token output contract; Stage A must be
materially compact rather than relying on truncation or repair.

## 4. Full-duplex foundation

### PersonaPlex

Primary sources:

- [PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models](https://arxiv.org/abs/2602.06053)
- [NVIDIA PersonaPlex repository](https://github.com/NVIDIA/personaplex)
- [NVIDIA PersonaPlex project page](https://research.nvidia.com/labs/adlr/personaplex/)

Evidence used by v5: PersonaPlex builds on Moshi, jointly represents concurrent
caller audio, agent text, and agent audio, and supports hybrid lexical role and
audio voice prompting.

Local decision: mutable typed per-turn state is added as a native temporal
condition. The PersonaPlex sources do not themselves establish that this local
control interface reaches 95 percent reliability.

### Moshi

Primary source:

- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)

Evidence used by v5: Moshi models user and agent streams concurrently and uses
delayed multistream text/audio generation suitable for overlap and low-latency
speech.

Local decision: preserve native delayed duplex context and supervise only the
current agent target while treating caller and prior-agent streams as context.

## 5. Text conditioning for a real-time speech model

### MoshiRAG

Primary sources:

- [MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](https://arxiv.org/abs/2604.12928)
- [Kyutai MoshiRAG repository](https://github.com/kyutai-labs/moshi-rag)

The checked-in research inspected the repository at commit
`8c6dfc101b7871baa428424bcdc583b74fb561d9`. It records frozen ARC encoding,
compression, projection to the temporal model width, and additive
`streaming_sum` consumption over native temporal steps. It also records
asynchronous retrieval/encoding, reference dropout, timing variation, broad
receiver adaptation, and a remaining reference-to-spoken-answer integration
gap.

Local decision: v5 uses the same class of native temporal injection for compact
target-free control frames, then trains the PersonaPlex temporal/text receiver
with four-role causal groups. MoshiRAG is evidence that native text influence is
viable, not evidence that v5's business-state schema or live reliability target
already works.

## 6. Spoken-dialogue alignment

### Aligning Spoken Dialogue Models from User Interactions

Primary source:

- [Aligning Spoken Dialogue Models from User Interactions](https://arxiv.org/abs/2506.21463)

Evidence used by v5: multistream preference construction can vary linguistic
and timing behavior, and text-token preference objectives can be more stable
than directly ranking audio-token likelihood.

Local decision: retain matched agent-audio likelihood while using text-stream
listwise causal discrimination across the four siblings.

### Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models

Primary source:

- [Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models](https://arxiv.org/abs/2606.11167)

Evidence used by v5: token likelihood alone does not directly optimize pause,
turn-taking, backchannel, and interruption behavior, and semantic and timing
outcomes should be measured separately.

Local decision: generated evaluation must independently score semantic
adherence, interruption behavior, timing, and audio quality before live
promotion.

### Align-SLM

Primary source:

- [Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback](https://arxiv.org/abs/2411.01834)

Evidence used by v5: automatic semantic feedback can help construct speech
preference data.

Limit: an automatic judge cannot independently certify its own training data or
replace calibrated human adjudication at the final reliability gate.

## 7. Full-duplex evaluation

Primary sources:

- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)
- [Full-Duplex-Bench-v3](https://arxiv.org/abs/2604.04847)

Evidence used by v5: evaluation must isolate pause handling, backchanneling,
turn taking, interruption, disfluency, self-correction, and chained tool tasks
with time-aligned measurements.

Local decision: v5 adds typed control revisions, stale-control rejection,
strict-renderer routing, and immutable Twilio cancellation semantics to those
full-duplex evaluation dimensions.

## 8. Claims not supported by these sources

The sources do not establish any of the following:

- A small local dataset is sufficient for 95 percent live control reliability.
- Teacher-forced group ranking predicts free-running speech behavior.
- Additive conditioning guarantees exact spoken wording.
- Correct control encoding guarantees correct spoken integration.
- Synthetic ASR/LLM judges can replace independent live adjudication.
- A successful unit test, smoke run, or 150-step checkpoint is production
  convergence.

Those claims require the generated and live evidence defined in
[ARCHITECTURE.md](ARCHITECTURE.md) and [TODO.md](TODO.md).
