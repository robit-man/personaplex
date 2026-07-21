# Primary-Source Research Synthesis

Research cutoff: 2026-07-19. Claims below distinguish published evidence from
local observation and design inference.

## 1. PersonaPlex

Primary sources:

- Roy et al., "PersonaPlex: Voice and Role Control for Full Duplex
  Conversational Speech Models," arXiv:2602.06053.
  https://arxiv.org/abs/2602.06053
- NVIDIA PersonaPlex source.
  https://github.com/NVIDIA/personaplex

Published evidence: PersonaPlex follows Moshi with concurrent user audio,
agent text, and agent audio streams. Its hybrid prompt temporally concatenates
voice audio and role text. Training masks prompt loss and downweights
nonsemantic audio tokens by 0.02. Synthetic service data is hierarchical; Dia
provides joint conversational audio and Chatterbox provides separately stitched
QA speech with positive or negative silence to model gaps and overlap.

Design implication: preserve native delayed duplex streams, voice prompting,
agent text loss, and agent audio loss. The existing role prompt demonstrates
static role conditioning, not mutable per-turn semantic control.

## 2. Moshi

Primary source:

- Defossez et al., "Moshi: a speech-text foundation model for real-time
  dialogue," arXiv:2410.00037.
  https://arxiv.org/abs/2410.00037

Published evidence: Moshi models separate user and model streams without hard
turn segmentation. Its inner-monologue text stream precedes audio tokens within
the delayed multistream architecture, supporting low-latency generation and
overlap.

Design implication: semantic alignment should act strongly through the text
stream while native audio SFT preserves acoustic realization. Caller audio must
remain context, not an optimization target.

## 3. MoshiRAG

Primary sources:

- Chien et al., "MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex
  Speech Language Models," arXiv:2604.12928.
  https://arxiv.org/abs/2604.12928
- Kyutai MoshiRAG source, inspected at commit
  `8c6dfc101b7871baa428424bcdc583b74fb561d9`.
  https://github.com/kyutai-labs/moshi-rag

Published evidence: MoshiRAG adds a retrieval trigger and reference text
encoder. Compressed reference embeddings are projected to model dimension and
added to native temporal input embeddings over successive 12.5 Hz steps.
ARC-Encoder with 4x compression and additive injection performs best among the
reported encoder/injection ablations. Training uses about 1.9 million synthetic
instances, simulated retrieval delay, reference dropout, and model fine-tuning.
The paper reports 3-6 percentage-point integration loss between correct
references and correct spoken answers, and strong sensitivity to ASR quality.

Local source evidence: `reference_with_time` is configured as a
`streaming_sum` condition. `LMGen.update_streaming_sum_tensors` queues one
sequence per batch slot and consumes one row before each executed temporal
step.

Design implication: use additive native temporal injection as v4's first
control mechanism; simulate update timing; keep ASR/state quality as a separate
reliability dimension; train explicit causal integration rather than assuming
an encoded fact will be spoken correctly.

## 4. Spoken-dialogue preference alignment

Primary source:

- Wu et al., "Aligning Spoken Dialogue Models from User Interactions,"
  arXiv:2506.21463.
  https://arxiv.org/abs/2506.21463

Published evidence: the work constructs more than 150,000 speech-dialogue
preference pairs covering linguistic and timing variation and applies offline
preference optimization. Its multistream DPO uses text-token probability;
including audio-token probability was unstable. It resynthesizes context while
preserving timing and reports that data-mix balance affects semantic and
temporal behavior.

Design implication: use text-stream pair ranking/DPO for causal semantic
preference, retain audio likelihood in matched SFT, and balance timing/content
failures instead of optimizing one scalar reward.

## 5. Multi-faceted full-duplex alignment

Primary source:

- "Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models,"
  arXiv:2606.11167.
  https://arxiv.org/abs/2606.11167

Published evidence: token likelihood alone does not directly optimize pause,
turn-taking, backchannel, and interruption behavior. The work uses GRPO on
short real-conversation segments for these four axes and combines timing reward
with an ASR/LLM semantic-quality reward. It applies the method to Moshi and
PersonaPlex and reports multi-turn generalization.

Design implication: generated-outcome alignment is required after SFT, and
timing reward must be separated from semantic reward to prevent content
degradation.

## 6. Full-Duplex-Bench series

Primary sources:

- Lin et al., "Full-Duplex-Bench," arXiv:2503.04721.
  https://arxiv.org/abs/2503.04721
- Lin et al., "Full-Duplex-Bench-v3," arXiv:2604.04847.
  https://arxiv.org/abs/2604.04847

Published evidence: the original benchmark isolates pause handling,
backchanneling, smooth turn taking, and user interruption with time-aligned ASR
and behavior metrics. V3 adds real human disfluencies and deterministic chained
tool tasks. Across evaluated systems, self-correction and multi-step reasoning
remain major failures.

Design implication: test control updates under pauses, false starts,
self-corrections, interruptions, and multi-step tool state. Evaluate tool
selection/arguments and spoken delivery separately.

## 7. Align-SLM

Primary source:

- Lin et al., "Align-SLM: Textless Spoken Language Models with Reinforcement
  Learning from AI Feedback," arXiv:2411.01834.
  https://arxiv.org/abs/2411.01834

Published evidence: speech continuations can be aligned from preference data
constructed with automatic semantic feedback and optimized with direct
preference methods.

Design implication: independent ASR/model feedback can construct post-training
preferences, but the final reliability gate still needs calibrated human
adjudication and cannot reuse its own training judge blindly.

## 8. Evidence strength and limits

| Source | Directness | Quality | Main limit |
| --- | --- | --- | --- |
| PersonaPlex paper/source | Direct architecture | High | No mutable turn-level semantic stream. |
| Moshi paper/source | Direct base architecture | High | No business-state control contract. |
| MoshiRAG paper/source | Direct injected text evidence | High | Retrieval accuracy remains below 95%; trains at much larger scale. |
| Spoken-dialogue preference paper | Direct multistream alignment | High | Preference objective is not typed control by itself. |
| Multi-faceted alignment | Direct duplex post-training | High | Very recent preprint; rewards require local reproduction. |
| Full-Duplex-Bench series | Direct evaluation | High | Benchmarks do not prove production control safety. |
| Align-SLM | Related speech preference | Moderate | Textless continuation differs from typed duplex control. |

## 9. Reconciled strategy

The sources do not support a claim that one frozen prefix adapter trained on a
few thousand targets will reach 95% live control reliability. Together they
support this staged design:

1. Native additive temporal control stream using pretrained lexical
   representations.
2. Causal paired SFT and text-token preference loss.
3. Limited temporal adaptation when adapter-only capacity is insufficient.
4. Generated-audio semantic and interactivity alignment.
5. Independent, preregistered full-duplex evaluation with live revision and
   cancellation semantics.

This is a design inference grounded in the sources, not a published guarantee.
