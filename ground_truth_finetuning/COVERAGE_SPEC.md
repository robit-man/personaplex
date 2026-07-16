# PersonaPlex controlled conversation coverage contract

This is the executable coverage contract for the 1,000-conversation synthetic
spoken-conversation corpus. It replaces topic-only generation. Every planned row carries a
`coverage` assignment and the Voryn generator persists that assignment in each
target record.

## V3 conversational breadth contract

The production v3 plan uses 100 concrete topic seeds across 20 families, then
crosses each seed with ten distinct context lenses. This creates 1,000 unique
scenario briefs rather than repeatedly paraphrasing a scheduling call. At least
85 percent of rows are outside service or administrative resolution. Required
families include casual peer exchange, polling, research interviews, learning,
work collaboration, technical troubleshooting, travel, shared household life,
creative review, community listening, wellbeing routines, repair/boundaries,
events, volunteering, science discussion, media, games, and environmental
problem solving.

Every row also has a length and cadence contract. The plan balances 6, 8, 10,
12, 14, 16, and 20 alternating turns across rapid reciprocal, conversational,
collaborative, repair, reflective, and interview-probe cadences. It does not
permit a generic introduction or early canned conclusion to substitute for a
multi-turn arc.

Research anchors inform the shape of this synthetic corpus only; none of their
audio or transcripts are copied into it. Switchboard demonstrates separate
telephone channels, diversified topics, and dialog-act annotation
(https://catalog.ldc.upenn.edu/LDC97S62). IEMOCAP motivates dyadic improvised
and scripted affective trajectories with multi-annotator labels
(https://sail.usc.edu/iemocap/). Interruption handling is modeled as an actual
timed phenomenon, not an intent label, following interruption-corpus work
(https://aclanthology.org/2024.lrec-main.176/) and current dual-channel
full-duplex benchmarks (https://arxiv.org/abs/2604.21406).

## Admission proportions

| Slice | Minimum | Rule |
| --- | ---: | --- |
| Trainable target turns carrying `ControlTrainingFrame` | 95% | The production plan targets 100%; unsteered samples are diagnostic baselines only. |
| Target turns with mutable state update source beyond the reducer | 80% | ASR, policy, task, tool, interruption, handoff, or timer source. |
| Conversations with actual caller barge-in and a recovery target turn | 20% | The 1,000-row plan assigns 200 `barge_in_recovery` rows. |
| Typed tool-result state transitions | 15% | Tool result is simulated only in this corpus and labeled as synthetic. |
| Strict renderer route examples | 2% holdout | Separate evaluation-only set; never represented as deterministic PersonaPlex output. |

## Required axes

The planner cycles every category below before repeating it. Quotas are written
to the plan summary and are checked before generation begins. The allocator
crosses dialogue act with turn pattern, then rotates speech style and control
source within each act block; high-interruption data therefore covers every
dialogue act instead of a fixed subset.

- Domains: 20 broad families including casual social life, polling/interviews,
  learning, work collaboration, technical maker support, consumer/local
  service, travel, shared household life, creative culture, community listening,
  wellbeing routines, administration, repair/boundaries, events, volunteering,
  research/science, media, games/hobbies, care logistics, and environment/public
  life. Service/admin rows are capped below 15 percent together.
- Dialogue acts: opening, clarification, fact verification, comparison,
  confirmation request, expectation setting, correction, constraint negotiation,
  bounded refusal, de-escalation, apology/repair, status explanation, evidence
  request, summary, handoff, escalation, next-step scheduling, interruption
  recovery, conditional closing, and open question.
- Caller postures: cooperative, conditionally compliant, skeptical,
  politely resistant, firm boundary, time pressured, confused, dissatisfied,
  handoff-oriented, and de-escalating.
- Speech styles: deliberate, brisk, hesitant, reassuring, concise, careful,
  frustrated-but-civil, confident, interruption-recovery, and natural overlap.
- Turn mechanisms: complete exchange, conditional clarification, actual barge-in
  plus cancellation, tool-result revision, and handoff/rejoin.
- Control sources: state reducer, finalized ASR (`asr_finalizer`), policy,
  task, tool result, interruption controller, handoff router, and timer. These
  are explicit contract values, not free-form labels.

## Silent steering requirements

Each target frame carries a bounded state tree: intent, verified facts,
commitments, uncertainty, policy constraints, tool results, caller posture,
compliance/resistance posture, next goal, and recovery state. The frame is
serialized into the adapter input. Target text/audio is a label only and must
not be present in that input.

Control updates are versioned and are applied at a following target boundary.
For a barge-in row, the target render is cropped at the acknowledged
`audibleEndedAtMs`; a caller turn starts before the original target render would
have ended; the following target frame has `recoveryExpected=true` and an
`interruption_recovery` update reason. A terminal truncation is rejected.

V3 rows additionally carry a seeded control-event program with exactly one
typed mutation before every target boundary. The mutation may originate from
finalized caller ASR, a verified synthetic tool result, policy, timer, handoff,
or interruption controller. It changes only the rolling state tree and typed
plan: facts, commitments, uncertainty, policy constraints, tool results, next
goal, or recovery status. The target label is never available to this program.

The generation system records an influence trace containing event ids, sources,
expected semantic effects, and state revision. An independent semantic verifier
receives the pre-turn frame, injected events, and candidate speech only after
generation; it must confirm natural realization of the update. Verifier failure
is a hard rejection. Caller ASR is compacted into a state atom before the
following target frame, providing the reverse information flow. This creates a
causal supervised relationship rather than a descriptive control label.

## Quality and split rules

- Chatterbox Turbo output must have provenance-approved reference audio,
  Whisper transcript, word alignment, WER/confidence pass, valid PCM/codec
  materialization, and semantic-control admission before it is trainable.
- The strict duplex exporter independently rechecks crop, overlap, recovery,
  frame/hash integrity, and label leakage.
- Split by voice pair, domain, and trajectory. Never put the same directed
  reference pair plus topic template in train and evaluation.
- Cap duplicate opening/closing patterns and repeated lexical n-grams during
  final curation. Generic name/company introductions are prohibited unless the
  planned dialogue act explicitly requires identity verification.
