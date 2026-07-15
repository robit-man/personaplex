# Training and release protocol

## What the historical artifact establishes

The checked-in log shows five teacher-token distillation epochs over 3,000
samples. It does not establish that the checkpoint is NF4, follows semantic
plans, preserves speech quality, performs ASR, handles interruptions, or is
safe for live calls. Do not publish it as a controllable hybrid model.

## Required dataset

Each example must retain reproducible provenance and include:

- licensed, consented caller and agent audio with speaker/voice permissions;
- aligned audio codec frames and text transcripts;
- full duplex timing: overlaps, silence, backchannels, interruptions, and
  turn boundaries;
- system role, versioned semantic plan, tool inputs/results, and policy;
- canonical response text plus factual/forbidden-claim annotations;
- split assignment by speaker, scenario, and source to prevent leakage.

Do not train on unlicensed cloned voices or production call recordings without
an explicit data-use basis and a deletion path.

## Training stages

1. Reproduce the pinned upstream model's speech and full-duplex baselines.
2. Train a small adapter/LoRA to condition the agent text stream on a structured
   semantic plan while freezing codec components initially.
3. Train with teacher-forced canonical response tokens aligned to agent audio;
   preserve user-audio and agent-audio streams rather than a text-only proxy.
4. Add ASR-grounded semantic rewards: required-entity recall, forbidden-claim
   rate, factual agreement, interruption behavior, and latency.
5. Evaluate against held-out speakers/scenarios and a no-regression full-duplex
   suite. Promote only checkpoints with a reproducibility manifest and model
   card.

The official PersonaPlex training regime is substantially larger than a
3,000-sample experiment. A local adapter pilot can be useful, but it must not
be represented as an equivalent retraining.

## Release requirements

Every candidate needs a base-model revision and license, code commit, data
manifest hashes, optimizer/configuration, hardware and wall time, checkpoint
hash, deterministic inference configuration, and complete evaluation report.
The first release should be an adapter plus a compatible base model rather than
a relabeled BF16 state dict.
