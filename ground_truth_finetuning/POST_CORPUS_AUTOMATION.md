# V7 post-corpus automation

`personaplex-v7-post-corpus.timer` polls the certified V7 corpus. It does nothing
until the configured target count is reached. At the threshold it performs one
immutable transition:

1. Snapshot only the explicitly allowlisted `*.certified.jsonl` artifact
   families, their referenced audio, and their duplex-timing sidecars.
2. Require V4 schema, accepted audio quality, target semantic certification,
   training eligibility, control frames, and safe audio paths.
3. Stop synthesis services after the snapshot so native encoding and training can
   use the configured GPUs.
4. Run export, pre-codec preparation, native encoding, and tensor certification.
5. Start distributed semantic-prefix training and private Hugging Face publication
   concurrently only when the tensor certificate says `certified_for_adapter_training`.

The coordinator uses `/srv/voxrn_cache/personaplex-systemd/personaplex-transition.env`.
Every location and policy is configurable there. The only hardware policy defaults
are the existing live host-memory guard and GPU admission; no memory capacity is
encoded in this automation. The HF token belongs only in the mode-`0600`
`personaplex-hf.env` file and is never committed.

`PERSONAPLEX_PYTHON` selects the interpreter used by finalization, training, and
publication. It must contain CUDA-enabled PyTorch, `sentencepiece==0.2`, and
`huggingface_hub`; the systemd units invoke a tracked launcher so child pipeline
commands inherit the same interpreter rather than silently falling back to the
host Python.

Use `systemd/personaplex-transition.env.example` as the tracked configuration
contract. The codec artifact is SHA-256 checked by native preparation against
the model contract before tensor encoding or training; a missing or mismatched
codec blocks the handoff without mutating the source corpus.

`PERSONAPLEX_CERTIFIED_ARTIFACT_PATTERNS` is a comma-separated filename
allowlist evaluated beneath each configured GPU's synthesis directory. Add a
new replenishment family there only after reviewing its schema, provenance, and
quality gates. The coordinator rejects path separators and parent traversal and
does not discover arbitrary historical certified artifacts.

The native exporter distinguishes conversation-integrity failures from
target-label failures. Missing audio, invalid timing, malformed timelines, and
failed interruption coverage reject the whole conversation. A target turn whose
own control state contains its target wording is quarantined by itself; valid
causally independent target turns from the same duplex conversation remain
available for agent-only supervision.

If export and pre-codec preparation finish but tensor encoding fails before
writing its artifact root, the next finalizer invocation resumes from that
immutable pre-codec manifest. It never reuses a partial tensor directory. This
prevents a recoverable source-contract or GPU-admission failure from needlessly
re-rendering the corpus while keeping every native tensor attempt fail-closed.
Partial tensor roots are atomically moved aside with an `incomplete-` suffix
before a clean re-encode; they remain available for incident analysis but can
never be mistaken for certified training input.

A failed tensor certificate with a complete encoded manifest resumes at
certification, not encoding or audio export. Its ASR rule requires an accepted
quality record and an explicit WER threshold; an over-threshold sample remains
admissible only when its recorded marginal-ASR adjudicator explicitly accepted
it. This preserves the upstream quality policy instead of silently widening it.

## Host memory policy

Install `systemd/user@.service.d/90-personaplex-resource-governor.conf` on a
host that uses the distribution `systemd-oomd` user-manager default. It
prevents the inherited 50% PSI kill policy from terminating synthesis workers
while host memory remains available. The live synthesis governor remains the
authoritative dynamic control: it measures `MemTotal` and `MemAvailable` for
the whole host, pauses at its configured 80% utilization threshold, and
resumes at 75%. The synthesis slice still has percentage-based `MemoryHigh`
and `MemoryMax` limits as a final cgroup safety net.
