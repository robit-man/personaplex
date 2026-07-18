# V7 post-corpus automation

`personaplex-v7-post-corpus.timer` polls the certified V7 corpus. It does nothing
until the configured target count is reached. At the threshold it performs one
immutable transition:

1. Snapshot only the explicitly allowlisted `*.certified.jsonl` artifact
   families and their referenced audio.
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

Use `systemd/personaplex-transition.env.example` as the tracked configuration
contract. The codec artifact is SHA-256 checked by native preparation against
the model contract before tensor encoding or training; a missing or mismatched
codec blocks the handoff without mutating the source corpus.

`PERSONAPLEX_CERTIFIED_ARTIFACT_PATTERNS` is a comma-separated filename
allowlist evaluated beneath each configured GPU's synthesis directory. Add a
new replenishment family there only after reviewing its schema, provenance, and
quality gates. The coordinator rejects path separators and parent traversal and
does not discover arbitrary historical certified artifacts.

## Host memory policy

Install `systemd/user@.service.d/90-personaplex-resource-governor.conf` on a
host that uses the distribution `systemd-oomd` user-manager default. It
prevents the inherited 50% PSI kill policy from terminating synthesis workers
while host memory remains available. The live synthesis governor remains the
authoritative dynamic control: it measures `MemTotal` and `MemAvailable` for
the whole host, pauses at its configured 80% utilization threshold, and
resumes at 75%. The synthesis slice still has percentage-based `MemoryHigh`
and `MemoryMax` limits as a final cgroup safety net.
