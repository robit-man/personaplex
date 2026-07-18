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

Training requests the configured GPU allowlist but can degrade to the number of
currently admitted GPUs, never below `PERSONAPLEX_TRAIN_MIN_WORLD_SIZE`. The
admission report records both requested and effective world size. Publication
attests the immutable certified-call count separately from the smaller set of
target turns that pass label-leak and native-tensor admission.

`PERSONAPLEX_TRAIN_MAX_WORLD_SIZE` defaults to `1`. The frozen 7B base is
replicated per rank, so this establishes a real checkpoint and evaluator result
without turning rank startup or DDP synchronization into the first deployment
gate. Raise it only after a measured distributed checkpoint proves rank loading
and synchronization on the current host.

The default training memory floor is derived from the inspected Moshi model-file
size multiplied by `PERSONAPLEX_TRAIN_MODEL_HEADROOM_RATIO` (default `1.50`,
calibrated from measured native adapter residency). Each GPU also retains
`PERSONAPLEX_TRAIN_GPU_RESERVE_RATIO` of its discovered total memory. The
immutable admission report records the effective per-GPU reserve, so this
policy scales without hard-coded server-memory assumptions.

Semantic-prefix training sets `NO_TORCH_COMPILE=1` by default through
`PERSONAPLEX_TRAIN_DISABLE_TORCH_COMPILE`. Moshi otherwise lazily creates a
large Inductor worker pool on its first forward pass, which is unsuitable for a
shared live-inference host. A launch records the effective choice; compilation
may only be re-enabled after a measured, bounded benchmark on that host.

`PERSONAPLEX_TRAIN_MAX_GPU_UTILIZATION_PCT` defaults to `85` for the shared
host lane. It is a configurable compute-co-tenancy policy, while the model-size
budget plus discovered VRAM reserve remains the hard admission condition.

Before the frozen model is loaded, rank zero alone SHA-256 verifies the native
weights against the model contract and broadcasts that result to the other
ranks. `startup.jsonl` records the preflight, model-load, adapter-init, and
run-contract stages in the attempt root. This avoids redundant multi-rank disk
hashing and makes a silent pre-artifact startup diagnosable without changing the
model or relaxing its provenance check.

Primary semantic-prefix preparation admits every quality-accepted, semantically
certified target turn, including V4 turns without a delayed-evidence frame.
V4 counterfactual provenance does not imply that every target has late evidence;
the separate evidence-stream stage selects the evidence-bearing subset. The
control serializer reserves its bounded input for the terminal flag, typed plan,
mutable state, recent audible context, and turn-taking. Training refuses a
held-out split unless that encoded context is effective and at least one genuine
model-selected terminal target is present.

`encode_controlled_native_adapter_tensors.py` supports deterministic modulo
shards. Each shard writes only its own manifest and ID-addressed tensor files;
`merge_controlled_native_tensor_shards.py` refuses missing, duplicate, or
foreign IDs before creating the manifest that certification consumes. This lets
the codec phase use dynamically admitted CUDA devices without weakening corpus
identity or provenance checks.

For an already prepared corrected corpus, use
`activate_prepared_controlv3.py` rather than re-entering the V7 snapshotter.
It certifies the merged native manifest, writes an isolated state that binds
the exact tensor root and native model contract, and may start only the named
control-v3 training unit. This prevents a timer from silently resuming an older
prepared root after a corrected corpus is produced.

Publication uses `HfApi.upload_large_folder` when available and falls back to
the compatible `upload_folder` API for older pinned Hugging Face clients. Both
paths publish only the staged, private, certified export.

## Host memory policy

Install `systemd/user@.service.d/90-personaplex-resource-governor.conf` on a
host that uses the distribution `systemd-oomd` user-manager default. It
prevents the inherited 50% PSI kill policy from terminating synthesis workers
while host memory remains available. The live synthesis governor remains the
authoritative dynamic control: it measures `MemTotal` and `MemAvailable` for
the whole host, pauses at its configured 80% utilization threshold, and
resumes at 75%. The synthesis slice still has percentage-based `MemoryHigh`
and `MemoryMax` limits as a final cgroup safety net.
