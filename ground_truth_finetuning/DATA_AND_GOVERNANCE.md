# Data, Provenance, and Governance

## 1. Dataset purpose

The corpus must teach a native audio model how a typed plan constrains the next agent turn while retaining natural voice, timing, interruption behavior, and dialogue continuity. It is not a generic speech dump, a scraped voice collection, or a text-only instruction corpus relabeled as speech control.

## 2. Eligibility

Every source conversation, voice prompt, transcript, synthetic derivative, and code artifact needs a manifest record. A record is eligible only when all of these are present:

- Source identifier and cryptographic content hash.
- License or written authorization that permits the intended training and distribution use.
- Documented speaker consent for voice modeling where a recognizable voice is retained.
- Collection date, jurisdiction, retention rule, and deletion/revocation contact.
- PII classification, redaction method, and redaction review result.
- Clear human/synthetic provenance label.
- Annotation author/tool version and quality-control status.

No political persuasion, impersonation, nonconsensual voice clone, covert recording, or unknown-rights source is eligible. Calls that include protected customer information require a separate approved data process; they are not automatically eligible because they exist in operational logs.

## 3. Corpus composition

The corpus has four separately labeled families:

| Family | Role | Minimum requirement |
| --- | --- | --- |
| `licensed_dialogue` | Natural multi-turn behavior and interruption timing. | Source rights, speaker consent if voice retained, reviewed semantic annotations. |
| `consented_voice_prompt` | Voice identity and timbre conditioning. | Per-speaker voice consent, approved use scope, revocation link. |
| `synthetic_controlled` | Broad plan coverage and counterfactual control pairs. | Generator/version/prompt lineage, human or rule-based audit, marked synthetic. |
| `adversarial_eval` | Held-out safety, entity, interruption, and stale-update tests. | Never fed back into training without a new split/version. |

Synthetic conversations expand coverage but do not establish human realism. They must be evaluated separately from real licensed-dialogue data.

## 4. Immutable split policy

Split by conversation lineage, speaker, organization, and scenario template. No adjacent turns from one source conversation may cross splits. A recommended initial split is 80% train, 10% validation, 10% test, plus a permanently frozen adversarial test suite.

The split function is deterministic:

```
split = hash(dataset_version + source_conversation_lineage) mod 100
```

The data manifest stores the exact split result. A record cannot be moved to improve a metric. New data creates a new dataset version and preserves old evaluation manifests.

## 5. Required item manifest

Each exported example has a JSON record similar to:

```json
{
  "schema_version": 1,
  "example_id": "sha256:...",
  "dataset_version": "gtft-v0.1",
  "split": "train",
  "conversation_lineage_id": "conv-...",
  "turn_index": 4,
  "provenance": {
    "kind": "licensed_dialogue",
    "source_uri": "restricted://...",
    "source_sha256": "...",
    "license_id": "...",
    "consent_id": "...",
    "contains_personal_data": false,
    "redaction_version": "..."
  },
  "audio": {
    "sample_rate_hz": 24000,
    "caller_sha256": "...",
    "agent_sha256": "...",
    "voice_prompt_sha256": "...",
    "speaker_scope_id": "..."
  },
  "semantics": {
    "plan_schema_version": 1,
    "plan_sha256": "...",
    "canonical_response_sha256": "...",
    "annotation_status": "double_reviewed"
  },
  "model_encoding": {
    "model_revision": "...",
    "codebook_layout": {"text": [0], "agent_audio": [1, 8], "caller_audio": [9, 16]},
    "delay_config_sha256": "...",
    "codes_sha256": "...",
    "text_alignment_sha256": "...",
    "target_mask_sha256": "..."
  }
}
```

The example manifest stores references and hashes, not raw secrets. Access-controlled locations must not be committed to a public repository.

## 6. Semantic annotation contract

Annotators create a plan for the next agent turn using only evidence available before that turn. They separately label the canonical response after the plan is complete. Annotation requires:

- Intent and dialogue act.
- Entity ledger, with evidence offsets or source turn references.
- Required facts, forbidden claims, required question(s), and escalation rules.
- Delivery controls such as language, register, assertiveness, interruption policy, and duration budget.
- Strict versus expressive decision and rationale.
- Uncertainty and ambiguity markers.

The canonical response text, human audio, and textual explanation are labels. None may be concatenated into the adapter's plan input. A validator rejects any serialized plan that contains canonical text or a matching canonical-text hash.

## 7. Audio and timing requirements

- Preserve original timestamp metadata and VAD/interruption events where available.
- Store caller and agent channels separately when they are separately available.
- Mark overlaps instead of forcing them into sequential turns.
- Resample only through a logged deterministic pipeline.
- Retain original lossless source when policy allows; derived WAV/codec files are reproducible outputs.
- For each turn, record caller-end, plan-ready, boundary, first-agent-audio, final-agent-audio, interruption, and end timestamps.

Synthetic timing must be generated from an explicit distribution that is fitted and reported against consented real-call timing. It cannot claim realism only because it has randomized pauses.

## 8. PersonaPlex encoding validity

An exporter must load the exact target model revision and inspect its runtime configuration. For every item it must:

1. Verify sampling rate, codec, codebook count, codebook layout, and delay configuration.
2. Encode caller and agent audio independently with recorded channel assignment.
3. Produce the text code stream using the model's actual tokenizer/conditioner.
4. Verify text/audio temporal alignment using the model-supported representation.
5. Construct an explicit agent-target mask.
6. Decode a deterministic sample and compare duration/tokens against the source.
7. Emit a validation report with every artifact hash.

A code file without a matched encoder revision, layout, and target mask is invalid training input.

## 9. Consent, voice identity, and revocation

Voice data has a separate consent ledger from dialogue content. The ledger names permitted purposes, model families, output distribution, retention period, and revocation mechanism. A revocation triggers a new corpus version excluding the source, derived prompts, cached codec codes, checkpoints that contain it when practical, and all published artifacts governed by the agreement.

Voice similarity is evaluated only using consented reference/target pairs. Speaker identities and raw clips are not published in metrics reports.

## 10. Dataset acceptance tests

Before any training run, the suite must reject a corpus with any of the following:

- Missing manifest field or artifact hash.
- Non-deterministic or cross-lineage split.
- Unverified text alignment.
- Caller stream included in the target mask.
- Canonical-text leakage into plan serialization.
- Missing consent/license/revocation record.
- Schema mismatch between corpus and adapter.
- Unredacted prohibited personal data.
- Unexplained timing or codec transformation.
