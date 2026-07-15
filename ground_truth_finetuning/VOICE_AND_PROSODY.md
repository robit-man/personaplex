# Voice Identity and Prosody

## 1. Separation of concerns

Voice identity, prosody, semantic content, and transport timing are separate controls. Treating them as one generic prompt is not sufficient for evaluation or safety.

| Concern | Primary owner | Guarantee |
| --- | --- | --- |
| Canonical wording | Semantic controller + strict renderer | Exact text when strict path passes ASR validation. |
| Natural turn-taking | PersonaPlex audio plane | Measured latency and interruption behavior. |
| Expressive semantic delivery | Prefix-adapted PersonaPlex | Measured plan compliance, not exact text. |
| Voice identity | Consented PersonaPlex voice prompt | Scoped to explicit consent and evaluated with consented comparisons. |
| Prosody/style | Plan delivery fields plus consented reference selection | Candidate behavior, evaluated independently. |

## 2. PersonaPlex voice prompts

PersonaPlex conditions agent voice with an audio prompt. The prompt has a speaker scope, approved use scope, source hash, transcript, codec/model revision, and revocation status. It is selected before call start or during an explicitly allowed change, never inferred from a caller voice without permission.

A voice prompt is not authorization to impersonate a speaker in any context. System policy must label synthetic voice output where applicable and prohibit use outside the consented scope.

## 3. Consented teacher and reference tools

Qwen3-TTS and TADA are potential research components, not interchangeable PersonaPlex replacements.

- **Qwen3-TTS Base:** May generate a consented reference clip from a short approved voice reference and known transcript. The official project describes fast voice cloning and fine-tuning capabilities. Use it to create or rank canonical voice-prompt candidates, not as proof that arbitrary style text controls Base cloning output.
- **Qwen3-TTS CustomVoice/VoiceDesign:** May be evaluated for explicit style or voice-design supervision only under its stated license, voice-consent, and benchmark constraints.
- **TADA:** Its aligned text-acoustic design is a useful candidate strict renderer or prosody teacher because timing/prosody can be studied on a direct representation. It does not share PersonaPlex latents by default and must not be injected into PersonaPlex without a trained, evaluated interface.

Any external engine is pinned by source revision, license, checkpoint provenance, and inference configuration. Generated teacher clips are labeled synthetic and retain their full lineage.

## 4. Voice-prompt curation procedure

1. Confirm consent, use scope, source quality, transcript, and absence of prohibited data.
2. Normalize through a deterministic logged audio pipeline.
3. Produce several approved prompt candidates with controlled transcript duration and phonetic coverage.
4. Run the frozen PersonaPlex model on a fixed neutral prompt set.
5. Score intelligibility, speaker similarity, duration stability, clipping, and turn-taking.
6. Select the prompt only on held-out utterances and record the scorecard.
7. On revocation, remove prompt and all derived artifacts according to `DATA_AND_GOVERNANCE.md`.

A subjective listening pick without a scorecard is not a production voice profile.

## 5. Prosody representation

The control plan uses bounded delivery fields that are observable and ethically limited:

```json
{
  "register": "calm_professional",
  "assertiveness": 0.45,
  "speaking_rate_bucket": "normal",
  "pause_density_bucket": "moderate",
  "emphasis_targets": ["Thursday", "afternoon"],
  "interruptibility": "yield_on_caller_speech"
}
```

Do not encode emotional manipulation, deceptive urgency, identity impersonation, or unbounded free-form delivery prompts as a hidden control channel. Prosody labels must be annotated from observable audio measures and reviewed for label agreement.

## 6. Training and evaluation use

Stage 1 semantic-prefix training conditions voice identity with an approved native voice prompt and treats delivery fields as plan inputs. It should not learn identity from the caller stream. Counterfactual tests hold semantic plan constant while changing only a delivery field, and hold delivery constant while changing required entities.

Voice metrics require consented same-speaker references and include speaker-embedding distance, intelligibility, duration, pitch/energy/rhythm distributions, and human preference with disclosure. A high similarity score cannot override failed semantic, safety, or turn-taking metrics.

## 7. Strict-mode rendering

The strict renderer has its own voice policy. It may use an approved TTS voice that is different from the expressive PersonaPlex prompt. If a requested voice cannot meet consent, quality, or latency requirements, select an approved neutral voice or transfer instead of silently substituting an unapproved clone.

## 8. Prohibited shortcuts

- Training on scraped celebrity, customer, or public-call voices without explicit rights.
- Claiming a voice clone is consented based solely on public availability.
- Treating arbitrary text style instructions as a verified cloning control.
- Mapping codec latents across unrelated engines without trained alignment and tests.
- Evaluating voice with the same source utterance used as its reference prompt.
