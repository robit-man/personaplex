# OpenRouter generative routing

The v5 whole-generative stack can use one to three OpenAI-compatible endpoints.
The checked runtime example binds Stage P, taxonomy-bound repair, Stage E, and
compact fanout to:

```text
https://openrouter.ai/api/v1/chat/completions
nvidia/nemotron-3-super-120b-a12b:free
```

`OPENROUTER_API_KEY` is read only from the process environment. There is no CLI
API-key option, keys are not written to checkpoints, and endpoint URLs containing
credentials or query strings are rejected. Super 120B remains bound to every
generative planner, taxonomy repair, scenario expansion, and compact-fanout role.

Adjudication uses Nemotron Super as the primary proposer, the exact
`nvidia/nemotron-3-ultra-550b-a55b:free` OpenRouter model as the independent
secondary proposer for both taxonomy and whole-blueprint findings, and local
Gemma only as the final evidence-bound boolean verifier. Super is allowed to
self-audit but cannot self-certify: admission requires an independently bound
proposer and a final verifier whose `{protocol, model}` identity differs from
the planner. The secondary route is configured only by
`PERSONAPLEX_LARGE_PROPOSER_ENDPOINT` and `PERSONAPLEX_LARGE_PROPOSER_MODEL`; it
uses only `OPENROUTER_API_KEY` and never falls back to the generative model.

`max_workers` remains capped at three. A single configured endpoint therefore
serves up to three concurrent generation workers; multiple endpoints rotate the
first attempt and provide transport failover.

## Strict-schema capability profile

The canonical Draft 2020-12 response schema remains the host admission contract.
For the exact OpenRouter/Nemotron binding above, the explicit
`openrouter_nemotron_free_grammar_v1` profile removes only `not` and
`uniqueItems` from the schema sent in `response_format`. These removals reflect
observed NVIDIA grammar errors. No other keyword is projected without new
provider evidence and a profile revision.

Each model-call record binds the capability-profile hash, canonical schema hash,
transport schema hash, and removed-keyword counts. Returned JSON is validated
against the untouched canonical schema, so projection cannot admit an output
that violates `not` or `uniqueItems`.

The exact Ultra 550B OpenRouter binding uses
`openrouter_nemotron_ultra_550b_prompt_schema_v1`. Reasoning remains explicitly
disabled, but `response_format` is omitted because this model does not support
it. The request instead appends the exact canonical schema plus a deterministic
minified-JSON-only contract. The raw returned JSON is parsed and always validated
on the host against the unchanged canonical schema. Call metadata binds profile,
model, schema, and prompt hashes without retaining prompt or response content.

Downstream scenario-contract scrutiny uses the same topology through
`PERSONAPLEX_SCENARIO_JUDGE_*` and `PERSONAPLEX_SCENARIO_SECONDARY_JUDGE_*`.
Both remote roles inherit `OPENROUTER_API_KEY`; repair continues through the
Super planner, while `PERSONAPLEX_SCENARIO_ADJUDICATOR_*` remains an independent
evidence-bound local decision role. Compact candidate and trajectory expansion
inherit `PERSONAPLEX_CASCADE_PLANNER_*`, so they cannot silently fall back to a
local 35B model.

## Stage-T canonical generation contract

Stage T uses canonical property names on the model wire. One-letter aliases are
forbidden because live generation showed semantic field shifts between submode,
participant relationship, resource, and tension. The canonical response budget
is 8192 tokens, while later Stage P and Stage E calls retain their tighter 4096
token budgets.

Every scenario ID receives its immutable host-assigned `interactionMode` in
`assignedInteractionModeByScenarioId` before generation. The submode must realize
that bound mode. Evaluating against an assignment hidden from the generator is a
protocol defect, not a repairable model-quality defect.

Exact structural failures are patched through model inference rather than host
text synthesis. A retry receives the complete rejected taxonomy, its content
hash, and exact per-ID forbidden duplicate values. It must return all twenty IDs
and may not repeat the rejected object. Once structurally valid, every repair
cycle rejudges the complete current twenty-anchor set; parent repair lineage
never suppresses a newly exposed confirmed finding.

## Retry policy

HTTP `408`, `425`, `429`, `500`, `502`, `503`, and `504`, transport failures, and
retriable top-level provider error envelopes are retried within a fixed attempt
bound. This includes HTTP 200 envelopes such as `{"error":{"code":502,...}}`.
`Retry-After` is honored on any retriable HTTP response or HTTP-200 provider
error envelope, including ResourceExhausted `502`; otherwise delay is
exponential. All delays are capped by
`PERSONAPLEX_GENERATIVE_RETRY_MAX_SECONDS`.

```text
PERSONAPLEX_GENERATIVE_TRANSPORT_ATTEMPTS=6
PERSONAPLEX_GENERATIVE_RETRY_BASE_SECONDS=1
PERSONAPLEX_GENERATIVE_RETRY_MAX_SECONDS=30
```

Diagnostics contain only endpoint, attempt, HTTP status, and error class. They do
not include authorization values, provider messages, prompts, or response text.
# Logical model identity and local physical routes

OpenRouter model names remain the logical checkpoint identity, while a physical
route may execute the same role locally. `ThreeEndpointStrictSchemaPlanner`
records `transportRoute.logicalModel`, `transportRoute.actualModel`, the actual
endpoint, and the actual schema profile on every call. A local route is omitted
from the logical binding for Stage-P/Stage-E generation so previously admitted
content remains exactly resumable; adjudication bindings include their physical
fallbacks so proposer independence is auditable.

`PERSONAPLEX_LOCAL_SUPER_*` binds the Ollama `nemotron-3-super:120b` artifact as
the preferred physical route for generation and the primary audit.
`PERSONAPLEX_LOCAL_SECONDARY_*` binds `nemotron-3-nano:30b` as the distinct
secondary proposer. The local Gemma verifier remains evidence-bound and cannot
propose content. Ollama requests use `reasoning_effort=none` and strict JSON
Schema. HTTP throttling and endpoint outages raise `ModelTransportUnavailable`;
they never consume semantic regeneration or repair attempts.
