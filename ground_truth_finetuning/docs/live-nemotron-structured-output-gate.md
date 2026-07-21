# Live Nemotron structured-output gate

## Observed failure, 2026-07-20

The local `nemotron-3-super:120b` route successfully returned strict JSON through
Ollama's OpenAI-compatible endpoint with reasoning disabled. Transport completion,
JSON parsing, and canonical schema validation were not the admission bottleneck.

Several schema-valid Stage-T anchors nevertheless contained mixed-language script
fragments, malformed token splices, dangling clauses, or values ending mid-word.
The prior typed judge vocabulary could only propose mode mismatch, field-role
misuse, semantic collapse, or general implausibility. Because the final verifier
is source-bound to proposed typed claims, it could not discover an unproposed
language or completeness defect. Some malformed anchors were therefore admitted.

## Ground-truth correction

Stage-T quality remains inference-judged. Host code must not use regexes, script
detection, word lists, or lexical acceptance heuristics as semantic truth.

The proposer and source-bound verifier now support three additional typed claims:

- `language_or_encoding_corruption`
- `incomplete_or_malformed_field`
- `unnatural_or_placeholder_content`

The full twenty-anchor view declares English, complete grammatical fields, natural
conversation setups, and absence of placeholder/meta residue as explicit quality
requirements. Each proposer emits at most one cluster per finding code with every
affected scenario ID. The independent verifier confirms or rejects each exact
code-and-ID claim from source-bound evidence. Confirmed IDs alone enter targeted
repair; untouched IDs retain immutable lineage.

## Attempt observability

Every OpenAI-compatible structured response is persisted under the run's
`.scenario_blueprint_v5/model_attempts` directory before downstream semantic
validation. Each trace records the stage name, topic, physical and logical model,
route, attempt, finish reason, schema/profile hashes, usage, response, response
hash, and status. It never stores request prompts, headers, API keys, or bearer
tokens. A compact JSON event with the trace hash is flushed to the service journal.

This separates the following failure classes with direct evidence:

- Transport or provider unavailable.
- Length termination.
- Malformed JSON.
- Canonical schema rejection.
- Canonical schema acceptance followed by downstream semantic rejection.

Raw checkpoint counts are not progress counts because immutable superseded stage
keys remain on disk. Progress reporting must use the active protocol/binding
lineage and unique topic IDs, not the number of JSON files in a checkpoint folder.

## Context-engineering conformance harness

`tools/evaluate_nemotron_context_contract.py` runs the local 120B model against
clean, defective, and counterfactually rebound source contexts with reasoning
disabled. Every case is sent through both OpenAI-compatible strict JSON Schema
output and a single terminal `submit_taxonomy_judgment` function call. The harness
measures transport, parse, schema, semantic-claim, clean-control, and latency
results separately and retains the normalized output for audit.

Tool calling is not enabled merely because Ollama exposes it. It must strictly
outperform schema output on semantic adherence; a tie selects schema because it
has fewer moving parts. Tool calls remain appropriate for genuine runtime actions
such as state lookup, handoff, and model-selected call termination. Bulk dataset
artifacts continue to use whichever typed transport wins the measured contract.

The preferred tool experiment uses native Ollama `think:false`, one report tool
per typed finding code, and a mutually exclusive `approve_taxonomy` tool. This
makes semantic selection part of the function name and keeps arguments limited to
source-bound scenario IDs. A response fails if it narrates reasoning, omits a
terminal tool, mixes approval with findings, repeats a finding tool, or supplies
arguments outside the canonical scenario-ID schema.

If a multi-label judgment suppresses later findings, the admissible fallback is
an atomic inference ensemble: one strict-schema call per finding definition over
the same immutable full-set view. Each detector returns only affected scenario
IDs or an empty array. Host code merges the typed detector outputs but never
infers semantic defects. This directly tests whether first-defect salience, rather
than model understanding, caused the monolithic judgment miss.

Only intrinsically local quality claims use pairwise atomic verification:
language/encoding corruption, incomplete or malformed fields, and unresolved
placeholder/meta content. Mode alignment, field-role validity, plausibility, and
semantic duplicate collapse remain full-set judgments because decomposing those
relational claims per candidate creates false positives.
