"""Export only certified V7 paired records as delayed EvidenceTrainingFrame examples.

The exporter is deliberately strict.  It rejects V3-shaped data, incomplete
counterfactual groups, non-identical replay prefixes, stale/noncausal evidence,
and any target without independent semantic/audio admission.  Target wording is
not carried into the output control/evidence inputs; it remains represented by
the approved target audio and label hash only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ground_truth_finetuning.training.contracts import (  # noqa: E402
    ContractError,
    assert_evidence_control_alignment,
    sha256_uri,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)


V7_SCHEMA = "voxrn.synthetic-conversation.v4"
OUTPUT_SCHEMA = "personaplex.v7.evidence-training-example.v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def is_shared_prefix_context(record: dict[str, Any]) -> bool:
    return (record.get("replay") or {}).get("role") == "shared_prefix_context_only"


def read_jsonl(paths: Iterable[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for source in paths:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if record.get("schema") != V7_SCHEMA:
                    continue
                group_id = record.get("counterfactualGroupId")
                branch_id = record.get("counterfactualBranchId")
                if not isinstance(group_id, str) or not group_id or not isinstance(branch_id, str) or not branch_id:
                    raise ValueError(f"{source}:{line_number}: V7 record is missing counterfactual lineage")
                record["_source_path"] = source
                groups[group_id][branch_id].append(record)
    return groups


def split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def target_ordinal_start(records: list[dict[str, Any]]) -> int:
    pivots = {record.get("counterfactualPivotTargetOrdinal") for record in records}
    if len(pivots) != 1 or not isinstance(next(iter(pivots)), int):
        raise ValueError("branch has inconsistent or missing pivot target ordinal")
    pivot_ordinal = next(iter(pivots))
    if pivot_ordinal < 1:
        raise ValueError("counterfactual pivot target ordinal must be positive")
    return pivot_ordinal * 2 - 1


def validate_replay_prefix(branches: dict[str, list[dict[str, Any]]]) -> None:
    if len(branches) != 2:
        raise ValueError("counterfactual group must contain exactly two branches")
    ordered = {branch: sorted(records, key=lambda row: int(row["turnIndex"])) for branch, records in branches.items()}
    starts = {branch: target_ordinal_start(records) for branch, records in ordered.items()}
    if len(set(starts.values())) != 1:
        raise ValueError("counterfactual branches disagree on pivot")
    pivot_start = next(iter(starts.values()))
    prefix_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for records in ordered.values():
        for record in records:
            if int(record["turnIndex"]) < pivot_start:
                prefix_by_turn[int(record["turnIndex"])].append(record)
    if set(prefix_by_turn) != set(range(pivot_start)):
        raise ValueError("counterfactual group has an incomplete shared prefix")
    for turn_index, records in prefix_by_turn.items():
        if len(records) != 2:
            raise ValueError(f"shared prefix turn {turn_index} does not have two branches")
        first, second = records
        for key in ("audioSha256", "text", "heardText", "control"):
            if first.get(key) != second.get(key):
                raise ValueError(f"shared prefix turn {turn_index} differs in {key}")
        if canonical(first.get("timing")) != canonical(second.get("timing")):
            raise ValueError(f"shared prefix turn {turn_index} differs in timing")
        replayed = [record for record in records if record.get("replay", {}).get("role") == "shared_prefix_context_only"]
        if len(replayed) != 1:
            raise ValueError(f"shared prefix turn {turn_index} lacks exactly one replay context record")
        replay_training = replayed[0].get("training") or {}
        if replay_training.get("eligible") or "shared_prefix_replay_context_only" not in replay_training.get("exclusionReasons", []):
            raise ValueError(f"shared prefix turn {turn_index} replay record is target-eligible")


def require_turn_admission(record: dict[str, Any]) -> None:
    quality = record.get("quality") or {}
    if not quality.get("accepted"):
        raise ValueError(f"turn {record.get('turnIndex')} failed audio/ASR admission")
    if record.get("speaker") == "caller":
        if (record.get("authenticity") or {}).get("status") != "batch_certified":
            raise ValueError(f"caller turn {record.get('turnIndex')} lacks independent authenticity certification")
        return
    if record.get("speaker") != "target":
        raise ValueError(f"unsupported speaker {record.get('speaker')!r}")
    if is_shared_prefix_context(record):
        training = record.get("training") or {}
        if training.get("eligible") or "shared_prefix_replay_context_only" not in training.get("exclusionReasons", []):
            raise ValueError(f"target turn {record.get('turnIndex')} is not quarantined replay context")
        return
    if not (record.get("training") or {}).get("eligible"):
        raise ValueError(f"target turn {record.get('turnIndex')} is not eligible")
    if (record.get("semanticAdherence") or {}).get("verificationStatus") != "batch_certified":
        raise ValueError(f"target turn {record.get('turnIndex')} lacks semantic certification")


def assert_record_frame_identity(record: dict[str, Any], control_frame: Any, evidence_frame: Any) -> None:
    """Reject a frame whose call identity differs from its enclosing V4 turn."""
    conversation_id = record.get("conversationId")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("target record has no conversation identity")
    if control_frame.conversation_id != conversation_id or evidence_frame.conversation_id != conversation_id:
        raise ValueError("target control/evidence frame belongs to a different conversation")


def evidence_examples(group_id: str, branches: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    validate_replay_prefix(branches)
    examples: list[dict[str, Any]] = []
    for branch_id, records in sorted(branches.items()):
        records.sort(key=lambda row: int(row["turnIndex"]))
        for record in records:
            require_turn_admission(record)
        for record in records:
            if record.get("speaker") != "target" or not (record.get("training") or {}).get("eligible"):
                continue
            control = record.get("control") or {}
            evidence_mapping = control.get("evidence")
            if not isinstance(evidence_mapping, dict):
                continue
            try:
                control_frame = validate_control_frame_mapping(control.get("frame") or {})
                evidence_frame = validate_evidence_frame_mapping(evidence_mapping)
                assert_record_frame_identity(record, control_frame, evidence_frame)
                assert_evidence_control_alignment(control_frame, evidence_frame)
            except ContractError as error:
                raise ValueError(f"branch {branch_id} target turn {record.get('turnIndex')}: {error}") from error
            counterfactual = evidence_frame.counterfactual
            if counterfactual.get("groupId") != group_id or counterfactual.get("branchId") != branch_id:
                raise ValueError(f"branch {branch_id} target turn {record.get('turnIndex')} has mismatched evidence lineage")
            source_path = record["_source_path"]
            audio_path = source_path.parent / str(record.get("audioPath") or "")
            if not record.get("audioPath") or not audio_path.is_file() or not record.get("audioSha256"):
                raise ValueError(f"branch {branch_id} target turn {record.get('turnIndex')} has no verified target audio")
            timeline_path = source_path.parent / str(record.get("duplexTimelinePath") or "")
            if not record.get("duplexTimelinePath") or not timeline_path.is_file():
                raise ValueError(f"branch {branch_id} target turn {record.get('turnIndex')} has no duplex timeline")
            lineage = {
                "group": group_id,
                "branch": branch_id,
                "conversation": record.get("conversationId"),
                "turn": record.get("turnIndex"),
                "control": control_frame.frame_hash,
                "evidence": evidence_frame.evidence_hash,
            }
            examples.append({
                "schema": OUTPUT_SCHEMA,
                "exampleId": sha256_uri(lineage),
                "split": split_for_group(group_id),
                "counterfactual": {
                    "groupId": group_id,
                    "branchId": branch_id,
                    "changedField": counterfactual.get("changedField"),
                    "pivotTargetOrdinal": record.get("counterfactualPivotTargetOrdinal"),
                },
                "conversation": {
                    "id": record.get("conversationId"),
                    "targetTurnIndex": record.get("turnIndex"),
                    "duplexTimelinePath": str(timeline_path),
                    "timing": record.get("timing"),
                },
                "controlFrame": control_frame.as_wire_dict(),
                "controlFrameHash": control_frame.frame_hash,
                "evidenceFrame": evidence_frame.as_wire_dict(),
                "evidenceFrameHash": evidence_frame.evidence_hash,
                "labels": {
                    "targetAudioPath": str(audio_path),
                    "targetAudioSha256": record.get("audioSha256"),
                    "targetLabelSha256": control.get("targetLabelSha256"),
                    "agentHeardTextSha256": sha256_uri(record.get("heardText") or ""),
                },
                "provenance": {
                    "kind": "synthetic_controlled",
                    "sourceManifest": str(source_path),
                    "voiceReferenceId": (record.get("voiceReference") or {}).get("id"),
                    "certificateStatus": (record.get("semanticAdherence") or {}).get("verificationStatus"),
                },
            })
    if not examples:
        raise ValueError("counterfactual group has no evidence-aligned eligible target turns")
    return examples


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Certified V7 paired JSONL bundle(s)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    groups = read_jsonl(args.inputs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for group_id, branches in sorted(groups.items()):
        try:
            examples.extend(evidence_examples(group_id, branches))
        except ValueError as error:
            rejected.append({"groupId": group_id, "reason": str(error)})

    examples_path = output_dir / "evidence_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(canonical(example) + "\n")
    manifest = {
        "schema": "personaplex.v7.evidence-export.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceFiles": [str(path.resolve()) for path in args.inputs],
        "counterfactualGroupsSeen": len(groups),
        "counterfactualGroupsAccepted": len({example["counterfactual"]["groupId"] for example in examples}),
        "examples": len(examples),
        "rejected": rejected,
        "targetTextInControlOrEvidence": False,
        "nextStep": "Encode accepted duplex timelines with the pinned PersonaPlex codec and build agent-only delayed-code masks.",
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "rejections.json", rejected)
    print(json.dumps(manifest, ensure_ascii=True))
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
