#!/usr/bin/env python3
"""Create exact shared-prefix native tensors from metadata-matched causal pairs.

Each output branch receives one authentic donor history through the pivot and
retains its own target-and-later native suffix.  This removes independently
rendered timing drift while preserving both branch labels and control frames.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.native_training import exact_text_contrast_masks


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    tensor = value.get(name) if isinstance(value, dict) else value
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path} does not contain {name}")
    return tensor


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def save_tensor(path: Path, name: str, value: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.contiguous()}, path)
    return hash_file(path)


def pair_member(pair: dict[str, Any], name: str) -> dict[str, Any]:
    value = pair.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{pair.get('pair_id')}: missing {name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = args.source_manifest.resolve()
    source_root = args.source_artifact_root.resolve()
    model_contract_path = args.model_contract.resolve()
    model_contract = json.loads(model_contract_path.read_text(encoding="utf-8"))
    text_stream_indices = model_contract.get("stream_layout", {}).get(
        "text_stream_indices"
    )
    if not isinstance(text_stream_indices, list) or len(text_stream_indices) != 1:
        raise ValueError("model contract must define exactly one text stream")
    text_stream_index = int(text_stream_indices[0])
    zero_token_id = model_contract.get("zero_token_id")
    if not isinstance(zero_token_id, int):
        raise ValueError("model contract lacks an integer zero_token_id")
    output_model_revision = model_contract.get("model_revision")
    if not isinstance(output_model_revision, str) or not output_model_revision:
        raise ValueError("model contract lacks a model_revision")
    required_mimi_hash = model_contract.get("mimi_weights_sha256")
    required_tokenizer_hash = model_contract.get("tokenizer_sha256")
    if not isinstance(required_mimi_hash, str) or not isinstance(required_tokenizer_hash, str):
        raise ValueError("model contract lacks codec/tokenizer identities")
    rows = load_jsonl(source_manifest)
    by_id = {str(row["example_id"]): row for row in rows}
    candidates = load_jsonl(args.candidate_pairs.resolve())
    output_rows: list[dict[str, Any]] = []
    output_pairs: list[dict[str, Any]] = []
    stale_ids: set[str] = set()
    model_revisions: set[str] = set()
    codec_artifacts: set[str] = set()
    seen_new_ids: set[str] = set()
    rejected_pairs: list[dict[str, str]] = []
    for pair in candidates:
        member_a = pair_member(pair, "member_a")
        member_b = pair_member(pair, "member_b")
        members = [member_a, member_b]
        source_members = [by_id[str(member["example_id"])] for member in members]
        source_tensors = []
        for member, row in zip(members, source_members):
            encoding = row["model_encoding"]
            codec = encoding.get("codec", {})
            if (
                codec.get("mimi_weights_sha256") != required_mimi_hash
                or codec.get("tokenizer_sha256") != required_tokenizer_hash
            ):
                raise ValueError(
                    f"{member['example_id']}: native codes do not match model-contract codec"
                )
            codes = load_tensor(source_root / encoding["codes_path"], "codes")
            target_mask = load_tensor(
                source_root / encoding["target_mask_path"], "target_mask"
            ).bool()
            if codes.ndim != 2 or target_mask.shape != codes.shape:
                raise ValueError(f"{member['example_id']}: invalid native tensors")
            source_tensors.append((codes, target_mask))
        try:
            exact_text_contrast_masks(
                source_tensors[0][0],
                source_tensors[0][1],
                source_tensors[1][0],
                source_tensors[1][1],
                text_stream_index=text_stream_index,
                zero_token_id=zero_token_id,
            )
        except ValueError as exc:
            rejected_pairs.append(
                {
                    "pair_id": str(pair.get("pair_id")),
                    "reason": "non_bidirectional_exact_target_contrast",
                    "detail": str(exc),
                }
            )
            continue
        voices = [
            json.dumps(row.get("provenance", {}).get("voice_pair"), sort_keys=True)
            for row in source_members
        ]
        if len(set(voices)) != 1:
            raise ValueError(f"{pair['pair_id']}: branch voice pair differs")
        donor_position = min(
            range(2), key=lambda index: str(members[index]["example_id"])
        )
        donor_row = source_members[donor_position]
        donor_encoding = donor_row["model_encoding"]
        donor_at = int(members[donor_position].get("prefix_at", donor_encoding["prefix_at"]))
        donor_codes = load_tensor(source_root / donor_encoding["codes_path"], "codes")
        if donor_codes.ndim != 2 or not 1 <= donor_at < donor_codes.shape[1]:
            raise ValueError(f"{pair['pair_id']}: donor native boundary is invalid")
        shared_prefix = donor_codes[:, :donor_at].contiguous()
        shared_hash = tensor_hash(shared_prefix)
        transformed_members: list[dict[str, Any]] = []
        for member, row, (codes, target_mask) in zip(
            members, source_members, source_tensors
        ):
            encoding = row["model_encoding"]
            member_at = int(member.get("prefix_at", encoding["prefix_at"]))
            if not 1 <= member_at < codes.shape[1]:
                raise ValueError(f"{member['example_id']}: invalid branch boundary")
            if target_mask[:, :member_at].any():
                raise ValueError(f"{member['example_id']}: target supervision exists before pivot")
            if not target_mask[:, member_at:].any():
                raise ValueError(f"{member['example_id']}: target supervision is absent after pivot")
            canonical_codes = torch.cat((shared_prefix, codes[:, member_at:]), dim=1)
            canonical_mask = torch.cat(
                (
                    torch.zeros(
                        target_mask.shape[0],
                        donor_at,
                        dtype=torch.bool,
                    ),
                    target_mask[:, member_at:].bool(),
                ),
                dim=1,
            )
            identity = {
                "source_example_id": member["example_id"],
                "pair_id": pair["pair_id"],
                "shared_prefix": shared_hash,
                "strategy": "canonical-native-prefix-v4",
            }
            new_id = "sha256:" + sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if new_id in seen_new_ids:
                raise ValueError(f"duplicate canonical example ID: {new_id}")
            seen_new_ids.add(new_id)
            stem = new_id.removeprefix("sha256:")
            codes_path = Path("tensors") / f"{stem}.pt"
            mask_path = Path("tensors") / f"{stem}.mask.pt"
            codes_sha = save_tensor(output_root / codes_path, "codes", canonical_codes)
            mask_sha = save_tensor(output_root / mask_path, "target_mask", canonical_mask)
            source_alignment = source_root / encoding["text_alignment_path"]
            alignment_path = Path("alignments") / f"{stem}.json"
            (output_root / alignment_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_alignment, output_root / alignment_path)
            new_row = deepcopy(row)
            new_row["example_id"] = new_id
            source_model_revision = str(new_row["model_encoding"]["model_revision"])
            new_row["model_encoding"].update(
                {
                    "codes_path": str(codes_path),
                    "codes_sha256": codes_sha,
                    "target_mask_path": str(mask_path),
                    "target_mask_sha256": mask_sha,
                    "text_alignment_path": str(alignment_path),
                    "text_alignment_sha256": hash_file(output_root / alignment_path),
                    "prefix_at": donor_at,
                    "model_revision": output_model_revision,
                }
            )
            new_row.setdefault("provenance", {})["canonical_counterfactual_prefix"] = {
                "schema": "personaplex.canonical-counterfactual-prefix.v4",
                "source_example_id": member["example_id"],
                "pair_id": pair["pair_id"],
                "donor_source_example_id": members[donor_position]["example_id"],
                "source_prefix_at": member_at,
                "canonical_prefix_at": donor_at,
                "shared_prefix_sha256": shared_hash,
                "transformation": "authentic_donor_prefix_plus_branch_native_suffix",
                "source_model_revision": source_model_revision,
                "output_model_revision": output_model_revision,
            }
            output_rows.append(new_row)
            transformed = deepcopy(member)
            transformed["source_example_id"] = transformed["example_id"]
            transformed["example_id"] = new_id
            transformed["prefix_at"] = donor_at
            transformed_members.append(transformed)
            if member.get("stale_example_id"):
                stale_ids.add(str(member["stale_example_id"]))
            model_revisions.add(output_model_revision)
            codec_artifacts.add(
                json.dumps(
                    {
                        "mimi_weights_sha256": encoding["codec"]["mimi_weights_sha256"],
                        "tokenizer_sha256": encoding["codec"]["tokenizer_sha256"],
                    },
                    sort_keys=True,
                )
            )
        canonical_pair = deepcopy(pair)
        canonical_pair["member_a"], canonical_pair["member_b"] = transformed_members
        canonical_pair["prefix_at"] = donor_at
        canonical_pair["shared_prefix_sha256"] = shared_hash
        canonical_pair["canonicalization"] = {
            "schema": "personaplex.causal-prefix-canonicalization.v4",
            "donor_source_example_id": members[donor_position]["example_id"],
            "strategy": "authentic_donor_prefix_plus_branch_native_suffix",
        }
        output_pairs.append(canonical_pair)
    for stale_id in sorted(stale_ids):
        if stale_id not in by_id:
            continue
        row = deepcopy(by_id[stale_id])
        encoding = row["model_encoding"]
        source_model_revision = str(encoding["model_revision"])
        encoding["codes_path"] = str((source_root / encoding["codes_path"]).resolve())
        encoding["target_mask_path"] = str((source_root / encoding["target_mask_path"]).resolve())
        encoding["text_alignment_path"] = str(
            (source_root / encoding["text_alignment_path"]).resolve()
        )
        encoding["model_revision"] = output_model_revision
        row.setdefault("provenance", {})["training_role"] = "stale_control_negative_only"
        row["provenance"]["model_rebinding"] = {
            "source_model_revision": source_model_revision,
            "output_model_revision": output_model_revision,
            "tensor_transformation": "none",
        }
        output_rows.append(row)
    output_rows.sort(key=lambda row: str(row["example_id"]))
    output_pairs.sort(key=lambda pair: str(pair["pair_id"]))
    manifest_path = output_root / "encoded_examples.jsonl"
    pair_path = output_root / "causal_pairs.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    pair_path.write_text(
        "".join(json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n" for pair in output_pairs),
        encoding="utf-8",
    )
    splits = {
        name: sum(pair["split"] == name for pair in output_pairs)
        for name in ("train", "validation", "test")
    }
    corpus_certificate = {
        "schema_version": 4,
        "kind": "personaplex-corpus-certificate",
        "status": "certified_for_adapter_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hash_file(manifest_path),
        "model_revisions": sorted(model_revisions),
        "codec_artifacts": [json.loads(value) for value in sorted(codec_artifacts)],
        "examples": len(output_rows),
        "causal_pair_examples": len(output_pairs) * 2,
        "stale_control_only_examples": len(output_rows) - len(output_pairs) * 2,
        "derivation": {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": hash_file(source_manifest),
            "candidate_pairs": str(args.candidate_pairs.resolve()),
            "candidate_pairs_sha256": hash_file(args.candidate_pairs.resolve()),
            "model_contract": str(model_contract_path),
            "model_contract_sha256": hash_file(model_contract_path),
            "model_rebinding": "native labels are codec/tokenizer-bound; tensor bytes are unchanged by LM revision rebinding",
            "shared_prefix_strategy": "authentic_donor_prefix_plus_branch_native_suffix",
            "target_masks_before_boundary": "all_false",
            "exact_target_contrast": {
                "algorithm": "exact_dynamic_programming_longest_common_subsequence",
                "text_stream_index": text_stream_index,
                "zero_token_id": zero_token_id,
                "candidate_pairs": len(candidates),
                "accepted_pairs": len(output_pairs),
                "rejected_pairs": rejected_pairs,
            },
        },
    }
    pair_certificate = {
        "schema_version": 4,
        "kind": "personaplex-causal-control-pair-index-certificate",
        "status": "certified_for_causal_control_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": hash_file(manifest_path),
        "pair_index": str(pair_path),
        "pair_index_sha256": hash_file(pair_path),
        "pairs": len(output_pairs),
        "pairs_by_split": splits,
        "shared_prefix_tensor_verification": True,
        "canonicalized_native_prefixes": len(output_pairs),
        "split_leakage_groups": 0,
        "exact_bidirectional_target_contrast_verified": True,
        "exact_target_contrast_rejections": rejected_pairs,
    }
    (output_root / "corpus_certificate.json").write_text(
        json.dumps(corpus_certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "pair_certificate.json").write_text(
        json.dumps(pair_certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "certified_for_causal_control_training",
                "pairs": len(output_pairs),
                "pairs_by_split": splits,
                "examples": len(output_rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
