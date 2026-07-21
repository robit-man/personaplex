import json
from pathlib import Path

import pytest
import torch

from ground_truth_finetuning.tools.train_native_moshirag_control import (
    CHECKPOINT_GATES,
    BRANCH_ALIGNMENT_SCHEMA,
    DATASET_SCHEMA,
    FRAME_DURATION_MS,
    GROUP_SCHEMA,
    SHARED_PREFIX_SCHEMA,
    NativeDatasetContract,
    NativeTensorLoader,
    ObjectiveWeights,
    TrainerContractError,
    build_group_batch,
    checkpoint_summary_record,
    compact_step_record,
    compose_causal_objective,
    deterministic_dropout_mask,
    hash_file,
    load_group_manifest,
    load_native_group,
    parse_allowed_physical_gpus,
    parse_meminfo,
    validate_worker_devices,
    verify_certified_pack,
    visible_physical_gpus,
)
from ground_truth_finetuning.training.causal_group_pack import (
    CERTIFICATE_SCHEMA,
    MANIFEST_SCHEMA,
    PACK_SCHEMA,
    TRAINER_BINDING_SCHEMA,
    content_hash,
)
from ground_truth_finetuning.training.native_moshirag_control import (
    NATIVE_MOSHIRAG_CONTROL_SCHEMA,
)


ROLES = ("verified_positive", "verified_negative", "uncertain", "superseded")


def tensor_ref(path: Path, key: str, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "key": key,
        "sha256": hash_file(path),
    }


def write_group_tensors(root: Path, namespace: str, shared_prefix_ref: dict) -> list[dict]:
    siblings = []
    for index, role in enumerate(ROLES):
        path = root / f"{namespace}-{index}.pt"
        suffix_frames = 4 + index
        codes = torch.arange(3 * suffix_frames, dtype=torch.long).reshape(3, suffix_frames) + index
        mask = torch.zeros_like(codes, dtype=torch.bool)
        mask[0, 1:3] = True
        mask[1, 1:3] = True
        control = torch.full((2 + index, 4), float(index + 1))
        torch.save({"suffix": codes, "suffix_mask": mask, "control": control}, path)
        suffix_ref = tensor_ref(path, "suffix", root)
        suffix_mask_ref = tensor_ref(path, "suffix_mask", root)
        control_ref = tensor_ref(path, "control", root)
        siblings.append(
            {
                "sibling_id": f"{namespace}-sibling-{index}",
                "control_role": role,
                "generation_id": f"generation-{namespace}-{index}",
                "control_revision": index + 1,
                "acknowledged_control_revision": index + 1,
                "probe_frame_index": 4,
                "probe_targets": {"evidence_status": index, "posture": index % 2},
                "native_suffix_codes": suffix_ref,
                "suffix_agent_target_mask": suffix_mask_ref,
                "control_stream": control_ref,
                "alignment": {
                    "schema": BRANCH_ALIGNMENT_SCHEMA,
                    "alignment_revision": index + 1,
                    "shared_prefix_sha256": shared_prefix_ref["sha256"],
                    "native_suffix_sha256": suffix_ref["sha256"],
                    "target_mask_sha256": suffix_mask_ref["sha256"],
                    "member_at_frame": 4,
                    "donor_at_frame": 4,
                    "suffix_start_frame": 4,
                    "suffix_end_frame": 4 + suffix_frames,
                    "control_available_frame": 3,
                    "control_active_frame": 3,
                    "retrieval_buffer_frames": 1,
                    "first_supervised_agent_frame": 5,
                    "cutoff_frame": None,
                    "cutoff_revision": None,
                    "cutoff_generation_id": None,
                },
            }
        )
    return siblings


def dataset_contract_mapping(manifest: Path, model_revision: str = "model-v1") -> dict:
    return {
        "schema": DATASET_SCHEMA,
        "status": "certified_for_native_moshirag_full_rank_training",
        "manifest_sha256": hash_file(manifest),
        "model_revision": model_revision,
        "native_control_schema": NATIVE_MOSHIRAG_CONTROL_SCHEMA,
        "sibling_count": 4,
        "sibling_roles": list(ROLES),
        "frame_duration_ms": FRAME_DURATION_MS,
        "num_codebooks": 3,
        "control_hidden_size": 4,
        "padding_token_id": 0,
        "stream_layout": {
            "text_stream_indices": [0],
            "agent_audio_stream_indices": [1],
            "caller_audio_stream_indices": [2],
        },
        "probe_slot_cardinalities": {"evidence_status": 4, "posture": 2},
        "split_policy": "group_and_leakage_component_disjoint",
        "packing": "one_shared_native_prefix_plus_branch_native_suffix",
    }


def write_manifest(root: Path, *, cross_split_component: bool = False) -> Path:
    records = []
    for split in ("train", "validation", "test"):
        prefix_path = root / f"{split}-shared-prefix.pt"
        prefix = torch.arange(12, dtype=torch.long).reshape(3, 4)
        torch.save({"shared_prefix": prefix}, prefix_path)
        prefix_ref = tensor_ref(prefix_path, "shared_prefix", root)
        records.append(
            {
                "schema": GROUP_SCHEMA,
                "group_id": f"group-{split}",
                "leakage_component_id": "shared" if cross_split_component else f"component-{split}",
                "split": split,
                "shared_prefix": {
                    "schema": SHARED_PREFIX_SCHEMA,
                    "common_input_id": f"common-{split}",
                    "native_pivot_frame": 4,
                    "window_start_frame": 0,
                    "window_end_frame": 4,
                    "native_codes": prefix_ref,
                },
                "siblings": write_group_tensors(root, split, prefix_ref),
            }
        )
    manifest = root / "groups.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def load_fixture(root: Path):
    manifest = write_manifest(root)
    contract = NativeDatasetContract.from_mapping(
        dataset_contract_mapping(manifest),
        manifest_sha256=hash_file(manifest),
        model_revision="model-v1",
    )
    groups = load_group_manifest(manifest, data_root=root, contract=contract)
    return contract, groups


def descriptor(path: Path, *, group_records: int | None = None) -> dict:
    value = {
        "path": str(path.resolve()),
        "sha256": hash_file(path),
        "sizeBytes": path.stat().st_size,
    }
    if group_records is not None:
        value["groupRecords"] = group_records
    return value


def write_certified_pack(
    root: Path,
    *,
    group_manifest: Path,
    data_contract: Path,
    model_contract: Path,
    split_override: tuple[str, str] | None = None,
) -> Path:
    pack_root = root / "certified-pack"
    pack_root.mkdir()
    trainer_rows = [json.loads(line) for line in group_manifest.read_text().splitlines()]
    assignments = [
        {
            "groupId": row["group_id"],
            "componentId": row["leakage_component_id"],
            "split": row["split"],
        }
        for row in trainer_rows
    ]
    if split_override is not None:
        group_id, split = split_override
        next(item for item in assignments if item["groupId"] == group_id)["split"] = split
    assignments.sort(key=lambda item: item["groupId"])
    split_hash = content_hash(assignments)
    component_splits = {item["componentId"]: item["split"] for item in assignments}
    split_counts = {
        split: sum(item["split"] == split for item in assignments)
        for split in ("train", "validation", "test")
    }
    listwise = [
        {
            "schema": f"{PACK_SCHEMA}.listwise-index",
            "groupId": item["groupId"],
            "split": item["split"],
            "componentId": item["componentId"],
            "premiseId": f"premise-{item['groupId']}",
            "lineageIdentifiers": [f"lineage-{item['groupId']}"],
            "templateId": f"template-{item['groupId']}",
            "controlOperator": {
                "id": f"operator-{item['groupId']}",
                "family": "semantic",
                "changedPaths": ["state.facts"],
            },
            "voicePair": {
                "id": f"voice-{item['groupId']}",
                "caller": "a",
                "agent": "b",
            },
            "commonInputRef": {
                "groupId": item["groupId"],
                "commonInputId": f"common-{item['groupId']}",
                "nativePivotFrame": 4,
            },
            "siblings": [{"role": role} for role in ROLES],
        }
        for item in assignments
    ]
    components = [
        {
            "componentId": item["componentId"],
            "groupIds": [item["groupId"]],
            "groupCount": 1,
            "leakageKeys": [f"lineage:{item['groupId']}"],
            "split": item["split"],
        }
        for item in assignments
    ]
    coverage = {
        "groups": len(assignments),
        "distinctPremises": len(assignments),
        "premiseIds": [f"premise-{item['groupId']}" for item in assignments],
        "splits": sorted(split_counts),
        "missingRequiredSplits": [],
        "accepted": True,
    }
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "status": "certified",
        "groupCount": len(assignments),
        "siblingCount": len(assignments) * 4,
        "componentCount": len(components),
        "splitCounts": split_counts,
        "componentSplits": component_splits,
        "splitAssignmentHash": split_hash,
        "requiredSiblingRoles": list(ROLES),
        "coveragePolicy": {
            "requiredSplits": ["train", "validation", "test"],
            "minimumDistinctPremises": 2,
        },
        "operatorFamilies": {"semantic": coverage},
        "changedPathSignatures": [
            {
                "signature": content_hash({"changedPaths": ["state.facts"]}),
                "changedPaths": ["state.facts"],
                **coverage,
            }
        ],
        "nativePivotAlignment": {
            "policy": "identical_native_pivot_frame_required",
            "certifiedGroups": len(assignments),
            "shiftedGroups": 0,
        },
        "reasons": [],
    }
    payloads = {
        "common_inputs.jsonl": "".join(
            json.dumps({"groupId": item["groupId"]}, sort_keys=True) + "\n"
            for item in assignments
        ),
        "listwise_groups.jsonl": "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in listwise
        ),
        "pairwise_diagnostics.jsonl": "".join(
            json.dumps({"groupId": item["groupId"], "edge": edge}) + "\n"
            for item in assignments
            for edge in range(6)
        ),
        "leakage_components.jsonl": "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in components
        ),
        "causal_coverage_certificate.json": json.dumps(
            certificate, indent=2, sort_keys=True
        )
        + "\n",
    }
    for name, payload in payloads.items():
        (pack_root / name).write_text(payload, encoding="utf-8")
    output_descriptors = [
        {
            "path": name,
            "sha256": hash_file(pack_root / name),
            "sizeBytes": (pack_root / name).stat().st_size,
        }
        for name in payloads
    ]
    source = root / "native_causal_groups_v5.jsonl"
    source.write_text(
        "".join(json.dumps({"groupId": item["groupId"]}) + "\n" for item in assignments),
        encoding="utf-8",
    )
    source_inputs = [{**descriptor(source), "groupRecords": len(assignments)}]
    binding = {
        "schema": TRAINER_BINDING_SCHEMA,
        "sourceGroupInputsHash": content_hash(source_inputs),
        "datasetContract": descriptor(data_contract),
        "groupManifest": descriptor(group_manifest, group_records=len(trainer_rows)),
        "modelContract": descriptor(model_contract),
        "modelRevision": "model-v1",
        "splitAssignmentHash": split_hash,
        "coverageCertificateSha256": hash_file(
            pack_root / "causal_coverage_certificate.json"
        ),
    }
    manifest_base = {
        "schema": MANIFEST_SCHEMA,
        "status": "certified",
        "inputs": source_inputs,
        "configuration": {
            "splitRatios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "splitSeed": "personaplex-native-causal-groups-v1",
            "requiredCoverageSplits": ["train", "validation", "test"],
            "minimumDistinctPremises": 2,
        },
        "counts": {
            "groups": len(assignments),
            "siblings": len(assignments) * 4,
            "components": len(components),
            "pairwiseDiagnostics": len(assignments) * 6,
        },
        "outputs": output_descriptors,
        "trainerBinding": binding,
    }
    manifest = {**manifest_base, "manifestId": content_hash(manifest_base)}
    manifest_path = pack_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def write_bound_contracts(root: Path, manifest: Path) -> tuple[Path, Path]:
    model_contract = root / "model-contract.json"
    model_contract.write_text(json.dumps({"model_revision": "model-v1"}) + "\n")
    data_contract = root / "dataset-contract.json"
    data_contract.write_text(
        json.dumps(dataset_contract_mapping(manifest), indent=2, sort_keys=True) + "\n"
    )
    return data_contract, model_contract


def test_group_loader_builds_complete_target_by_control_matrix(tmp_path: Path) -> None:
    contract, groups = load_fixture(tmp_path)
    loaded = load_native_group(
        groups[0], loader=NativeTensorLoader(tmp_path), contract=contract
    )
    batch = build_group_batch(loaded, torch.device("cpu"))
    assert batch.matrix_codes.shape == (16, 3, 11)
    assert batch.matrix_controls.shape == (16, 5, 4)
    assert batch.matched_codes.shape == (4, 3, 11)
    assert batch.diagonal_indices.tolist() == [0, 5, 10, 15]
    assert torch.equal(batch.matrix_codes[0, :, :4], batch.matrix_codes[15, :, :4])
    assert batch.matrix_codes[0, 0, 4].item() == 0
    assert batch.matrix_codes[4, 0, 4].item() == 1
    assert batch.matrix_controls[0, 0, 0].item() == 1
    assert batch.matrix_controls[1, 0, 0].item() == 2
    assert batch.matrix_controls[3, 4].eq(4).all()
    assert batch.probe_targets["evidence_status"].tolist() == [0, 1, 2, 3]


def test_manifest_fails_closed_for_absent_control_tensor(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0]["siblings"][0]["control_stream"]["path"] = "absent.pt"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    contract = NativeDatasetContract.from_mapping(
        dataset_contract_mapping(manifest),
        manifest_sha256=hash_file(manifest),
        model_revision="model-v1",
    )
    with pytest.raises(TrainerContractError, match="required native tensor is absent"):
        load_group_manifest(manifest, data_root=tmp_path, contract=contract)


def test_manifest_rejects_leakage_component_crossing_split(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, cross_split_component=True)
    contract = NativeDatasetContract.from_mapping(
        dataset_contract_mapping(manifest),
        manifest_sha256=hash_file(manifest),
        model_revision="model-v1",
    )
    with pytest.raises(TrainerContractError, match="crosses dataset splits"):
        load_group_manifest(manifest, data_root=tmp_path, contract=contract)


def test_certified_pack_gate_binds_manifest_certificate_sources_splits_and_model(
    tmp_path: Path,
) -> None:
    group_manifest = write_manifest(tmp_path)
    data_contract, model_contract = write_bound_contracts(tmp_path, group_manifest)
    pack_manifest = write_certified_pack(
        tmp_path,
        group_manifest=group_manifest,
        data_contract=data_contract,
        model_contract=model_contract,
    )
    proof = verify_certified_pack(
        pack_manifest,
        data_contract_path=data_contract,
        group_manifest_path=group_manifest,
        model_contract_path=model_contract,
    )
    assert proof.manifest_id.startswith("sha256:")
    assert proof.group_manifest_sha256 == hash_file(group_manifest)
    assert proof.dataset_contract_sha256 == hash_file(data_contract)
    assert proof.model_contract_sha256 == hash_file(model_contract)
    assert proof.coverage_certificate_sha256 == hash_file(
        pack_manifest.parent / "causal_coverage_certificate.json"
    )


def test_certified_pack_gate_rejects_fully_hashed_split_mismatch(tmp_path: Path) -> None:
    group_manifest = write_manifest(tmp_path)
    data_contract, model_contract = write_bound_contracts(tmp_path, group_manifest)
    pack_manifest = write_certified_pack(
        tmp_path,
        group_manifest=group_manifest,
        data_contract=data_contract,
        model_contract=model_contract,
        split_override=("group-train", "validation"),
    )
    with pytest.raises(TrainerContractError, match="split assignment does not match"):
        verify_certified_pack(
            pack_manifest,
            data_contract_path=data_contract,
            group_manifest_path=group_manifest,
            model_contract_path=model_contract,
        )


def test_certified_pack_gate_rejects_certificate_and_source_tampering(tmp_path: Path) -> None:
    group_manifest = write_manifest(tmp_path)
    data_contract, model_contract = write_bound_contracts(tmp_path, group_manifest)
    pack_manifest = write_certified_pack(
        tmp_path,
        group_manifest=group_manifest,
        data_contract=data_contract,
        model_contract=model_contract,
    )
    certificate = pack_manifest.parent / "causal_coverage_certificate.json"
    certificate.write_text(certificate.read_text() + "\n")
    with pytest.raises(
        TrainerContractError,
        match=r"certified pack output causal_coverage_certificate\.json (?:size|hash) mismatch",
    ):
        verify_certified_pack(
            pack_manifest,
            data_contract_path=data_contract,
            group_manifest_path=group_manifest,
            model_contract_path=model_contract,
        )

    other_root = tmp_path / "source-case"
    other_root.mkdir()
    group_manifest = write_manifest(other_root)
    data_contract, model_contract = write_bound_contracts(other_root, group_manifest)
    pack_manifest = write_certified_pack(
        other_root,
        group_manifest=group_manifest,
        data_contract=data_contract,
        model_contract=model_contract,
    )
    source = other_root / "native_causal_groups_v5.jsonl"
    source.write_text(source.read_text() + "{}\n")
    with pytest.raises(TrainerContractError, match="source input.*mismatch"):
        verify_certified_pack(
            pack_manifest,
            data_contract_path=data_contract,
            group_manifest_path=group_manifest,
            model_contract_path=model_contract,
        )


def test_native_group_rejects_control_that_arrives_at_response(tmp_path: Path) -> None:
    contract, groups = load_fixture(tmp_path)
    sibling = groups[0].siblings[0]
    object.__setattr__(sibling.alignment, "control_available_frame", 6)
    with pytest.raises(TrainerContractError, match="strictly before"):
        load_native_group(
            groups[0], loader=NativeTensorLoader(tmp_path), contract=contract
        )

    contract, groups = load_fixture(tmp_path)
    sibling = groups[0].siblings[0]
    object.__setattr__(sibling.alignment, "control_available_frame", 5)
    with pytest.raises(TrainerContractError, match="strictly before"):
        load_native_group(
            groups[0], loader=NativeTensorLoader(tmp_path), contract=contract
        )


def test_native_group_rejects_stale_alignment_and_cutoff_metadata(tmp_path: Path) -> None:
    contract, groups = load_fixture(tmp_path)
    sibling = groups[0].siblings[0]
    object.__setattr__(sibling.alignment, "member_at_frame", 3)
    with pytest.raises(TrainerContractError, match="alignment is stale"):
        load_native_group(
            groups[0], loader=NativeTensorLoader(tmp_path), contract=contract
        )

    contract, groups = load_fixture(tmp_path)
    sibling = groups[0].siblings[0]
    object.__setattr__(sibling.alignment, "cutoff_frame", 7)
    object.__setattr__(sibling.alignment, "cutoff_revision", sibling.control_revision)
    object.__setattr__(sibling.alignment, "cutoff_generation_id", sibling.generation_id)
    with pytest.raises(TrainerContractError, match="cropped exactly"):
        load_native_group(
            groups[0], loader=NativeTensorLoader(tmp_path), contract=contract
        )


def test_causal_objective_is_differentiable_and_rewards_diagonal() -> None:
    nll = torch.full((4, 4), 1.2, requires_grad=True)
    with torch.no_grad():
        nll.diagonal().fill_(0.2)
    probe = torch.tensor(0.4, requires_grad=True)
    dropout = torch.tensor(0.3, requires_grad=True)
    total, matched, listwise = compose_causal_objective(
        nll,
        probe,
        dropout,
        weights=ObjectiveWeights(1.0, 0.25, 0.1, 0.1),
        temperature=0.2,
    )
    assert matched.item() == pytest.approx(0.2)
    assert listwise.item() < 0.1
    total.backward()
    assert nll.grad is not None
    assert probe.grad is not None
    assert dropout.grad is not None


def test_dropout_is_deterministic_without_process_rng() -> None:
    ids = [f"sibling-{index}" for index in range(4)]
    first = deterministic_dropout_mask(
        ids,
        probability=0.5,
        seed=7,
        step=11,
        rank=2,
        micro_step=0,
        device=torch.device("cpu"),
    )
    second = deterministic_dropout_mask(
        ids,
        probability=0.5,
        seed=7,
        step=11,
        rank=2,
        micro_step=0,
        device=torch.device("cpu"),
    )
    assert torch.equal(first, second)


def test_host_ram_admission_uses_discovered_total_and_available() -> None:
    snapshot = parse_meminfo(
        "MemTotal:       1000 kB\nMemAvailable:    250 kB\nSwapTotal: 0 kB\n"
    )
    assert snapshot["total_bytes"] == 1000 * 1024
    assert snapshot["available_bytes"] == 250 * 1024
    assert snapshot["used_ratio"] == pytest.approx(0.75)


def test_worker_device_mapping_rejects_cpu_and_outside_physical_ceiling() -> None:
    assert visible_physical_gpus(
        {"CUDA_VISIBLE_DEVICES": "2,0,1"}, cuda_device_count=3
    ) == (2, 0, 1)
    assert validate_worker_devices(
        environ={"CUDA_VISIBLE_DEVICES": "2,0,1"},
        cuda_device_count=3,
        world_size=3,
        local_rank=1,
        allowed_physical_gpus=(0, 1, 2),
    ) == 0
    with pytest.raises(TrainerContractError, match="outside the allowlist"):
        validate_worker_devices(
            environ={"CUDA_VISIBLE_DEVICES": "0,1,3"},
            cuda_device_count=3,
            world_size=3,
            local_rank=0,
            allowed_physical_gpus=(0, 1, 2),
        )
    with pytest.raises(TrainerContractError, match="restricted"):
        parse_allowed_physical_gpus("0,1,4")


def test_checkpoint_telemetry_keeps_heldout_and_train_namespaces_separate() -> None:
    heldout = {"strictGroupPassRate": 0.96, "probeAccuracy": 0.97, "groups": 20}
    train = {"strictGroupPassRate": 0.99, "probeAccuracy": 1.0, "groups": 20}
    record = checkpoint_summary_record(
        step=150,
        checkpoint="checkpoint-step-000150",
        heldout=heldout,
        train=train,
        minimum_group_pass_rate=0.95,
        minimum_probe_accuracy=0.95,
    )
    assert CHECKPOINT_GATES == (100, 125, 150)
    assert record["heldout"] == heldout
    assert record["train"] == train
    assert record["teacherForcedGate"]["passed"] is True
    step = compact_step_record(
        step=1,
        reduced=[3.0, 2.0, 1.0, 0.9, 0.6, 1.5, 0.3, 120, 240, 3, 2.1],
        world_size=3,
        duration_seconds=2.0,
        host_ram_used_ratio=0.5,
    )
    assert step["loss"] == 1.0
    assert step["textTokens"] == 120
    assert step["globalShardedGradNorm"] == pytest.approx(0.7)
