from __future__ import annotations

from hashlib import sha256
import json

import pytest

from ground_truth_finetuning.tools.pack_native_causal_groups import main
from ground_truth_finetuning.training.causal_group_pack import (
    CAUSAL_GROUP_ROLES,
    CausalGroupPackError,
    PackConfig,
    build_leakage_components,
    content_hash,
    normalize_causal_group,
    pack_causal_groups,
)


def digest(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


def file_digest(path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def write_trainer_binding_inputs(tmp_path, result):
    trainer_manifest = tmp_path / "native_moshirag_groups_v2.jsonl"
    rows = [
        {
            "schema": "personaplex.native-moshirag-group.v2-shared-prefix",
            "group_id": item["groupId"],
            "leakage_component_id": item["componentId"],
            "split": item["split"],
        }
        for item in result.listwise_groups
    ]
    trainer_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    model_contract = tmp_path / "model_contract.json"
    model_contract.write_text(json.dumps({"model_revision": "model-v1"}) + "\n")
    data_contract = tmp_path / "native_moshirag_dataset_v2.json"
    data_contract.write_text(
        json.dumps(
            {
                "schema": "personaplex.native-moshirag-dataset.v2-shared-prefix",
                "status": "certified_for_native_moshirag_full_rank_training",
                "manifest_sha256": file_digest(trainer_manifest),
                "model_revision": "model-v1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return data_contract, trainer_manifest, model_contract


def make_group(
    index: int,
    *,
    premise: str | None = None,
    template: str | None = None,
    operator: str | None = None,
    family: str = "semantic",
    changed_paths: tuple[str, ...] = ("state.facts",),
    voices: tuple[str, str] | None = None,
) -> dict:
    premise = premise or f"premise-{index}"
    template = template or f"template-{index}"
    operator = operator or f"operator-{index}"
    voices = voices or (f"caller-{index}", f"agent-{index}")
    common = {
        "audio": {"path": f"audio/common-{index}.wav", "sha256": digest(f"common-{index}")},
        "context": {"premiseId": premise, "priorTurns": [f"prior context {index}"]},
    }
    siblings = []
    for revision, role in enumerate(CAUSAL_GROUP_ROLES, start=1):
        siblings.append(
            {
                "role": role,
                "exampleId": f"group-{index}:{role}",
                "commonInput": common,
                "alignment": {
                    "nativePivotFrame": 120 + index,
                    "memberAtFrame": 120 + index,
                    "donorAtFrame": 120 + index,
                    "controlAvailableFrame": 119 + index,
                    "targetStartFrame": 120 + index,
                },
                "controlInput": {
                    "revision": revision,
                    "evidenceStatus": role,
                    "nextGoal": f"respond according to state {revision}",
                },
                "target": {
                    "text": f"Natural branch {revision} for premise number {index}.",
                    "audio": {
                        "path": f"audio/group-{index}-{role}.wav",
                        "sha256": digest(f"target-{index}-{role}"),
                    },
                },
            }
        )
    return {
        "schema": "personaplex.native-causal-group.v5",
        "groupId": f"causal-group-{index}",
        "premiseId": premise,
        "lineageIdentifiers": [f"topic-{index}", premise, f"trajectory-{index}"],
        "templateId": template,
        "controlOperator": {
            "id": operator,
            "family": family,
            "changedPaths": list(changed_paths),
        },
        "voicePair": {"caller": voices[0], "agent": voices[1]},
        "commonInput": common,
        "siblings": siblings,
    }


def certified_groups() -> list[dict]:
    return [make_group(index) for index in range(3)]


def test_packs_common_input_once_and_pairwise_edges_are_diagnostic_only() -> None:
    result = pack_causal_groups(certified_groups())
    assert result.certificate["status"] == "certified"
    assert result.certificate["splitCounts"] == {"train": 1, "validation": 1, "test": 1}
    assert len(result.common_inputs) == 3
    assert len(result.listwise_groups) == 3
    assert len(result.pairwise_diagnostics) == 18
    for group in result.listwise_groups:
        assert [sibling["role"] for sibling in group["siblings"]] == list(CAUSAL_GROUP_ROLES)
        assert "commonInput" not in group
        assert all("commonInput" not in sibling for sibling in group["siblings"])
        assert set(group["commonInputRef"]) == {"groupId", "commonInputId", "nativePivotFrame"}
        assert group["commonInputRef"]["nativePivotFrame"] >= 120
    assert all(edge["diagnosticOnly"] is True for edge in result.pairwise_diagnostics)


def test_union_find_closes_transitive_template_operator_and_voice_pair_leakage() -> None:
    first = make_group(0, template="shared-template")
    second = make_group(1, template="shared-template", voices=("shared-caller", "shared-agent"))
    third = make_group(2, voices=("shared-caller", "shared-agent"))
    fourth = make_group(3)
    normalized = [normalize_causal_group(group) for group in (first, second, third, fourth)]
    components = build_leakage_components(normalized)
    member_sets = {frozenset(component["groupIds"]) for component in components}
    assert frozenset({"causal-group-0", "causal-group-1", "causal-group-2"}) in member_sets
    assert frozenset({"causal-group-3"}) in member_sets


def test_split_assignment_is_deterministic_and_never_splits_a_component() -> None:
    groups = certified_groups() + [
        make_group(3, template="shared-template"),
        make_group(4, template="shared-template"),
    ]
    forward = pack_causal_groups(groups)
    reverse = pack_causal_groups(list(reversed(groups)))
    forward_splits = {group["groupId"]: group["split"] for group in forward.listwise_groups}
    reverse_splits = {group["groupId"]: group["split"] for group in reverse.listwise_groups}
    assert forward_splits == reverse_splits
    assert forward_splits["causal-group-3"] == forward_splits["causal-group-4"]


def test_rejects_invalid_roles_and_exact_target_leakage() -> None:
    invalid_roles = make_group(0)
    invalid_roles["siblings"][-1]["role"] = "verified_positive"
    with pytest.raises(CausalGroupPackError, match="sibling role"):
        normalize_causal_group(invalid_roles)

    leaked = make_group(1)
    target = leaked["siblings"][0]["target"]["text"]
    leaked["siblings"][0]["controlInput"]["hiddenVerbatim"] = f"Say exactly: {target}"
    with pytest.raises(CausalGroupPackError, match="target text leaked"):
        normalize_causal_group(leaked)


def test_rejects_stale_donor_member_alignment_instead_of_copying_it() -> None:
    stale = make_group(0)
    stale["siblings"][2]["alignment"]["memberAtFrame"] += 4
    with pytest.raises(CausalGroupPackError, match="conflicting native pivot frames"):
        normalize_causal_group(stale)

    stale_donor = make_group(1)
    stale_donor["siblings"][1]["alignment"]["donorAtFrame"] += 3
    with pytest.raises(CausalGroupPackError, match="donor/member pivot mismatch"):
        normalize_causal_group(stale_donor)


def test_rejects_causal_coverage_that_does_not_repeat_by_premise_and_split() -> None:
    groups = [
        make_group(0, changed_paths=("state.facts",)),
        make_group(1, changed_paths=("plan.delivery",)),
        make_group(2, changed_paths=("turnTaking.eventType",)),
    ]
    result = pack_causal_groups(groups)
    assert result.certificate["status"] == "rejected"
    assert any("changed-path signature" in reason for reason in result.certificate["reasons"])


def test_cli_writes_input_hashed_immutable_manifest(tmp_path) -> None:
    source = tmp_path / "groups.json"
    payload = json.dumps({"groups": certified_groups()}, indent=2, sort_keys=True) + "\n"
    source.write_text(payload, encoding="utf-8")
    result = pack_causal_groups(certified_groups())
    data_contract, trainer_manifest, model_contract = write_trainer_binding_inputs(
        tmp_path, result
    )
    output = tmp_path / "packed"
    arguments = [
        "--input",
        str(source),
        "--output-dir",
        str(output),
        "--trainer-data-contract",
        str(data_contract),
        "--trainer-group-manifest",
        str(trainer_manifest),
        "--model-contract",
        str(model_contract),
    ]
    assert main(arguments) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "certified"
    assert manifest["inputs"][0]["sha256"] == digest(payload)
    assert manifest["counts"] == {
        "groups": 3,
        "siblings": 12,
        "components": 3,
        "pairwiseDiagnostics": 18,
    }
    certificate = json.loads(
        (output / "causal_coverage_certificate.json").read_text(encoding="utf-8")
    )
    binding = manifest["trainerBinding"]
    assert binding["groupManifest"]["sha256"] == file_digest(trainer_manifest)
    assert binding["datasetContract"]["sha256"] == file_digest(data_contract)
    assert binding["modelContract"]["sha256"] == file_digest(model_contract)
    assert binding["coverageCertificateSha256"] == file_digest(
        output / "causal_coverage_certificate.json"
    )
    assert binding["splitAssignmentHash"] == certificate["splitAssignmentHash"]
    assert binding["sourceGroupInputsHash"] == content_hash(manifest["inputs"])
    assert main(arguments) == 0


def test_cli_rejects_trainer_projection_that_differs_from_pack(tmp_path) -> None:
    groups = certified_groups()
    source = tmp_path / "groups.json"
    source.write_text(json.dumps({"groups": groups}) + "\n", encoding="utf-8")
    result = pack_causal_groups(groups)
    data_contract, trainer_manifest, model_contract = write_trainer_binding_inputs(
        tmp_path, result
    )
    rows = [json.loads(line) for line in trainer_manifest.read_text().splitlines()]
    rows[0]["split"] = "validation" if rows[0]["split"] != "validation" else "train"
    trainer_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    data = json.loads(data_contract.read_text())
    data["manifest_sha256"] = file_digest(trainer_manifest)
    data_contract.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "rejected-pack"
    assert main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--trainer-data-contract",
            str(data_contract),
            "--trainer-group-manifest",
            str(trainer_manifest),
            "--model-contract",
            str(model_contract),
        ]
    ) == 2
    assert not output.exists()


def test_required_coverage_splits_are_configurable_but_remain_explicit() -> None:
    result = pack_causal_groups(
        [make_group(0), make_group(1)],
        PackConfig(required_coverage_splits=("train", "validation")),
    )
    assert result.certificate["status"] == "certified"
    assert result.certificate["coveragePolicy"]["requiredSplits"] == ["train", "validation"]
