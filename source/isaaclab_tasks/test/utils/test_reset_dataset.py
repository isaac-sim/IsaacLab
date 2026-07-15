# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-agnostic reset-dataset utilities."""

from copy import deepcopy

import pytest
import torch

from isaaclab_tasks.utils.reset_dataset import (
    reset_dataset_collect_batches,
    reset_dataset_content_digest,
    reset_dataset_digest,
    reset_dataset_save_atomic,
    reset_dataset_validate_header,
)

_FORMAT = "example_reset_states"
_SCHEMA_VERSION = 3
_CONTRACT = {"generator": {"seed": 7, "workspace": (-1.0, 1.0)}}


def _payload() -> dict:
    payload = {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "contract_sha256": reset_dataset_digest(_CONTRACT),
        "metadata": {"state_count": 2},
        "states": {
            "position": torch.tensor(((0.0, 1.0), (2.0, 3.0)), dtype=torch.float32),
            "category": torch.tensor((0, 1), dtype=torch.int64),
        },
    }
    payload["content_sha256"] = reset_dataset_content_digest(payload)
    return payload


def _validate(payload: dict) -> None:
    reset_dataset_validate_header(
        payload,
        expected_format=_FORMAT,
        expected_schema_version=_SCHEMA_VERSION,
        expected_contract=_CONTRACT,
    )


def test_reset_dataset_digest_is_stable_over_mapping_order_and_tensor_device_metadata():
    first = {
        "nested": [{"enabled": True, "gain": 0.25}, None],
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    second = {
        "tensor": first["tensor"].clone(),
        "nested": [{"gain": 0.25, "enabled": True}, None],
    }

    assert reset_dataset_digest(first) == reset_dataset_digest(second)


@pytest.mark.parametrize(
    "changed",
    [
        {"value": True},
        {"value": [1, 2]},
        {"value": torch.tensor((1, 2), dtype=torch.int32)},
        {"value": torch.tensor(((1, 2),), dtype=torch.int64)},
    ],
)
def test_reset_dataset_digest_preserves_type_shape_and_dtype_boundaries(changed):
    baseline = {"value": (1, 2)}

    assert reset_dataset_digest(changed) != reset_dataset_digest(baseline)


def test_reset_dataset_digest_supports_scalar_and_bfloat16_tensors():
    payload = {
        "scalar": torch.tensor(2.0, dtype=torch.float64),
        "bfloat16": torch.tensor((1.0, 2.0), dtype=torch.bfloat16),
    }

    assert len(reset_dataset_digest(payload)) == 64


def test_reset_dataset_content_digest_excludes_only_its_own_top_level_field():
    payload = _payload()
    original = reset_dataset_content_digest(payload)
    payload["content_sha256"] = "not-the-content-digest"

    assert reset_dataset_content_digest(payload) == original

    payload["metadata"]["content_sha256"] = "nested-data-remains-content"
    assert reset_dataset_content_digest(payload) != original


def test_reset_dataset_validate_header_returns_common_mappings():
    payload = _payload()

    metadata, states = reset_dataset_validate_header(
        payload,
        expected_format=_FORMAT,
        expected_schema_version=_SCHEMA_VERSION,
        expected_contract=_CONTRACT,
    )

    assert metadata is payload["metadata"]
    assert states is payload["states"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(format="other"), "format"),
        (lambda payload: payload.update(schema_version=4), "schema version"),
        (lambda payload: payload.update(metadata=[]), "metadata"),
        (lambda payload: payload.update(states=[]), "states"),
        (lambda payload: payload.update(contract_sha256="wrong"), "contract digest"),
        (lambda payload: payload["states"]["position"].add_(1.0), "content digest"),
    ],
)
def test_reset_dataset_validate_header_rejects_invalid_envelopes(mutation, message):
    payload = _payload()
    mutation(payload)

    with pytest.raises((TypeError, ValueError), match=message):
        _validate(payload)


def test_reset_dataset_save_atomic_validates_and_round_trips(tmp_path):
    payload = _payload()
    calls = []

    def validator(candidate):
        calls.append(candidate)
        _validate(candidate)

    output_path = tmp_path / "nested" / "states.pt"
    saved_path = reset_dataset_save_atomic(payload, output_path, validator=validator)
    restored = torch.load(saved_path, map_location="cpu", weights_only=False)

    assert calls == [payload]
    assert saved_path == output_path.resolve()
    assert reset_dataset_content_digest(restored) == payload["content_sha256"]
    assert not list(output_path.parent.glob(".*.tmp"))


def test_reset_dataset_save_atomic_does_not_replace_destination_when_validation_fails(tmp_path):
    destination = tmp_path / "states.pt"
    destination.write_bytes(b"existing")

    with pytest.raises(ValueError, match="invalid dataset"):
        reset_dataset_save_atomic(
            _payload(),
            destination,
            validator=lambda _payload: (_ for _ in ()).throw(ValueError("invalid dataset")),
        )

    assert destination.read_bytes() == b"existing"


def test_reset_dataset_collect_batches_trims_the_final_accepted_batch():
    evaluated_ranges = []

    def evaluate(candidate_ids: range) -> list[int]:
        evaluated_ranges.append(candidate_ids)
        return [candidate_id for candidate_id in candidate_ids if candidate_id % 2 == 0]

    batches, evaluated_count = reset_dataset_collect_batches(
        5,
        batch_size=4,
        max_candidate_count=12,
        evaluate_batch=evaluate,
        batch_count=len,
        batch_slice=lambda batch, count: batch[:count],
    )

    assert batches == [[0, 2], [4, 6], [8]]
    assert evaluated_ranges == [range(0, 4), range(4, 8), range(8, 12)]
    assert evaluated_count == 12


def test_reset_dataset_collect_batches_reports_candidate_budget_exhaustion():
    with pytest.raises(RuntimeError, match="accepted 2/3.*6 candidates"):
        reset_dataset_collect_batches(
            3,
            batch_size=2,
            max_candidate_count=6,
            evaluate_batch=lambda candidate_ids: [value for value in candidate_ids if value % 5 == 0],
            batch_count=len,
            batch_slice=lambda batch, count: batch[:count],
        )


def test_reset_dataset_collect_batches_rejects_impossible_callback_counts():
    with pytest.raises(ValueError, match="outside the candidate range"):
        reset_dataset_collect_batches(
            1,
            batch_size=2,
            max_candidate_count=2,
            evaluate_batch=lambda _candidate_ids: "invalid",
            batch_count=lambda _batch: 3,
            batch_slice=lambda batch, _count: batch,
        )


def test_reset_dataset_content_tampering_does_not_mutate_fixture():
    payload = _payload()
    modified = deepcopy(payload)
    modified["states"]["position"][0, 0] = 9.0

    assert payload["states"]["position"][0, 0] == 0.0
    assert reset_dataset_content_digest(modified) != payload["content_sha256"]
