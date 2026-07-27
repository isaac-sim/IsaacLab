# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assertions for comparing articulation traces across joint orderings."""

from typing import Any

import torch

_ORDERING_TRACE_FIELDS = (
    "joint_pos",
    "joint_vel",
    "computed_torque",
    "applied_torque",
    "adapter_computed_effort",
    "adapter_applied_effort",
)
_ORDERING_TRACE_TOLERANCES = {
    "joint_pos": (2e-3, 1e-3),
    "joint_vel": (1e-2, 1e-2),
    "computed_torque": (1e-3, 1e-3),
    "applied_torque": (1e-3, 1e-3),
    "adapter_computed_effort": (1e-3, 1e-3),
    "adapter_applied_effort": (1e-3, 1e-3),
}


def assert_articulation_ordering_trace_matches(
    identity_result: dict[str, Any],
    reordered_result: dict[str, Any],
    requested_joint_names: tuple[str, ...],
) -> None:
    """Assert that two traces command and observe the same physical joints.

    Args:
        identity_result: Trace recorded with backend joint ordering.
        reordered_result: Trace recorded with an explicit public joint ordering.
        requested_joint_names: Explicit public joint-name ordering used for the reordered trace.
    """
    assert identity_result["joint_names"] == identity_result["backend_joint_names"]
    assert reordered_result["joint_names"] == requested_joint_names
    assert reordered_result["backend_joint_names"] == identity_result["backend_joint_names"]

    installed_ordering = reordered_result["joint_ordering"]
    assert installed_ordering is not None
    assert installed_ordering["user_names"] == requested_joint_names
    assert installed_ordering["backend_names"] == identity_result["backend_joint_names"]
    expected_user_to_backend = tuple(
        identity_result["backend_joint_names"].index(name) for name in requested_joint_names
    )
    expected_backend_to_user = tuple(
        requested_joint_names.index(name) for name in identity_result["backend_joint_names"]
    )
    assert installed_ordering["user_to_backend_indices"] == expected_user_to_backend
    assert installed_ordering["backend_to_user_indices"] == expected_backend_to_user

    canonical_joint_names = tuple(identity_result["backend_joint_names"])
    identity_result = _canonicalize_ordering_result(identity_result, canonical_joint_names)
    reordered_result = _canonicalize_ordering_result(reordered_result, canonical_joint_names)

    assert identity_result["joint_names"] == reordered_result["joint_names"] == canonical_joint_names
    for field_name in ("target_pos", "target_vel", "effort_target"):
        identity_values = identity_result[field_name]
        reordered_values = reordered_result[field_name]
        if identity_values is None or reordered_values is None:
            assert identity_values is reordered_values
            continue
        torch.testing.assert_close(
            identity_values,
            reordered_values,
            rtol=0.0,
            atol=0.0,
            msg=f"{field_name} does not request the same physical command",
        )

    for field_name in _ORDERING_TRACE_FIELDS:
        atol, rtol = _ORDERING_TRACE_TOLERANCES[field_name]
        for step_index, (identity_values, reordered_values) in enumerate(
            zip(identity_result[field_name], reordered_result[field_name], strict=True)
        ):
            torch.testing.assert_close(
                identity_values,
                reordered_values,
                atol=atol,
                rtol=rtol,
                msg=f"{field_name} diverged at step {step_index}",
            )


def _canonicalize_ordering_result(result: dict[str, Any], canonical_joint_names: tuple[str, ...]) -> dict[str, Any]:
    """Gather public and adapter traces into one physical joint-name order."""
    canonical_result = dict(result)
    for field_name in _ORDERING_TRACE_FIELDS:
        source_names = result["adapter_joint_names"] if field_name.startswith("adapter_") else result["joint_names"]
        source_indices = tuple(source_names.index(name) for name in canonical_joint_names)
        canonical_result[field_name] = [
            values.index_select(
                1,
                torch.tensor(source_indices, dtype=torch.long, device=values.device),
            )
            for values in result[field_name]
        ]

    public_indices = tuple(result["joint_names"].index(name) for name in canonical_joint_names)
    for field_name in ("target_pos", "target_vel", "effort_target"):
        values = result[field_name]
        if values is not None:
            canonical_result[field_name] = values.index_select(
                1,
                torch.tensor(public_indices, dtype=torch.long, device=values.device),
            )

    canonical_result["joint_names"] = canonical_joint_names
    canonical_result["backend_joint_names"] = canonical_joint_names
    canonical_result["adapter_joint_names"] = canonical_joint_names
    return canonical_result
