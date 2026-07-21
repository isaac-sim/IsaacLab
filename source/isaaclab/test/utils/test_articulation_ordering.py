# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import torch


def _make_trace(
    joint_names: tuple[str, ...],
    adapter_joint_names: tuple[str, ...],
    values: torch.Tensor,
    adapter_values: torch.Tensor,
    target_values: torch.Tensor,
) -> dict:
    return {
        "joint_names": joint_names,
        "backend_joint_names": ("hip", "knee"),
        "joint_ordering": {
            "user_names": joint_names,
            "backend_names": ("hip", "knee"),
            "user_to_backend_indices": tuple(("hip", "knee").index(name) for name in joint_names),
            "backend_to_user_indices": tuple(joint_names.index(name) for name in ("hip", "knee")),
        },
        "adapter_joint_names": adapter_joint_names,
        "joint_pos": [values],
        "joint_vel": [values],
        "computed_torque": [values],
        "applied_torque": [values],
        "adapter_computed_effort": [adapter_values],
        "adapter_applied_effort": [adapter_values],
        "target_pos": target_values,
        "target_vel": target_values,
        "effort_target": target_values,
    }


def test_assert_articulation_ordering_trace_matches_canonicalizes_public_and_adapter_axes() -> None:
    helper = importlib.import_module("isaaclab.test.utils.articulation_ordering")
    identity_trace = _make_trace(
        ("hip", "knee"),
        ("hip", "knee"),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
    )
    reordered_trace = _make_trace(
        ("knee", "hip"),
        ("knee", "hip"),
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[4.0, 3.0]]),
        torch.tensor([[6.0, 5.0]]),
    )

    helper.assert_articulation_ordering_trace_matches(identity_trace, reordered_trace, ("knee", "hip"))
