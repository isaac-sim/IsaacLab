# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton IK action prototype construction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.envs.mdp.actions.newton_ik_actions import (
    NewtonInverseKinematicsAction,
    _build_isolated_prototype_model,
    _finalize_prototype_model,
    _resolve_prototype_articulation_path,
)


class _Geometry:
    pass


class _Builder:
    def __init__(self) -> None:
        self.shape_source = [_Geometry(), None]

    def finalize(self, *, device: str):
        del device
        return self.shape_source


def test_resolve_prototype_articulation_path_includes_nested_root() -> None:
    path = _resolve_prototype_articulation_path("/World/envs/env_0", "/Robot", "/Geometry/arm")

    assert path == "/World/envs/env_0/Robot/Geometry/arm"


def test_finalize_prototype_model_isolates_shared_geometry() -> None:
    builder = _Builder()
    original_shape_source = builder.shape_source
    original_geometry = builder.shape_source[0]

    finalized_shape_source = _finalize_prototype_model(builder, "cpu")

    assert builder.shape_source is original_shape_source
    assert builder.shape_source[0] is original_geometry
    assert finalized_shape_source is not original_shape_source
    assert finalized_shape_source[0] is not original_geometry
    assert finalized_shape_source[1] is None


def test_build_isolated_prototype_model_imports_only_asset_subtree(monkeypatch) -> None:
    calls = {}

    class IsolatedBuilder:
        def add_usd(self, stage, **kwargs) -> None:
            calls["stage"] = stage
            calls["import"] = kwargs

        def finalize(self, *, device: str):
            calls["device"] = device
            return "isolated-model"

    monkeypatch.setattr(
        "isaaclab_newton.envs.mdp.actions.newton_ik_actions.ModelBuilder",
        IsolatedBuilder,
    )
    stage = object()

    model = _build_isolated_prototype_model(stage, "/World/envs/env_0/YamRight", "cuda:0")

    assert model == "isolated-model"
    assert calls == {
        "stage": stage,
        "import": {
            "root_path": "/World/envs/env_0/YamRight",
            "floating": False,
            "load_visual_shapes": False,
            "load_static_visual_shapes": False,
            "verbose": False,
        },
        "device": "cuda:0",
    }


def test_first_cuda_graph_capture_is_replayed_before_the_target_is_written(monkeypatch) -> None:
    """Warp capture records kernels, so the first action must explicitly replay them."""
    events = []

    class FakeCapture:
        graph = "captured-graph"

        def __init__(self, *, device: str):
            events.append(("capture_enter", device))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("capture_exit")

    action = object.__new__(NewtonInverseKinematicsAction)
    action.cfg = SimpleNamespace(use_cuda_graph=True)
    action._env = SimpleNamespace(device="cuda:0")
    action._ik_graph = None
    action._solve_and_gather = lambda: events.append("solve_recorded")
    action._joint_pos_des = "joint-target"
    action._joint_ids = "joint-ids"
    action._asset = SimpleNamespace(set_joint_position_target_index=lambda **kwargs: events.append(("write", kwargs)))
    monkeypatch.setattr(
        "isaaclab_newton.envs.mdp.actions.newton_ik_actions.wp.ScopedCapture",
        FakeCapture,
    )
    monkeypatch.setattr(
        "isaaclab_newton.envs.mdp.actions.newton_ik_actions.wp.capture_launch",
        lambda graph: events.append(("launch", graph)),
    )

    action.apply_actions()

    assert events == [
        ("capture_enter", "cuda:0"),
        "solve_recorded",
        "capture_exit",
        ("launch", "captured-graph"),
        ("write", {"target": "joint-target", "joint_ids": "joint-ids"}),
    ]


def test_fixed_base_root_orientation_validation_is_cached() -> None:
    action = object.__new__(NewtonInverseKinematicsAction)
    action._root_orientations_validated = False
    root_quaternions = torch.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)))
    action._asset = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=SimpleNamespace(torch=root_quaternions)),
    )

    action._validate_matching_root_orientations()

    assert action._root_orientations_validated
    # Fixed-base roots are immutable. A second action therefore takes the
    # synchronization-free fast path instead of inspecting the tensor again.
    action._asset.data.root_quat_w.torch = torch.full_like(root_quaternions, torch.nan)
    action._validate_matching_root_orientations()


def test_fixed_base_root_orientation_validation_rejects_mismatched_clones() -> None:
    action = object.__new__(NewtonInverseKinematicsAction)
    action._root_orientations_validated = False
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=torch.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0))))
        ),
    )

    with pytest.raises(RuntimeError, match=r"root orientations differ in env ids \[1\]"):
        action._validate_matching_root_orientations()

    assert not action._root_orientations_validated
