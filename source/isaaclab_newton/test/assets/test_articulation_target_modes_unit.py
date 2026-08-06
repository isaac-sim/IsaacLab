# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less unit tests for replicated articulation target-mode configuration."""

from types import SimpleNamespace

from isaaclab_newton.assets.articulation import articulation as articulation_module
from newton import JointTargetMode, JointType

from isaaclab.actuators import ImplicitActuatorCfg


def test_identical_replicated_articulations_reuse_resolution_and_update_every_dof(monkeypatch):
    """Cached mode resolution is applied to every replicated articulation."""
    resolve_call_count = 0
    original_resolve_matching_names = articulation_module.resolve_matching_names

    def counted_resolve_matching_names(*args, **kwargs):
        nonlocal resolve_call_count
        resolve_call_count += 1
        return original_resolve_matching_names(*args, **kwargs)

    monkeypatch.setattr(
        articulation_module,
        "_resolve_articulation_root_prim_path_expr",
        lambda cfg: "/World/envs/env_.*/Robot",
    )
    monkeypatch.setattr(articulation_module, "resolve_matching_names", counted_resolve_matching_names)
    builder = SimpleNamespace(
        articulation_label=["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"],
        articulation_start=[0, 2],
        articulation_end=[2, 4],
        joint_type=[JointType.REVOLUTE] * 4,
        joint_qd_start=[0, 1, 2, 3],
        joint_label=[
            "/World/envs/env_0/Robot/left",
            "/World/envs/env_0/Robot/right",
            "/World/envs/env_1/Robot/left",
            "/World/envs/env_1/Robot/right",
        ],
        joint_target_mode=[int(JointTargetMode.NONE)] * 4,
        joint_target_ke=[0.0] * 4,
        joint_target_kd=[0.0] * 4,
    )
    cfg = SimpleNamespace(
        actuators={
            "left": ImplicitActuatorCfg(joint_names_expr=["left"], stiffness=10.0, damping=0.0),
            "right": ImplicitActuatorCfg(joint_names_expr=["right"], stiffness=0.0, damping=2.0),
        }
    )

    articulation_module._configure_builder_joint_target_modes(builder, cfg)

    assert builder.joint_target_mode == [
        int(JointTargetMode.POSITION),
        int(JointTargetMode.VELOCITY),
        int(JointTargetMode.POSITION),
        int(JointTargetMode.VELOCITY),
    ]
    assert resolve_call_count == 3
