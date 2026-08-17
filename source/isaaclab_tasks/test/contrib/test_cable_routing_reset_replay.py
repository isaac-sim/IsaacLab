# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused reset-curve and success-conditioned replay tests."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (
    CABLE_SEGMENT_LENGTH,
    ROUTE_AXIAL_CUTOFF,
    YamCableBoardSceneCfg,
)
from isaaclab_tasks.contrib.cable_routing.mdp.reset_curves import (
    generate_route_conditioned_cable_poses,
    validate_route_conditioned_cable_poses,
)
from isaaclab_tasks.contrib.cable_routing.mdp.reset_replay import (
    CableResetReplay,
    CableResetReplayCfg,
    SceneStateBuffer,
    finite_scene_state_rows,
)

_ROUTE_OPTIONS = (
    ((0, -1),),
    ((1, 1),),
    ((0, -1), (1, 1)),
    ((0, 1),),
    ((1, -1),),
    ((0, 1), (1, 1)),
    ((0, -1), (1, -1)),
)


def _default_scene_tensors(num_envs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scene_cfg = YamCableBoardSceneCfg()
    origins = torch.zeros(num_envs, 3)
    control_points = torch.tensor(scene_cfg.cable.spawn.positions, dtype=torch.float32)
    control_points += torch.tensor(scene_cfg.cable.init_state.pos, dtype=torch.float32)
    direction = torch.nn.functional.normalize(control_points[1:] - control_points[:-1], dim=-1)
    quaternion = torch.stack(
        (-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0]), torch.ones_like(direction[:, 0])),
        dim=-1,
    )
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    poses = torch.cat((0.5 * (control_points[1:] + control_points[:-1]), quaternion), dim=-1)
    poses = poses.unsqueeze(0).expand(num_envs, -1, -1).clone()
    pegs = torch.tensor(
        (scene_cfg.peg_0.init_state.pos, scene_cfg.peg_1.init_state.pos),
        dtype=torch.float32,
    )
    return poses, pegs.unsqueeze(0).expand(num_envs, -1, -1).clone(), origins


def test_route_conditioned_curves_support_every_goal_and_active_stage() -> None:
    """Every configured route/stage produces valid, nonterminal reset geometry."""
    route_ids = torch.tensor((0, 1, 2, 2, 3, 4, 5, 5, 6, 6))
    active_steps = torch.tensor((0, 0, 0, 1, 0, 0, 0, 1, 0, 1))
    active_winding = torch.linspace(1.10, 2.35, len(route_ids))
    default_poses, peg_positions, origins = _default_scene_tensors(len(route_ids))

    poses, progress = generate_route_conditioned_cable_poses(
        default_poses,
        peg_positions,
        origins,
        route_ids,
        active_steps,
        active_winding,
        _ROUTE_OPTIONS,
        rest_length=CABLE_SEGMENT_LENGTH,
        completion_winding=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
        curve_projection_iterations=50,
        max_rejection_attempts=96,
        generator=torch.Generator().manual_seed(73),
    )
    valid, validated_progress = validate_route_conditioned_cable_poses(
        poses,
        peg_positions,
        origins,
        route_ids,
        active_steps,
        active_winding,
        _ROUTE_OPTIONS,
        rest_length=CABLE_SEGMENT_LENGTH,
        completion_winding=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        radial_cutoff=0.05,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
    )

    assert bool(valid.all())
    assert bool(((progress > 0.0) & (progress < 1.0)).all())
    torch.testing.assert_close(progress, validated_progress)


def test_scene_state_buffer_round_trips_arbitrary_rows_and_rejects_nonfinite_state() -> None:
    state = {
        "articulation": {
            "robot": {
                "root_pose": torch.arange(21, dtype=torch.float32).reshape(3, 7),
                "joint_position": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            }
        },
        "cable_object": {
            "cable": {
                "segment_pose": torch.arange(84, dtype=torch.float32).reshape(3, 4, 7),
                "segment_velocity": torch.arange(72, dtype=torch.float32).reshape(3, 4, 6),
            }
        },
    }
    buffer = SceneStateBuffer(5, state)
    buffer.store_rows(torch.tensor([4, 0, 2]), state, torch.tensor([1, 2, 0]))
    restored = buffer.gather(torch.tensor([0, 2, 4]))
    expected = torch.tensor([2, 0, 1])

    torch.testing.assert_close(
        restored["articulation"]["robot"]["joint_position"],
        state["articulation"]["robot"]["joint_position"][expected],
    )
    torch.testing.assert_close(
        restored["cable_object"]["cable"]["segment_velocity"],
        state["cable_object"]["cable"]["segment_velocity"][expected],
    )

    state["cable_object"]["cable"]["segment_pose"][1, 0, 0] = torch.nan
    assert torch.equal(finite_scene_state_rows(state), torch.tensor((True, False, True)))


def test_replay_credits_only_sources_and_samples_the_requested_route() -> None:
    env = SimpleNamespace(device="cpu", num_envs=4)
    replay = CableResetReplay(CableResetReplayCfg(buffer_size=8), env)
    replay.env_source[:] = torch.tensor((1, -1, 1, 4))
    replay.credit(torch.arange(4), torch.tensor((True, True, False, True)))

    replay.route_id[:] = torch.tensor((0, 0, 0, 0, 1, 1, 1, 1))
    requested_routes = torch.tensor((0, 1)).repeat(64)
    sources = replay.sample_sources(requested_routes)

    assert replay.monitor.success_size.sum() == 3
    assert replay.monitor.success_rate[1] == 0.5
    assert replay.monitor.success_rate[4] == 1.0
    assert torch.equal(replay.route_id[sources], requested_routes)
