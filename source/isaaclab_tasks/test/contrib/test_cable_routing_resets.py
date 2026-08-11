# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cable-routing procedural reset helpers."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (
    BOARD_SIZE,
    CABLE_SEGMENT_LENGTH,
    PEG_RADIUS,
    ROUTE_AXIAL_CUTOFF,
    CableRoutingSceneCfg,
)
from isaaclab_tasks.contrib.cable_routing.mdp.events import (
    BENCHMARK_GRID_DIRECTIONS,
    cable_capsule_clearance_mask,
    cable_capsule_self_clearance_mask,
    cable_unrouted_mask,
    generate_collision_free_cable_poses,
    reset_cable_state,
    reset_peg_offsets,
    sample_benchmark_grid_offsets,
    sample_board_frame_se2,
    sample_cable_heading_offsets,
    shape_cable_poses_planar,
    transform_cable_poses_se2,
)
from isaaclab_tasks.contrib.cable_routing.mdp.route_metrics import benchmark_winding_angle


class _Proxy:
    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class _FakeRigidObject:
    def __init__(self, default_pose: torch.Tensor, default_velocity: torch.Tensor):
        root_pose_w = default_pose.clone()
        self.data = SimpleNamespace(
            default_root_pose=_Proxy(default_pose),
            default_root_vel=_Proxy(default_velocity),
            root_pose_w=_Proxy(root_pose_w),
        )
        self.pose_w = torch.full_like(default_pose, -9.0)
        self.velocity_w = torch.full_like(default_velocity, -9.0)
        self.pose_writes: list[torch.Tensor] = []
        self.velocity_writes: list[torch.Tensor] = []

    def write_root_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.pose_w[env_ids] = root_pose
        self.data.root_pose_w.torch[env_ids] = root_pose
        self.pose_writes.append(env_ids.clone())

    def write_root_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.velocity_w[env_ids] = root_velocity
        self.velocity_writes.append(env_ids.clone())


class _FakeCableObject:
    def __init__(self, default_pose: torch.Tensor, default_velocity: torch.Tensor):
        self.data = SimpleNamespace(
            default_segment_pose_w=_Proxy(default_pose),
            default_segment_velocity_w=_Proxy(default_velocity),
        )
        self.pose_w = torch.full_like(default_pose, -9.0)
        self.velocity_w = torch.full_like(default_velocity, -9.0)
        self.pose_writes: list[torch.Tensor] = []
        self.velocity_writes: list[torch.Tensor] = []

    def write_segment_pose_to_sim_index(self, *, segment_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.pose_w[env_ids] = segment_pose
        self.pose_writes.append(env_ids.clone())

    def write_segment_velocity_to_sim_index(self, *, segment_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.velocity_w[env_ids] = segment_velocity
        self.velocity_writes.append(env_ids.clone())


class _FakeScene:
    def __init__(self, env_origins: torch.Tensor, assets: dict[str, object]):
        self.env_origins = env_origins
        self._assets = assets

    def __getitem__(self, name: str):
        return self._assets[name]


def _fake_env(env_origins: torch.Tensor, **assets):
    return SimpleNamespace(
        device=env_origins.device,
        num_envs=len(env_origins),
        scene=_FakeScene(env_origins, assets),
    )


def _identity_poses(num_envs: int, num_segments: int) -> torch.Tensor:
    poses = torch.zeros(num_envs, num_segments, 7)
    poses[..., 6] = 1.0
    return poses


def _straight_cable_poses(env_origins: torch.Tensor, num_segments: int, rest_length: float) -> torch.Tensor:
    """Create connected cable capsules whose local Z axes point along world X."""
    poses = _identity_poses(len(env_origins), num_segments)
    poses[..., 0] = env_origins[:, None, 0] + (torch.arange(num_segments) - 0.5 * (num_segments - 1)) * rest_length
    poses[..., 1] = env_origins[:, None, 1]
    poses[..., 2] = env_origins[:, None, 2] + 0.05
    poses[..., 3:7] = torch.tensor((0.0, 2**-0.5, 0.0, 2**-0.5))
    return poses


def _planar_control_points_to_segment_poses(control_points_w: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Convert planar cable control points to COM-centered capsule poses."""
    starts = control_points_w[:-1]
    ends = control_points_w[1:]
    directions = torch.nn.functional.normalize(ends - starts, dim=-1)
    # Newton cable capsules use local +Z as their centerline axis. The shortest-arc
    # quaternion from +Z to a planar direction is normalize((-dy, dx, 0, 1)).
    quaternions = torch.stack(
        (-directions[:, 1], directions[:, 0], torch.zeros_like(directions[:, 0]), torch.ones_like(directions[:, 0])),
        dim=-1,
    )
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    poses = torch.cat((0.5 * (starts + ends), quaternions), dim=-1)
    return poses.unsqueeze(0).expand(num_envs, -1, -1).clone()


def test_sample_benchmark_grid_offsets_is_deterministic_and_uses_exact_nonzero_set() -> None:
    generator_a = torch.Generator().manual_seed(17)
    generator_b = torch.Generator().manual_seed(17)
    offsets_a = sample_benchmark_grid_offsets(1024, 3, grid_pitch=0.01, generator=generator_a)
    offsets_b = sample_benchmark_grid_offsets(1024, 3, grid_pitch=0.01, generator=generator_b)

    torch.testing.assert_close(offsets_a, offsets_b)
    directions = torch.round(offsets_a.reshape(-1, 2) / 0.01).to(torch.int64)
    observed = {tuple(direction) for direction in directions.tolist()}
    assert observed == set(BENCHMARK_GRID_DIRECTIONS)
    assert not bool((directions == 0).all(dim=-1).any())


def test_sample_board_frame_se2_respects_axis_specific_ranges() -> None:
    translation, yaw = sample_board_frame_se2(
        128,
        translation_jitter=((-0.03, -0.01), (0.02, 0.04)),
        yaw_jitter=(0.1, 0.2),
        generator=torch.Generator().manual_seed(3),
    )

    assert bool(((translation[:, 0] >= -0.03) & (translation[:, 0] <= -0.01)).all())
    assert bool(((translation[:, 1] >= 0.02) & (translation[:, 1] <= 0.04)).all())
    assert bool(((yaw >= 0.1) & (yaw <= 0.2)).all())


def test_transform_cable_poses_se2_rotates_about_each_environment_origin() -> None:
    poses = _identity_poses(2, 2)
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, -3.0, 1.0]])
    poses[0, :, :3] = torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, 0.3]])
    poses[1, :, :3] = origins[1] + torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, 0.3]])
    translation = torch.tensor([[0.1, -0.2], [-0.4, 0.5]])
    yaw = torch.full((2,), torch.pi / 2)

    transformed = transform_cable_poses_se2(poses, origins, translation, yaw)

    expected_local = torch.tensor([[[0.1, 0.8, 0.2], [-0.9, -0.2, 0.3]], [[-0.4, 1.5, 0.2], [-1.4, 0.5, 0.3]]])
    expected_positions = expected_local + origins[:, None, :]
    torch.testing.assert_close(transformed[..., :3], expected_positions, atol=1.0e-6, rtol=0.0)
    expected_quat = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]).expand(2, 2, 4)
    torch.testing.assert_close(transformed[..., 3:7], expected_quat, atol=1.0e-6, rtol=0.0)


def test_planar_shape_reset_is_smooth_bounded_and_keeps_segments_connected() -> None:
    num_envs, num_segments, rest_length = 8, 25, 0.01
    poses = _identity_poses(num_envs, num_segments)
    poses[..., 0] = torch.arange(num_segments) * rest_length
    # The cable spawner orients each capsule's local Z axis along the curve tangent.
    poses[..., 3:7] = torch.tensor((0.0, 2**-0.5, 0.0, 2**-0.5))
    heading = sample_cable_heading_offsets(
        num_envs,
        num_segments,
        max_heading_offset=0.1,
        generator=torch.Generator().manual_seed(19),
    )

    shaped = shape_cable_poses_planar(poses, rest_length, heading)

    assert bool((heading.abs() <= 0.1 + 1.0e-6).all())
    torch.testing.assert_close(heading[:, (0, -1)], torch.zeros(num_envs, 2), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(shaped[..., :3].mean(dim=1), poses[..., :3].mean(dim=1), atol=1.0e-6, rtol=0.0)
    axes = torch.zeros_like(shaped[..., :3])
    axes[..., 2] = 1.0
    directions = quat_apply(shaped[..., 3:7], axes)
    segment_starts = shaped[..., :3] - 0.5 * rest_length * directions
    segment_ends = shaped[..., :3] + 0.5 * rest_length * directions
    torch.testing.assert_close(segment_ends[:, :-1], segment_starts[:, 1:], atol=1.0e-6, rtol=0.0)
    assert not torch.equal(shaped[0, ..., :3], poses[0, ..., :3])


def test_cable_capsule_clearance_checks_fixture_surfaces_and_board_bounds() -> None:
    rest_length = 0.01
    origins = torch.zeros(3, 3)
    poses = _straight_cable_poses(origins, num_segments=10, rest_length=rest_length)
    pegs = torch.tensor([[[0.0, 0.04, 0.05]], [[0.0, 0.0, 0.05]], [[0.0, 0.04, 0.05]]])
    # The third cable crosses the board's right inset boundary.
    poses[2, :, 0] += 0.01

    clear = cable_capsule_clearance_mask(
        poses,
        pegs,
        origins,
        rest_length=rest_length,
        cable_radius=0.003,
        peg_radius=0.0125,
        fixture_clearance=0.002,
        board_bounds_b=((-0.06, 0.06), (-0.06, 0.06)),
        board_clearance=0.002,
    )

    assert torch.equal(clear, torch.tensor([True, False, False]))


def test_cable_capsule_self_clearance_rejects_non_neighbor_crossing() -> None:
    poses = _identity_poses(2, 3)
    poses[..., :3] = torch.tensor(((0.0, 0.0, 0.05), (0.4, 0.4, 0.05), (0.0, 0.0, 0.05)))
    # Segment 0 points along X and segment 2 along Y, so those non-neighbors
    # cross in environment 0. Moving segment 2 beyond segment 0's endpoint in
    # environment 1 supplies the clear control case.
    poses[:, 0, 3:7] = torch.tensor((0.0, 2**-0.5, 0.0, 2**-0.5))
    poses[:, 2, 3:7] = torch.tensor((-(2**-0.5), 0.0, 0.0, 2**-0.5))
    poses[1, 2, 0] = 0.08

    clear = cable_capsule_self_clearance_mask(
        poses,
        rest_length=0.1,
        cable_radius=0.003,
        self_clearance=0.00025,
    )

    assert torch.equal(clear, torch.tensor([False, True]))


def test_collision_free_generator_handles_randomized_pegs_and_preserves_rest_length() -> None:
    num_envs, num_segments, rest_length = 16, 20, 0.01
    origins = torch.zeros(num_envs, 3)
    origins[:, 0] = 2.0 * torch.arange(num_envs)
    default_poses = _straight_cable_poses(origins, num_segments, rest_length)
    generator = torch.Generator().manual_seed(9)
    peg_offsets = sample_benchmark_grid_offsets(num_envs, 2, grid_pitch=0.01, generator=generator)
    peg_base_b = torch.tensor([[-0.03, 0.0, 0.05], [0.04, 0.0, 0.05]])
    peg_positions = origins[:, None, :] + peg_base_b[None]
    peg_positions[..., :2] += peg_offsets
    board_bounds = ((-0.15, 0.15), (-0.10, 0.10))

    poses, _, _ = generate_collision_free_cable_poses(
        default_poses,
        peg_positions,
        origins,
        translation_jitter=(0.0, 0.0),
        yaw_jitter=(0.0, 0.0),
        rest_length=rest_length,
        board_bounds_b=board_bounds,
        max_rejection_attempts=512,
        generator=generator,
    )

    clear = cable_capsule_clearance_mask(
        poses,
        peg_positions,
        origins,
        rest_length=rest_length,
        board_bounds_b=board_bounds,
    )
    assert bool(clear.all())
    axes = torch.zeros_like(poses[..., :3])
    axes[..., 2] = 1.0
    directions = quat_apply(poses[..., 3:7], axes)
    segment_starts = poses[..., :3] - 0.5 * rest_length * directions
    segment_ends = poses[..., :3] + 0.5 * rest_length * directions
    # Poses are float32 world coordinates. At the test's furthest origin (x=30 m),
    # one ULP is already about 3.8e-6 m, so connectivity must be judged relative
    # to coordinate magnitude rather than with a sub-ULP absolute tolerance.
    coordinate_scale = max(float(poses[..., :3].abs().amax()), 1.0)
    roundoff_atol = 2.0 * torch.finfo(poses.dtype).eps * coordinate_scale
    torch.testing.assert_close(segment_ends[:, :-1], segment_starts[:, 1:], atol=roundoff_atol, rtol=0.0)


def test_actual_reset_is_clear_self_clear_and_unrouted_for_all_benchmark_offset_pairs() -> None:
    scene_cfg = CableRoutingSceneCfg()
    offset_pairs = torch.tensor(
        list(itertools.product(BENCHMARK_GRID_DIRECTIONS, repeat=2)),
        dtype=torch.float32,
    )
    num_envs = len(offset_pairs)
    env_origins = torch.zeros(num_envs, 3)
    env_origins[:, 0] = torch.linspace(-30.0, 30.0, num_envs)
    env_origins[:, 1] = torch.linspace(24.0, -24.0, num_envs)

    control_points = torch.tensor(scene_cfg.cable.spawn.positions, dtype=torch.float32)
    control_points += torch.tensor(scene_cfg.cable.init_state.pos, dtype=torch.float32)
    default_poses = _planar_control_points_to_segment_poses(control_points, num_envs)
    default_poses[..., :3] += env_origins[:, None, :]
    peg_positions = (
        torch.tensor(
            (scene_cfg.peg_0.init_state.pos, scene_cfg.peg_1.init_state.pos),
            dtype=torch.float32,
        )
        .expand(num_envs, -1, -1)
        .clone()
    )
    peg_positions += env_origins[:, None, :]
    peg_positions[..., :2] += 0.01 * offset_pairs

    board_bounds = ((-0.5 * BOARD_SIZE[0], 0.5 * BOARD_SIZE[0]), (-0.5 * BOARD_SIZE[1], 0.5 * BOARD_SIZE[1]))
    cable_radius = 0.5 * scene_cfg.cable.spawn.physics_material.thickness
    peg_radius = PEG_RADIUS
    poses, _, _ = generate_collision_free_cable_poses(
        default_poses,
        peg_positions,
        env_origins,
        translation_jitter=((-0.002, 0.002), (-0.002, 0.002)),
        yaw_jitter=(-0.02, 0.02),
        rest_length=CABLE_SEGMENT_LENGTH,
        max_heading_offset=0.0,
        num_shape_modes=3,
        cable_radius=cable_radius,
        peg_radius=peg_radius,
        board_bounds_b=board_bounds,
        generator=torch.Generator().manual_seed(2026),
    )

    assert bool(
        cable_capsule_clearance_mask(
            poses,
            peg_positions,
            env_origins,
            rest_length=CABLE_SEGMENT_LENGTH,
            cable_radius=cable_radius,
            peg_radius=peg_radius,
            board_bounds_b=board_bounds,
        ).all()
    )
    assert bool(
        cable_unrouted_mask(
            poses,
            peg_positions,
            radial_cutoff=0.05,
            axial_cutoff=ROUTE_AXIAL_CUTOFF,
            max_abs_winding=0.5,
        ).all()
    )
    winding = benchmark_winding_angle(
        poses[..., :3],
        peg_positions,
        radial_cutoff=0.05,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
    )
    assert float(winding.abs().amax()) <= 0.5
    assert not bool((winding.abs() >= 2.6).any())
    assert bool(
        cable_capsule_self_clearance_mask(
            poses,
            rest_length=CABLE_SEGMENT_LENGTH,
            cable_radius=cable_radius,
            self_clearance=0.00025,
        ).all()
    )

    axes = torch.zeros_like(poses[..., :3])
    axes[..., 2] = 1.0
    directions = quat_apply(poses[..., 3:7], axes)
    segment_starts = poses[..., :3] - 0.5 * CABLE_SEGMENT_LENGTH * directions
    segment_ends = poses[..., :3] + 0.5 * CABLE_SEGMENT_LENGTH * directions
    coordinate_scale = max(float(poses[..., :3].abs().amax()), 1.0)
    roundoff_atol = 2.0 * torch.finfo(poses.dtype).eps * coordinate_scale
    torch.testing.assert_close(segment_ends[:, :-1], segment_starts[:, 1:], atol=roundoff_atol, rtol=0.0)
    local_positions = poses[..., :2] - env_origins[:, None, :2]
    assert torch.unique(local_positions.flatten(start_dim=1), dim=0).shape[0] == num_envs


def test_collision_aware_reset_isolated_to_selected_environment_mask() -> None:
    num_envs, num_segments, rest_length = 6, 20, 0.01
    origins = torch.zeros(num_envs, 3)
    origins[:, 0] = 2.0 * torch.arange(num_envs)
    default_poses = _straight_cable_poses(origins, num_segments, rest_length)
    cable = _FakeCableObject(default_poses.clone(), torch.zeros(num_envs, num_segments, 6))

    peg_default_0 = torch.zeros(num_envs, 7)
    peg_default_1 = torch.zeros(num_envs, 7)
    peg_default_0[:, :3] = torch.tensor((-0.03, 0.0, 0.05))
    peg_default_1[:, :3] = torch.tensor((0.04, 0.0, 0.05))
    peg_default_0[:, 6] = 1.0
    peg_default_1[:, 6] = 1.0
    peg_0 = _FakeRigidObject(peg_default_0, torch.zeros(num_envs, 6))
    peg_1 = _FakeRigidObject(peg_default_1, torch.zeros(num_envs, 6))
    # The cable reset reads live world-frame fixture poses after their own reset event.
    peg_0.data.root_pose_w.torch[:, :3] += origins
    peg_1.data.root_pose_w.torch[:, :3] += origins
    env = _fake_env(origins, cable=cable, peg_0=peg_0, peg_1=peg_1)
    env_mask = torch.tensor([False, True, False, True, False, True])

    reset_cable_state(
        env,
        env_mask,
        translation_jitter=(0.0, 0.0),
        yaw_jitter=(0.0, 0.0),
        rest_length=rest_length,
        board_bounds_b=((-0.15, 0.15), (-0.10, 0.10)),
        generator=torch.Generator().manual_seed(21),
    )

    selected = env_mask.nonzero(as_tuple=False).squeeze(-1)
    unselected = (~env_mask).nonzero(as_tuple=False).squeeze(-1)
    live_pegs = torch.stack(
        (peg_0.data.root_pose_w.torch[selected, :3], peg_1.data.root_pose_w.torch[selected, :3]), dim=1
    )
    assert bool(
        cable_capsule_clearance_mask(
            cable.pose_w[selected],
            live_pegs,
            origins[selected],
            rest_length=rest_length,
            board_bounds_b=((-0.15, 0.15), (-0.10, 0.10)),
        ).all()
    )
    torch.testing.assert_close(cable.pose_w[unselected], torch.full((len(unselected), num_segments, 7), -9.0))
    assert torch.equal(cable.pose_writes[0], selected)


def test_reset_peg_offsets_writes_only_selected_environments() -> None:
    num_envs = 4
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    default_pose = torch.zeros(num_envs, 7)
    default_pose[:, 6] = 1.0
    default_pose[:, 2] = 0.4
    default_velocity = torch.arange(num_envs * 6, dtype=torch.float32).reshape(num_envs, 6)
    peg_0 = _FakeRigidObject(default_pose.clone(), default_velocity.clone())
    peg_1 = _FakeRigidObject(default_pose.clone(), default_velocity.clone())
    env = _fake_env(env_origins, peg_0=peg_0, peg_1=peg_1)
    env_ids = torch.tensor([3, 1])
    base_positions = ((0.1, -0.2, 0.5), (-0.15, 0.25, 0.6))

    offsets = reset_peg_offsets(
        env,
        env_ids,
        base_positions_b=base_positions,
        generator=torch.Generator().manual_seed(11),
    )

    for asset_index, asset in enumerate((peg_0, peg_1)):
        expected_position = env_origins[env_ids] + torch.tensor(base_positions[asset_index])
        expected_position[:, :2] += offsets[:, asset_index]
        torch.testing.assert_close(asset.pose_w[env_ids, :3], expected_position)
        torch.testing.assert_close(asset.pose_w[env_ids, 3:7], default_pose[env_ids, 3:7])
        torch.testing.assert_close(asset.velocity_w[env_ids], default_velocity[env_ids])
        torch.testing.assert_close(asset.pose_w[torch.tensor([0, 2])], torch.full((2, 7), -9.0))
        torch.testing.assert_close(asset.velocity_w[torch.tensor([0, 2])], torch.full((2, 6), -9.0))
        assert torch.equal(asset.pose_writes[0], env_ids)
        assert torch.equal(asset.velocity_writes[0], env_ids)

    directions = torch.round(offsets.reshape(-1, 2) / 0.01).to(torch.int64)
    assert all(tuple(direction) in BENCHMARK_GRID_DIRECTIONS for direction in directions.tolist())


def test_reset_cable_state_restores_defaults_without_neighbor_writes() -> None:
    num_envs, num_segments = 4, 3
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    default_pose = _identity_poses(num_envs, num_segments)
    for env_id in range(num_envs):
        default_pose[env_id, :, 0] = env_origins[env_id, 0] + torch.arange(num_segments) * 0.01
        default_pose[env_id, :, 1] = 0.2
        default_pose[env_id, :, 2] = 0.5
    default_velocity = torch.arange(num_envs * num_segments * 6, dtype=torch.float32).reshape(num_envs, num_segments, 6)
    cable = _FakeCableObject(default_pose.clone(), default_velocity.clone())
    env = _fake_env(env_origins, cable=cable)
    env_ids = torch.tensor([3, 1])

    translation, yaw = reset_cable_state(
        env,
        env_ids,
        asset_cfg=SceneEntityCfg("cable"),
        fixture_asset_names=(),
        translation_jitter=(0.0, 0.0),
        yaw_jitter=(0.0, 0.0),
        max_heading_offset=0.0,
        board_bounds_b=None,
    )

    torch.testing.assert_close(translation, torch.zeros(2, 2))
    torch.testing.assert_close(yaw, torch.zeros(2))
    torch.testing.assert_close(cable.pose_w[env_ids], default_pose[env_ids])
    torch.testing.assert_close(cable.velocity_w[env_ids], default_velocity[env_ids])
    torch.testing.assert_close(cable.pose_w[torch.tensor([0, 2])], torch.full((2, num_segments, 7), -9.0))
    torch.testing.assert_close(cable.velocity_w[torch.tensor([0, 2])], torch.full((2, num_segments, 6), -9.0))
    assert torch.equal(cable.pose_writes[0], env_ids)
    assert torch.equal(cable.velocity_writes[0], env_ids)


def test_reset_cable_state_defers_to_enabled_named_full_scene_replay(monkeypatch) -> None:
    """An enabled command-owned replay bypasses ordinary curve generation and cable writes."""
    num_envs, num_segments = 3, 2
    env_origins = torch.zeros(num_envs, 3)
    cable = _FakeCableObject(_identity_poses(num_envs, num_segments), torch.zeros(num_envs, num_segments, 6))
    env = _fake_env(env_origins, cable=cable)
    replay = SimpleNamespace(cfg=SimpleNamespace(enabled=True))
    requested_terms: list[str] = []
    env.command_manager = SimpleNamespace(
        get_term=lambda name: requested_terms.append(name) or SimpleNamespace(reset_replay=replay)
    )

    def unexpected_curve_generation(*args, **kwargs):
        raise AssertionError("ordinary cable generation must be skipped when full-scene replay owns the reset")

    monkeypatch.setattr(
        "isaaclab_tasks.contrib.cable_routing.mdp.events.generate_collision_free_cable_poses",
        unexpected_curve_generation,
    )
    translation, yaw = reset_cable_state(
        env,
        torch.tensor([2, 0]),
        full_scene_replay_command_name="route",
    )

    assert requested_terms == ["route"]
    torch.testing.assert_close(translation, torch.zeros(2, 2))
    torch.testing.assert_close(yaw, torch.zeros(2))
    assert not cable.pose_writes
    assert not cable.velocity_writes


@pytest.mark.parametrize(
    "reset_replay",
    [None, SimpleNamespace(cfg=SimpleNamespace(enabled=False))],
)
def test_reset_cable_state_keeps_ordinary_reset_when_named_replay_is_disabled(reset_replay) -> None:
    """A named term without active replay still receives a valid standalone cable reset."""
    num_envs, num_segments = 2, 2
    env_origins = torch.zeros(num_envs, 3)
    default_pose = _identity_poses(num_envs, num_segments)
    default_pose[..., 0] = torch.tensor([[0.00, 0.01], [0.02, 0.03]])
    cable = _FakeCableObject(default_pose.clone(), torch.zeros(num_envs, num_segments, 6))
    env = _fake_env(env_origins, cable=cable)
    env.command_manager = SimpleNamespace(get_term=lambda _name: SimpleNamespace(reset_replay=reset_replay))

    reset_cable_state(
        env,
        torch.tensor([1]),
        fixture_asset_names=(),
        translation_jitter=(0.0, 0.0),
        yaw_jitter=(0.0, 0.0),
        max_heading_offset=0.0,
        board_bounds_b=None,
        full_scene_replay_command_name="route",
    )

    assert torch.equal(cable.pose_writes[0], torch.tensor([1]))
    assert torch.equal(cable.velocity_writes[0], torch.tensor([1]))


def test_reset_cable_state_accepts_boolean_environment_mask() -> None:
    num_envs, num_segments = 4, 2
    env_origins = torch.zeros(num_envs, 3)
    default_pose = _identity_poses(num_envs, num_segments)
    default_pose[..., 0] = torch.arange(num_envs)[:, None]
    default_velocity = torch.zeros(num_envs, num_segments, 6)
    cable = _FakeCableObject(default_pose.clone(), default_velocity)
    env = _fake_env(env_origins, cable=cable)

    reset_cable_state(
        env,
        torch.tensor([False, True, False, True]),
        fixture_asset_names=(),
        translation_jitter=(0.0, 0.0),
        yaw_jitter=(0.0, 0.0),
        max_heading_offset=0.0,
        board_bounds_b=None,
    )

    expected_ids = torch.tensor([1, 3])
    torch.testing.assert_close(cable.pose_w[expected_ids], default_pose[expected_ids])
    torch.testing.assert_close(cable.pose_w[torch.tensor([0, 2])], torch.full((2, num_segments, 7), -9.0))
    assert torch.equal(cable.pose_writes[0], expected_ids)


def test_reset_helpers_accept_empty_environment_selection() -> None:
    env_origins = torch.zeros(2, 3)
    default_pose = torch.zeros(2, 7)
    default_pose[:, 6] = 1.0
    peg = _FakeRigidObject(default_pose, torch.zeros(2, 6))
    cable = _FakeCableObject(_identity_poses(2, 2), torch.zeros(2, 2, 6))
    env = _fake_env(env_origins, peg_0=peg, cable=cable)

    peg_offsets = reset_peg_offsets(env, torch.empty(0, dtype=torch.long), asset_names=("peg_0",))
    translation, yaw = reset_cable_state(env, torch.empty(0, dtype=torch.long))

    assert peg_offsets.shape == (0, 1, 2)
    assert translation.shape == (0, 2)
    assert yaw.shape == (0,)
    assert not peg.pose_writes
    assert not cable.pose_writes


@pytest.mark.parametrize(
    ("translation_jitter", "yaw_jitter"),
    [(((0.1, -0.1), (0.0, 0.0)), (0.0, 0.0)), (((0.0, 0.0), (0.0, 0.0)), (0.2, -0.2))],
)
def test_sample_board_frame_se2_rejects_reversed_ranges(translation_jitter, yaw_jitter) -> None:
    with pytest.raises(ValueError, match="ordered"):
        sample_board_frame_se2(1, translation_jitter=translation_jitter, yaw_jitter=yaw_jitter)
