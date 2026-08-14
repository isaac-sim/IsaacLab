# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cable route curves and success-conditioned reset replay."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (
    CABLE_RADIUS,
    CABLE_SEGMENT_LENGTH,
    ROUTE_AXIAL_CUTOFF,
    CableRoutingSceneCfg,
)
from isaaclab_tasks.contrib.cable_routing.mdp import commands as cable_commands
from isaaclab_tasks.contrib.cable_routing.mdp.commands import CableRoutingCommand
from isaaclab_tasks.contrib.cable_routing.mdp.events import (
    BENCHMARK_GRID_DIRECTIONS,
    cable_capsule_clearance_mask,
    cable_capsule_self_clearance_mask,
)
from isaaclab_tasks.contrib.cable_routing.mdp.reset_curves import (
    generate_route_conditioned_cable_poses,
    planar_vertices_to_segment_poses,
    tangent_point_energy,
    validate_route_conditioned_cable_poses,
)
from isaaclab_tasks.contrib.cable_routing.mdp.reset_replay import (
    CableResetReplay,
    CableResetReplayCfg,
    SceneStateBuffer,
    active_step_progress_from_route_progress,
    finite_scene_state_rows,
)
from isaaclab_tasks.contrib.cable_routing.mdp.route_metrics import (
    benchmark_local_cable_spans,
    benchmark_winding_angle,
    ordered_route_state,
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


def test_active_step_progress_is_recovered_from_whole_route_progress() -> None:
    route_ids = torch.tensor((0, 1, 2, 2))
    active_steps = torch.tensor((0, 0, 0, 1))
    expected_active_progress = torch.tensor((0.40, 0.72, 0.56, 0.91))
    route_lengths = torch.tensor((1, 1, 2))
    route_progress = (active_steps + expected_active_progress) / route_lengths[route_ids]

    actual = active_step_progress_from_route_progress(
        route_progress,
        route_ids,
        active_steps,
        route_lengths,
    )

    torch.testing.assert_close(actual, expected_active_progress)


@pytest.mark.parametrize(
    ("route_ids", "active_steps", "expected_error"),
    (
        (torch.tensor((7,)), torch.tensor((0,)), IndexError),
        (torch.tensor((2,)), torch.tensor((2,)), IndexError),
        (torch.tensor((0.0,)), torch.tensor((0,)), TypeError),
    ),
)
def test_active_step_progress_rejects_invalid_route_metadata(
    route_ids: torch.Tensor,
    active_steps: torch.Tensor,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        active_step_progress_from_route_progress(
            torch.tensor((0.5,)),
            route_ids,
            active_steps,
            (1, 1, 2),
        )


def _default_scene_tensors(num_envs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scene_cfg = CableRoutingSceneCfg()
    origins = torch.zeros(num_envs, 3)
    control_points = torch.tensor(scene_cfg.cable.spawn.positions, dtype=torch.float32)
    control_points += torch.tensor(scene_cfg.cable.init_state.pos, dtype=torch.float32)
    edge = control_points[1:] - control_points[:-1]
    direction = torch.nn.functional.normalize(edge, dim=-1)
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
    pegs = pegs.unsqueeze(0).expand(num_envs, -1, -1).clone()
    return poses, pegs, origins


def _route_tensors(route_ids: torch.Tensor, active_steps: torch.Tensor) -> tuple[torch.Tensor, ...]:
    peg_indices = torch.zeros((len(route_ids), 2), dtype=torch.long)
    directions = torch.zeros((len(route_ids), 2))
    valid_steps = torch.zeros((len(route_ids), 2), dtype=torch.bool)
    for route_id, route in enumerate(_ROUTE_OPTIONS):
        rows = (route_ids == route_id).nonzero(as_tuple=False).squeeze(-1)
        for step, (peg_index, direction) in enumerate(route):
            peg_indices[rows, step] = peg_index
            directions[rows, step] = direction
            valid_steps[rows, step] = True
    return peg_indices, directions, valid_steps, active_steps


def test_tangent_point_energy_penalizes_a_crossing_curve() -> None:
    straight = torch.tensor([[[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    crossing = torch.tensor([[[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.5, -1.0]]])

    straight_energy = tangent_point_energy(straight)
    crossing_energy = tangent_point_energy(crossing)

    assert torch.isfinite(straight_energy).all()
    assert torch.isfinite(crossing_energy).all()
    assert float(crossing_energy[0]) > float(straight_energy[0])


def test_route_conditioned_curves_cover_all_fixture_offsets_and_remain_nonterminal() -> None:
    offset_pairs = list(itertools.product(BENCHMARK_GRID_DIRECTIONS, repeat=2))
    num_envs = 4 * len(offset_pairs)
    default_poses, peg_positions, origins = _default_scene_tensors(num_envs)
    peg_positions[..., :2] += 0.01 * torch.tensor(offset_pairs, dtype=torch.float32).repeat_interleave(4, dim=0)
    route_ids = torch.tensor((0, 1, 2, 2)).repeat(len(offset_pairs))
    active_steps = torch.tensor((0, 0, 0, 1)).repeat(len(offset_pairs))
    # Exercise the lower edge of the configured 40%--92% frontier range for
    # every route/stage; lower partial arcs can overlap their approach strand.
    active_winding = torch.full((4 * len(offset_pairs),), 1.08)

    poses, start_progress = generate_route_conditioned_cable_poses(
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
        max_rejection_attempts=48,
        generator=torch.Generator().manual_seed(7),
    )

    assert bool(
        cable_capsule_clearance_mask(
            poses,
            peg_positions,
            origins,
            rest_length=CABLE_SEGMENT_LENGTH,
        ).all()
    )
    assert bool(cable_capsule_self_clearance_mask(poses, rest_length=CABLE_SEGMENT_LENGTH).all())
    local_axis = torch.zeros_like(poses[..., :3])
    local_axis[..., 2] = 1.0
    half_edge = 0.5 * CABLE_SEGMENT_LENGTH * quat_apply(poses[..., 3:7], local_axis)
    starts = poses[..., :3] - half_edge
    ends = poses[..., :3] + half_edge
    torch.testing.assert_close(ends[:, :-1], starts[:, 1:], atol=2.0e-6, rtol=0.0)

    # For equal-length adjacent chords, the discrete circumradius is
    # l / sqrt(2 (1 - cos(theta))). The generator's final exact-length
    # reconstruction must retain the same 1.2-segment bend target as the
    # geometric projector instead of reintroducing a sharp corner.
    tangent = torch.nn.functional.normalize(quat_apply(poses[..., 3:7], local_axis)[..., :2], dim=-1)
    adjacent_cosine = (tangent[:, 1:] * tangent[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    bend_radius = CABLE_SEGMENT_LENGTH / torch.sqrt((2.0 * (1.0 - adjacent_cosine)).clamp_min(1.0e-12))
    minimum_bend_radius = max(1.2 * CABLE_SEGMENT_LENGTH, 2.0 * CABLE_RADIUS)
    assert float(bend_radius.amin()) >= minimum_bend_radius - 1.0e-6

    peg_indices, directions, valid_steps, active_steps = _route_tensors(route_ids, active_steps)
    winding = benchmark_winding_angle(
        poses[..., :3],
        peg_positions,
        radial_cutoff=0.05,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
    )
    span_count, local_length = benchmark_local_cable_spans(
        poses[..., :3],
        peg_positions,
        radial_cutoff=0.05,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
    )
    eligible = torch.gather((span_count == 1) & (local_length <= 0.25), 1, peg_indices)
    directed_progress, _, prefix, success = ordered_route_state(
        winding,
        peg_indices,
        directions,
        valid_steps,
        completion_threshold=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        completion_mask=eligible,
    )
    active_progress = torch.gather(directed_progress, 1, active_steps[:, None]).squeeze(1)
    assert torch.equal(prefix, active_steps)
    assert not bool(success.any())
    assert bool((active_progress > 0.0).all())
    assert bool((active_progress < 1.0).all())
    assert bool(((start_progress > 0.0) & (start_progress < 1.0)).all())

    floating_poses = poses.clone()
    floating_poses[..., 2] += 2.0 * ROUTE_AXIAL_CUTOFF
    floating_valid, _ = validate_route_conditioned_cable_poses(
        floating_poses,
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
    assert not bool(floating_valid.any())


def test_route_conditioned_curves_support_all_seven_goal_programs() -> None:
    """Every configured route and active stage must construct a valid reset curve."""
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
    torch.testing.assert_close(progress, validated_progress)


def test_reset_program_strata_cover_seven_routes_independently() -> None:
    """Route, active step, progress, and interaction phase form independent strata."""
    command = object.__new__(CableRoutingCommand)
    command._env = SimpleNamespace(device="cpu")
    command.cfg = SimpleNamespace(
        allowed_route_ids=tuple(range(7)),
        route_options=_ROUTE_OPTIONS,
        completion_winding=2.6,
        reset_replay=SimpleNamespace(buffer_size=7 * 2 * 16 * 3, active_progress_range=(0.4, 0.92)),
    )
    command.reset_replay = SimpleNamespace(cfg=command.cfg.reset_replay)

    route_ids, active_steps, active_winding, phases = command._sample_reset_programs()

    for route_id, route in enumerate(_ROUTE_OPTIONS):
        rows = route_ids == route_id
        assert int(rows.sum()) == 2 * 16 * 3
        assert torch.equal(torch.bincount(phases[rows], minlength=3), torch.full((3,), 2 * 16))
        assert len(torch.unique(active_winding[rows])) == 16
        for active_step in range(len(route)):
            step_rows = rows & (active_steps == active_step)
            expected = 2 * 16 * 3 // len(route)
            assert int(step_rows.sum()) == expected
            assert torch.equal(
                torch.bincount(phases[step_rows], minlength=3),
                torch.full((3,), expected // 3),
            )


def test_reset_replay_retries_rotate_rows_across_scratch_environments() -> None:
    """A rejected bank row must not remain pinned to one clone on every retry."""
    all_env_ids = torch.arange(256)
    pending_local = torch.tensor((3, 163, 255))
    assignments = torch.stack(
        [cable_commands._reset_replay_scratch_env_ids(all_env_ids, pending_local, attempt, 64) for attempt in range(64)]
    )

    torch.testing.assert_close(assignments[0], pending_local)
    torch.testing.assert_close(assignments[1], (pending_local + 4) % len(all_env_ids))
    assert bool((torch.sort(assignments, dim=0).values[1:] != torch.sort(assignments, dim=0).values[:-1]).all())
    assert bool((torch.sort(assignments, dim=1).values[:, 1:] != torch.sort(assignments, dim=1).values[:, :-1]).all())


def test_first_valid_curve_is_invariant_to_far_clone_origin() -> None:
    """An accepted first proposal must not lose bend precision at a distant clone origin."""
    default_poses, default_pegs, _ = _default_scene_tensors(1)
    peg_positions_b = default_pegs.clone()
    peg_positions_b[..., :2] = torch.tensor(
        (
            (0.05500030517578125, -0.06500005722045898),
            (-0.034999847412109375, 0.07500004768371582),
        )
    )
    route_ids = torch.tensor((2,))
    active_steps = torch.tensor((1,))
    active_winding = torch.tensor((1.0822498798370361,))

    def generate_at_origin(origin_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        poses_w = default_poses.clone()
        poses_w[..., :3] += origin_w[:, None]
        pegs_w = peg_positions_b + origin_w[:, None]
        return generate_route_conditioned_cable_poses(
            poses_w,
            pegs_w,
            origin_w,
            route_ids,
            active_steps,
            active_winding,
            _ROUTE_OPTIONS,
            rest_length=CABLE_SEGMENT_LENGTH,
            completion_winding=2.6,
            maximum_completion_winding=2.0 * torch.pi + 0.25,
            axial_cutoff=ROUTE_AXIAL_CUTOFF,
            curve_projection_iterations=0,
            max_rejection_attempts=1,
            generator=torch.Generator().manual_seed(0),
        )

    origin_zero = torch.zeros((1, 3))
    far_origin = torch.tensor(((8.25, 0.75, 0.0),))
    poses_zero, progress_zero = generate_at_origin(origin_zero)
    poses_far, progress_far = generate_at_origin(far_origin)

    # Before the local-frame construction fix, a proposal accepted at the
    # origin could fail the bend gate at this exact distant clone.
    torch.testing.assert_close(
        poses_far[..., :3] - far_origin[:, None],
        poses_zero[..., :3],
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(poses_far[..., 3:7], poses_zero[..., 3:7], atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(progress_far, progress_zero, atol=1.0e-5, rtol=0.0)

    valid, _ = validate_route_conditioned_cable_poses(
        poses_far,
        peg_positions_b + far_origin[:, None],
        far_origin,
        route_ids,
        active_steps,
        active_winding,
        _ROUTE_OPTIONS,
        rest_length=CABLE_SEGMENT_LENGTH,
        completion_winding=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
    )
    assert bool(valid.all())


def test_route_conditioned_curve_generation_is_seed_deterministic() -> None:
    default_poses, peg_positions, origins = _default_scene_tensors(4)
    route_ids = torch.tensor((0, 1, 2, 2))
    active_steps = torch.tensor((0, 0, 0, 1))
    active_winding = torch.tensor((1.0, 1.4, 1.8, 2.2))
    kwargs = dict(
        rest_length=CABLE_SEGMENT_LENGTH,
        completion_winding=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
        curve_projection_iterations=50,
        max_rejection_attempts=48,
    )

    # Policy playback commonly holds an outer inference-mode context. The
    # forward-only Warp projector must remain deterministic without autograd.
    with torch.inference_mode():
        poses_a, progress_a = generate_route_conditioned_cable_poses(
            default_poses,
            peg_positions,
            origins,
            route_ids,
            active_steps,
            active_winding,
            _ROUTE_OPTIONS,
            generator=torch.Generator().manual_seed(31),
            **kwargs,
        )
    poses_b, progress_b = generate_route_conditioned_cable_poses(
        default_poses,
        peg_positions,
        origins,
        route_ids,
        active_steps,
        active_winding,
        _ROUTE_OPTIONS,
        generator=torch.Generator().manual_seed(31),
        **kwargs,
    )

    torch.testing.assert_close(poses_a, poses_b)
    torch.testing.assert_close(progress_a, progress_b)


def test_route_validator_rejects_a_sharp_adjacent_segment_turn() -> None:
    default_poses, peg_positions, origins = _default_scene_tensors(1)
    route_ids = torch.tensor((0,))
    active_steps = torch.tensor((0,))
    active_winding = torch.tensor((1.08,))
    poses, _ = generate_route_conditioned_cable_poses(
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
        max_rejection_attempts=256,
        generator=torch.Generator().manual_seed(7),
    )
    kinked = poses.clone()
    half_yaw = 0.5 * math.radians(50.0)
    yaw = poses.new_tensor(((0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),))
    kinked[:, 5, 3:7] = quat_mul(yaw, kinked[:, 5, 3:7])

    def validate(minimum_bend_radius: float) -> torch.Tensor:
        valid, _ = validate_route_conditioned_cable_poses(
            kinked,
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
            minimum_bend_radius=minimum_bend_radius,
        )
        return valid

    # All pre-existing collision and topology predicates still accept this
    # far-from-fixture capsule. Only the physically useful 12 mm bend target
    # rejects its 50-degree material-frame kink.
    assert bool(validate(0.5 * CABLE_SEGMENT_LENGTH + 1.0e-6))
    assert not bool(validate(max(1.2 * CABLE_SEGMENT_LENGTH, 2.0 * CABLE_RADIUS)))


def test_planar_pose_conversion_transfers_authored_material_roll_profile() -> None:
    default_poses, _, _ = _default_scene_tensors(2)
    segment_count = default_poses.shape[1]
    heading = torch.linspace(-0.35, 1.10, segment_count).repeat(2, 1)
    heading[1] *= -1.0
    direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    vertices = torch.zeros((2, segment_count + 1, 2))
    vertices[:, 0] = torch.tensor((-0.12, -0.14))
    vertices[:, 1:] = vertices[:, :1] + torch.cumsum(CABLE_SEGMENT_LENGTH * direction, dim=1)
    z_w = default_poses[..., 2].mean(dim=1)

    transferred = planar_vertices_to_segment_poses(vertices, z_w, default_poses[..., 3:7])

    def material_spin_profile(poses: torch.Tensor) -> torch.Tensor:
        local_z = torch.zeros_like(poses[..., :3])
        local_z[..., 2] = 1.0
        tangent = quat_apply(poses[..., 3:7], local_z)
        half_axis = 0.5 * CABLE_SEGMENT_LENGTH * tangent
        centerline = torch.cat(
            ((poses[:, :1, :3] - half_axis[:, :1]), poses[..., :3] + half_axis),
            dim=1,
        )
        bishop = planar_vertices_to_segment_poses(centerline[..., :2], centerline[..., 2].mean(dim=1))
        spin = quat_mul(poses[..., 3:7], quat_conjugate(bishop[..., 3:7]))
        axial_sine = (spin[..., :3] * tangent).sum(dim=-1)
        return torch.nn.functional.normalize(torch.stack((axial_sine, spin[..., 3]), dim=-1), dim=-1)

    reference_spin = material_spin_profile(default_poses)
    transferred_spin = material_spin_profile(transferred)
    # Spin quaternions q and -q encode the same material frame.
    alignment = (reference_spin * transferred_spin).sum(dim=-1).abs()
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=2.0e-5, rtol=0.0)


def test_scene_state_buffer_stores_and_gathers_complete_nested_state() -> None:
    example = {
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
    buffer = SceneStateBuffer(5, example)
    buffer.store(1, example, torch.tensor([2, 0]))

    restored = buffer.gather(torch.tensor([2, 1]))

    torch.testing.assert_close(
        restored["articulation"]["robot"]["root_pose"], example["articulation"]["robot"]["root_pose"][[0, 2]]
    )
    torch.testing.assert_close(
        restored["cable_object"]["cable"]["segment_pose"],
        example["cable_object"]["cable"]["segment_pose"][[0, 2]],
    )


def test_finite_scene_state_rows_checks_every_asset_and_field() -> None:
    state = {
        "articulation": {
            "left": {
                "joint_position": torch.zeros(4, 6),
                "joint_velocity": torch.zeros(4, 6),
            },
            "right": {"root_velocity": torch.zeros(4, 6)},
        },
        "cable_object": {"cable": {"segment_pose": torch.zeros(4, 8, 7)}},
    }
    state["articulation"]["left"]["joint_velocity"][1, 0] = torch.nan
    state["articulation"]["right"]["root_velocity"][2, 3] = torch.inf
    state["cable_object"]["cable"]["segment_pose"][3, 4, 0] = torch.nan

    valid = finite_scene_state_rows(state)

    assert torch.equal(valid, torch.tensor((True, False, False, False)))


def test_scene_state_buffer_stores_arbitrary_destination_rows() -> None:
    example = {
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
    buffer = SceneStateBuffer(5, example)

    buffer.store_rows(torch.tensor([4, 0, 2]), example, torch.tensor([1, 2, 0]))
    restored = buffer.gather(torch.tensor([0, 2, 4]))

    expected_env_ids = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        restored["articulation"]["robot"]["joint_position"],
        example["articulation"]["robot"]["joint_position"][expected_env_ids],
    )
    torch.testing.assert_close(
        restored["cable_object"]["cable"]["segment_velocity"],
        example["cable_object"]["cable"]["segment_velocity"][expected_env_ids],
    )


def test_scene_state_buffer_validates_row_indices_once_per_store(monkeypatch) -> None:
    """Index validation must not synchronize once for every scene-state field."""
    example = {
        "articulation": {
            "robot": {
                "root_pose": torch.zeros(3, 7),
                "joint_position": torch.zeros(3, 4),
            }
        },
        "cable_object": {
            "cable": {
                "segment_pose": torch.zeros(3, 4, 7),
                "segment_velocity": torch.zeros(3, 4, 6),
            }
        },
    }
    buffer = SceneStateBuffer(5, example)
    original_any = torch.Tensor.any
    original_aminmax = torch.aminmax
    destination_validation_count = 0
    source_validation_count = 0

    def counted_any(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal destination_validation_count
        destination_validation_count += 1
        return original_any(tensor, *args, **kwargs)

    def counted_aminmax(tensor: torch.Tensor, *args, **kwargs):
        nonlocal source_validation_count
        source_validation_count += 1
        return original_aminmax(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "any", counted_any)
    monkeypatch.setattr(torch, "aminmax", counted_aminmax)

    buffer.store_rows(torch.tensor([4, 0]), example, torch.tensor([1, 2]))

    assert destination_validation_count == 1
    assert source_validation_count == 1


def test_scene_state_buffer_source_index_error_names_first_incompatible_field() -> None:
    """One-shot source validation retains the field-specific bounds diagnostic."""
    example = {
        "rigid_object": {"board": {"root_pose": torch.zeros(3, 7)}},
        "cable_object": {"cable": {"segment_pose": torch.zeros(2, 4, 7)}},
    }
    buffer = SceneStateBuffer(5, example)

    with pytest.raises(
        IndexError,
        match=r"env_ids must lie in \[0, 2\) for state field cable_object/cable/segment_pose; got \[2\]",
    ):
        buffer.store_rows(torch.tensor([0]), example, torch.tensor([2]))


@pytest.mark.parametrize(
    ("destination_rows", "env_ids", "expected_error"),
    (
        (torch.tensor([[0]]), torch.tensor([0]), ValueError),
        (torch.tensor([0, 1]), torch.tensor([0]), ValueError),
        (torch.tensor([0.0]), torch.tensor([0]), TypeError),
        (torch.tensor([0]), torch.tensor([0.0]), TypeError),
        (torch.tensor([-1]), torch.tensor([0]), IndexError),
        (torch.tensor([5]), torch.tensor([0]), IndexError),
        (torch.tensor([0]), torch.tensor([3]), IndexError),
    ),
)
def test_scene_state_buffer_store_rows_validates_indices(
    destination_rows: torch.Tensor,
    env_ids: torch.Tensor,
    expected_error: type[Exception],
) -> None:
    example = {"rigid_object": {"board": {"root_pose": torch.zeros(3, 7)}}}
    buffer = SceneStateBuffer(5, example)

    with pytest.raises(expected_error):
        buffer.store_rows(destination_rows, example, env_ids)


def test_scene_state_buffer_store_rows_validates_state_field_shape() -> None:
    example = {"rigid_object": {"board": {"root_pose": torch.zeros(3, 7)}}}
    incompatible = {"rigid_object": {"board": {"root_pose": torch.zeros(3, 6)}}}
    buffer = SceneStateBuffer(5, example)

    with pytest.raises(ValueError, match="trailing shape"):
        buffer.store_rows(torch.tensor([1]), incompatible, torch.tensor([0]))


def test_reset_replay_settle_validation_defaults_are_probe_grounded() -> None:
    cfg = CableResetReplayCfg()

    assert cfg.completed_winding == pytest.approx(4.0)
    assert cfg.settle_steps == 16
    assert cfg.max_settle_attempts == 64
    assert cfg.max_donor_fraction == pytest.approx(0.10)
    assert cfg.max_settle_linear_speed == pytest.approx(0.15)
    assert cfg.max_settle_angular_speed == pytest.approx(15.0)
    assert cfg.max_segment_length_relative_error == pytest.approx(0.15)
    assert cfg.restore_clearance == pytest.approx(5.0e-6)
    assert cfg.post_settle_progress_tolerance == pytest.approx(0.35)
    assert cfg.maximum_settled_active_progress == pytest.approx(0.92)
    assert cfg.curve_projection_iterations == 50
    assert cfg.repulsive_iterations is None
    assert cfg.max_curve_attempts == 512


def test_restore_clearance_survives_float32_cross_clone_relocation() -> None:
    """A banked tangent capsule must stay clear after moving between far clone origins."""
    combined_radius = 0.003 + 0.0125
    angle = 0.02912256307899952
    source_origin = torch.tensor(((-11.25, -6.75, 0.0),), dtype=torch.float32)
    destination_origin = torch.tensor(((-11.25, -11.25, 0.0),), dtype=torch.float32)
    peg_position_b = torch.tensor((((0.045, -0.055, 0.7681),),), dtype=torch.float32)
    radial = torch.tensor((math.cos(angle), math.sin(angle)), dtype=torch.float32)
    tangent = torch.tensor((-math.sin(angle), math.cos(angle)), dtype=torch.float32)
    quaternion = torch.nn.functional.normalize(
        torch.tensor((-tangent[1], tangent[0], 0.0, 1.0), dtype=torch.float32),
        dim=0,
    )

    def author_and_restore(margin: float) -> tuple[torch.Tensor, ...]:
        local_pose = torch.zeros((1, 1, 7), dtype=torch.float32)
        local_pose[0, 0, :2] = peg_position_b[0, 0, :2] + (combined_radius + margin) * radial
        local_pose[0, 0, 2] = peg_position_b[0, 0, 2]
        local_pose[0, 0, 3:] = quaternion
        source_pose = local_pose.clone()
        source_pose[..., :3] += source_origin[:, None]
        source_peg = peg_position_b + source_origin[:, None]
        stored_pose = source_pose.clone()
        stored_pose[..., :3] -= source_origin[:, None]
        stored_peg = source_peg.clone()
        stored_peg[..., :3] -= source_origin[:, None]
        restored_pose = stored_pose.clone()
        restored_pose[..., :3] += destination_origin[:, None]
        restored_peg = stored_peg + destination_origin[:, None]
        return source_pose, source_peg, restored_pose, restored_peg

    def is_clear(poses: torch.Tensor, pegs: torch.Tensor, origin: torch.Tensor) -> bool:
        return bool(
            cable_capsule_clearance_mask(
                poses,
                pegs,
                origin,
                rest_length=0.01,
                cable_radius=0.003,
                peg_radius=0.0125,
                fixture_clearance=0.0,
                board_bounds_b=None,
            ).item()
        )

    source_pose, source_peg, restored_pose, restored_peg = author_and_restore(0.0)
    assert is_clear(source_pose, source_peg, source_origin)
    assert not is_clear(restored_pose, restored_peg, destination_origin)

    # The measured loss for this worst-case source/destination clone pair is
    # below 1 um; retain a 5x reserve without rejecting genuinely clear cable
    # states that settle tens of micrometers from the collision surface.
    source_pose, source_peg, restored_pose, restored_peg = author_and_restore(5.0e-6)
    assert is_clear(source_pose, source_peg, source_origin)
    assert is_clear(restored_pose, restored_peg, destination_origin)


def test_post_settle_validation_applies_restore_clearance(monkeypatch) -> None:
    """Replay acceptance must apply its relocation reserve to fixtures and board edges."""
    replay_cfg = CableResetReplayCfg()
    replay_cfg.robot_targets.enabled = False
    command = object.__new__(CableRoutingCommand)
    command.reset_replay = SimpleNamespace(cfg=replay_cfg)
    command.cfg = SimpleNamespace(
        peg_names=("peg_0", "peg_1"),
        route_options=_ROUTE_OPTIONS,
        completion_winding=2.6,
        maximum_completion_winding=2.0 * torch.pi + 0.25,
        radial_cutoff=0.05,
        axial_cutoff=ROUTE_AXIAL_CUTOFF,
        maximum_local_cable_length=0.25,
        settled_cable_bounds_b=((-0.18, 0.18), (-0.13, 0.13)),
    )
    command._cable_rest_length_m = CABLE_SEGMENT_LENGTH

    def tensor_proxy(value: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(torch=value)

    poses = torch.zeros((1, 2, 7))
    poses[..., 6] = 1.0
    velocities = torch.zeros((1, 2, 6))
    command.cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=tensor_proxy(poses),
            segment_velocity_w=tensor_proxy(velocities),
        )
    )

    class FakeScene(dict):
        env_origins = torch.zeros((1, 3))

        def get_state(self, *, is_relative: bool) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
            assert is_relative
            return {"rigid_object": {"board": {"root_pose": torch.zeros((1, 7))}}}

    scene = FakeScene(
        {
            name: SimpleNamespace(data=SimpleNamespace(root_pose_w=tensor_proxy(torch.zeros((1, 7)))))
            for name in command.cfg.peg_names
        }
    )
    command._env = SimpleNamespace(scene=scene)
    captured: dict[str, object] = {}

    def capture_validation(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        return torch.ones(1, dtype=torch.bool), torch.zeros(1)

    monkeypatch.setattr(cable_commands, "validate_route_conditioned_cable_poses", capture_validation)
    monkeypatch.setattr(cable_commands, "cable_relative_joint_gap", lambda *args: torch.zeros((1, 1)))
    robot_condition = SimpleNamespace()

    valid, _ = command._post_settle_replay_validity(
        torch.tensor((0,)),
        torch.tensor((0,)),
        torch.tensor((0,)),
        torch.tensor((1.0,)),
        robot_condition,
    )

    assert bool(valid.item())
    assert captured["fixture_clearance"] == replay_cfg.restore_clearance
    assert captured["board_clearance"] == replay_cfg.restore_clearance


def test_reset_replay_deprecates_repulsive_iteration_alias() -> None:
    with pytest.deprecated_call(match="curve_projection_iterations"):
        cfg = CableResetReplayCfg(repulsive_iterations=7)

    assert cfg.curve_projection_iterations == 7


@pytest.mark.parametrize(
    "overrides",
    (
        {"settle_steps": 0},
        {"max_settle_attempts": 0},
        {"max_donor_fraction": -0.01},
        {"max_donor_fraction": 1.0},
        {"max_donor_fraction": float("nan")},
        {"max_settle_linear_speed": 0.0},
        {"max_settle_linear_speed": float("nan")},
        {"max_settle_angular_speed": float("inf")},
        {"max_segment_length_relative_error": -0.01},
        {"max_segment_length_relative_error": 1.0},
        {"restore_clearance": 0.0},
        {"restore_clearance": -1.0e-4},
        {"restore_clearance": float("nan")},
        {"restore_clearance": float("inf")},
        {"post_settle_progress_tolerance": -0.01},
        {"post_settle_progress_tolerance": 1.01},
        {"maximum_settled_active_progress": 0.91},
        {"maximum_settled_active_progress": 1.0},
        {"maximum_settled_active_progress": float("nan")},
    ),
)
def test_reset_replay_settle_validation_rejects_invalid_values(overrides: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        CableResetReplayCfg(**overrides)


def test_reset_replay_uses_lift_monitor_and_credits_only_replay_sources() -> None:
    env = SimpleNamespace(device="cpu", num_envs=4)
    replay = CableResetReplay(
        CableResetReplayCfg(buffer_size=8),
        env,
    )
    replay.env_source[:] = torch.tensor((1, -1, 1, 4))

    assert replay.requested_active_progress.shape == (8,)
    assert replay.start_progress.shape == (8,)

    replay.credit(torch.arange(4), torch.tensor((True, True, False, True)))
    replay.route_id[:] = torch.tensor((0, 0, 0, 0, 1, 1, 1, 1))
    requested_routes = torch.tensor((0, 1)).repeat(64)
    sources = replay.sample_sources(requested_routes)

    assert replay.monitor.success_size[1] == 2
    assert replay.monitor.success_rate[1] == 0.5
    assert replay.monitor.success_size[4] == 1
    assert replay.monitor.success_rate[4] == 1.0
    assert replay.monitor.success_size.sum() == 3
    assert bool(((sources >= 0) & (sources < 8)).all())
    assert torch.equal(replay.route_id[sources], requested_routes)
