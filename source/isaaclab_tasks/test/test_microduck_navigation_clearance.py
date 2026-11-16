# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for the swept circular footprint used by navigation reward gates."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg

from isaaclab_tasks.contrib.microduck_obstacle.mdp.commands import MicroDuckNavigationCommand, navigation_progress
from isaaclab_tasks.contrib.microduck_obstacle.mdp.events import OBSTACLE_CELLS, OBSTACLE_LAYOUT_NAMES
from isaaclab_tasks.contrib.microduck_obstacle.mdp.geometry import ObstacleFootprintCfg, requested_path_clearance
from isaaclab_tasks.contrib.microduck_obstacle.mdp.rewards import (
    NavigationGoalArrival,
    NavigationObstacleBypass,
    navigation_gated_progress,
    navigation_velocity_tracking,
    navigation_yaw_tracking,
    obstacle_approach_risk,
)


def _make_env(commands, centers, positions=None, headings=None):
    """Provide scene state only; clearance and reward calculations use production code."""
    commands = torch.tensor(commands, dtype=torch.float32)
    processed_commands = torch.zeros_like(commands)
    count = len(commands)
    positions = torch.zeros(count, 3) if positions is None else torch.tensor(positions, dtype=torch.float32)
    headings = torch.zeros(count) if headings is None else torch.tensor(headings, dtype=torch.float32)
    rotation = torch.zeros(count, 4)
    rotation[:, 2] = torch.sin(headings / 2)
    rotation[:, 3] = torch.cos(headings / 2)
    velocity = torch.zeros_like(commands)
    velocity[:, 0] = commands[:, 0] * headings.cos() - commands[:, 1] * headings.sin()
    velocity[:, 1] = commands[:, 0] * headings.sin() + commands[:, 1] * headings.cos()
    angular_velocity = torch.zeros_like(commands)
    angular_velocity[:, 2] = commands[:, 2]
    scene = {
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=SimpleNamespace(torch=positions),
                root_link_quat_w=SimpleNamespace(torch=rotation),
                root_link_lin_vel_w=SimpleNamespace(torch=velocity),
                root_link_ang_vel_w=SimpleNamespace(torch=angular_velocity),
            )
        )
    }
    for name in OBSTACLE_CELLS:
        center = torch.tensor(centers, dtype=torch.float32) if name == "cylinder" else torch.full((count, 3), 10.0)
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=center), root_quat_w=SimpleNamespace(torch=rotation)),
            cfg=SimpleNamespace(spawn=SimpleNamespace(radius=0.025, size=(0.1, 0.1, 0.1))),
        )
    return SimpleNamespace(
        device="cpu",
        cfg=SimpleNamespace(
            obstacle_footprints={
                name: ObstacleFootprintCfg(radius=0.025)
                if name in ("cylinder", "thin_post")
                else ObstacleFootprintCfg(size=(0.1, 0.1))
                for name in OBSTACLE_CELLS
            }
        ),
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
        action_manager=SimpleNamespace(get_term=lambda _name: SimpleNamespace(velocity_commands=processed_commands)),
    )


def test_progress_gate_suppresses_only_positive_blocked_progress():
    """Gate positive progress by closing speed, retain reverse penalties, and permit safe motion."""
    env = _make_env(
        [(0.25, 0.0, 0.0)] * 4 + [(0.0, 0.0, 0.0)],
        [(0.3, 0.0, 0.0), (0.82, 0.0, 0.0), (1.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.3, 0.0, 0.0)],
    )
    env.scene["robot"].data.root_link_lin_vel_w.torch[3, 0] = -0.125
    torch.testing.assert_close(navigation_progress(env), torch.tensor([1.0, 1.0, 1.0, -0.5, 0.0]))
    values = navigation_gated_progress(env)
    assert values[0] == 0.0
    # At 0.675 m clearance, the 2 s horizon permits (0.675 - 0.2) / 2 = 0.2375 m/s.
    # Excess closing is 0.0125 m/s, so risk is (0.0125 / 0.2)^2 = 1/256.
    torch.testing.assert_close(values[1:], torch.tensor([255 / 256, 1.0, -0.5, 0.0]), atol=1e-6, rtol=0.0)


def test_turning_path_releases_tracking_rewards_at_obstacles():
    """A half-circle ends at the obstacle; the same obstacle misses the straight path."""
    # At 0.25 m/s and +/- pi/2 rad/s, 0.5 m of travel makes a semicircle
    # of diameter 1/pi m. Both end footprints intersect their respective poles.
    env = _make_env(
        [(0.25, 0.0, math.pi / 2), (0.25, 0.0, -math.pi / 2), (0.25, 0.0, 0.0)],
        [(0.0, 1 / math.pi, 0.0), (0.0, -1 / math.pi, 0.0), (0.0, 1 / math.pi, 0.0)],
    )
    clearance = requested_path_clearance(env)
    assert (clearance[:2] < -0.14).all()
    assert clearance[2] > 0.12
    # Actual velocities exactly match the request. Suppression must come from
    # the obstructed requested path, not velocity tracking error.
    for reward in (navigation_velocity_tracking, navigation_yaw_tracking):
        values = reward(env)
        torch.testing.assert_close(values[:2], torch.zeros(2))
        assert values[2] > 0.45


@pytest.mark.parametrize("reward", [navigation_velocity_tracking, navigation_yaw_tracking])
def test_tracking_requires_progress_for_moving_requests_but_preserves_stop_and_turn(reward):
    """Matching heading while stopped, moving sideways or retreating cannot earn movement credit."""
    env = _make_env(
        [(0.25, 0.0, 0.0)] * 5 + [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)],
        [(10.0, 10.0, 0.0)] * 7,
    )
    env.scene["robot"].data.root_link_lin_vel_w.torch[:] = torch.tensor(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.25, 0.0),
            (-0.25, 0.0, 0.0),
            (0.125, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
    )
    values = reward(env)
    assert torch.isfinite(values).all()
    torch.testing.assert_close(values[:3], torch.zeros(3))
    assert 0.0 < values[3] < values[4]
    torch.testing.assert_close(values[4:], torch.ones(3))


def test_straight_path_respects_world_pose_and_body_lateral_velocity():
    """Known world-space line endpoints retain their signed obstacle clearance."""
    env = _make_env(
        [(0.25, 0.0, 0.0), (0.0, 0.25, 0.0)],
        [(2.0, 3.8, 0.0), (-1.8, 2.0, 0.0)],
        positions=[(2.0, 3.0, 0.0), (-1.0, 2.0, 0.0)],
        headings=[math.pi / 2, math.pi / 2],
    )
    # Endpoint-to-pole distance 0.3 m, minus pole/robot radii and the 0.05 m margin.
    torch.testing.assert_close(requested_path_clearance(env), torch.full((2,), 0.105), atol=1e-6, rtol=0.0)


def test_stationary_footprint_and_negligible_twists_stay_finite():
    """Pure turns cannot sweep a circular footprint; tiny inputs must not divide by zero."""
    env = _make_env(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0),
            (1e-12, -1e-12, 0.8),
            (0.25, 0.0, 1e-12),
            (0.25, 0.0, 0.0),
            (2e-6, 0.0, 0.8),
        ],
        [(0.8, 0.0, 0.0)] * 6,
    )
    clearance = requested_path_clearance(env)
    assert torch.isfinite(clearance).all()
    torch.testing.assert_close(clearance[:3], torch.full((3,), 0.655), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(clearance[3:5], torch.full((2,), 0.105), atol=1e-6, rtol=0.0)
    # Just above the stationary threshold, repeated tight circles remain near
    # the current footprint while retaining the moving-path sampling margin.
    assert 0.60499 < clearance[5] < 0.60501


def test_sampled_clearance_bounds_dense_integrated_trajectories():
    """Half the arc spacing bounds overestimation between samples, including reverse travel."""
    commands = [(0.22, 0.06, 0.5), (0.3, -0.1, -1.0), (-0.2, 0.07, 0.4), (0.2, 0.0, 1e-10)]
    centers = [(0.2, 0.19, 0.0), (0.27, -0.15, 0.0), (-0.31, 0.11, 0.0), (0.23, 0.11, 0.0)]
    env = _make_env(commands, centers)
    coarse = requested_path_clearance(env).double()

    # An independent midpoint quadrature integrates body velocity through time;
    # it does not reuse the production closed-form path or clearance helper.
    steps = 20000
    reference = []
    for (vx, vy, yaw_rate), center in zip(commands, centers):
        duration = 0.5 / math.hypot(vx, vy)
        dt = duration / steps
        heading = yaw_rate * (torch.arange(steps, dtype=torch.float64) + 0.5) * dt
        velocity = torch.stack((vx * heading.cos() - vy * heading.sin(), vx * heading.sin() + vy * heading.cos()), -1)
        points = torch.cat((torch.zeros(1, 2, dtype=torch.float64), (velocity * dt).cumsum(dim=0)))
        pole_distance = (points - torch.tensor(center[:2], dtype=torch.float64)).norm(dim=-1).amin()
        reference.append(pole_distance - 0.025 - 0.12)
    dense = torch.stack(reference)
    assert (coarse <= dense + 1e-6).all(), (coarse, dense)
    assert (coarse >= dense - 0.050001).all(), (coarse, dense)


def test_approach_risk_prefers_braking_at_walls_without_penalizing_open_progress():
    """Known wall gaps distinguish closing speeds, stopped, parallel and retreating motion."""
    env = _make_env(
        [(0.18, 0.0, 0.0)] * 10,
        [(10.0, 10.0, 0.0)] * 10,
        headings=[0.0] * 6 + [math.pi / 2, 0.0, 0.0, 0.0],
    )
    wall = env.scene["obstacle"]
    wall.cfg.spawn.size = (0.1, 2.0, 1.0)
    env.cfg.obstacle_footprints["obstacle"].size = (0.1, 2.0)
    wall.data.root_pos_w.torch[:] = torch.tensor(
        [
            [0.32, 0.0, 0.0],
            [0.32, 0.0, 0.0],
            [0.32, 0.0, 0.0],
            [0.32, 0.0, 0.0],
            [0.32, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 0.32, 0.0],
            [0.32, 0.0, 0.0],
            [0.42, 0.0, 0.0],
            [0.32, 0.0, 0.0],
        ]
    )
    velocity = env.scene["robot"].data.root_link_lin_vel_w.torch
    velocity[:] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.18, 0.0, 0.0],
            [0.09, 0.0, 0.0],
            [-0.18, 0.0, 0.0],
            [0.0, 0.18, 0.0],
            [0.18, 0.0, 0.0],
            [0.0, 0.18, 0.0],
            [0.03, 0.0, 0.0],
            [0.18, 0.0, 0.0],
            [0.30, 0.0, 0.0],
        ]
    )
    # Near wall: 0.32 - half-thickness 0.05 - footprint 0.12 = 0.15 m clearance,
    # inside the task's 0.20 m margin. Slower closing must reduce cost here too.
    risk = obstacle_approach_risk(env, safety_distance=0.2)
    assert 0.0 < risk[7] < risk[2] < risk[1] < 1.0, "Slower approach inside the margin must reduce cost."
    # The 0.25 m gap leaves 0.05 m over the margin; 0.30 m/s closing saturates the cost.
    torch.testing.assert_close(
        risk,
        torch.tensor([0.0, 0.81, 0.2025, 0.0, 0.0, 0.0, 0.81, 0.0225, 0.600625, 1.0]),
        atol=5e-6,
        rtol=0.0,
    )
    assert torch.equal(risk, obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=0.0))
    # Use the task's progress and approach terms with their configured weights.
    shaped = 5.0 * navigation_gated_progress(env) - 4.0 * risk
    # A slow approach can beat stopping; full-speed closing inside the margin cannot.
    assert shaped[2] > shaped[7] > shaped[0] > shaped[1]
    assert shaped[5] > shaped[0]  # Clear forward progress still beats stopping.
    assert shaped[3] < shaped[0]  # Retreat is not rewarded merely for removing approach risk.

    # Processed commands are independent of operator requests and actual motion.
    # Penalize a pending approach even before acceleration, while retaining measured
    # risk after a stop command. Agreeing retreat/parallel motion and open travel stay free.
    processed = env.action_manager.get_term("joint_pos").velocity_commands
    processed[[0, 2, 5, 6], 0] = 0.18
    processed[3, 0] = -0.18
    processed[4] = torch.tensor([0.0, 0.18, 0.75])
    velocity[6] = 0.0  # Body-forward command at pi/2 must predict world +Y motion.
    command_risk = obstacle_approach_risk(env, safety_distance=0.2)
    torch.testing.assert_close(
        command_risk,
        torch.tensor([0.81, 0.81, 0.81, 0.0, 0.0, 0.0, 0.81, 0.0225, 0.600625, 1.0]),
        atol=5e-6,
        rtol=0.0,
    )
    assert torch.equal(command_risk, obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=0.0))
    weighted_risk = obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=1.0)
    torch.testing.assert_close(
        weighted_risk,
        torch.tensor([1.62, 0.81, 1.62, 0.0, 0.0, 0.0, 1.62, 0.0225, 0.600625, 1.0]),
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.isfinite(weighted_risk).all()

    # A 1 m/s^2 limit over the native 0.02 s control permits this one-step
    # processed decrease. Measured closing masks it unless command credit is separate.
    velocity[[1, 2], 0] = 0.18
    processed[[1, 2], 0] = torch.tensor([0.18, 0.16])
    processed[9, 0] = 0.30  # Both branches saturate: the combined cost must reach two.
    masked_risk = obstacle_approach_risk(env, safety_distance=0.2)
    assert masked_risk[1] == masked_risk[2]
    credited_risk = obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=1.0)
    assert credited_risk[2] < credited_risk[1], (
        "A slower processed command must retain credit during measured approach."
    )
    torch.testing.assert_close(credited_risk[[1, 2, 9]], torch.tensor([1.62, 1.45, 2.0]), atol=5e-6, rtol=0.0)
    torch.testing.assert_close(
        obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=0.5)[[1, 2]],
        torch.tensor([1.215, 1.13]),
        atol=5e-6,
        rtol=0.0,
    )

    # Measured motion closes on the front wall; a perpendicular command closes on
    # a side wall. Reduce each branch over obstacles before adding their costs.
    side_wall = env.scene["wall_right"]
    side_wall.cfg.spawn.size = (2.0, 0.1, 1.0)
    env.cfg.obstacle_footprints["wall_right"].size = (2.0, 0.1)
    side_wall.data.root_pos_w.torch[9] = torch.tensor([0.0, 0.32, 0.0])
    processed[9] = torch.tensor([0.0, 0.16, 0.0])
    torch.testing.assert_close(
        obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=1.0)[9],
        torch.tensor(1.64),
        atol=5e-6,
        rtol=0.0,
    )
    for weight in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            obstacle_approach_risk(env, safety_distance=0.2, commanded_risk_weight=weight)


def test_destination_commands_reaim_persist_stop_and_reset_independently():
    """Turning/resampling must not evade a fixed request; arrival and partial resets are isolated."""
    from isaaclab_tasks.contrib.microduck_obstacle.obstacle_avoidance_env_cfg import MicroDuckNavigationCommandsCfg

    count = 512
    env = _make_env([(0.0, 0.0, 0.0)] * count, [(10.0, 10.0, 0.0)] * count)
    env.num_envs = count
    env.sim = SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda _term: None))
    env.step_dt = 0.02
    env.episode_length_buf = torch.zeros(count, dtype=torch.long)
    env.microduck_obstacle_layout = torch.ones(count, dtype=torch.long)  # Open: reset heading defines destination.
    env.microduck_front_obstacle_index = torch.zeros(count, dtype=torch.long)
    env.termination_manager = SimpleNamespace(
        get_term=lambda _name: torch.zeros(count, dtype=torch.bool), terminated=torch.zeros(count, dtype=torch.bool)
    )
    cfg = MicroDuckNavigationCommandsCfg().base_velocity
    cfg.rel_destination_envs = 0.5
    term = MicroDuckNavigationCommand(cfg, env)
    env.command_manager.get_command = lambda _name: term.command
    torch.manual_seed(19)
    term.reset()
    assert 0.4 < term.is_destination_env.float().mean() < 0.6
    ids = term.is_destination_env.nonzero().flatten()[:3]
    robot = env.scene["robot"].data
    target = term.destination_w.clone()
    # Duck 0 turns left 90 degrees; duck 1 faces backwards; duck 2 reaches its target.
    robot.root_link_quat_w.torch[ids[0]] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    robot.root_link_quat_w.torch[ids[1]] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    robot.root_link_pos_w.torch[ids[2]] = target[ids[2]]
    term.compute(0.02)
    assert term.command[ids[0], 2] == -0.75  # Turn back toward the original world destination.
    assert term.command[ids[0], 0] < 1e-6
    assert term.command[ids[1], 0] == 0.0
    torch.testing.assert_close(term.command[ids[2]], torch.zeros(3))
    # Force ordinary command resampling; destination and chosen episode mode must persist.
    mode = term.is_destination_env.clone()
    term.time_left[:] = 0.0
    term.compute(0.02)
    torch.testing.assert_close(term.destination_w, target)
    torch.testing.assert_close(term.is_destination_env, mode)
    assert torch.isfinite(term.command).all()
    assert (term.command[:, 0] >= 0.0).all() and (term.command[:, 0] <= 0.35).all()
    assert (term.command[:, 1] == 0.0).all() and (term.command[:, 2].abs() <= 0.75).all()
    # Arrival is latched even if standing drift carries the duck outside the tolerance.
    robot.root_link_pos_w.torch[ids[2], 0] += 0.4
    term.compute(0.02)
    torch.testing.assert_close(term.command[ids[2]], torch.zeros(3))
    # Reset one duck at a new position; other destinations/modes/arrival flags survive.
    cfg.rel_destination_envs = 1.0
    before_arrival = term.destination_reached.clone()
    robot.root_link_pos_w.torch[ids[1], 0] += 1.0
    term.reset(ids[1:2])
    keep = torch.arange(count) != ids[1]
    torch.testing.assert_close(term.destination_w[keep], target[keep])
    torch.testing.assert_close(term.is_destination_env[keep], mode[keep])
    torch.testing.assert_close(term.destination_reached[keep], before_arrival[keep])
    assert not torch.allclose(term.destination_w[ids[1]], target[ids[1]])
    # Explicitly disabling encounter commands restores joystick-only episodes.
    cfg.rel_destination_envs = 0.0
    term.reset()
    assert not term.is_destination_env.any()
    # A long slalom can put its final cylinder at the old straight-ahead endpoint.
    cfg.rel_destination_envs = 1.0
    env.microduck_obstacle_layout[0] = 3
    robot.root_link_pos_w.torch[0] = 0.0
    robot.root_link_quat_w.torch[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    for name, xy in (("crate", (0.8, 0.2)), ("tall_box", (1.5, -0.2)), ("cylinder", (2.5, 0.03))):
        env.scene[name].data.root_pos_w.torch[0, :2] = torch.tensor(xy)
    term.reset(torch.tensor([0]))
    assert term.destination_w[0, 1] < -0.45
    # Cylinder radius 0.12 m plus robot footprint radius 0.12 m.
    assert (term.destination_w[0, :2] - torch.tensor([2.5, 0.03])).norm() > 0.24

    # A shuffled box occupies the frontal slot while the named obstacle is off to the side.
    env.microduck_obstacle_layout[0] = OBSTACLE_LAYOUT_NAMES.index("blocked")
    env.microduck_front_obstacle_index[0] = tuple(OBSTACLE_CELLS).index("tall_box")
    env.scene["tall_box"].data.root_pos_w.torch[0, :2] = torch.tensor([0.6, 0.0])
    env.scene["obstacle"].data.root_pos_w.torch[0, :2] = torch.tensor([0.6, 0.8])
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.destination_w[0, :2], torch.tensor([2.6, 0.0]))

    # Shuffling the former thin/low asset names into parked slots must not move the goal.
    env.microduck_obstacle_layout[0] = OBSTACLE_LAYOUT_NAMES.index("thin_low")
    env.scene["thin_post"].data.root_pos_w.torch[0, :2] = torch.tensor([-1.6, 1.6])
    env.scene["low_block"].data.root_pos_w.torch[0, :2] = torch.tensor([-1.6, -1.6])
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.destination_w[0, :2], torch.tensor([2.6, 0.0]))

    # All layout anchors, including rear/side/coincident ones, start ahead at arbitrary reset headings.
    generator = torch.Generator().manual_seed(271)
    headings = 2 * math.pi * torch.rand(count, generator=generator) - math.pi
    headings[:4] = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2])
    robot.root_link_quat_w.torch[:, 2] = (headings / 2).sin()
    robot.root_link_quat_w.torch[:, 3] = (headings / 2).cos()
    robot.root_link_pos_w.torch[:, :2] = 1000 * torch.rand(count, 2, generator=generator)
    env.microduck_obstacle_layout[:] = torch.arange(count) % len(OBSTACLE_LAYOUT_NAMES)
    anchors = torch.tensor([[1.0, 0.4], [-1.0, 0.4], [-1.0, -0.4], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])
    anchor_id = (torch.arange(count) // len(OBSTACLE_LAYOUT_NAMES)) % len(anchors)
    local_anchor = anchors[anchor_id]
    world_anchor = robot.root_link_pos_w.torch.clone()
    world_anchor[:, 0] += local_anchor[:, 0] * headings.cos() - local_anchor[:, 1] * headings.sin()
    world_anchor[:, 1] += local_anchor[:, 0] * headings.sin() + local_anchor[:, 1] * headings.cos()
    for name in OBSTACLE_CELLS:
        env.scene[name].data.root_pos_w.torch[:] = world_anchor
    term.reset()
    offset = term.destination_w - robot.root_link_pos_w.torch
    initial_forward = offset[:, 0] * headings.cos() + offset[:, 1] * headings.sin()
    initial_lateral = -offset[:, 0] * headings.sin() + offset[:, 1] * headings.cos()
    assert (initial_forward > 0.0).all()
    torch.testing.assert_close(offset[:, :2].norm(dim=-1), torch.full((count,), 2.6), atol=1e-4, rtol=0.0)
    assert torch.isfinite(term.command).all()
    anchored = (env.microduck_obstacle_layout != OBSTACLE_LAYOUT_NAMES.index("open")) & (
        ~torch.isin(
            env.microduck_obstacle_layout,
            torch.tensor(
                [
                    OBSTACLE_LAYOUT_NAMES.index("slalom"),
                    OBSTACLE_LAYOUT_NAMES.index("scattered"),
                    OBSTACLE_LAYOUT_NAMES.index("thin_low"),
                ]
            ),
        )
    )
    side = anchored & (local_anchor[:, 1] != 0)
    assert (initial_lateral[side] * local_anchor[side, 1] > 0.0).all()
    # Front anchors keep their original direction; coincident anchors fall back to body forward.
    front = anchored & (anchor_id == 0)
    torch.testing.assert_close(
        initial_lateral[front] / initial_forward[front],
        torch.full_like(initial_forward[front], 0.4),
        atol=1e-4,
        rtol=0.0,
    )
    zero = anchored & (anchor_id == 5)
    torch.testing.assert_close(initial_forward[zero], torch.full_like(initial_forward[zero], 2.6), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(initial_lateral[zero], torch.zeros_like(initial_lateral[zero]), atol=1e-4, rtol=0.0)
    target = term.destination_w.clone()
    term.reset(torch.empty(0, dtype=torch.long))
    torch.testing.assert_close(term.destination_w, target, atol=0.0, rtol=0.0)
    # A later half-turn may put the fixed world target behind; it must not select a new target.
    robot.root_link_quat_w.torch[:, 2] = ((headings + math.pi) / 2).sin()
    robot.root_link_quat_w.torch[:, 3] = ((headings + math.pi) / 2).cos()
    term.time_left[:] = 0.0
    term.compute(0.02)
    torch.testing.assert_close(term.destination_w, target, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dt", [0.02, 0.04])
def test_completion_rewards_are_one_shot_safe_and_reset_per_environment(dt):
    """Exercise real event terms on safe detours, shortcuts, failures and partial resets."""
    count = 7
    env = _make_env([(0.25, 0.0, 0.0)] * count, [(10.0, 10.0, 0.0)] * count)
    env.num_envs, env.step_dt = count, dt
    position = env.scene["robot"].data.root_link_pos_w.torch
    position[:, 0] = -0.5
    env.scene["obstacle"].data.root_pos_w.torch.zero_()
    env.scene["obstacle"].cfg.spawn.size = (0.1, 0.6, 0.2)
    env.cfg.obstacle_footprints["obstacle"].size = (0.1, 0.6)
    env.microduck_obstacle_layout = torch.full((count,), OBSTACLE_LAYOUT_NAMES.index("barrier"))
    env.command_manager.get_command("base_velocity")[4, 0] = -0.25
    failed, boundary = torch.zeros(count, dtype=torch.bool), torch.zeros(count, dtype=torch.bool)
    env.termination_manager = SimpleNamespace(terminated=failed, get_term=lambda name: boundary)
    command = SimpleNamespace(
        is_destination_env=torch.ones(count, dtype=torch.bool),
        destination_w=torch.tensor([[0.6, 0.0, 0.0]] * count),
        cfg=SimpleNamespace(destination_tolerance=0.2),
    )
    command.is_destination_env[4] = False
    env.command_manager.get_term = lambda name: command
    goal = NavigationGoalArrival(RewardTermCfg(func=NavigationGoalArrival, weight=10.0), env)
    bypass = NavigationObstacleBypass(RewardTermCfg(func=NavigationObstacleBypass, weight=5.0), env)
    goal.reset()
    bypass.reset()
    assert not goal(env).any()
    assert not bypass(env).any()
    totals = torch.zeros(count)
    # 0/1 detour left/right; 2 tunnels directly through; 3 fails; 4 is ineligible;
    # 5 crosses a boundary timeout; 6 encounters a nonfinite position then recovers.
    for x, y in [(-0.3, 0.5), (0.3, 0.5), (0.5, 0.0)]:
        position[:, 0], position[:, 1] = x, y
        position[1, 1] = -y
        position[2, 1] = 0.0
        if x == -0.3:
            position[6, 0] = float("nan")
        failed[3], boundary[5] = True, True
        totals += bypass(env) * dt * 5.0
        if x != 0.5:
            assert not goal(env).any()
    torch.testing.assert_close(totals, torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert not bypass(env).any()
    arrival = goal(env) * dt * 10.0
    torch.testing.assert_close(arrival, torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0]))
    assert not goal(env).any()
    # Later native failure, boundary exit or nonfinite state revokes arrival once.
    failed[0], boundary[1] = True, True
    position[2, 0] = float("nan")
    torch.testing.assert_close(goal(env) * dt * 10.0, torch.tensor([-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0]))
    assert not goal(env).any()
    failed[0], boundary[1] = False, False
    position[2, 0] = 0.5
    assert not goal(env).any()  # Recovery without reset cannot earn back the credit.
    # Returning to the start and bypassing again cannot farm the event.
    for x, y in [(-0.5, 0.0), (-0.3, 0.5), (0.3, 0.5), (0.5, 0.0)]:
        position[:, 0], position[:, 1] = x, y
        assert not bypass(env).any()
    ids = torch.tensor([0])
    position[0, 0] = -0.5
    goal.reset(ids)
    bypass.reset(ids)
    assert not bypass(env).any()
    for x, y in [(-0.3, 0.5), (0.3, 0.5), (0.5, 0.0)]:
        position[0, 0], position[0, 1] = x, y
        value = bypass(env) * dt * 5.0
        assert value[1:].sum() == 0.0
        if x == 0.3:
            assert value[0] == 5.0
    torch.testing.assert_close(goal(env) * dt * 10.0, torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_barrier_bonus_respects_rotated_geometry_and_initial_request():
    """World rotation preserves the witness; a request away from the barrier cannot earn it."""
    angle = 0.7
    env = _make_env([(0.25, 0.0, 0.0), (-0.25, 0.0, 0.0)], [(10.0, 10.0, 0.0)] * 2, headings=[angle] * 2)
    env.num_envs, env.step_dt = 2, 0.02
    position = env.scene["robot"].data.root_link_pos_w.torch
    origin = torch.tensor([1.0, 2.0, 0.0])
    env.scene["obstacle"].data.root_pos_w.torch[:] = origin
    env.scene["obstacle"].cfg.spawn.size = (0.1, 0.6, 0.2)
    env.cfg.obstacle_footprints["obstacle"].size = (0.1, 0.6)
    env.microduck_obstacle_layout = torch.full((2,), OBSTACLE_LAYOUT_NAMES.index("barrier"))
    env.termination_manager = SimpleNamespace(
        terminated=torch.zeros(2, dtype=torch.bool), get_term=lambda name: torch.zeros(2, dtype=torch.bool)
    )
    # The USD-style asset has no primitive spawn.size/radius and an unrelated name.
    env.scene["solid_prop"] = env.scene.pop("obstacle")
    env.scene["solid_prop"].cfg = SimpleNamespace(spawn=SimpleNamespace(usd_path="compact_prop.usd"))
    env.cfg.obstacle_footprints["solid_prop"] = env.cfg.obstacle_footprints.pop("obstacle")
    del env.microduck_obstacle_layout
    position[:] = origin + torch.tensor([-0.5 * math.cos(angle), -0.5 * math.sin(angle), 0.0])
    reward = NavigationObstacleBypass(RewardTermCfg(func=NavigationObstacleBypass, weight=5.0), env)
    # Commands reset after reward state: the first step must use the new request.
    request = env.command_manager.get_command("base_velocity")
    request[0, 0] = -0.25
    reward.reset()
    request[0, 0] = 0.25
    assert not reward(env).any()
    total = torch.zeros(2)
    # Include a parallel segment inside the slab to exercise the zero-delta branch.
    for x, y in [(-0.3, 0.5), (0.0, 0.5), (0.0, 0.6), (0.3, 0.6), (0.5, 0.0)]:
        position[:] = origin + torch.tensor(
            [x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle), 0.0]
        )
        total += reward(env) * env.step_dt * 5.0
    torch.testing.assert_close(total, torch.tensor([5.0, 0.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"size": (1.0, 1.0), "radius": 0.1},
        {"size": (0.0, 1.0)},
        {"radius": float("nan")},
        {"radius": -1.0},
        {"size": (1.0,)},
        {"radius": 0.1, "center": (float("inf"), 0.0)},
    ],
)
def test_invalid_obstacle_footprints_fail_at_configuration(kwargs):
    with pytest.raises(ValueError):
        ObstacleFootprintCfg(**kwargs)


def test_geometry_is_independent_of_asset_names_and_usd_root_offsets():
    """A renamed USD-style prop with an offset pivot gives the same reward geometry."""
    from isaaclab.utils.math import quat_apply, yaw_quat

    from isaaclab_tasks.contrib.microduck_obstacle.mdp.geometry import obstacle_clearances

    env = _make_env([(0.2, 0.0, 0.0)] * 2, [(0.5, 0.0, 0.0)] * 2, headings=[0.0, 0.7])
    before = [term(env) for term in (obstacle_clearances, navigation_gated_progress, obstacle_approach_risk)]
    for index, name in enumerate(list(env.cfg.obstacle_footprints)):
        asset = env.scene.pop(name)
        footprint = env.cfg.obstacle_footprints.pop(name)
        footprint.center = (0.2, -0.1)
        offset = torch.tensor([[0.2, -0.1, 0.0]] * 2)
        asset.data.root_pos_w.torch -= quat_apply(yaw_quat(asset.data.root_quat_w.torch), offset)
        asset.cfg = SimpleNamespace(spawn=SimpleNamespace(usd_path=f"prop_{index}.usd"))
        env.scene[f"prop_{index}"] = asset
        env.cfg.obstacle_footprints[f"prop_{index}"] = footprint
    for expected, term in zip(before, (obstacle_clearances, navigation_gated_progress, obstacle_approach_risk)):
        torch.testing.assert_close(term(env), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("axis,sign", [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)])
def test_bypass_selects_and_keeps_nearest_obstacle_from_any_entry_face(axis, sign):
    command = [0.0, 0.0, 0.0]
    command[axis] = sign * 0.25
    env = _make_env([command], [(10.0, 10.0, 0.0)])
    env.num_envs, env.step_dt = 1, 0.02
    env.cfg.obstacle_footprints = {
        "near_prop": ObstacleFootprintCfg(size=(0.2, 0.2)),
        "far_prop": ObstacleFootprintCfg(size=(0.2, 0.2)),
    }
    env.scene["near_prop"] = env.scene.pop("obstacle")
    env.scene["far_prop"] = env.scene.pop("crate")
    env.scene["near_prop"].data.root_pos_w.torch.zero_()
    env.scene["far_prop"].data.root_pos_w.torch.zero_()
    env.scene["far_prop"].data.root_pos_w.torch[:, axis] = sign
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([False]), get_term=lambda name: torch.tensor([False])
    )
    position = env.scene["robot"].data.root_link_pos_w.torch
    position[:, axis] = -0.5 * sign
    reward = NavigationObstacleBypass(RewardTermCfg(func=NavigationObstacleBypass, weight=5.0), env)
    reward.reset()
    assert not reward(env).any()
    # Change the requested direction after arming. The target must remain the near prop.
    env.command_manager.get_command("base_velocity").zero_()
    total = 0.0
    for u, v in [(-0.3, 0.4), (0.3, 0.4), (0.5, 0.0)]:
        position[:, axis], position[:, 1 - axis] = u * sign, v
        total += (reward(env) * env.step_dt * 5.0).item()
    assert total == pytest.approx(5.0)


def test_circle_bounding_box_corner_is_not_a_bypass_encounter():
    """A diagonal ray can intersect the enclosing square while missing the circle."""
    env = _make_env([(0.25, 0.25, 0.0)], [(0.0, 0.0, 0.0)], positions=[(-0.5, -0.15, 0.0)])
    env.num_envs, env.step_dt = 1, 0.02
    env.cfg.obstacle_footprints = {"round_prop": ObstacleFootprintCfg(radius=0.2)}
    env.scene["round_prop"] = env.scene.pop("cylinder")
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([False]), get_term=lambda name: torch.tensor([False])
    )
    reward = NavigationObstacleBypass(RewardTermCfg(func=NavigationObstacleBypass, weight=5.0), env)
    reward.reset()
    assert not reward(env).any()
    position = env.scene["robot"].data.root_link_pos_w.torch
    for x, y in [(-0.4, 0.5), (0.4, 0.5), (0.5, 0.0)]:
        position[:] = torch.tensor([x, y, 0.0])
        assert not reward(env).any()


def test_circle_bypass_pays_once_after_safe_detour():
    env = _make_env([(0.25, 0.0, 0.0)], [(0.0, 0.0, 0.0)], positions=[(-0.5, 0.0, 0.0)])
    env.num_envs, env.step_dt = 1, 0.02
    env.cfg.obstacle_footprints = {"round_prop": ObstacleFootprintCfg(radius=0.2)}
    env.scene["round_prop"] = env.scene.pop("cylinder")
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([False]), get_term=lambda name: torch.tensor([False])
    )
    reward = NavigationObstacleBypass(RewardTermCfg(func=NavigationObstacleBypass, weight=5.0), env)
    reward.reset()
    assert not reward(env).any()
    position = env.scene["robot"].data.root_link_pos_w.torch
    total = 0.0
    for x, y in [(-0.4, 0.5), (0.4, 0.5), (0.5, 0.0), (0.6, 0.0)]:
        position[:] = torch.tensor([x, y, 0.0])
        total += (reward(env) * env.step_dt * 5.0).item()
    assert total == pytest.approx(5.0)
