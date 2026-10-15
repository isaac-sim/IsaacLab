# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain references and observation diagnostics for the moving-patch task."""

import math
from types import SimpleNamespace

import torch
import warp as wp

from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.env_cfg.reward_cfg import G1RewardsCfg
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mdp.rewards import foot_touch_down_angle_penalty, metric_sliderbar
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mpm_env import G1MovingPatchEnv


def test_soft_landing_penalizes_only_commanded_touchdowns():
    """Persistent support is free; new foot contacts pay force magnitude, including turning in place."""
    cfg = G1RewardsCfg().contact_impulse
    env = SimpleNamespace(
        foot_contact_force=torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]] * 3),
        foot_first_contact=torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]),
        command_manager=SimpleNamespace(
            get_command=lambda _: torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
        ),
        extras={"log": {}},
    )
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([5.0, 17.0, 0.0]))
    torch.testing.assert_close(env.extras["log"]["Metrics/landing_force_mean"], torch.tensor(39.0 / 5))
    env.foot_first_contact.zero_()
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.zeros(3))
    assert env.extras["log"]["Metrics/landing_force_mean"] == 0


def test_stance_pitch_follows_scanner_slope_and_contact_order():
    """Fit current terrain slopes and penalize contacting feet throughout stance."""
    # Flat, toe-down, heel-down, swing, uphill, cross-slope, downhill, diagonal,
    # all missed, too few hits, and collinear hits.
    slopes = torch.tensor([[0.0, 0.0]] * 4 + [[0.25, 0.0], [0.25, -0.3], [0.25, 0.0], [0.25, 0.1]] + [[0.0, 0.0]] * 3)
    yaw = torch.tensor([0.0] * 5 + [math.pi / 2, math.pi, math.pi / 4] + [0.0] * 3)
    elevation = torch.tensor(
        [
            0.0,
            -0.35,
            0.35,
            -0.35,
            math.atan(0.25),
            -math.atan(0.3),
            -math.atan(0.25),
            math.atan(0.35 / math.sqrt(2)),
            -0.35,
            -0.35,
            -0.35,
        ]
    )
    num_envs = len(slopes)
    x, y = torch.meshgrid(torch.tensor([-0.1, 0.0, 0.1]), torch.tensor([-0.1, 0.0, 0.1]), indexing="ij")
    # A yawed scanner and shuffled rays must still yield world-frame slopes.
    xy = torch.stack((x.flatten(), y.flatten()), dim=-1)
    rotation = torch.tensor([[math.cos(0.4), -math.sin(0.4)], [math.sin(0.4), math.cos(0.4)]])
    xy = (xy @ rotation.T)[torch.tensor([4, 2, 8, 0, 1, 3, 5, 7, 6])]
    hits = torch.zeros(num_envs, 9, 3)
    hits[..., :2] = xy
    hits[..., 2] = (xy.unsqueeze(0) * slopes.unsqueeze(1)).sum(dim=-1)
    hits += torch.tensor([10.0, -5.0, 3.0])
    hits[4, 0] = float("nan")  # Partial misses still leave a usable plane.
    hits[5, 8] = float("inf")
    hits[8] = float("inf")
    hits[9, 2:] = float("nan")
    hits[10, :, 1] = 0.0  # All remaining points lie on a line.
    pitch = torch.stack([-elevation, torch.full_like(elevation, 0.8)], dim=-1)
    quaternions = quat_from_euler_xyz(torch.zeros_like(pitch), pitch, yaw[:, None].expand_as(pitch))
    robot = SimpleNamespace(
        num_bodies=2,
        find_bodies=lambda *args, **kwargs: ([0, 1], ["left_ankle_roll_link", "right_ankle_roll_link"]),
        data=SimpleNamespace(body_quat_w=SimpleNamespace(torch=quaternions)),
    )
    contact = torch.tensor([[1.0, 0.0]] * num_envs)
    contact[3, 0] = 0.0
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={
            "robot": robot,
            "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits))),
        },
        cfg=SimpleNamespace(foot_body_expr=".*_ankle_roll_link"),
        foot_contact=contact,
        foot_first_contact=torch.zeros_like(contact),
    )
    cfg = G1RewardsCfg().stance_foot_angle_penalty
    cfg.params["asset_cfg"].body_ids = [1, 0]  # Reward selection can differ from contact storage order.
    term = cfg.func(cfg, env)
    expected = torch.zeros(num_envs)
    expected[1:3] = (0.35 - cfg.params["angle_tolerance"]) ** 2
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1.0e-6, rtol=1.0e-5)
    # The original term remains touchdown-only; stance continues after that flag clears.
    touchdown_cfg = cfg.copy()
    touchdown_cfg.func = foot_touch_down_angle_penalty
    touchdown_term = touchdown_cfg.func(touchdown_cfg, env)
    torch.testing.assert_close(touchdown_term(env, **touchdown_cfg.params), torch.zeros(num_envs))
    env.foot_first_contact.copy_(contact)
    torch.testing.assert_close(touchdown_term(env, **touchdown_cfg.params), expected, atol=1.0e-6, rtol=1.0e-5)
    env.foot_first_contact.zero_()
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1.0e-6, rtol=1.0e-5)

    # Both feet can contribute during double support.
    contact[0, 1] = 1.0
    double_support = expected.clone()
    double_support[0] = (0.8 - cfg.params["angle_tolerance"]) ** 2
    torch.testing.assert_close(term(env, **cfg.params), double_support, atol=1.0e-6, rtol=1.0e-5)
    contact[0, 1] = 0.0

    # Use the scanner's latest values, not a cached slope from the first call.
    hits[0, :, 2] = 0.3 * hits[0, :, 0]
    expected[0] = (math.atan(0.3) - cfg.params["angle_tolerance"]) ** 2
    torch.testing.assert_close(term(env, **cfg.params), expected, atol=1.0e-6, rtol=1.0e-5)
    contact[:] = 0.0
    quaternions[0, 0] = float("nan")
    contact[0, 0] = 1.0
    torch.testing.assert_close(term(env, **cfg.params), torch.zeros(num_envs))


def test_base_height_uses_supporting_floor():
    """Translating both terrain and robot preserves rewards, with no sand-depth correction."""
    root_pos = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 3.8]])
    hits = torch.zeros(2, 2, 3)
    hits[:, :, 2] = torch.tensor([[-0.1, 0.1], [2.9, 3.1]])
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=SimpleNamespace(torch=root_pos),
                )
            ),
            "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=hits))),
        }
    )
    rewards = G1RewardsCfg()
    torch.testing.assert_close(rewards.base_height.func(env, **rewards.base_height.params), torch.full((2,), 0.0025))


def _sample(env):
    return env.sample.clone()


def test_contact_force_collection_replaces_nonfinite_components():
    """Invalid feedback cannot poison contact observations or modify the solver's buffer."""
    feedback = torch.tensor(
        [
            [float("nan"), float("inf"), float("-inf"), 1.0, 2.0, 3.0],
            [3.0, -4.0, 0.0, 4.0, 5.0, 6.0],
        ]
    )
    env = SimpleNamespace(
        _coupling_forces=wp.from_torch(feedback, dtype=wp.spatial_vector),
        _coupling_force_body_ids=torch.tensor([[1, 0]]),
        device="cpu",
        cfg=SimpleNamespace(foot_contact_force_threshold=1.0),
        extras={},
    )
    G1MovingPatchEnv._refresh_contact_forces(env)
    torch.testing.assert_close(env._foot_contact_force, torch.tensor([[[3.0, -4.0, 0.0], [0.0, 0.0, 0.0]]]))
    torch.testing.assert_close(env._foot_contact, torch.tensor([[True, False]]))
    assert torch.isnan(feedback[0, 0]) and torch.isposinf(feedback[0, 1]) and torch.isneginf(feedback[0, 2])
    metrics = env.extras["_observation_metrics"]
    assert metrics["Metrics/mpm/contact_force_nonfinite_fraction"] == 0.5
    # A later clean read (e.g. after reset) must not erase the pre-reset diagnostic.
    feedback[0, :3] = 0
    G1MovingPatchEnv._refresh_contact_forces(env)
    assert metrics["Metrics/mpm/contact_force_nonfinite_fraction"] == 0.5


def test_observation_metrics_capture_invalid_values_without_advancing_history():
    """Diagnostics expose current spikes and invalid components without changing observation history."""
    env = SimpleNamespace(
        num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True), extras={}, sample=torch.ones(2, 3)
    )
    group = ObservationGroupCfg(history_length=3, concatenate_terms=True)
    group.sample = ObservationTermCfg(func=_sample)
    env.observation_manager = ObservationManager({"privileged": group}, env)
    before = env.observation_manager.compute(update_history=True)["privileged"].clone()
    env.sample = torch.tensor([[2.0, float("nan"), float("inf")], [-4.0, 6.0, float("nan")]])
    torch.testing.assert_close(metric_sliderbar(env, ["sample"]), torch.zeros(2))
    metrics = env.extras["_observation_metrics"]
    prefix = "Metrics/observations/privileged/sample"
    for name, expected in {
        "mean": [-1.0, 6.0, 0.0],
        "abs_max": [4.0, 6.0, 0.0],
        "nonfinite_fraction": [0.0, 0.5, 1.0],
    }.items():
        torch.testing.assert_close(
            torch.stack([metrics[f"{prefix}/{name}_{i}"] for i in range(3)]), torch.tensor(expected)
        )
    assert torch.isnan(env.sample[0, 1]) and torch.isinf(env.sample[0, 2])
    torch.testing.assert_close(env.observation_manager.compute(update_history=False)["privileged"], before)
    env.sample = torch.full((2, 3), 7.0)
    after = env.observation_manager.compute(update_history=True)["privileged"]
    torch.testing.assert_close(after, torch.tensor([[1.0] * 6 + [7.0] * 3] * 2))


def test_standing_contact_penalty_allows_recovery():
    """Commands disable the standing term; measured motion reduces the cost of recovery steps."""
    cfg = G1RewardsCfg().break_contact
    contact = torch.tensor([[True, True], [True, False], [False, False], [False, False], [False, False]])
    velocity = torch.zeros(5, 3)
    velocity[4, 0] = cfg.params["recovery_linear_velocity"]
    commands = torch.zeros(5, 7)
    commands[3, 2] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=SimpleNamespace(torch=velocity),
            root_ang_vel_b=SimpleNamespace(torch=torch.zeros(5, 3)),
            projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]] * 5)),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        foot_contact=contact,
        command_manager=SimpleNamespace(get_command=lambda _: commands),
    )
    torch.testing.assert_close(cfg.func(env, **cfg.params), torch.tensor([0.0, 1.0, 2.0, 0.0, 2.0 / math.e]))
