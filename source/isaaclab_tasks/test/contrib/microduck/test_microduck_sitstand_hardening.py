# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral contracts for the SitStand seated perturbations and neck load cost."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.sitstand.sitstand_env_cfg import MicroDuckSitStandFlatEnvCfg

pytestmark = pytest.mark.unit


class _Robot:
    """State-only articulation double; the write records what the event sends to simulation."""

    def __init__(self, count):
        self.data = SimpleNamespace(
            root_link_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(count, 1)),
            root_link_vel_w=SimpleNamespace(torch=torch.zeros(count, 6)),
            projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1)),
        )
        self.actuators = SimpleNamespace(applied_effort=SimpleNamespace(torch=torch.zeros(count, 8)))
        self.writes = []

    def write_root_link_velocity_to_sim_index(self, root_velocity, env_ids):
        self.writes.append((root_velocity.clone(), env_ids.clone()))


class _Commands:
    def __init__(self, count):
        self.posture = SimpleNamespace(command=torch.zeros(count, 3), alpha=torch.zeros(count))
        self.head = SimpleNamespace(
            command=torch.full((count, 4), 0.25),
            dim=4,
            cfg=SimpleNamespace(ranges=((-0.5, 0.5),) * 4),
            time_left=torch.full((count,), 4.0),
        )

    def get_term(self, name):
        return self.posture if name == "base_velocity" else self.head

    def get_command(self, name):
        return self.get_term(name).command


def _env(count):
    return SimpleNamespace(
        num_envs=count, device="cpu", scene={"robot": _Robot(count)}, command_manager=_Commands(count)
    )


def test_head_jerk_requires_a_seated_target_and_finished_ramp_without_resetting_dwell():
    env = _env(5)
    posture, head = env.command_manager.posture, env.command_manager.head
    posture.command[:, 0] = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    posture.alpha[:] = torch.tensor([1.0, 0.979, 1.0, 0.98, 1.0])

    mdp.seated_head_command_jerk(env, torch.arange(4), prob=1.0, flip_frac=1.0)

    torch.testing.assert_close(head.command[:, 0], torch.tensor([-0.25, 0.25, 0.25, -0.25, 0.25]))
    torch.testing.assert_close(head.time_left, torch.full((5,), 4.0))


def test_head_jerk_uses_live_ranges_and_is_disabled_at_zero_probability():
    env = _env(2)
    posture, head = env.command_manager.posture, env.command_manager.head
    posture.command[:, 0] = 1.0
    posture.alpha[:] = 1.0
    mdp.seated_head_command_jerk(env, torch.arange(2), prob=0.0)
    torch.testing.assert_close(head.command, torch.full((2, 4), 0.25))

    head.cfg.ranges = ((0.1, 0.1), (-0.2, -0.2), (0.3, 0.3), (-0.4, -0.4))
    mdp.seated_head_command_jerk(env, torch.arange(2), prob=1.0, flip_frac=0.0)
    torch.testing.assert_close(head.command, torch.tensor([[0.1, -0.2, 0.3, -0.4]]).repeat(2, 1))


def test_backward_tip_adds_world_rotated_angular_velocity_only_to_seated_environments():
    env = _env(4)
    robot, posture = env.scene["robot"], env.command_manager.posture
    posture.command[:, 0] = torch.tensor([1.0, 1.0, 0.0, 1.0])
    posture.alpha[:] = torch.tensor([1.0, 1.0, 1.0, 0.979])
    robot.data.root_link_vel_w.torch[:] = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    # A 90-degree world yaw maps body -y to world +x. Quaternion order is xyzw.
    robot.data.root_link_quat_w.torch[1] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])

    mdp.seated_backward_tip(env, torch.arange(4), ang_vel_range=(3.0, 3.0), lateral_frac=0.0)

    velocity, ids = robot.writes[0]
    assert ids.tolist() == [0, 1]
    torch.testing.assert_close(
        velocity, torch.tensor([[1.0, 2.0, 3.0, 0.1, -2.8, 0.3], [1.0, 2.0, 3.0, 3.1, 0.2, 0.3]])
    )
    mdp.seated_backward_tip(env, torch.arange(4), ang_vel_range=(0.0, 0.0))
    assert len(robot.writes) == 1


def test_neck_strain_prices_only_backward_lean_and_fades_with_the_slewed_posture():
    env = _env(7)
    robot, posture = env.scene["robot"], env.command_manager.posture
    posture.alpha[:] = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 1.0])
    angles = torch.deg2rad(torch.tensor([5.0, 30.0, -30.0, 18.5, 30.0, 30.0, 0.0]))
    robot.data.projected_gravity_b.torch[:, 0] = -torch.sin(angles)
    robot.data.projected_gravity_b.torch[:, 2] = -torch.cos(angles)
    # Huge leg torque must not affect the four selected head joints.
    robot.actuators.applied_effort.torch[:] = 9.0
    robot.actuators.applied_effort.torch[:, [4, 1, 6, 2]] = 0.15
    selection = SceneEntityCfg("robot", joint_ids=[4, 1, 6, 2])

    cost = mdp.neck_torque_when_tilted(env, command_name="base_velocity", asset_cfg=selection)

    torch.testing.assert_close(cost, torch.tensor([0.0, 2.25, 0.0, 1.125, 0.0, 1.125, 0.0]))
    weight = MicroDuckSitStandFlatEnvCfg().rewards.neck_strain_leaned.weight
    assert weight == -0.5
    assert bool((cost * weight <= 0.0).all())


def test_seated_disturbance_curricula_change_live_event_parameters_at_exact_boundaries():
    cfg = MicroDuckSitStandFlatEnvCfg()
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda name: getattr(cfg.events, name)))
    assert cfg.events.seated_head_jerk.interval_range_s == (0.3, 0.8)
    assert cfg.events.seated_tip.interval_range_s == (3.0, 6.0)
    assert cfg.events.seated_tip.params["prob"] == 0.5
    assert cfg.events.seated_tip.params["lateral_frac"] == 0.3
    cases = [
        (0, 0.0, (0.0, 0.0)),
        (1500 * 24 - 1, 0.0, (0.0, 0.0)),
        (1500 * 24, 0.2, (1.0, 2.5)),
        (2000 * 24 - 1, 0.2, (1.0, 2.5)),
        (2000 * 24, 0.4, (1.5, 3.25)),
        (2500 * 24, 0.4, (2.0, 4.0)),
    ]
    for step, head_probability, tip_range in cases:
        env.common_step_counter = step
        for term in (cfg.curriculum.seated_head_jerk_prob, cfg.curriculum.seated_tip_range):
            term.func(env, [], **term.params)
        assert cfg.events.seated_head_jerk.params["prob"] == head_probability
        assert cfg.events.seated_tip.params["ang_vel_range"] == tip_range
