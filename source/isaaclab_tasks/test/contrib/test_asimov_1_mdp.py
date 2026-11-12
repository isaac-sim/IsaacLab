# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Asimov-1 MDP terms that keep state or combine several physical quantities."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.asimov_1.mdp import rewards

TARGET_HEIGHT = 0.1
FEET = ["left_foot", "right_foot"]


def _proxy(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


class _Scene(dict):
    """Scene lookup by name, with the sensors as an attribute."""

    sensors: dict


class _FootScene:
    """Two environments with two feet whose heights and contact forces are set directly."""

    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.step_dt = 0.02
        self.extras = {}
        self.foot_z = torch.zeros(2, 2)
        self.force_z = torch.zeros(2, 2)
        self.first_contact = torch.zeros(2, 2)
        self.command = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2, 2, 1)
        robot_data = SimpleNamespace(
            body_link_pos_w=SimpleNamespace(torch=None),
            body_link_quat_w=_proxy(quat),
            body_link_lin_vel_w=_proxy(torch.zeros(2, 2, 3)),
            body_link_ang_vel_w=_proxy(torch.zeros(2, 2, 3)),
        )
        self._robot = SimpleNamespace(data=robot_data, body_names=FEET, device="cpu")
        sensor_data = SimpleNamespace(net_forces_w=SimpleNamespace(torch=None))
        self._sensor = SimpleNamespace(
            data=sensor_data, body_names=FEET, compute_first_contact=lambda dt: _proxy(self.first_contact)
        )
        self.scene = _Scene(robot=self._robot)
        self.scene.sensors = {"contact_forces": self._sensor}
        self.command_manager = SimpleNamespace(get_command=lambda name: self.command)
        self.asset_cfg = SceneEntityCfg("robot", body_names=FEET)
        self.asset_cfg.body_ids = [0, 1]
        self.sensor_cfg = SceneEntityCfg("contact_forces", body_names=FEET)
        self.sensor_cfg.body_ids = [0, 1]

    def set_feet(self, foot_z, in_contact, first_contact=None):
        self.foot_z = torch.as_tensor(foot_z, dtype=torch.float32)
        in_contact = torch.as_tensor(in_contact, dtype=torch.float32)
        position = torch.zeros(2, 2, 3)
        position[..., 2] = self.foot_z
        self._robot.data.body_link_pos_w.torch = position
        force = torch.zeros(2, 2, 3)
        force[..., 2] = 100.0 * in_contact
        self._sensor.data.net_forces_w.torch = force
        self.first_contact = torch.zeros(2, 2) if first_contact is None else torch.as_tensor(first_contact).float()


def _swing_height_term(env: _FootScene) -> tuple[rewards.feet_swing_height, dict]:
    params = {
        "target_height": TARGET_HEIGHT,
        "offset": (0.0, 0.0, 0.0),
        "command_name": "base_velocity",
        "command_threshold": 0.05,
        "sensor_cfg": env.sensor_cfg,
        "asset_cfg": env.asset_cfg,
    }
    term = rewards.feet_swing_height.__new__(rewards.feet_swing_height)
    rewards.feet_swing_height.__init__(term, SimpleNamespace(params=params), env)
    return term, params


def _swing(env: _FootScene, term, params, ground: list[float], peak: float, landing: list[float]) -> torch.Tensor:
    """Swing the left foot from ``ground`` to ``peak`` above it and land it at ``landing``; return the landing cost."""
    env.set_feet([[g, g] for g in ground], [[1, 1], [1, 1]])
    term(env, **params)
    env.set_feet([[g + peak, g] for g in ground], [[0, 1], [0, 1]])
    term(env, **params)
    env.set_feet([[z, g] for z, g in zip(landing, ground)], [[1, 1], [1, 1]], first_contact=[[1, 0], [1, 0]])
    return term(env, **params)


def test_swing_height_is_measured_from_the_last_ground_contact():
    """A swing is measured from where the foot last stood, at any elevation and after stepping up."""
    env = _FootScene()
    term, params = _swing_height_term(env)

    # a 0.04 m swing that lands on a step 0.3 m higher, on ground at two elevations
    step_up = _swing(env, term, params, ground=[0.0, 2.5], peak=0.04, landing=[0.3, 2.8])
    # the next swing starts on the step and reaches the target height above it
    on_step = _swing(env, term, params, ground=[0.3, 2.8], peak=TARGET_HEIGHT, landing=[0.3, 2.8])

    torch.testing.assert_close(step_up, torch.full((2,), (0.04 / TARGET_HEIGHT - 1.0) ** 2))
    torch.testing.assert_close(on_step, torch.zeros(2), rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("reset_ids", [torch.tensor([0]), slice(0, 1)])
def test_swing_height_partial_reset_drops_the_interrupted_swing_only(reset_ids):
    """A swing cut short by a reset does not reach the next episode; the other environment keeps its swing."""
    env = _FootScene()
    term, params = _swing_height_term(env)
    env.set_feet([[0.0, 0.0], [0.0, 0.0]], [[1, 1], [1, 1]])
    term(env, **params)
    # both environments swing the left foot to the target height
    env.set_feet([[TARGET_HEIGHT, 0.0], [TARGET_HEIGHT, 0.0]], [[0, 1], [0, 1]])
    term(env, **params)

    # environment 0 resets mid-swing and restarts 0.3 m higher with its left foot in the air
    term.reset(reset_ids)
    env.set_feet([[0.32, 0.3], [0.04, 0.0]], [[0, 1], [0, 1]])
    term(env, **params)
    # the new episode only rises 0.04 m above where it started
    env.set_feet([[0.36, 0.3], [0.04, 0.0]], [[0, 1], [0, 1]])
    term(env, **params)
    env.set_feet([[0.3, 0.3], [0.0, 0.0]], [[1, 1], [1, 1]], first_contact=[[1, 0], [1, 0]])
    cost = term(env, **params)

    # environment 0 is scored on its new 0.04 m swing, environment 1 on the target-height swing it kept
    expected = torch.tensor([(0.04 / TARGET_HEIGHT - 1.0) ** 2, 0.0])
    torch.testing.assert_close(cost, expected, rtol=0.0, atol=1e-6)


def test_swing_height_full_reset_drops_every_swing():
    """A reset without environment indices clears every environment."""
    env = _FootScene()
    term, params = _swing_height_term(env)
    env.set_feet([[0.0, 0.0], [0.0, 0.0]], [[1, 1], [1, 1]])
    term(env, **params)
    env.set_feet([[TARGET_HEIGHT, 0.0], [TARGET_HEIGHT, 0.0]], [[0, 1], [0, 1]])
    term(env, **params)

    term.reset()
    env.set_feet([[0.0, 0.0], [0.0, 0.0]], [[1, 1], [1, 1]], first_contact=[[1, 0], [1, 0]])
    cost = term(env, **params)

    torch.testing.assert_close(cost, torch.ones(2))


def test_swing_height_rejects_feet_in_a_different_order():
    """The robot and contact-sensor selections must list the feet in the same order."""
    env = _FootScene()
    env.sensor_cfg.body_ids = [1, 0]
    with pytest.raises(ValueError, match="must match in order"):
        _swing_height_term(env)


def _momentum_env(pos, lin_vel, ang_vel, mass, inertia, quat=None) -> SimpleNamespace:
    num_bodies = len(mass)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(1, num_bodies, 1) if quat is None else quat
    data = SimpleNamespace(
        body_mass=_proxy(torch.tensor([mass])),
        body_inertia=_proxy(torch.tensor([inertia]).reshape(1, num_bodies, 9)),
        body_com_pos_w=_proxy(torch.tensor([pos])),
        body_com_lin_vel_w=_proxy(torch.tensor([lin_vel])),
        body_com_ang_vel_w=_proxy(torch.tensor([ang_vel])),
        body_link_quat_w=_proxy(quat),
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


def test_angular_momentum_includes_link_rotation_and_orbital_motion():
    """A link spinning in place and two links orbiting their common center each contribute their momentum."""
    diagonal = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    asset_cfg = SceneEntityCfg("robot")

    # one link spinning about its own center of mass has no orbital momentum: L = I * omega
    spinning = _momentum_env(
        pos=[[0.0, 0.0, 0.0]], lin_vel=[[0.0, 0.0, 0.0]], ang_vel=[[0.0, 0.0, 2.0]], mass=[5.0], inertia=[diagonal]
    )
    torch.testing.assert_close(
        rewards.whole_body_angular_momentum(spinning, asset_cfg), torch.tensor([[0.0, 0.0, 6.0]])
    )

    # the inertia axes follow the link: the same spin about world z with the link rolled 30 deg about x
    rolled = _momentum_env(
        pos=[[0.0, 0.0, 0.0]],
        lin_vel=[[0.0, 0.0, 0.0]],
        ang_vel=[[0.0, 0.0, 2.0]],
        mass=[5.0],
        inertia=[diagonal],
        quat=torch.tensor([[[math.sin(math.pi / 12), 0.0, 0.0, math.cos(math.pi / 12)]]]),
    )
    torch.testing.assert_close(
        rewards.whole_body_angular_momentum(rolled, asset_cfg),
        torch.tensor([[0.0, -math.sin(math.pi / 3), 5.5]]),
        rtol=0.0,
        atol=1e-5,
    )

    # two point-like links circling their common center, which is away from the origin and moving,
    # without spinning: L = sum(m * r x v) about the common center
    zero_inertia = [[0.0] * 3] * 3
    orbiting = _momentum_env(
        pos=[[6.0, -2.0, 1.0], [4.0, -2.0, 1.0]],
        lin_vel=[[0.5, 3.5, 0.0], [0.5, -2.5, 0.0]],
        ang_vel=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        mass=[2.0, 2.0],
        inertia=[zero_inertia, zero_inertia],
    )
    torch.testing.assert_close(
        rewards.whole_body_angular_momentum(orbiting, asset_cfg), torch.tensor([[0.0, 0.0, 12.0]]), rtol=0.0, atol=1e-5
    )
