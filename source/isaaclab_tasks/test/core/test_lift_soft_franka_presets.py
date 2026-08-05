# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free gates for the Franka soft-beam and cloth lift presets and their MDP terms."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.utils import resolve_presets

##
# Preset resolution
##


def test_supported_physics_presets_resolve():
    """The soft task keeps both PhysX and Newton; the cloth task keeps only Newton."""
    soft_newton = resolve_presets(FrankaSoftEnvCfg(), ("newton_mjwarp_vbd_proxy",))
    assert type(soft_newton.sim.physics).__name__ == "NewtonCfg"

    soft_physx = resolve_presets(FrankaSoftEnvCfg(), ("isaacsim_physx",))
    assert type(soft_physx.sim.physics).__name__ == "PhysxCfg"
    # the PhysX beam rests on the table via explicit collision offsets instead of the 20 mm default
    collision_props = soft_physx.scene.deformable.spawn.collision_props
    assert collision_props and collision_props[0].rest_offset == pytest.approx(0.0025)

    cloth_newton = resolve_presets(FrankaClothEnvCfg(), ("newton_mjwarp_vbd_proxy",))
    assert type(cloth_newton.sim.physics).__name__ == "NewtonCfg"


##
# MDP term math (fake scene / manager, no simulator)
##


class _FakeScene:
    """Minimal ``env.scene`` supporting ``scene[name]`` and ``env_origins``."""

    def __init__(self, assets, env_origins=None):
        self._assets = assets
        self.env_origins = env_origins

    def __getitem__(self, key):
        return self._assets[key]


class _FakeEventManager:
    """Round-trips a single event term cfg through ``get_term_cfg`` / ``set_term_cfg``."""

    def __init__(self, params):
        self._cfg = SimpleNamespace(params=params)

    def get_term_cfg(self, name):
        return self._cfg

    def set_term_cfg(self, name, cfg):
        self._cfg = cfg


def _gravity_env(step):
    params = {"gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])}
    return SimpleNamespace(common_step_counter=step, event_manager=_FakeEventManager(params))


def test_gravity_range_linear_validates_step_order():
    """A non-increasing step window is rejected loudly."""
    with pytest.raises(ValueError, match="end_step must be greater"):
        mdp.gravity_range_linear(
            _gravity_env(0),
            [],
            "variable_gravity",
            start_gravity_z=-1e-4,
            end_gravity_z=-9.81,
            start_step=10,
            end_step=10,
        )


def test_gravity_range_linear_clamps_and_interpolates():
    """Gravity clamps outside the window, interpolates inside, and is written back to the event term."""
    kwargs = dict(event_name="g", start_gravity_z=-1.0, end_gravity_z=-9.0, start_step=100, end_step=300)

    # before the window: clamped to the start value
    assert mdp.gravity_range_linear(_gravity_env(0), [], **kwargs)["gravity_z"] == pytest.approx(-1.0)
    # midpoint: halfway between start and end
    assert mdp.gravity_range_linear(_gravity_env(200), [], **kwargs)["gravity_z"] == pytest.approx(-5.0)

    # after the window: clamped to the end value and pushed into the event cfg
    env = _gravity_env(10_000)
    assert mdp.gravity_range_linear(env, [], **kwargs)["gravity_z"] == pytest.approx(-9.0)
    ramped = env.event_manager.get_term_cfg("g").params["gravity_distribution_params"]
    assert ramped[0] == [0.0, 0.0, pytest.approx(-9.0)]


def _articulation(joint_vel, actuators):
    return SimpleNamespace(
        data=SimpleNamespace(joint_vel=SimpleNamespace(torch=joint_vel)),
        actuators=actuators,
    )


def test_joint_vel_out_of_sim_limit_triggers_per_actuator():
    """A joint over its actuator's sim limit terminates; the joint_ids selection is respected."""
    # env 0 within limits; env 1 exceeds the arm limit (2.0) on joint index 2
    joint_vel = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 9.0, 0.5]])
    actuators = {
        "arm": SimpleNamespace(joint_indices=[0, 1, 2], velocity_limit_sim=2.0),
        "hand": SimpleNamespace(joint_indices=[3], velocity_limit_sim=1.0),
    }
    env = SimpleNamespace(scene=_FakeScene({"robot": _articulation(joint_vel, actuators)}))

    out = mdp.joint_vel_out_of_sim_limit(env, SimpleNamespace(name="robot", joint_ids=None))
    assert out.tolist() == [False, True]

    # excluding the offending joint clears the violation
    out = mdp.joint_vel_out_of_sim_limit(env, SimpleNamespace(name="robot", joint_ids=[0, 1, 3]))
    assert out.tolist() == [False, False]


def test_deformable_outside_bounds_covers_z():
    """A node dropped below the z floor terminates even when x and y stay inside."""
    env_origins = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # env 0 node 0 falls to env-frame z = -0.5 (below the floor); env 1 stays inside the box
    nodal_pos_w = torch.tensor(
        [
            [[10.5, 0.0, -0.5], [10.5, 0.0, 0.2]],
            [[0.5, 0.0, 0.2], [0.5, 0.0, 0.3]],
        ]
    )
    asset = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=SimpleNamespace(torch=nodal_pos_w)))
    env = SimpleNamespace(scene=_FakeScene({"deformable": asset}, env_origins=env_origins))

    out = mdp.deformable_outside_bounds(
        env,
        x_bounds=(0.0, 1.0),
        y_bounds=(-0.5, 0.5),
        z_bounds=(-0.02, 1.0),
        asset_cfg=SimpleNamespace(name="deformable"),
    )
    assert out.tolist() == [True, False]
