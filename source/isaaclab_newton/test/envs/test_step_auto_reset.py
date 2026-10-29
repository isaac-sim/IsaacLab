# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stepping without auto-reset, on Newton.

Each environment holds a cube resting on the ground. The odd environments terminate at once, and all environments
time out after two steps.
"""

from collections.abc import Iterator

import pytest
import torch
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _odd_environments(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device) % 2 == 1


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )


@configclass
class _PolicyCfg(ObservationGroupCfg):
    cube = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})


@configclass
class _ObservationsCfg:
    policy = _PolicyCfg()


@configclass
class _ActionsCfg:
    pass


@configclass
class _EventsCfg:
    reset = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class _TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    odd = DoneTerm(func=_odd_environments)


@configclass
class _EnvCfg(ManagerBasedRLEnvCfg):
    scene = _SceneCfg(num_envs=4, env_spacing=3.0)
    sim = SimulationCfg(dt=0.01, physics=NewtonCfg(use_cuda_graph=False))
    decimation = 2
    episode_length_s = 0.04
    observations = _ObservationsCfg()
    actions = _ActionsCfg()
    events = _EventsCfg()
    rewards = {}
    terminations = _TerminationsCfg()


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def env(request) -> Iterator[ManagerBasedRLEnv]:
    cfg = _EnvCfg()
    cfg.sim.device = request.param
    env = ManagerBasedRLEnv(cfg)
    try:
        yield env
    finally:
        env.close()


def test_step_without_auto_reset_leaves_resets_to_the_caller(env: ManagerBasedRLEnv, monkeypatch):
    """Without auto-reset, terminated environments continue their episodes until the caller or a visualizer resets
    them."""
    env.reset()
    actions = torch.zeros(env.num_envs, 0, device=env.device)
    for _ in range(3):
        _, _, terminated, time_outs, _ = env.step(actions, auto_reset=False)
    assert terminated.tolist() == [False, True, False, True]
    assert time_outs.all()
    assert env.episode_length_buf.tolist() == [3, 3, 3, 3]

    env.reset(env_ids=torch.tensor([1, 3], device=env.device))
    assert env.episode_length_buf.tolist() == [3, 0, 3, 0]

    # A reset requested from a visualizer resets the terminated environments too: they were not reset in the step.
    monkeypatch.setattr(env.sim, "consume_reset_request", lambda: True)
    env.step(actions, auto_reset=False)
    assert env.episode_length_buf.tolist() == [0, 0, 0, 0]
