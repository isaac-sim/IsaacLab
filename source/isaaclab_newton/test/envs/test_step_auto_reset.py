# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stepping without auto-reset, on Newton.

Each environment holds a cube resting on the ground. The odd environments terminate at once, and environment 0
terminates at the first step of its episodes only.
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


def _first_step_of_environment_0(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (torch.arange(env.num_envs, device=env.device) == 0) & (env.episode_length_buf == 1)


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
    odd = DoneTerm(func=_odd_environments)
    first_step = DoneTerm(func=_first_step_of_environment_0)


@configclass
class _EnvCfg(ManagerBasedRLEnvCfg):
    scene = _SceneCfg(num_envs=4, env_spacing=3.0)
    sim = SimulationCfg(dt=0.01, physics=NewtonCfg(use_cuda_graph=False))
    decimation = 2
    episode_length_s = 1.0
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


def test_step_without_auto_reset_keeps_ended_episodes_pending(env: ManagerBasedRLEnv, monkeypatch):
    """Without auto-reset, environments whose episodes ended stay pending, even once their termination no longer
    holds, until a step with auto-reset, the caller or a visualizer resets them."""
    env.reset()
    actions = torch.zeros(env.num_envs, 0, device=env.device)
    _, _, terminated, _, _ = env.step(actions, auto_reset=False)
    assert terminated.tolist() == [True, True, False, True]
    _, _, terminated, _, _ = env.step(actions, auto_reset=False)
    assert terminated.tolist() == [False, True, False, True]
    assert env.pending_reset_buf.tolist() == [True, True, False, True]
    assert env.episode_length_buf.tolist() == [2, 2, 2, 2]

    env.step(actions)
    assert env.episode_length_buf.tolist() == [0, 0, 3, 0]
    assert not env.pending_reset_buf.any()

    env.step(actions, auto_reset=False)
    env.reset(env_ids=torch.tensor([1], device=env.device))
    assert env.pending_reset_buf.tolist() == [True, False, False, True]

    # A reset requested from a visualizer resets every environment, the pending ones included.
    monkeypatch.setattr(env.sim, "consume_reset_request", lambda: True)
    env.step(actions, auto_reset=False)
    assert env.episode_length_buf.tolist() == [0, 0, 0, 0]
    assert not env.pending_reset_buf.any()
