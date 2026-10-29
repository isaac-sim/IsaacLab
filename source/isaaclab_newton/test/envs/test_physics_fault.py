# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics faults reset only the failed environment, on Newton with MuJoCo Warp.

Each environment holds a cube that starts high above the ground. MuJoCo Warp has room for fewer constraints per
world than a cube resting on the ground needs.
"""

from collections.abc import Iterator

import pytest
import torch
from isaaclab_newton.envs.mdp import physics_fault
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 20.0)),
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
    fault = DoneTerm(func=physics_fault)


@configclass
class _EnvCfg(ManagerBasedRLEnvCfg):
    scene = _SceneCfg(num_envs=2, env_spacing=3.0)
    sim = SimulationCfg(
        dt=0.01,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=8),
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
            use_cuda_graph=False,
        ),
    )
    decimation = 2
    episode_length_s = 10.0
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


def test_fault_resets_only_the_failed_environment(env: ManagerBasedRLEnv):
    """A non-finite state or a world's constraint overflow resets that environment; the other keeps running."""
    env.reset()
    actions = torch.zeros(env.num_envs, 0, device=env.device)
    _, _, terminated, _, _ = env.step(actions)
    assert not terminated.any()

    cube = env.scene["cube"]
    pose = cube.data.root_pose_w.torch[1:].clone()
    pose[:, 0] = float("nan")
    cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([1], device=env.device))
    _, _, terminated, _, _ = env.step(actions)
    assert terminated.tolist() == [False, True]
    assert env.episode_length_buf.tolist() == [2, 0]
    assert torch.isfinite(cube.data.root_pose_w.torch).all()

    # Resting on the ground overflows the constraint capacity of the first world only.
    pose = cube.data.root_pose_w.torch[:1].clone()
    pose[:, 2] = 0.05
    cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=env.device))
    _, _, terminated, _, _ = env.step(actions)
    assert terminated.tolist() == [True, False]
    _, _, terminated, _, _ = env.step(actions)
    assert not terminated.any()
