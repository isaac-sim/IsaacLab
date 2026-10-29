# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success of environments that each run one of three tasks on a kinematic box and two balls, on Newton.

The tasks hold the same objects: ``first_in`` succeeds with the first ball in the box, ``second_in`` with the second
ball in it, and ``first_out`` with the first ball out of it. A ball away from them stands in for the gripper link.
"""

from collections.abc import Iterator

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.heterogeneous_tasks import mdp
from isaaclab_tasks.contrib.heterogeneous_tasks.mdp.utils import GRIPPER

from heterogeneous_tasks_test_utils import place  # isort: skip

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _object(spawn: sim_utils.SpawnerCfg, pos: tuple[float, float, float], kinematic: bool = False) -> RigidObjectCfg:
    spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic)
    spawn.mass_props = sim_utils.MassCfg(mass=0.1)
    spawn.collision_props = sim_utils.UsdPhysicsCollisionCfg()
    init_state = RigidObjectCfg.InitialStateCfg(pos=pos)
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/object", spawn=spawn, init_state=init_state)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    gripper = _object(sim_utils.SphereCfg(radius=0.03), (1.5, 1.0, 0.03)).replace(
        prim_path=f"{{ENV_REGEX_NS}}/{GRIPPER}"
    )
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/[^/]+", filter_prim_paths_expr=["{ENV_REGEX_NS}/[^/]+"], update_period=0.0
    )


@configclass
class _PolicyCfg(ObservationGroupCfg):
    gripper = ObservationTermCfg(func=isaaclab_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("gripper")})


@configclass
class _ObservationsCfg:
    policy = _PolicyCfg()


@configclass
class _ActionsCfg:
    pass


@configclass
class _TerminationsCfg:
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)


@configclass
class _EnvCfg(ManagerBasedRLEnvCfg):
    scene = _SceneCfg(num_envs=4, env_spacing=3.0)
    sim = SimulationCfg(
        dt=1.0 / 120.0,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
            use_cuda_graph=False,
        ),
    )
    decimation = 2
    episode_length_s = 100.0
    observations = _ObservationsCfg()
    actions = _ActionsCfg()
    events = {}
    rewards = {}
    terminations = _TerminationsCfg()


def _task(func, ball: str) -> _EnvCfg:
    """A task of the box and the balls that succeeds by ``func`` of one ball and the box."""
    task = _EnvCfg()
    task.scene.gripper = None
    task.scene.box = _object(sim_utils.CuboidCfg(size=(0.3, 0.3, 0.1)), (0.0, 0.0, 0.05), kinematic=True)
    task.scene.ball = _object(sim_utils.SphereCfg(radius=0.03), (0.5, 0.0, 0.03))
    task.scene.ball_b = _object(sim_utils.SphereCfg(radius=0.03), (1.0, 0.0, 0.03))
    params = {"objects": [SceneEntityCfg(ball)], "container": SceneEntityCfg("box")}
    task.terminations.success = DoneTerm(func=func, params=params)
    return task


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def env(request) -> Iterator[ManagerBasedRLEnv]:
    cfg = _EnvCfg(episode_length_s=0.0)
    cfg.sim.device = request.param
    cfg.terminations.success = DoneTerm(func=mdp.task_success)
    cfg.tasks = {
        "first_in": _task(mdp.object_in_container, "ball"),
        "second_in": _task(mdp.object_in_container, "ball_b"),
        "first_out": _task(mdp.object_outside_of, "ball"),
    }
    env = ManagerBasedRLEnv(cfg)
    try:
        yield env
    finally:
        env.close()


def test_success_follows_the_predicate_and_objects_of_each_task(env: ManagerBasedRLEnv):
    """One term evaluates each environment by its task: the predicate, and the ball it names."""
    assert env.termination_manager.active_terms == ["time_out", "success"]
    env.task_manager.set_task([0, 1, 2, 3], [0, 1, 2, 2])
    env.reset()
    on, beside, away = (-0.06, 0.0, 0.15), (0.5, 0.0, 0.03), (1.0, 0.0, 0.03)
    # The first ball is object_1 and the second object_2 in every task.
    place(env, "object_1", [on, on, on, beside])
    place(env, "object_2", [away, away, (0.06, 0.0, 0.15), away])
    for _ in range(30):
        env.step(torch.zeros(env.num_envs, 0, device=env.device), auto_reset=False)
    assert env.termination_manager.get_term("success").tolist() == [True, False, False, True]
