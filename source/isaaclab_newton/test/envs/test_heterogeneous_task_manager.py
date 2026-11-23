# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environments that each run one of three tasks, on Newton.

The ``cube`` task has a table and a cube and times out after two steps. The ``spheres`` task has a wider table
and two spheres, and succeeds at once. The ``cube_left`` task has the objects of the ``cube`` task with the cube
placed elsewhere. All tables are kinematic.
"""

from collections.abc import Iterator

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    EventTermCfg,
    HeterogeneousTaskManager,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _object(name: str, spawn: sim_utils.SpawnerCfg, pos: tuple[float, float, float], mass: float) -> RigidObjectCfg:
    spawn.mass_props = sim_utils.MassCfg(mass=mass)
    spawn.collision_props = sim_utils.UsdPhysicsCollisionCfg()
    spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg()
    init_state = RigidObjectCfg.InitialStateCfg(pos=pos)
    return RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn, init_state=init_state)


def _table(width: float) -> RigidObjectCfg:
    table = _object("table", sim_utils.CuboidCfg(size=(width, width, 0.1)), (0.0, 0.0, 0.05), 10.0)
    table.spawn.rigid_props.kinematic_enabled = True
    return table


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())


def _task_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.task_manager.task_ids.unsqueeze(1).float()


@configclass
class _PolicyCfg(ObservationGroupCfg):
    task = ObservationTermCfg(func=_task_ids)


@configclass
class _ObservationsCfg:
    policy = _PolicyCfg()


@configclass
class _EventsCfg:
    reset = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class _ActionsCfg:
    pass


@configclass
class _TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _EnvCfg(ManagerBasedRLEnvCfg):
    scene = _SceneCfg(num_envs=4, env_spacing=3.0)
    sim = SimulationCfg(
        dt=0.01,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            collision_cfg=NewtonCollisionPipelineCfg(broad_phase="sap"),
            use_cuda_graph=False,
        ),
    )
    decimation = 2
    observations = _ObservationsCfg()
    actions = _ActionsCfg()
    events = _EventsCfg()
    rewards = {}
    terminations = _TerminationsCfg()


@configclass
class _CubeTaskCfg(_EnvCfg):
    episode_length_s = 0.04

    def __post_init__(self):
        self.scene.table = _table(0.6)
        self.scene.cube = _object("cube", sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)), (0.0, 0.0, 0.2), 1.0)
        self.terminations.success = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": -1.0, "asset_cfg": SceneEntityCfg("cube")}
        )


@configclass
class _SpheresTaskCfg(_EnvCfg):
    episode_length_s = 1.0

    def __post_init__(self):
        self.scene.table = _table(0.8)
        self.scene.ball_a = _object("ball_a", sim_utils.SphereCfg(radius=0.05), (0.2, 0.0, 0.2), 2.0)
        self.scene.ball_b = _object("ball_b", sim_utils.SphereCfg(radius=0.05), (-0.2, 0.0, 0.2), 2.0)
        self.terminations.success = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 10.0, "asset_cfg": SceneEntityCfg("ball_b")}
        )


@configclass
class _CubeLeftTaskCfg(_CubeTaskCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.cube.init_state.pos = (0.0, 0.2, 0.2)


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def env(request) -> Iterator[ManagerBasedRLEnv]:
    tasks = {"cube": _CubeTaskCfg(), "spheres": _SpheresTaskCfg(), "cube_left": _CubeLeftTaskCfg()}
    cfg = _EnvCfg(episode_length_s=0.0)
    HeterogeneousTaskManager.compose_tasks(cfg, tasks)
    cfg.sim.device = request.param
    env = ManagerBasedRLEnv(cfg)
    try:
        yield env
    finally:
        env.close()


def test_environments_run_their_tasks(env: ManagerBasedRLEnv):
    """Reset places each task's objects, shared ones at the task's own pose; terminations and time-outs follow the
    task of each environment."""
    assert env.task_manager.task_names == ["cube", "spheres", "cube_left"]
    assert set(env.termination_manager.active_terms) == {
        "time_out",
        "cube_success",
        "spheres_success",
        "cube_left_success",
    }
    assert env.max_episode_length == 50
    # The success terms of the tasks name the slots of their objects; no task has a failure term.
    terms = env.task_manager.task_terms("success")
    assert [terms[task].params["asset_cfg"].name for task in range(3)] == ["object_1", "object_2", "object_1"]
    assert env.task_manager.task_terms("failure") == {}

    env.task_manager.set_task([0, 1, 2, 3], [0, 1, 0, 1])
    observations, _ = env.reset()
    # The observation terms of the environment read the task of each environment.
    assert observations["policy"][:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    # The second slot holds the cube or the first sphere.
    masses = env.scene["object_1"].data.body_mass.torch[:, 0]
    torch.testing.assert_close(masses, torch.tensor([1.0, 2.0, 1.0, 2.0], device=env.device))
    # The third slot holds the placeholder or the second sphere.
    masses = env.scene["object_2"].data.body_mass.torch[:, 0]
    torch.testing.assert_close(masses, torch.tensor([0.01, 2.0, 0.01, 2.0], device=env.device))
    # The cube task parks its placeholder in the third slot; the spheres task places its second sphere there.
    positions = env.scene["object_2"].data.root_pos_w.torch - env.scene.env_origins
    assert positions[0, 1] < -40.0 and positions[2, 1] < -40.0
    torch.testing.assert_close(positions[[1, 3]], torch.tensor([[-0.2, 0.0, 0.2]] * 2, device=env.device))

    actions = torch.zeros(env.num_envs, 0, device=env.device)
    _, _, terminated, time_outs, _ = env.step(actions)
    assert terminated.tolist() == [False, True, False, True] and not time_outs.any()
    _, _, terminated, time_outs, _ = env.step(actions)
    assert terminated.tolist() == [False, True, False, True]
    assert time_outs.tolist() == [True, False, True, False]
    # Both tables stay kinematic.
    torch.testing.assert_close(
        env.scene["object_0"].data.root_pos_w.torch[:, 2], torch.full((4,), 0.05, device=env.device)
    )

    # cube_left shares the cube task's candidate, the first of the second slot, and places the cube at its own pose;
    # the task ids may be tensors of any integer dtype.
    env.task_manager.set_task(torch.tensor([0], dtype=torch.int32), torch.tensor([2], dtype=torch.int32))
    env.reset()
    assert env.task_manager.candidate_ids("object_1")[:2].tolist() == [0, 1]
    cube = env.scene["object_1"]
    assert cube.data.body_mass.torch[0, 0].item() == 1.0
    position = cube.data.root_pos_w.torch[0] - env.scene.env_origins[0]
    torch.testing.assert_close(position, torch.tensor([0.0, 0.2, 0.2], device=env.device))


def test_reset_switches_only_the_environments_whose_task_changed(env: ManagerBasedRLEnv):
    """A reset keeps the candidate selection when every environment runs the same task again, so viewers and the
    shape BVH are not refreshed, and switches the objects of the environments whose task changed."""
    env.task_manager.set_task([0, 1, 2, 3], [0, 1, 0, 1])
    env.reset()
    backend = env.sim.backend_manager.backend
    version = backend.geometry_change_version
    env.reset()
    assert backend.geometry_change_version == version
    env.task_manager.set_task([1], [0])
    env.reset()
    assert backend.geometry_change_version > version
    assert env.scene["object_1"].data.body_mass.torch[:, 0].tolist() == [1.0, 1.0, 1.0, 2.0]
