# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success terms of the VLA benchmark tasks on a kinematic box and two balls, on Newton.

The box (0.3 m wide, 0.1 m high) is a container open to the air above it and a support. In the first environment
the first ball rests on the box, in the second it rests on the ground beside it. A third ball, away from them, stands
in for the gripper link. A bar (0.04 m wide, 0.12 m high) stands up or lies down.
"""

from collections.abc import Iterator

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationManager
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


def _ball(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
    )


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )
    ball = _ball("ball")
    ball_b = _ball("ball_b")
    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bar",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.12),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.0, 0.06)),
    )
    gripper = _ball(GRIPPER).replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.0, 0.03)))
    contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/[^/]+", filter_prim_paths_expr=["{ENV_REGEX_NS}/[^/]+"], update_period=0.0
    )


@configclass
class _PolicyCfg(ObservationGroupCfg):
    ball = ObservationTermCfg(func=isaaclab_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball")})


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
    scene = _SceneCfg(num_envs=2, env_spacing=3.0)
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


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def env(request) -> Iterator[ManagerBasedRLEnv]:
    cfg = _EnvCfg()
    cfg.sim.device = request.param
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    try:
        yield env
    finally:
        env.close()


def _term(env: ManagerBasedRLEnv, func, **params):
    """Add a success term through a termination manager, which checks its parameters; return its evaluation."""
    manager = TerminationManager({"success": DoneTerm(func=func, params=params)}, env)

    def evaluate() -> list[bool]:
        manager.compute()
        return manager.get_term("success").tolist()

    return evaluate


def _settle(env: ManagerBasedRLEnv, steps: int = 30, each_step=None) -> None:
    for _ in range(steps):
        env.step(torch.zeros(env.num_envs, 0, device=env.device))
        if each_step is not None:
            each_step()


def test_container_support_and_stack(env: ManagerBasedRLEnv):
    """A ball resting on the box is in it (open top), on top of it and stacked on it; beside the box it is not."""
    ball, box = [SceneEntityCfg("ball")], SceneEntityCfg("box")
    place(env, "ball", [(0.0, 0.0, 0.15), (0.5, 0.0, 0.05)])
    place(env, "ball_b", [(1.0, 0.0, 0.05), (1.0, 0.0, 0.05)])
    _settle(env)
    assert _term(env, mdp.object_in_container, objects=ball, container=box, require_contact=True)() == [True, False]
    assert _term(env, mdp.object_outside_of, objects=ball, container=box)() == [False, True]
    assert _term(env, mdp.object_on_top, objects=ball, reference=box)() == [True, False]
    assert _term(env, mdp.stacked, objects=[SceneEntityCfg("ball"), box])() == [True, False]
    both = [SceneEntityCfg("ball"), SceneEntityCfg("ball_b")]
    assert _term(env, mdp.object_in_container, objects=both, container=box, logical="any")() == [True, False]
    assert _term(env, mdp.object_in_container, objects=both, container=box, logical="choose", count=1)() == [
        True,
        False,
    ]


def test_placement_order(env: ManagerBasedRLEnv):
    """Placing the first ball and then the second holds; the reverse order does not."""
    in_order = _term(
        env,
        mdp.objects_placed_in_container_in_order,
        objects=[SceneEntityCfg("ball"), SceneEntityCfg("ball_b")],
        container=SceneEntityCfg("box"),
        require_gripper_detached=False,
    )
    place(env, "ball", [(-0.06, 0.0, 0.15), (0.5, 0.0, 0.05)])
    place(env, "ball_b", [(1.0, 0.0, 0.05), (0.06, 0.0, 0.15)])
    _settle(env, each_step=in_order)
    assert in_order() == [False, False]
    place(env, "ball", [(-0.06, 0.0, 0.15), (-0.06, 0.0, 0.15)])
    place(env, "ball_b", [(0.06, 0.0, 0.15), (0.06, 0.0, 0.15)])
    _settle(env, each_step=in_order)
    assert in_order() == [True, False]


def test_upright(env: ManagerBasedRLEnv):
    """A bar standing on the box is upright; lying on it, it is on top and in it but not upright."""
    bar, box = [SceneEntityCfg("bar")], SceneEntityCfg("box")
    place(env, "ball", [(0.5, 0.0, 0.05), (0.5, 0.0, 0.05)])
    place(env, "ball_b", [(1.0, 0.0, 0.05), (1.0, 0.0, 0.05)])
    lying = (0.7071068, 0.0, 0.0, 0.7071068)
    place(env, "bar", [(0.0, 0.0, 0.17), (0.0, 0.0, 0.13)], [(0.0, 0.0, 0.0, 1.0), lying])
    _settle(env)
    for func, params in ((mdp.object_on_top, {"reference": box}), (mdp.object_in_container, {"container": box})):
        assert _term(env, func, objects=bar, **params)() == [True, True]
        assert _term(env, func, objects=bar, **params, require_upright=True)() == [True, False]
