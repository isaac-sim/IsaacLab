# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A task written for a physics backend, run on the world-model backend: articulation, reset event and terminations."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.events import reset_joints_by_offset
from isaaclab.envs.mdp.observations import joint_pos, joint_pos_rel
from isaaclab.envs.mdp.terminations import time_out
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.test.utils import test_devices
from isaaclab.utils import configclass

from isaaclab_neural.mdp import at_chunk_boundary, is_chunk_boundary
from isaaclab_neural.renderers import NeuralRendererCfg

from .conftest import DT, FakeModel, H, N, T, W, world_model_sim

JOINTS = ["slider_to_cart", "cart_to_pole"]


def author_robot(path):
    """A fixed-base cart (prismatic) with a pole (revolute): two joints, no external assets."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in ("cart", "pole"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(stage, f"/Robot/{name}").GetPrim())
    slider = UsdPhysics.PrismaticJoint.Define(stage, "/Robot/slider_to_cart")
    slider.CreateAxisAttr("X")
    slider.CreateLowerLimitAttr(-1.0)
    slider.CreateUpperLimitAttr(2.0)
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/cart_to_pole")
    hinge.CreateAxisAttr("Y")
    hinge.CreateLowerLimitAttr(-90.0)  # degrees; the upper limit is not authored
    UsdPhysics.FixedJoint.Define(stage, "/Robot/mount")
    stage.Save()


def event_env(sim, scene, decimation=1):
    """The parts of an environment that event and termination terms read."""
    return SimpleNamespace(
        sim=sim,
        scene=scene,
        num_envs=N,
        device=sim.device,
        cfg=SimpleNamespace(decimation=decimation),
    )


@contextlib.contextmanager
def robot_world(tmp_path, joint_ordering=None, device="cpu"):
    """The robot and a camera on the world-model backend, reset as an environment resets."""
    usd = tmp_path / "robot.usda"
    author_robot(usd)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"cart_to_pole": 0.5}),
            soft_joint_pos_limit_factor=0.5,
            joint_ordering=joint_ordering,
            actuators={},
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=H,
            width=W,
            data_types=["rgb"],
            frame_history_length=T,
            spawn=PinholeCameraCfg(focal_length=13.6, clipping_range=(0.001, 20.0)),
            renderer_cfg=NeuralRendererCfg(),
        )

    ordered = joint_ordering or JOINTS
    states = [np.array([0.1, 0.2])[[JOINTS.index(name) for name in ordered]]]
    with world_model_sim(tmp_path, states=states, device=device) as sim:
        scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()
        scene.reset()
        yield sim, scene


@pytest.fixture
def world(tmp_path):
    with robot_world(tmp_path) as world:
        yield world


@contextlib.contextmanager
def no_gpu_sync(device):
    """On a CUDA device, make torch raise on operations that wait for the GPU."""
    if not device.startswith("cuda"):
        yield
        return
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def env_step(sim, scene):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)
    scene["camera"].data


def test_joint_names_and_limits_come_from_the_usd_asset(world):
    robot = world[1]["robot"]
    assert robot.joint_names == JOINTS
    limits = robot.data.joint_pos_limits.torch
    # The slider is [-1, 2] m; the hinge has -90 deg, authored in degrees, and no upper limit, so it is unbounded.
    assert torch.allclose(limits[0, 0], torch.tensor([-1.0, 2.0]))
    assert limits[0, 1, 0].item() == pytest.approx(-np.pi / 2) and limits[0, 1, 1] == float("inf")
    soft = robot.data.soft_joint_pos_limits.torch
    assert torch.allclose(soft[0, 0], torch.tensor([-0.25, 1.25]))  # factor 0.5 around the middle
    assert soft[0, 1, 1] == float("inf")


def test_joint_ordering_orders_the_joints_and_their_limits_like_a_physics_backend(tmp_path):
    with robot_world(tmp_path, joint_ordering=JOINTS[::-1]) as (sim, scene):
        robot = scene["robot"]
        assert robot.joint_names == JOINTS[::-1]
        assert torch.allclose(robot.data.joint_pos.torch, torch.tensor([[0.2, 0.1]] * N))
        assert torch.allclose(robot.data.joint_pos_limits.torch[0, :, 0], torch.tensor([-np.pi / 2, -1.0]))


def test_a_reset_starts_from_the_record_and_standard_reset_events_keep_it(world):
    sim, scene = world
    env = event_env(sim, scene)
    robot = SceneEntityCfg("robot")
    robot.resolve(scene)
    assert torch.allclose(joint_pos(env, robot), torch.tensor([[0.1, 0.2]] * N))
    # The record is the episode's default pose, so the observation relative to it starts at zero.
    assert torch.allclose(joint_pos_rel(env, robot), torch.zeros(N, 2))
    # A physics task's reset event starts from the default pose too, so it keeps the recorded state.
    reset_joints_by_offset(env, torch.arange(N), position_range=(0.0, 0.0), velocity_range=(0.0, 0.0), asset_cfg=robot)
    assert torch.allclose(joint_pos(env, robot), torch.tensor([[0.1, 0.2]] * N))


@pytest.mark.parametrize("device", test_devices())
def test_a_standard_action_term_drives_the_model(tmp_path, device):
    with robot_world(tmp_path, device=device) as (sim, scene):
        env = SimpleNamespace(scene=scene, num_envs=N, device=device)
        term = JointPositionAction(
            JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=False), env
        )
        actions = [torch.full((N, 2), 0.1 * (k + 1), device=device) for k in range(T)]
        robot, env_ids = scene["robot"], torch.tensor([0], device=device)
        FakeModel.resets.clear()
        # Resets and the steps inside a chunk must not wait for the device; the model call at the end of the chunk may.
        with no_gpu_sync(device):
            robot.reset(env_ids=env_ids)
            for action in actions[:-1]:
                term.process_actions(action)
                term.apply_actions()
                env_step(sim, scene)
        term.process_actions(actions[-1])
        term.apply_actions()
        env_step(sim, scene)
        # Before it predicted the chunk, the model restarted every environment reset since the last chunk: the
        # scene reset of both and this one.
        assert [env_ids for env_ids, *_ in FakeModel.resets] == [[0, 1]]
        ((targets, start),) = FakeModel.steps
        assert np.allclose(targets[0, :, 0], [0.1, 0.2, 0.3, 0.4])
        assert np.allclose(start, [[0.1, 0.2]] * N)
        # The joint state is the last commanded target, and the camera shows the generated chunk.
        assert torch.allclose(scene["robot"].data.joint_pos.torch.cpu(), torch.full((N, 2), 0.4))
        assert (scene["camera"].data.output["rgb"].torch == 13).all()


def test_what_needs_a_simulated_body_raises(world):
    robot = world[1]["robot"]
    with pytest.raises(NotImplementedError, match="world-model backend"):
        robot.data.root_link_pose_w
    with pytest.raises(NotImplementedError, match="world-model backend"):
        robot.write_root_pose_to_sim_index(root_pose=torch.zeros(N, 7))
    with pytest.raises(NotImplementedError, match="world-model backend"):
        robot.is_fixed_base


def test_a_termination_at_chunk_boundary_fires_only_when_a_chunk_completes(world):
    sim, scene = world
    # Episodes past their time limit inside a chunk, as after a random first episode length.
    env = event_env(sim, scene)
    env.episode_length_buf, env.max_episode_length = torch.full((N,), 5), 3
    # Configured as in a task, through the round trip that command-line overrides take.
    cfg = DoneTerm(func=at_chunk_boundary, params={"term": time_out}, time_out=True)
    cfg.from_dict(cfg.to_dict())
    fired = []
    for _ in range(2 * T):
        env_step(sim, scene)
        fired.append(bool(cfg.func(env, **cfg.params).all()))
        assert fired[-1] == bool(is_chunk_boundary(env).all())
    assert [i + 1 for i, f in enumerate(fired) if f] == [T, 2 * T]


def test_a_termination_at_chunk_boundary_checks_the_timing_first(world):
    sim, scene = world
    with pytest.raises(ValueError, match="must divide"):
        at_chunk_boundary(event_env(sim, scene, decimation=3), term=time_out)
