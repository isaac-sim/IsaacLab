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
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.observations import joint_pos, joint_pos_rel
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass

from isaaclab_neural.mdp import at_chunk_boundary, is_chunk_boundary, reset_from_records
from isaaclab_neural.renderers import WorldModelRendererCfg

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


def event_env(sim, scene, decimation=1, episode_steps=3 * T):
    """The parts of an environment that the reset event reads."""
    return SimpleNamespace(
        sim=sim,
        scene=scene,
        cfg=SimpleNamespace(decimation=decimation),
        step_dt=decimation * DT,
        max_episode_length=episode_steps,
        num_envs=N,
        device=sim.device,
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
            renderer_cfg=WorldModelRendererCfg(),
        )

    ordered = joint_ordering or JOINTS
    states = [np.array([0.1, 0.2])[[JOINTS.index(name) for name in ordered]]]
    with world_model_sim(tmp_path, states=states, device=device) as sim:
        scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        sim.reset()
        scene.reset()
        env = event_env(sim, scene)
        reset_from_records(None, env)(env, slice(None))
        yield sim, scene


@pytest.fixture
def world(tmp_path):
    with robot_world(tmp_path) as world:
        yield world


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


def test_the_reset_event_sets_the_recorded_joint_state(world):
    env = SimpleNamespace(scene=world[1])
    assert torch.allclose(joint_pos(env, SceneEntityCfg("robot")), torch.tensor([[0.1, 0.2]] * N))
    # Relative to the default pose of the articulation, as the observation term of a physics task computes it.
    assert torch.allclose(joint_pos_rel(env, SceneEntityCfg("robot")), torch.tensor([[0.1, -0.3]] * N))
    sim, scene = world
    robot = scene["robot"]
    robot.write_joint_position_to_sim_index(position=torch.tensor([[0.7, 0.8]] * N))
    robot.set_joint_position_target_index(target=torch.tensor([[0.9, 1.0]] * N))
    reset_env = event_env(sim, scene)
    reset_from_records(None, reset_env)(reset_env, torch.tensor([0]))
    # A partial reset restores both shared buffers, keeps the other episode, and leaves the configured default alone.
    torch.testing.assert_close(robot.data.joint_pos.torch, torch.tensor([[0.1, 0.2], [0.7, 0.8]]))
    torch.testing.assert_close(robot.data.joint_pos_target.torch, torch.tensor([[0.1, 0.2], [0.9, 1.0]]))
    torch.testing.assert_close(robot.data.default_joint_pos.torch, torch.tensor([[0.0, 0.5]] * N))


@pytest.mark.parametrize("decimation", [1, 2, T])
def test_a_standard_action_term_drives_the_model(world, decimation):
    sim, scene = world
    env = SimpleNamespace(scene=scene, num_envs=N, device="cpu")
    term = JointPositionAction(
        JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=False), env
    )
    sim.backend_manager.set_joint_target(np.zeros((N, 2), dtype=np.float32))
    torch.testing.assert_close(scene["robot"].data.joint_pos_target.torch, torch.zeros(N, 2))
    for k in range(T // decimation):
        target = torch.full((N, 2), 0.1 * (k + 1))
        term.process_actions(target)
        for _ in range(decimation):
            term.apply_actions()
            # Actions reach the manager before write_data_to_sim, and observations see a step before scene.update.
            torch.testing.assert_close(torch.as_tensor(sim.backend_manager.joint_target), target)
            scene.write_data_to_sim()
            sim.step(render=False)
            torch.testing.assert_close(scene["robot"].data.joint_pos.torch, target)
            scene.update(DT)
            scene["camera"].data
    ((targets, start),) = FakeModel.steps
    assert np.allclose(targets[0, :, 0], np.repeat(np.arange(1, T // decimation + 1) * 0.1, decimation))
    assert np.allclose(start, [[0.1, 0.2]] * N)
    # The joint state is the last commanded target, and the camera shows the generated chunk.
    assert torch.allclose(scene["robot"].data.joint_pos.torch, target)
    assert (scene["camera"].data.output["rgb"].torch == 13).all()


def test_what_needs_a_simulated_body_raises(world):
    robot = world[1]["robot"]
    with pytest.raises(NotImplementedError, match="world-model backend"):
        robot.data.root_link_pose_w
    with pytest.raises(NotImplementedError, match="world-model backend"):
        robot.write_root_pose_to_sim_index(root_pose=torch.zeros(N, 7))


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_masked_joint_writes_preserve_other_entries_without_host_sync(tmp_path, device):
    with robot_world(tmp_path, device=device) as (sim, scene):
        robot = scene["robot"]
        env_mask = wp.from_torch(torch.tensor([True, False], device=device))
        joint_mask = wp.from_torch(torch.tensor([False, True], device=device))
        empty_mask = wp.from_torch(torch.zeros(N, dtype=torch.bool, device=device))
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        previous_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            robot.write_joint_position_to_sim_mask(position=values, env_mask=env_mask, joint_mask=joint_mask)
            state = robot.data.joint_pos.torch.clone()
            robot.set_joint_position_target_mask(target=values, env_mask=env_mask, joint_mask=joint_mask)
            robot.set_joint_position_target_mask(target=values, env_mask=empty_mask)
            targets = robot.data.joint_pos_target.torch.clone()
            robot.set_joint_position_target_mask(target=values)
            robot.reset(env_mask=env_mask)
            reset_targets = robot.data.joint_pos_target.torch.clone()
            sim.step(render=False)
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
        expected = torch.tensor([[0.1, 2.0], [0.1, 0.2]], device=device)
        torch.testing.assert_close(state, expected)
        torch.testing.assert_close(targets, expected)
        # Reset clears selected targets to the current state, without resetting the recorded episode or defaults.
        expected_reset = torch.tensor([[0.1, 2.0], [3.0, 4.0]], device=device)
        torch.testing.assert_close(reset_targets, expected_reset)
        torch.testing.assert_close(robot.data.joint_pos.torch, expected_reset)
        torch.testing.assert_close(robot.data.default_joint_pos.torch, torch.tensor([[0.0, 0.5]] * N, device=device))


def test_the_reset_event_checks_the_timing_when_the_environment_is_created(world):
    sim, scene = world
    for decimation in (1, 2, T):
        reset_from_records(None, event_env(sim, scene, decimation=decimation))
    with pytest.raises(ValueError, match="must divide"):
        reset_from_records(None, event_env(sim, scene, decimation=3))
    with pytest.raises(ValueError, match="not a whole number"):
        reset_from_records(None, event_env(sim, scene, decimation=2, episode_steps=3 * T + 1))


@pytest.mark.parametrize("decimation", [1, 2, T])
def test_a_termination_at_chunk_boundary_fires_only_when_a_chunk_completes(world, decimation):
    sim, scene = world
    env = event_env(sim, scene, decimation=decimation)
    always = at_chunk_boundary(lambda env: torch.ones(N, dtype=torch.bool))
    fired = []
    for _ in range(2 * T // decimation):
        for _ in range(decimation):
            env_step(sim, scene)
        fired.append(bool(always(env).all()))
        assert fired[-1] == bool(is_chunk_boundary(env).all())
    assert [i + 1 for i, f in enumerate(fired) if f] == [T // decimation, 2 * T // decimation]
