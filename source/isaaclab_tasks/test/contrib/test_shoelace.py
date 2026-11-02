# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused reset, reward, and material tests for the shoelace task."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_tasks.contrib.shoelace.mdp.events import reset_shoe_position
from isaaclab_tasks.contrib.shoelace.mdp.rewards import dense_task_reward, grasp_hold_reward, pregrasp_progress_reward
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.shoelace_physics import configure_shoelace_builder


@pytest.fixture
def grasp_env() -> SimpleNamespace:
    """Stationary wrists with open opposing pads and both tails at the fixed TCPs."""
    cfg = ShoelaceEnvCfg()
    term_cfg = cfg.rewards.dense_task
    scene = {"finger_tail_contacts": SimpleNamespace(data=torch.full((1, 4), 0.002))}
    for robot_cfg in cfg.observations.policy.tails_to_tcp.params["robot_cfgs"]:
        robot_cfg.body_ids = [0]
        scene[robot_cfg.name] = SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0]]])),
                body_quat_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])),
                root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            )
        )
    for term in (cfg.rewards.dense_task, cfg.rewards.pregrasp, cfg.rewards.grasp_hold):
        for robot_cfg in term.params["robot_cfgs"]:
            robot_cfg.body_ids = [0]
        for i, finger_cfg in enumerate(term.params["finger_cfgs"]):
            finger_cfg.body_ids = [1 + i % 2]
    for robot in (scene["robot_left"], scene["robot_right"]):
        tcp = torch.tensor(cfg.tcp_body_offsets[0][0])
        fingers = tcp + torch.tensor([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
        fingers -= torch.tensor(cfg.finger_body_offsets[:2])
        robot.data.body_pos_w.torch = torch.cat((robot.data.body_pos_w.torch, fingers[None]), dim=1)
        robot.data.body_quat_w.torch = robot.data.body_quat_w.torch.expand(-1, 3, -1)
        robot.data.body_link_vel_w = SimpleNamespace(torch=torch.zeros(1, 3, 6))
    for cable_cfg in term_cfg.params["cable_cfgs"]:
        pose = torch.zeros(1, 6, 7)
        pose[..., :3] = tcp
        pose[..., 6] = 1.0
        scene[cable_cfg.name] = SimpleNamespace(
            data=SimpleNamespace(
                segment_pose_w=SimpleNamespace(torch=pose),
                segment_velocity_w=SimpleNamespace(torch=torch.zeros(1, 6, 6)),
            )
        )
    return SimpleNamespace(
        cfg=cfg,
        scene=scene,
        num_envs=1,
        device="cpu",
        step_dt=1 / 30,
        extras={"log": {}},
        termination_manager=SimpleNamespace(
            dones=torch.zeros(1, dtype=torch.bool), get_term=lambda name: torch.zeros(1, dtype=torch.bool)
        ),
    )


@pytest.mark.parametrize("term_name", ["dense_task", "pregrasp"])
def test_approach_penalizes_tail_offset_inside_open_fingers(grasp_env: SimpleNamespace, term_name: str) -> None:
    """Moving along the open pinch segment still loses fixed-TCP alignment credit."""
    env = grasp_env
    term_cfg = getattr(env.cfg.rewards, term_name)
    term_class = dense_task_reward if term_name == "dense_task" else pregrasp_progress_reward
    reward = term_class(term_cfg, env)
    assert reward(env, **term_cfg.params).item() == 0.0
    for cable_cfg in term_cfg.params["cable_cfgs"]:
        env.scene[cable_cfg.name].data.segment_pose_w.torch[..., 0] += 0.006
    assert reward(env, **term_cfg.params).item() < 0.0
    for cable_cfg in term_cfg.params["cable_cfgs"]:
        env.scene[cable_cfg.name].data.segment_pose_w.torch[..., 0] -= 0.006
    assert reward(env, **term_cfg.params).item() > 0.0


def test_grasp_hold_uses_tcp_speed_instead_of_finger_closure_speed(grasp_env: SimpleNamespace) -> None:
    """Closing fingers alone does not penalize retention; tail motion relative to the TCP does."""
    env = grasp_env
    cfg = env.cfg.rewards.grasp_hold
    env.scene["finger_tail_contacts"].data.zero_()
    for name in ("robot_left", "robot_right"):
        robot = env.scene[name].data
        robot.body_pos_w.torch[:, 1, 0] += 0.0085
        robot.body_pos_w.torch[:, 2, 0] -= 0.0085
    reward = grasp_hold_reward(cfg, env)
    held = reward(env, **cfg.params)
    assert held.item() > 0.9
    for name in ("robot_left", "robot_right"):
        velocity = env.scene[name].data.body_link_vel_w.torch
        velocity[:, 1, 0], velocity[:, 2, 0] = 0.1, -0.1
    reward.reset()
    torch.testing.assert_close(reward(env, **cfg.params), held)
    for cable_cfg in cfg.params["cable_cfgs"]:
        env.scene[cable_cfg.name].data.segment_velocity_w.torch[..., 0] = 0.1
    reward.reset()
    assert reward(env, **cfg.params).item() < held.item() * 0.5


def test_pregrasp_contact_alone_earns_no_progress(grasp_env: SimpleNamespace) -> None:
    """A new collision candidate cannot earn pregrasp credit without alignment or closure."""
    env = grasp_env
    cfg = env.cfg.rewards.pregrasp
    reward = pregrasp_progress_reward(cfg, env)
    assert reward(env, **cfg.params).item() == 0.0
    env.scene["finger_tail_contacts"].data.zero_()
    assert reward(env, **cfg.params).item() == 0.0


@pytest.mark.parametrize("env_ids", [slice(1, 4, 2), torch.tensor([3, 1])])
def test_shoe_reset_accepts_slices_and_preserves_unselected_poses(env_ids: torch.Tensor | slice) -> None:
    """Apply the same translation to selected shoe and cable poses without expanding slices."""
    defaults = torch.zeros(4, 7)
    defaults[:, 6] = 1.0
    origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shoe_pose = defaults.clone()
    cable_poses = [defaults[:, None].expand(-1, 3, -1).clone() for _ in range(2)]
    cable_poses[1][..., 1] = 0.5
    initial_cable_poses = [pose.clone() for pose in cable_poses]
    shoe = SimpleNamespace(
        data=SimpleNamespace(default_root_pose=SimpleNamespace(torch=defaults)),
        write_root_pose_to_sim_index=lambda *, root_pose, env_ids: shoe_pose.__setitem__(env_ids, root_pose),
    )
    cables = [
        SimpleNamespace(
            data=SimpleNamespace(default_segment_pose_w=SimpleNamespace(torch=pose.clone())),
            write_segment_pose_to_sim_index=lambda *, segment_pose, env_ids, buffer=pose: buffer.__setitem__(
                env_ids, segment_pose
            ),
        )
        for pose in cable_poses
    ]

    class Scene(dict):
        env_origins = origins

    env = SimpleNamespace(device="cpu", scene=Scene(shoe=shoe, shoelace_left=cables[0], shoelace_right=cables[1]))
    reset_shoe_position(env, env_ids, {"x": (0.1, 0.1), "y": (-0.2, -0.2)})
    expected_shoe = defaults.clone()
    offset = torch.tensor([0.1, -0.2, 0.0])
    expected_shoe[env_ids, :3] += origins[env_ids] + offset
    torch.testing.assert_close(shoe_pose, expected_shoe)
    for actual, expected in zip(cable_poses, initial_cable_poses, strict=True):
        expected[env_ids, :, :3] += offset
        torch.testing.assert_close(actual, expected)


def test_cable_friction_uses_each_bound_usd_material() -> None:
    """Preserve distinct authored cable friction through model construction without external assets."""
    stage = Usd.Stage.CreateInMemory()
    builder = newton.ModelBuilder()
    builder.begin_world()
    cfg = ShoelaceEnvCfg()
    cfg.scene.clone_cfg.clone_template = "/Lab/Cells/Cell{}"
    expected = {}
    for side, friction in (("Left", 0.23), ("Right", 0.37)):
        path = f"/Lab/Cells/Cell0/ShoelaceScene/Shoelace{side}/geometry/mesh"
        curve = UsdGeom.BasisCurves.Define(stage, path).GetPrim()
        points = np.array([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0)])
        segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1).mean()
        for name in ("referenceSegmentLength", "segmentLength"):
            curve.CreateAttribute(f"shoelace:{name}", Sdf.ValueTypeNames.Double).Set(segment_length)
        material = UsdShade.Material.Define(stage, f"{path}/ContactMaterial")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(friction)
        UsdShade.MaterialBindingAPI.Apply(curve).Bind(material, materialPurpose="physics")
        bodies, _ = builder.add_rod(rod=newton.Rod(points, radius=0.001), label=path, body_frame_origin="start")
        for body in bodies:
            expected.update({shape: friction for shape in builder.body_shapes[body]})
    builder.end_world()

    configure_shoelace_builder(builder, stage, 0, np.array([0.0, 0.0, 0.0, 1.0]), cfg)
    model = builder.finalize(device="cpu")
    for shape, friction in expected.items():
        assert model.shape_material_mu.numpy()[shape] == pytest.approx(friction)
