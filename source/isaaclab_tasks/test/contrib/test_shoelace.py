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
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_tasks.contrib.shoelace.mdp.events import reset_shoe_position
from isaaclab_tasks.contrib.shoelace.mdp.grasp import shoelace_grasp_quality
from isaaclab_tasks.contrib.shoelace.mdp.rewards import dense_task_reward, grasp_hold_reward, pregrasp_progress_reward
from isaaclab_tasks.contrib.shoelace.shoelace_contacts import FingerTailContacts
from isaaclab_tasks.contrib.shoelace.shoelace_env_cfg import ShoelaceEnvCfg
from isaaclab_tasks.contrib.shoelace.shoelace_physics import configure_shoelace_builder


@pytest.fixture
def grasp_env() -> SimpleNamespace:
    """Stationary wrists with open opposing pads and both tails at the fixed TCPs."""
    cfg = ShoelaceEnvCfg()
    term_cfg = cfg.rewards.dense_task
    scene = {
        "finger_tail_contacts": SimpleNamespace(
            data=torch.full((1, 4), 0.002), pinch_distance=torch.full((1, 4, 3), 0.002)
        )
    }
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
    for i, finger_cfg in enumerate(cfg.rewards.pregrasp.params["finger_cfgs"]):
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


def test_grasp_hold_uses_surface_contacts_and_tcp_speed(grasp_env: SimpleNamespace) -> None:
    """Surface contacts can retain a tail away from pad landmarks, but missing contact and slip cannot."""
    env = grasp_env
    cfg = env.cfg.rewards.grasp_hold
    contacts = env.scene["finger_tail_contacts"].pinch_distance
    reward = grasp_hold_reward(cfg, env)
    assert reward(env, **cfg.params).item() == 0.0
    contacts.zero_()
    for name in ("robot_left", "robot_right"):
        # The sensor measures the collision surface, not these arbitrary reference points.
        env.scene[name].data.body_pos_w.torch[:, 1:, 1] += 0.03
    reward.reset()
    held = reward(env, **cfg.params)
    assert held.item() > 0.9
    for name in ("robot_left", "robot_right"):
        velocity = env.scene[name].data.body_link_vel_w.torch
        velocity[:, 1, 0], velocity[:, 2, 0] = 0.1, -0.1
    reward.reset()
    torch.testing.assert_close(reward(env, **cfg.params), held)
    contacts[:, [1, 3]] = 0.002
    reward.reset()
    assert reward(env, **cfg.params).item() == 0.0
    contacts.fill_(-0.0005)
    reward.reset()
    torch.testing.assert_close(reward(env, **cfg.params), held)
    contacts.fill_(-0.004)
    reward.reset()
    assert reward(env, **cfg.params).item() < 0.01
    contacts.zero_()
    for cable_cfg in cfg.params["cable_cfgs"]:
        env.scene[cable_cfg.name].data.segment_velocity_w.torch[..., 0] = 0.1
    reward.reset()
    assert reward(env, **cfg.params).item() < held.item() * 0.5

    for cable_cfg in cfg.params["cable_cfgs"]:
        env.scene[cable_cfg.name].data.segment_velocity_w.torch.zero_()
    reward.reset()
    retention = torch.stack([reward(env, **cfg.params) for _ in range(90)])
    assert retention.sum().item() * env.step_dt == pytest.approx(2.0)
    assert retention[-1].item() == 0.0
    contacts.fill_(0.002)
    reward(env, **cfg.params)
    contacts.zero_()
    assert reward(env, **cfg.params).item() == 0.0  # Regrasping cannot renew the episode budget.


def test_pregrasp_contact_alone_earns_no_progress(grasp_env: SimpleNamespace) -> None:
    """A new collision candidate cannot earn pregrasp credit without alignment or closure."""
    env = grasp_env
    cfg = env.cfg.rewards.pregrasp
    reward = pregrasp_progress_reward(cfg, env)
    assert reward(env, **cfg.params).item() == 0.0
    env.scene["finger_tail_contacts"].data.zero_()
    assert reward(env, **cfg.params).item() == 0.0


def test_grasp_quality_rejects_supporting_contacts(grasp_env: SimpleNamespace) -> None:
    """Real box/capsule contacts accept pinching off-centre but reject support on the outer faces."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    fingers, tails = [], []
    for arm, robot in enumerate(("RobotLeft", "RobotRight")):
        pair = []
        for finger, name in enumerate(("panda_leftfinger", "panda_rightfinger")):
            body = builder.add_body(label=f"/Env/{robot}/{name}")
            builder.add_shape_box(
                body,
                hx=0.0015,
                hy=0.008,
                hz=0.005,
                xform=wp.transform(wp.vec3(0.02 * (finger + 1), 0, 0), wp.quat_identity()),
            )
            builder.add_shape_box(body, as_site=True)
            pair.append(body)
        fingers.append(pair)
        chain = []
        for _ in range(3):
            body = builder.add_body()
            builder.add_shape_capsule(
                body,
                radius=0.0015,
                half_height=0.002,
                xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.pi / 2)),
            )
            chain.append(body)
        tails.append(chain)
    builder.end_world()
    model = builder.finalize(device="cpu")
    state = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    manager = SimpleNamespace(
        backend=SimpleNamespace(model=model, state_0=state, contacts=contacts),
        register_post_step_callback=lambda callback: None,
        unregister_post_step_callback=lambda callback: None,
    )
    observer = FingerTailContacts(model, [(tails[1], tails[0])], manager)
    observer.initialize()
    sensor = SimpleNamespace(data=observer.signed_distance, pinch_distance=observer.pinch_distance)
    grasp_env.scene["finger_tail_contacts"] = sensor
    params = grasp_env.cfg.rewards.grasp_hold.params
    quality_params = {
        k: params[k]
        for k in ("contact_std", "relative_speed_std", "cable_cfgs", "contact_penetration_tolerance", "robot_cfgs")
    }
    # The cable remains inside the projected finger span even in the support case.
    for spacing, cable_y, cable_z, split_segment, expected_grasp in (
        (0.003, 0.0, 0.0, None, True),
        (0.003, 0.006, 0.0, None, True),
        (0.0015, 0.0, 0.0065, None, False),
        (0.005, 0.0, 0.0, 2, False),
        (0.005, 0.0, 0.0, 1, True),
    ):
        poses = np.zeros((model.body_count, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        for arm in range(2):
            for finger, body in enumerate(fingers[arm]):
                poses[body, :3] = ((2 * finger - 1) * spacing - 0.02 * (finger + 1), arm, 0.0)
            for segment, body in enumerate(tails[arm]):
                poses[body, :3] = (0.0, arm + cable_y, cable_z + segment)
            if split_segment is not None:
                poses[tails[arm][0], :3] = (-0.002, arm, 0.0)
                poses[tails[arm][split_segment], :3] = (0.002, arm, 0.0)
        if cable_y:
            frame = wp.transform(wp.vec3(0.1, 0.2, 0.3), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.7))
            poses = np.array([wp.transform_multiply(frame, wp.transform(p[:3], p[3:])) for p in poses])
        state.body_q.assign(poses)
        pipeline.collide(state, contacts)
        if cable_y:
            # Contact ordering must not change which way a finger's outward normal points.
            for name in ("shape", "point", "margin"):
                a, b = (getattr(contacts, f"rigid_contact_{name}{i}") for i in range(2))
                previous = a.numpy().copy()
                a.assign(b.numpy())
                b.assign(previous)
            contacts.rigid_contact_normal.assign(-contacts.rigid_contact_normal.numpy())
        observer.update()
        # Both configurations really have bilateral surface contact; rejection must be geometric.
        assert (sensor.data.abs() < 0.0002).all()
        quality, finite = shoelace_grasp_quality(grasp_env, **quality_params)
        assert finite.all()
        if expected_grasp:
            assert (quality > 0.99).all()
        else:
            assert (quality == 0.0).all()
    observer.reset(torch.tensor([0]))
    assert (sensor.data == 0.002).all() and (sensor.pinch_distance == 0.002).all()


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
