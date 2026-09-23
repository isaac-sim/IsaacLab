# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

from types import SimpleNamespace

import pytest
import torch

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.managers import CommandTerm
from isaaclab.sim import select_usd_variants, use_stage
from isaaclab.utils.math import quat_box_minus

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.adr_curriculum import CurriculumCfg
from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FrankaLiftEnvCfg, FrankaReorientEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.mdp.commands import pose_commands
from isaaclab_tasks.core.lift.mdp.commands.pose_commands import (
    CableUniformPoseCommand,
    DeformableUniformPoseCommand,
    ObjectUniformPoseCommand,
)
from isaaclab_tasks.core.lift.mdp.utils import collect_collision_meshes
from isaaclab_tasks.utils.hydra import resolve_presets


class _MarkerSpy:
    def __init__(self, _cfg=None):
        self.calls: list[tuple[tuple, dict]] = []

    def set_visibility(self, _visible: bool) -> None:
        pass

    def visualize(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class _FakeScene(dict):
    def __init__(self, environment_ids: torch.Tensor, **assets):
        super().__init__(assets)
        self._ALL_INDICES = environment_ids
        self.env_origins = torch.zeros((len(environment_ids), 3))


def test_rigid_lift_motion_regularization_follows_success_driven_adr() -> None:
    """Motion penalties should grow continuously with successful episodes, not elapsed steps."""
    curriculum = CurriculumCfg()
    assert curriculum.adr.func is mdp.DifficultyScheduler
    difficulty = SimpleNamespace(difficulty_frac=0.0)
    env = SimpleNamespace(
        common_step_counter=100_000,
        curriculum_manager=SimpleNamespace(cfg=SimpleNamespace(adr=SimpleNamespace(func=difficulty))),
    )

    for term_name in ("action_rate", "joint_vel"):
        term = getattr(curriculum, term_name)
        assert term.func is mdp.modify_term_cfg
        assert term.params["address"] == f"rewards.{term_name}.weight"
        assert term.params["modify_fn"] is mdp.difficulty_interpolate_float
        params = term.params["modify_params"]
        assert params == {"initial_value": -1e-4, "final_value": -1e-1, "difficulty_term_str": "adr"}
        for fraction in (0.0, 0.02, 0.5, 1.0):
            difficulty.difficulty_frac = fraction
            weight = term.params["modify_fn"](env, torch.arange(2), -1e-4, **params)
            assert weight == pytest.approx(-1e-4 + fraction * (-1e-1 + 1e-4))


def test_abnormal_robot_state_ignores_nominal_velocity_excursions() -> None:
    """The instability guard should leave policy exploration below twice the solver limit intact."""
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_vel=SimpleNamespace(torch=torch.tensor([[15.0, 100.0], [20.0, 100.0], [20.1, 0.0]])),
            joint_vel_limits=SimpleNamespace(torch=torch.full((3, 2), 10.0)),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    arm_only = SimpleNamespace(name="robot", joint_ids=[0])

    assert mdp.abnormal_robot_state(env, arm_only).tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("selected_presets", "expected_physics"),
    [
        ((), "mujoco"),
        (("newton_mjwarp_vbd_proxy",), "mujoco"),
        (("isaacsim_physx",), "physx"),
        (("physx",), "physx"),
    ],
)
def test_franka_soft_robot_physics_variant_matches_backend(
    selected_presets: tuple[str, ...], expected_physics: str
) -> None:
    """The Franka USD physics payload must match the selected simulation backend."""
    cfg = resolve_presets(FrankaSoftEnvCfg(), selected=selected_presets)

    assert cfg.scene.robot.spawn.variants == {"Physics": expected_physics, "Colliders": "gripper_only"}


def test_franka_lift_physx_runtimes_share_the_same_mdp() -> None:
    """Isaac Sim PhysX and OvPhysX should differ only in runtime configuration."""
    isaacsim_cfg = resolve_presets(FrankaLiftEnvCfg(), selected=("isaacsim_physx", "cube"))
    ovphysx_cfg = resolve_presets(FrankaLiftEnvCfg(), selected=("ovphysx", "cube"))
    assert isaacsim_cfg.sim.physics.enable_external_forces_every_iteration
    assert ovphysx_cfg.sim.physics.enable_external_forces_every_iteration
    isaacsim = isaacsim_cfg.to_dict()
    ovphysx = ovphysx_cfg.to_dict()

    for section in ("scene", "observations", "actions", "commands", "rewards", "terminations", "events", "curriculum"):
        assert ovphysx[section] == isaacsim[section], section


@pytest.mark.parametrize("cfg_type", [FrankaLiftEnvCfg, FrankaReorientEnvCfg])
def test_franka_rigid_tasks_select_gripper_only_colliders(cfg_type) -> None:
    """Rigid tasks avoid the arm colliders that intersect the ground during reset sampling."""
    cfg = resolve_presets(cfg_type(), selected=())
    stage = Usd.Stage.CreateInMemory()
    robot = stage.DefinePrim("/Robot", "Xform")
    colliders = robot.GetVariantSets().AddVariantSet("Colliders")
    for selection, prim_path, prim_type in (
        ("convex_hulls", "/Robot/link1_c/link1_c", "Mesh"),
        ("primitives", "/Robot/link1_capsule", "Capsule"),
        ("gripper_only", "/Robot/gripper_capsule", "Capsule"),
    ):
        colliders.AddVariant(selection)
        colliders.SetVariantSelection(selection)
        with colliders.GetVariantEditContext():
            stage.DefinePrim(prim_path, prim_type)
    colliders.SetVariantSelection("primitives")

    select_usd_variants("/Robot", cfg.scene.robot.spawn.variants or {}, stage=stage)

    assert stage.GetPrimAtPath("/Robot/gripper_capsule").IsValid()
    assert not stage.GetPrimAtPath("/Robot/link1_c/link1_c").IsValid()
    assert not stage.GetPrimAtPath("/Robot/link1_capsule").IsValid()


def test_franka_tasks_use_distinct_lift_and_reorient_bootstraps() -> None:
    """Lift should retain broad starts while Reorient starts from held objects."""
    reorient = FrankaReorientEnvCfg()
    lift = FrankaLiftEnvCfg()

    assert reorient.commands.object_pose.difficulty_term == "adr"
    assert reorient.commands.object_pose.initial_position_distance == pytest.approx(0.0)
    reorient_reset = reorient.events.conditional_reset.params
    assert reorient_reset["terms"]["reset_object_to_target"].func is mdp.reset_to_grasp
    assert reorient_reset["terms"]["reset_object_to_target"].params["probability"] == pytest.approx(1.0)
    assert "object_robot_clearance" not in reorient_reset["valid_criteria"]
    assert reorient.actions.arm_action.joint_names == ["panda_joint.*"]
    assert reorient.actions.arm_action.scale == pytest.approx(0.03)
    assert reorient.actions.gripper_action.joint_names == ["panda_finger_joint1"]
    assert reorient.terminations.abnormal_robot.func is mdp.abnormal_robot_state
    assert lift.commands.object_pose.difficulty_term is None
    assert lift.actions.action.joint_names == [".*"]
    assert lift.terminations.abnormal_robot.func is mdp.abnormal_robot_state
    lift_reset = lift.events.conditional_reset.params
    lift_reset_terms = list(lift_reset["terms"])
    assert lift_reset_terms.index("reset_object_to_target") > lift_reset_terms.index("reset_robot_wrist_joint")
    lift_target_reset = lift_reset["terms"]["reset_object_to_target"]
    assert lift_target_reset.func == "isaaclab_tasks.core.lift.mdp.events:reset_to_target"
    assert lift_target_reset.params["probability"] == pytest.approx(0.25)
    assert lift_target_reset.params["pose_range"] == {
        "x": [-0.02, 0.02],
        "y": [-0.02, 0.02],
        "z": [0.08, 0.12],
    }
    assert lift_target_reset.params["velocity_range"] == {}
    assert "object_robot_clearance" in lift_reset["valid_criteria"]

    lift.play_mode()
    assert lift.events.conditional_reset.params["terms"]["reset_object_to_target"].params[
        "probability"
    ] == pytest.approx(0.25)


def test_pose_command_curriculum_preserves_full_difficulty_goal() -> None:
    """Pose goals should start locally and equal the sampled goal at maximum difficulty."""
    scheduler = SimpleNamespace(
        cfg=SimpleNamespace(params={"min_difficulty": 0, "max_difficulty": 10}),
        current_difficulties=torch.tensor([0.0, 10.0]),
    )
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    object_pos_b = torch.tensor([[0.4, 0.0, 0.5], [0.4, 0.0, 0.5]])
    env = SimpleNamespace(
        device="cpu", curriculum_manager=SimpleNamespace(cfg=SimpleNamespace(adr=SimpleNamespace(func=scheduler)))
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)),
            root_quat_w=SimpleNamespace(torch=identity),
        )
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=object_pos_b),
            root_quat_w=SimpleNamespace(torch=identity),
        )
    )
    command = object.__new__(ObjectUniformPoseCommand)
    command._env = env
    command.robot = robot
    command.object = obj
    command.cfg = SimpleNamespace(difficulty_term="adr", initial_position_distance=0.06)
    full_goal = torch.tensor(
        [
            [0.4, 0.0, 0.7, 0.0, 0.0, 0.5646425, 0.8253356],
            [0.4, 0.0, 0.7, 0.0, 0.0, 0.5646425, 0.8253356],
        ]
    )
    command.pose_command_b = full_goal.clone()

    command._apply_difficulty_curriculum(torch.arange(2))

    assert torch.linalg.norm(command.pose_command_b[0, :3] - object_pos_b[0]).item() == pytest.approx(0.06)
    assert torch.linalg.norm(quat_box_minus(command.pose_command_b[0:1, 3:], identity[0:1])).item() == pytest.approx(
        0.0
    )
    torch.testing.assert_close(command.pose_command_b[1], full_goal[1])


def test_reset_clearance_ignores_disabled_collision_geometry() -> None:
    """Disabled colliders and visual-only geometry must not reject reset candidates."""
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Object", "Xform")
    for name, enabled in (("default_enabled", None), ("explicit_enabled", True), ("disabled", False)):
        prim = UsdGeom.Cube.Define(stage, f"/Object/{name}").GetPrim()
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        if enabled is not None:
            collision.CreateCollisionEnabledAttr(enabled)
    UsdGeom.Cube.Define(stage, "/Object/visual_only")

    with use_stage(stage):
        meshes = collect_collision_meshes(root, lambda prim: (prim.GetName(), root))

    assert set(meshes) == {"default_enabled", "explicit_enabled"}


def _make_vision_camera(data_type: str, images: torch.Tensor) -> mdp.vision_camera:
    """Build a ``vision_camera`` term around a fake single-data-type camera sensor."""
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(data_types=[data_type]), data=SimpleNamespace(output={data_type: images})
    )
    term = object.__new__(mdp.vision_camera)
    term.sensor = sensor
    term.sensor_type = data_type
    term._is_depth = data_type in ("distance_to_image_plane", "depth")
    return term


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must map fixed inputs to fixed outputs, independent of per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0]).view(1, 1, 1, 3)
    depth = torch.tensor([0.0, 2.0]).view(1, 1, 2, 1)
    env = SimpleNamespace()

    rgb_obs = _make_vision_camera("rgb", rgb)(env, sensor_cfg=None)
    depth_obs = _make_vision_camera("depth", depth)(env, sensor_cfg=None)

    # channel-first output with the value range mapped to [-0.5, 0.5)
    assert rgb_obs.shape == (1, 3, 1, 1)
    assert torch.allclose(rgb_obs.flatten(), torch.tensor([-0.5, 0.0, 0.5]))
    assert depth_obs.shape == (1, 1, 1, 2)
    assert torch.allclose(depth_obs.flatten(), torch.tanh(torch.tensor([0.0, 2.0]) / 2) - 0.5)


def test_lift_pose_markers_forward_environment_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lift pose, goal, and success marker should retain its environment ownership."""
    num_envs = 3
    environment_ids = torch.arange(num_envs)
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 3] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    root_pose_w = torch.cat((root_pos_w, identity_quat), dim=-1)

    robot = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        ),
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
            root_link_pose_w=SimpleNamespace(torch=root_pose_w),
        )
    )
    success_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_pos_w)))
    scene = _FakeScene(environment_ids, robot=robot, object=object_asset, table=success_asset)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
    cfg = SimpleNamespace(
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="table",
        success_visualizer_cfg=object(),
        goal_pose_visualizer_cfg=object(),
        curr_pose_visualizer_cfg=object(),
        position_only=True,
        cmd_kind=None,
        element_names=None,
    )

    def _initialize_command_term(command, command_cfg, command_env) -> None:
        command.cfg = command_cfg
        command._env = command_env
        command.metrics = {}

    monkeypatch.setattr(CommandTerm, "__init__", _initialize_command_term)
    monkeypatch.setattr(pose_commands, "VisualizationMarkers", _MarkerSpy)

    command = ObjectUniformPoseCommand(cfg, env)
    command._set_debug_vis_impl(True)
    command._debug_vis_callback(None)
    command.cfg.position_only = False
    command._debug_vis_callback(None)
    command._update_metrics()
    DeformableUniformPoseCommand._update_metrics(command)
    command._segment_position_w = lambda: root_pos_w
    CableUniformPoseCommand._update_metrics(command)
    CableUniformPoseCommand._debug_vis_callback(command, None)

    expected_call_counts = {
        command.success_visualizer: 4,
        command.goal_visualizer: 3,
        command.curr_visualizer: 3,
    }
    for visualizer, expected_count in expected_call_counts.items():
        assert len(visualizer.calls) == expected_count
        for _, kwargs in visualizer.calls:
            assert torch.equal(kwargs["environment_ids"], environment_ids)


def test_lift_point_cloud_markers_repeat_environment_ids_per_point() -> None:
    """Flattened point-cloud markers should retain env-major ownership."""
    num_envs = 3
    num_points = 4
    identity_quat = torch.zeros((num_envs, 4))
    identity_quat[:, 3] = 1.0
    root_pos_w = torch.zeros((num_envs, 3))
    points_local = torch.arange(num_envs * num_points * 3, dtype=torch.float32).view(num_envs, num_points, 3)

    term = object.__new__(mdp.object_point_cloud_b)
    term.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.ref_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos_w),
            root_quat_w=SimpleNamespace(torch=identity_quat),
        )
    )
    term.points_local = points_local
    term.points_w = torch.zeros_like(points_local)
    term.visualizer = _MarkerSpy()
    term._marker_env_ids = torch.arange(num_envs).repeat_interleave(num_points)
    env = SimpleNamespace(num_envs=num_envs)

    points_b = term(env, num_points=num_points, visualize=True)

    # identity poses: the points in the reference frame are the local points
    assert torch.allclose(points_b, points_local)
    assert len(term.visualizer.calls) == 1
    _, kwargs = term.visualizer.calls[0]
    assert torch.equal(kwargs["translations"], term.points_w.view(-1, 3))
    assert torch.equal(kwargs["environment_ids"], torch.arange(num_envs).repeat_interleave(num_points))
