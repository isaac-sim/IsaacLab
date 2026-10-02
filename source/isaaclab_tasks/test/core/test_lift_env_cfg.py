# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the dexterous Lift tasks."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.assets import Asset
from isaaclab.managers import CommandTerm, ObservationTermCfg, SceneEntityCfg
from isaaclab.sim import use_stage
from isaaclab.utils import instantiate
from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft.franka_cable_env_cfg import FrankaCableEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_camera_env_cfg import KukaAllegroLiftCameraEnvCfg
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

    assert cfg.scene.robot.spawn.variants == {"Physics": expected_physics, "Colliders": "primitives"}


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


def _make_vision_camera(data_type: str, images: torch.Tensor) -> tuple[mdp.vision_camera, SimpleNamespace]:
    """Build a ``vision_camera`` term around a fake single-data-type camera sensor."""
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(data_types=[data_type]),
        data=SimpleNamespace(output={data_type: ProxyArray(wp.from_torch(images))}),
    )
    env = SimpleNamespace(num_envs=images.shape[0], device="cpu", scene=SimpleNamespace(sensors={"camera": sensor}))
    cfg = ObservationTermCfg(func=mdp.vision_camera, params={"sensor_cfg": SceneEntityCfg("camera")})
    return mdp.vision_camera(cfg, env), env


def test_legacy_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must map fixed inputs to fixed outputs, independent of per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0]).view(1, 1, 1, 3)
    depth = torch.tensor([0.0, 2.0]).view(1, 1, 2, 1)
    rgb_term, rgb_env = _make_vision_camera("rgb", rgb)
    depth_term, depth_env = _make_vision_camera("depth", depth)

    rgb_obs = rgb_term(rgb_env, sensor_cfg=None)
    depth_obs = depth_term(depth_env, sensor_cfg=None)

    # channel-first output with the value range mapped to [-0.5, 0.5)
    assert rgb_obs.shape == (1, 3, 1, 1)
    assert torch.allclose(rgb_obs.flatten(), torch.tensor([-0.5, 0.0, 0.5]))
    assert depth_obs.shape == (1, 1, 1, 2)
    assert torch.allclose(depth_obs.flatten(), torch.tanh(torch.tensor([0.0, 2.0]) / 2) - 0.5)


@pytest.mark.parametrize("data_type", ["rgb", "depth", "albedo", "semantic_segmentation"])
def test_camera_normalization_is_stationary(data_type: str) -> None:
    """Configured camera terms keep raw images, and the policy applies stationary normalization."""
    from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import CameraImageNormalizer
    from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_camera_env_cfg import KukaAllegroLiftCameraEnvCfg

    cfg = resolve_presets(KukaAllegroLiftCameraEnvCfg(), {"duo_camera", f"{data_type}128"})
    cfg.validate()
    if data_type == "depth":
        images = torch.tensor([0.0, 2.0, float("nan")]).view(1, 1, 3, 1)
        expected = torch.tanh(torch.tensor([0.0, 1.0, 20.0])) - 0.5
    else:
        images = torch.tensor([0, 51, 255, 255], dtype=torch.uint8).view(1, 1, 1, 4)
        if data_type == "rgb":
            images = images[..., :3]
        expected = torch.tensor([-0.5, -0.3, 0.5, 0.5])
        if data_type != "semantic_segmentation":
            expected = expected[:3]
    sensor = SimpleNamespace(data=SimpleNamespace(output={data_type: ProxyArray(wp.from_torch(images))}))
    env = SimpleNamespace(
        num_envs=1, device="cpu", scene=SimpleNamespace(sensors={"base_camera": sensor, "wrist_camera": sensor})
    )
    for term_cfg in (cfg.observations.base_image.object_observation_b, cfg.observations.wrist_image.wrist_observation):
        term = term_cfg.func(term_cfg, env)
        raw = term(env, **term_cfg.params)
        assert raw.dtype == images.dtype
        assert raw.shape == (1, 1, 1, 3) if data_type == "depth" else raw.shape == (1, len(expected), 1, 1)
        output = CameraImageNormalizer(raw.dtype == torch.uint8)(raw)
        torch.testing.assert_close(output.flatten(), expected)


def _make_pose_command(
    monkeypatch: pytest.MonkeyPatch,
    num_envs: int,
    success_asset: object,
    *,
    command_cfg: mdp.ObjectUniformPoseCommandCfg | None = None,
) -> tuple[ObjectUniformPoseCommand, torch.Tensor, list[torch.Tensor]]:
    """Build a pose command around fake assets and spies; returns it, the root positions, and material colors."""
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
    material = SimpleNamespace(is_per_env=True)
    scene = _FakeScene(environment_ids, robot=robot, object=object_asset, table=success_asset, table_material=material)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
    cfg = command_cfg or SimpleNamespace(
        class_type=ObjectUniformPoseCommand,
        asset_name="robot",
        object_name="object",
        success_vis_asset_name="table",
        success_visualizer_cfg=object(),
        success_vis_material_name="table_material",
        success_vis_colors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        goal_pose_visualizer_cfg=object(),
        curr_pose_visualizer_cfg=object(),
        position_only=True,
        cmd_kind=None,
        element_names=None,
    )
    scene[cfg.object_name] = object_asset
    if isinstance(cfg, mdp.CableUniformPoseCommandCfg):
        object_asset.num_segments = cfg.segment_index + 1
        object_asset.data.segment_pose_w = SimpleNamespace(
            torch=root_pose_w[:, None, :].expand(-1, object_asset.num_segments, -1)
        )

    def _initialize_command_term(command, command_cfg, command_env) -> None:
        command.cfg = command_cfg
        command._env = command_env
        command.metrics = {}

    colors: list[torch.Tensor] = []
    monkeypatch.setattr(CommandTerm, "__init__", _initialize_command_term)
    monkeypatch.setattr(pose_commands, "VisualizationMarkers", _MarkerSpy)
    monkeypatch.setattr(
        pose_commands.VisualMaterial, "write_channels", lambda _, channels: colors.append(channels["color"])
    )
    return instantiate(cfg, env), root_pos_w, colors


@pytest.mark.parametrize(
    ("env_cfg_type", "command_name"),
    [
        (KukaAllegroLiftCameraEnvCfg, "object_pose"),
        (FrankaSoftEnvCfg, "deformable_pose"),
        (FrankaCableEnvCfg, "cable_pose"),
    ],
)
def test_lift_table_colors_update_only_in_play(
    monkeypatch: pytest.MonkeyPatch, env_cfg_type: type, command_name: str
) -> None:
    """Training updates metrics without color writes; play updates success colors."""
    cfg = resolve_presets(env_cfg_type())
    command, _, colors = _make_pose_command(monkeypatch, 2, None, command_cfg=getattr(cfg.commands, command_name))
    command.pose_command_b[:, 0] = torch.tensor([1.0, 0.0])
    command._update_metrics()
    torch.testing.assert_close(command.metrics["position_error"], torch.tensor([1.0, 0.0]))
    assert colors == []

    cfg.play_mode()
    command, _, colors = _make_pose_command(monkeypatch, 2, None, command_cfg=getattr(cfg.commands, command_name))
    command.pose_command_b[:, 0] = torch.tensor([1.0, 0.0])
    command._update_metrics()
    failure, success = command.cfg.success_vis_colors
    torch.testing.assert_close(colors[-1], torch.tensor([[failure, success]]))


@pytest.mark.parametrize("static", [False, True])
def test_lift_pose_markers_forward_environment_ids(monkeypatch: pytest.MonkeyPatch, static: bool) -> None:
    """Every lift pose, goal, and success marker should retain its environment ownership."""
    num_envs = 3
    environment_ids = torch.arange(num_envs)
    success_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.zeros((num_envs, 3)))))
    if static:
        success_asset = object.__new__(Asset)
        success_asset.cfg = SimpleNamespace(init_state=SimpleNamespace(pos=(0.5, 0.0, 0.0)))
    command, root_pos_w, colors = _make_pose_command(monkeypatch, num_envs, success_asset)
    command._set_debug_vis_impl(True)
    command._debug_vis_callback(None)
    command.cfg.position_only = False
    command._debug_vis_callback(None)
    # Discard construction-time placement, as happens before a backend is active.
    command.success_visualizer.calls.clear()
    command.cfg.position_only = True
    command.pose_command_b[:, 0] = torch.tensor([1.0, 0.0, 1.0])
    command._update_metrics()
    assert torch.equal(command.success_visualizer.calls[0][1]["marker_indices"], torch.tensor([0, 1, 0]))
    failure, success = command.cfg.success_vis_colors
    torch.testing.assert_close(colors[-1], torch.tensor([[failure, success, failure]]))
    DeformableUniformPoseCommand._update_metrics(command)
    command._segment_position_w = lambda: root_pos_w
    CableUniformPoseCommand._update_metrics(command)
    CableUniformPoseCommand._debug_vis_callback(command, None)

    expected_call_counts = {
        command.success_visualizer: 3,
        command.goal_visualizer: 3,
        command.curr_visualizer: 3,
    }
    for visualizer, expected_count in expected_call_counts.items():
        assert len(visualizer.calls) == expected_count
        for _, kwargs in visualizer.calls:
            assert torch.equal(kwargs["environment_ids"], environment_ids)
    if static:
        for args, _ in command.success_visualizer.calls:
            torch.testing.assert_close(args[0], torch.tensor([[0.5, 0.0, 0.0]] * num_envs))


def test_lift_contact_terms_match_per_sensor_reference() -> None:
    """Contact gating and counting from stacked sensor forces match a per-sensor evaluation."""
    num_envs, threshold = 3, 1.0
    names = ["thumb", "index", "middle"]
    # The rows cover a finger without thumb, thumb with finger, and thumb alone.
    magnitudes = {"thumb": [0.0, 2.0, 2.0], "index": [2.0, 0.0, 0.0], "middle": [0.0, 2.0, 0.0]}
    forces = {
        name: torch.tensor([[value, 0.0, 0.0] for value in values]).reshape(num_envs, 1, 1, 3)
        for name, values in magnitudes.items()
    }
    sensors = {
        name: SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=ProxyArray(wp.from_torch(force))))
        for name, force in forces.items()
    }
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=SimpleNamespace(sensors=sensors))
    assert torch.equal(mdp.contacts(env, threshold, "thumb", names[1:]), torch.tensor([False, True, False]))
    torch.testing.assert_close(mdp.contact_count(env, threshold, names), torch.tensor([1 / 3, 2 / 3, 1 / 3]))
    assert not mdp.contacts(env, threshold, "thumb", []).any()


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
