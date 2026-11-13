# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark`` config, motion replay, and kitchen spawner.

No Kit/GPU required: the config is loaded and resolved through the registry, the motion replay runs against a mocked
articulation, and the spawner runs on a small stand-in for the coffee kitchen.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.benchmark.benchmark_mode import BENCHMARK_MODE
from isaaclab_tasks.benchmark.coffee_kitchen_camera.coffee_kitchen_camera_env import (
    CoffeeKitchenCameraEnv,
    CoffeeMotionPlayback,
    frame_indices,
    pose_from_recording,
    spread_offsets,
)
from isaaclab_tasks.benchmark.coffee_kitchen_camera.coffee_kitchen_camera_env_cfg import (
    KITCHEN_CAMERA_COUNT,
    KITCHEN_LIGHT_COUNT,
    CoffeeMotionPlaybackCfg,
)
from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import ARM_JOINT_NAMES, COFFEE_READY_POSITIONS
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark"
_DATA_TYPES = (
    "rgb",
    "depth",
    "albedo",
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
    "semantic_segmentation",
)
_RESOLUTIONS = (64, 128, 256)


def _load_cfg():
    return load_cfg_from_registry(_TASK, "env_cfg_entry_point")


def test_scene_has_no_powder():
    cfg = _load_cfg()

    assert cfg.scene.coffee is None and cfg.scene.water is None


@pytest.mark.skipif(BENCHMARK_MODE is not None, reason="BENCHMARK_MODE overrides the default mode")
def test_default_mode_is_render():
    assert _load_cfg().benchmark_mode == "render"


def test_pose_event_fires_after_physics_on_every_step():
    """A global interval event with a zero interval runs on every step, after physics and before observations."""
    event = _load_cfg().events.recorded_pose

    assert event.func.endswith("coffee_kitchen_camera_env:pose_from_recording")
    assert (event.mode, event.interval_range_s, event.is_global_time) == ("interval", (0.0, 0.0), True)


def test_recording_matches_robot_and_physics():
    """The shipped recording drives the demo's arm joints from its ready pose at the environment's physics rate."""
    cfg = _load_cfg()
    motion = json.loads(Path(cfg.actions.arm_motion.motion_file).read_text(encoding="utf-8"))
    positions, velocities, targets = (torch.tensor(motion[key]) for key in ("positions", "velocities", "targets"))

    assert motion["joint_names"] == list(ARM_JOINT_NAMES)
    assert motion["decimation"] == cfg.decimation
    assert motion["physics_hz"] * cfg.sim.dt == pytest.approx(1.0)
    assert motion["rate_hz"] * cfg.sim.dt * cfg.decimation == pytest.approx(1.0)
    frames, joints = positions.shape
    assert velocities.shape == (frames, joints)
    assert targets.shape == (frames - 1, cfg.decimation, joints)
    assert all(tensor.isfinite().all() for tensor in (positions, velocities, targets))
    assert positions[0].tolist() == pytest.approx(COFFEE_READY_POSITIONS, abs=1e-5)


@pytest.mark.parametrize(
    ("num_envs", "expected"),
    [(1, [0]), (4, [0, 3, 6, 9]), (5, [0, 2, 4, 7, 9]), (16, [0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11])],
)
def test_spread_offsets_cover_the_recording_evenly(num_envs, expected):
    assert sorted(spread_offsets(num_envs, 12, seed=42).tolist()) == expected


def test_spread_offsets_order_follows_the_seed():
    assert torch.equal(spread_offsets(64, 1000, seed=7), spread_offsets(64, 1000, seed=7))
    assert not torch.equal(spread_offsets(64, 1000, seed=7), spread_offsets(64, 1000, seed=8))


def test_frame_indices_loop_over_the_recording():
    assert frame_indices(torch.tensor([0, 3, 9]), step=5, length=10).tolist() == [5, 8, 4]


_LENGTH, _DECIMATION, _HAND_JOINTS = 4, 2, 2


def _write_motion(path: Path) -> None:
    """A recording whose frame ``f`` holds every arm joint at ``f``, moving at ``f / 10``; substep ``s`` targets
    ``f + s / 10``."""
    recording = {
        "rate_hz": 100,
        "decimation": _DECIMATION,
        "joint_names": list(ARM_JOINT_NAMES),
        "positions": [[float(frame)] * len(ARM_JOINT_NAMES) for frame in range(_LENGTH)],
        "velocities": [[frame / 10] * len(ARM_JOINT_NAMES) for frame in range(_LENGTH)],
        "targets": [
            [[frame + substep / 10] * len(ARM_JOINT_NAMES) for substep in range(_DECIMATION)]
            for frame in range(_LENGTH - 1)
        ],
    }
    path.write_text(json.dumps(recording), encoding="utf-8")


def _make_playback(
    tmp_path: Path, mode: str, step: int, decimation: int = _DECIMATION
) -> tuple[CoffeeMotionPlayback, Mock, SimpleNamespace]:
    """The replay term on a mocked articulation and simulation, whose calls are recorded in order on the mock."""
    motion_file = tmp_path / "motion.json"
    _write_motion(motion_file)
    events = Mock()
    robot = events.robot
    robot.joint_names = [*(f"hand_{index}" for index in range(_HAND_JOINTS)), *ARM_JOINT_NAMES]
    robot.data.default_joint_pos.torch = torch.full((_LENGTH, _HAND_JOINTS + len(ARM_JOINT_NAMES)), -1.0)
    events.sim.backend_manager.handles_decimation.return_value = False
    env = SimpleNamespace(
        cfg=SimpleNamespace(decimation=decimation, seed=3, benchmark_mode=mode),
        step_dt=0.01,
        num_envs=_LENGTH,
        device="cpu",
        common_step_counter=step,
        scene={"robot": robot},
        sim=events.sim,
    )
    playback = CoffeeMotionPlayback(CoffeeMotionPlaybackCfg(motion_file=str(motion_file)), env)
    events.reset_mock()
    return playback, events, env


def test_playback_rejects_a_recording_made_at_another_step_rate(tmp_path):
    with pytest.raises(ValueError, match="decimation"):
        _make_playback(tmp_path, "physics_render", step=0, decimation=_DECIMATION + 1)


def _arm(values: torch.Tensor) -> torch.Tensor:
    return values[:, _HAND_JOINTS:]


def test_physics_render_sends_each_substeps_recorded_target(tmp_path):
    """One environment per frame: the targets follow the recording, and the last frame restarts at the first."""
    playback, events, env = _make_playback(tmp_path, "physics_render", step=2)
    frames = frame_indices(spread_offsets(_LENGTH, _LENGTH, seed=3), env.common_step_counter, _LENGTH)
    (wrapping,) = (frames == _LENGTH - 1).nonzero().flatten().tolist()

    playback.process_actions(torch.zeros(_LENGTH, 0))
    for _ in range(_DECIMATION):
        playback.apply_actions()

    target_calls = events.robot.actuators.target_command.set_position_index.call_args_list
    substep_targets = [call.kwargs["value"] for call in target_calls if "joint_ids" in call.kwargs]
    assert len(substep_targets) == _DECIMATION
    for substep, value in enumerate(substep_targets):
        expected = (frames + substep / 10).float().unsqueeze(1).expand(-1, len(ARM_JOINT_NAMES)).clone()
        expected[wrapping] = 0.0
        assert torch.allclose(value, expected)
    events.robot.write_joint_position_to_sim_index.assert_called_once()
    written = events.robot.write_joint_position_to_sim_index.call_args.kwargs
    assert written["env_ids"] == [wrapping]
    assert torch.equal(_arm(written["position"]), torch.zeros(1, len(ARM_JOINT_NAMES)))
    events.sim.forward.assert_called_once()


@pytest.mark.parametrize(
    ("mode", "lazy_sensor_update", "renderer_type", "pumps_app_update", "error"),
    [
        ("physics", True, "ovrtx", False, "benchmark_mode must be one of"),
        (None, True, "ovrtx", False, "benchmark_mode must be one of"),
        ("render", False, "ovrtx", False, "lazy_sensor_update=True"),
        ("render", True, "isaac_rtx", True, "requires no visualizer"),
        ("physics_render", True, "isaac_rtx", True, None),
        ("render", True, "ovrtx", True, None),
    ],
)
def test_camera_env_rejects_modes_it_cannot_run(
    monkeypatch, mode, lazy_sensor_update, renderer_type, pumps_app_update, error
):
    """Unknown modes would freeze the arm; direct posing needs rendering to follow the pose write."""
    cfg = _load_cfg()
    cfg.benchmark_mode = mode
    cfg.scene.lazy_sensor_update = lazy_sensor_update

    def initialize(self, *args, **kwargs):
        self.sim = SimpleNamespace(
            render_context=SimpleNamespace(renderer_types=(renderer_type,)),
            visualizers=[SimpleNamespace(pumps_app_update=lambda: pumps_app_update)],
        )

    close = Mock()
    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", initialize)
    monkeypatch.setattr(CoffeeKitchenCameraEnv, "close", close)

    if error:
        with pytest.raises(ValueError, match=error):
            CoffeeKitchenCameraEnv(cfg)
    else:
        CoffeeKitchenCameraEnv(cfg)
    assert close.called == (error == "requires no visualizer")


def test_render_mode_poses_after_physics_then_updates_the_spoon_and_transforms(tmp_path):
    playback, events, env = _make_playback(tmp_path, "render", step=5)
    playback.process_actions(torch.zeros(_LENGTH, 0))
    playback.apply_actions()
    assert not events.mock_calls, "render mode leaves the actuator targets alone before physics"

    camera_env = SimpleNamespace(
        cfg=env.cfg,
        action_manager=SimpleNamespace(get_term=lambda name: playback),
        sync_spoon=events.sync_spoon,
        sim=events.sim,
    )
    camera_env.pose_from_recording = lambda: CoffeeKitchenCameraEnv.pose_from_recording(camera_env)
    pose_from_recording(camera_env, None)

    assert [call[0] for call in events.mock_calls] == [
        "robot.write_joint_position_to_sim_index",
        "robot.write_joint_velocity_to_sim_index",
        "robot.actuators.target_command.set_position_index",
        "sync_spoon",
        "sim.forward",
    ]
    frames = frame_indices(playback._offsets_host, env.common_step_counter, _LENGTH).float()
    position = events.robot.write_joint_position_to_sim_index.call_args.kwargs["position"]
    assert torch.equal(_arm(position), frames.unsqueeze(1).expand(-1, len(ARM_JOINT_NAMES)))
    velocity = events.robot.write_joint_velocity_to_sim_index.call_args.kwargs["velocity"]
    assert torch.equal(velocity, torch.zeros_like(velocity))


def test_pose_event_does_nothing_in_physics_render_mode():
    env = SimpleNamespace(cfg=SimpleNamespace(benchmark_mode="physics_render"), pose_from_recording=Mock())

    pose_from_recording(env, None)

    env.pose_from_recording.assert_not_called()


def test_reset_starts_physics_render_from_the_recorded_state(tmp_path):
    playback, events, env = _make_playback(tmp_path, "physics_render", step=0)

    playback.reset(None)

    frames = frame_indices(playback._offsets_host, 0, _LENGTH).float().unsqueeze(1)
    position = events.robot.write_joint_position_to_sim_index.call_args.kwargs["position"]
    velocity = events.robot.write_joint_velocity_to_sim_index.call_args.kwargs["velocity"]
    assert torch.equal(_arm(position), frames.expand(-1, len(ARM_JOINT_NAMES)))
    assert torch.allclose(_arm(velocity), (frames / 10).expand(-1, len(ARM_JOINT_NAMES)))
    assert torch.equal(velocity[:, :_HAND_JOINTS], torch.zeros(_LENGTH, _HAND_JOINTS))


@pytest.fixture(scope="module")
def kitchen_presets():
    """Collect every preset once, before any preset selection mutates the config in place."""
    return collect_presets(_load_cfg())


def test_camera_presets_resolve_to_expected_data_types(kitchen_presets):
    """Each camera preset requests the data type and resolution it is named after (``default`` is ``rgb256``)."""
    camera_presets = kitchen_presets["scene.camera"]
    expected = {
        f"{data_type}{resolution}": (data_type, resolution) for data_type in _DATA_TYPES for resolution in _RESOLUTIONS
    }
    assert set(camera_presets) == {"default", *expected}
    expected["default"] = ("rgb", 256)
    for preset_name, resolved in camera_presets.items():
        data_type, resolution = expected[preset_name]
        assert resolved.data_types == [data_type]
        assert (resolved.width, resolved.height) == (resolution, resolution)


def test_one_preset_selects_camera_and_image_term():
    cfg = resolve_presets(_load_cfg(), selected=("simple_shading_full_mdl64",))

    assert cfg.scene.camera.data_types == ["simple_shading_full_mdl"]
    assert (cfg.scene.camera.width, cfg.scene.camera.height) == (64, 64)
    assert cfg.observations.camera_image.image.params["data_type"] == "simple_shading_full_mdl"
    assert cfg.observations.camera_image.image.params["sensor_cfg"].name == "camera"


@pytest.fixture
def stage():
    """A new in-memory stage, current only for the test so the previous stage is restored afterwards."""
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        yield stage


def _write_kitchen(path: Path, light_count: int) -> None:
    """A stand-in for the coffee kitchen: geometry, a ceiling, and the expected lights and cameras."""
    asset = Usd.Stage.CreateNew(str(path))
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Kitchen").GetPrim())
    floor = UsdGeom.Mesh.Define(asset, "/Kitchen/KitchenRoom/Floor")
    # As in the hosted kitchen: the normals primvar holds the interpolation, the normals attribute the normals.
    floor.GetPrim().CreateAttribute("primvars:normals", Sdf.ValueTypeNames.TokenArray).Set(["faceVarying"])
    floor.CreateNormalsAttr([(0.0, 0.0, 1.0)])
    UsdGeom.Mesh.Define(asset, "/Kitchen/KitchenRoom/Ceiling")
    # As in the hosted kitchen's appliances: a disabled door body hinged by a joint outside any articulation.
    door = UsdGeom.Xform.Define(asset, "/Kitchen/Fridge/Door").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(door).CreateRigidBodyEnabledAttr(False)
    UsdPhysics.RevoluteJoint.Define(asset, "/Kitchen/Fridge/Joints/Hinge")
    # As in the hosted kitchen's appliances: an invisible grasp-annotation curve.
    UsdGeom.BasisCurves.Define(asset, "/Kitchen/Fridge/Grasp/grasp_identifier/grasp_identifier_curve").MakeInvisible()
    for index in range(light_count):
        UsdLux.SphereLight.Define(asset, f"/Kitchen/Lights/Light{index}")
    for index in range(KITCHEN_CAMERA_COUNT):
        UsdGeom.Camera.Define(asset, f"/Kitchen/Cameras/Camera{index}")
    asset.Save()


def _spawn_kitchen(stage: Usd.Stage, path: Path) -> None:
    """Spawn the kitchen into two environments, each with a table whose studio floor the spawner hides."""
    for env_index in range(2):
        UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_index}/Table/StudioFloor")
    cfg = _load_cfg().scene.kitchen.spawn
    cfg.usd_path = str(path)
    cfg.func("/World/envs/env_.*/Kitchen", cfg)


def test_kitchen_spawner_removes_ceiling_lights_cameras_and_token_normals(tmp_path, stage):
    """Every environment's copy keeps its geometry; loses the ceiling, lights, cameras, curves, appliance joints
    and bodies; gets its normals' interpolation from the token primvar; and hides the floor."""
    kitchen_path = tmp_path / "kitchen.usda"
    _write_kitchen(kitchen_path, KITCHEN_LIGHT_COUNT)
    _spawn_kitchen(stage, kitchen_path)

    for env_index in range(2):
        env = f"/World/envs/env_{env_index}"
        kitchen_floor = stage.GetPrimAtPath(f"{env}/Kitchen/KitchenRoom/Floor")
        assert kitchen_floor.IsActive()
        assert UsdGeom.PrimvarsAPI(kitchen_floor).GetPrimvar("normals").Get() is None
        assert len(UsdGeom.Mesh(kitchen_floor).GetNormalsAttr().Get()) == 1
        assert UsdGeom.Mesh(kitchen_floor).GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/KitchenRoom/Ceiling").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/Lights/Light0").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/Cameras/Camera0").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/Fridge/Joints/Hinge").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/Fridge/Grasp/grasp_identifier/grasp_identifier_curve").IsActive()
        door = stage.GetPrimAtPath(f"{env}/Kitchen/Fridge/Door")
        assert door.IsActive() and not door.HasAPI(UsdPhysics.RigidBodyAPI)
        floor = UsdGeom.Imageable(stage.GetPrimAtPath(f"{env}/Table/StudioFloor"))
        assert floor.ComputeVisibility() == UsdGeom.Tokens.invisible


def test_kitchen_spawner_rejects_changed_light_count(tmp_path, stage):
    """A kitchen bundle with an unexpected light count fails instead of adding lights to every environment."""
    kitchen_path = tmp_path / "kitchen.usda"
    _write_kitchen(kitchen_path, KITCHEN_LIGHT_COUNT + 1)

    with pytest.raises(RuntimeError, match=f"{KITCHEN_LIGHT_COUNT + 1} lights"):
        _spawn_kitchen(stage, kitchen_path)
