# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render cases for the foveated camera, shared by the render tests of each renderer.

A camera looks down at the ground plane and renders distance. The ground distance along any camera
ray is known in closed form, so the images check the sampling geometry without reference images.

Each case takes a backend: the physics configuration, the renderer configuration class, and the camera class the
renderer selects. Import this module after the test has launched its simulation app, if it needs one.
"""

import math

import numpy as np
import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import angular_directions
from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.test.utils.devices import resolve_test_sim_device
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

DEVICE = resolve_test_sim_device()
WARMUP_STEPS = 4
RESOLUTION = 32

# The physical camera the retina looks through: a 640x480 pinhole.
WIDTH, HEIGHT = 640, 480
FOCAL, CX, CY = 550.0, 319.5, 239.5
CLIPPING_RANGE = (0.001, 20.0)
CAMERA_MODEL = {"model": "pinhole", "image_size": (HEIGHT, WIDTH), "intrinsics": (FOCAL, FOCAL, CX, CY)}

# The same camera as the spawned camera's OpenCV model, from which the foveated camera derives its camera model.
SPAWNED_CAMERA = dict(
    camera_model=None,
    spawn=sim_utils.PinholeCameraCfg(
        clipping_range=CLIPPING_RANGE,
        distortion=sim_utils.OpenCvPinholeDistortionCfg(fx=FOCAL, fy=FOCAL, cx=CX, cy=CY, image_size=(WIDTH, HEIGHT)),
    ),
)

# The camera looks down at the ground plane; its optical axis meets the ground at the target.
CAM_EYE = (0.0, 0.0, 2.5)
CAM_TARGET = (1.75, 0.0, 0.0)


@configclass
class _GroundSceneCfg(InteractiveSceneCfg):
    """A ground plane, a foveated camera, and an off-screen body; Newton cannot build an empty model."""

    ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.001),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )
    camera = FoveatedCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=CLIPPING_RANGE),
        camera_model=CAMERA_MODEL,
        foveated_res=RESOLUTION,
        mask_to_sensor_fov=True,
    )


def _look_at(eye, target) -> tuple[float, float, float, float]:
    """OpenGL camera orientation (x, y, z, w) at ``eye`` looking at ``target`` with world +Z up."""
    matrix = create_rotation_matrix_from_view(torch.tensor([eye]), torch.tensor([target]), up_axis="Z")
    return tuple(quat_from_matrix(matrix)[0].tolist())


def _ground_distance_along_gaze(tilt_deg: float, pan_deg: float, eye=CAM_EYE, target=CAM_TARGET) -> float:
    """Distance [m] from the camera to the ground along the gaze (tilt, pan) [deg]; up and left are positive."""
    tilt, pan = math.radians(tilt_deg), math.radians(pan_deg)
    # OpenGL camera frame: +X right, +Y up, looking along -Z.
    ray_camera = np.array((-math.cos(tilt) * math.sin(pan), math.sin(tilt), -math.cos(tilt) * math.cos(pan)))
    eye = np.asarray(eye)
    forward = np.asarray(target) - eye
    z_axis = -forward / np.linalg.norm(forward)
    x_axis = np.cross(np.array((0.0, 0.0, 1.0)), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    ray_world = np.column_stack((x_axis, y_axis, z_axis)) @ ray_camera
    return float(-eye[2] / ray_world[2])


def _cell_angles(field: torch.Tensor, fov: float) -> torch.Tensor:
    """Angles (tilt, pan) [deg] of each grid cell's ray from the gaze, from fovi's angular projection."""
    # Each cell's camera-frame ray (OpenCV: x right, y down, z forward).
    x, y, z = angular_directions(field.double(), fov).unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1).rad2deg().float()


def _render(
    backend: tuple,
    gaze: torch.Tensor,
    moved_mounts: dict[int, tuple] | None = None,
    later_gaze: torch.Tensor | None = None,
    **camera_overrides,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Render ground distance for one environment per row of gaze angles (tilt, pan) [deg].

    With ``moved_mounts``, each listed environment's camera moves to its (eye, target) after its gaze is set; a target
    of None moves only its position.
    With ``later_gaze``, the gaze changes to it after the first frames, and the camera renders one more frame.

    Returns:
        The distance images [m], shape (num_envs, H, W), the visual-field position of each grid cell, shape
        (H, W, 2), the angles (tilt, pan) [deg] of each grid cell's ray from the gaze, shape (H, W, 2), and the
        retina's field of view [deg].
    """
    physics_cfg, renderer_cfg, camera_class = backend
    sim_cfg = SimulationCfg(physics=physics_cfg, device=DEVICE)
    with build_simulation_context(device=DEVICE, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = _GroundSceneCfg(num_envs=len(gaze), env_spacing=20.0)
        scene_cfg.camera.renderer_cfg = renderer_cfg()
        rot = _look_at(CAM_EYE, CAM_TARGET)
        scene_cfg.camera.offset = FoveatedCameraCfg.OffsetCfg(pos=CAM_EYE, rot=rot, convention="opengl")
        for name, value in camera_overrides.items():
            setattr(scene_cfg.camera, name, value)
        scene = InteractiveScene(scene_cfg)
        camera: FoveatedCamera = scene["camera"]
        assert isinstance(camera, camera_class)
        sim.reset()
        camera.set_gaze_angles(gaze.to(sim.device))
        for env, (eye, target) in (moved_mounts or {}).items():
            position = scene.env_origins[env : env + 1] + torch.tensor([eye], device=sim.device)
            # Without a target, only the position moves and the mount keeps its orientation.
            orientation = None if target is None else torch.tensor([_look_at(eye, target)], device=sim.device)
            camera.set_world_poses(position, orientation, env_ids=[env], convention="opengl")
        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(sim.get_physics_dt(), force_recompute=True)
        if later_gaze is not None:
            # The first frame rendered after the command must already show the new gaze.
            camera.set_gaze_angles(later_gaze.to(sim.device))
            sim.step()
            camera.update(sim.get_physics_dt(), force_recompute=True)
        distance = camera.data.output["distance_to_camera"].torch[..., 0].detach().cpu().float().clone()
        coords = camera.sampling_coords
        field = coords.as_grid(coords.cartesian, sample_dim=0).detach().cpu().float().clone()
        return distance, field, _cell_angles(field, coords.fov), float(coords.fov)


def foveated_camera_follows_gaze(backend: tuple):
    """The retina's centre sees the ground along the gaze, from the first frame after the gaze changes.

    Both environments render along the optical axis, then one turns 16 degrees to the right, as a gaze action does
    every step. The camera is the spawned camera's OpenCV model. The reference is the analytic ground distance along
    each gaze direction.
    """
    later = torch.tensor([[0.0, 0.0], [0.0, -16.0]])
    distance, field, _, _ = _render(backend, torch.zeros(2, 2), later_gaze=later, **SPAWNED_CAMERA)

    assert distance.shape == (2, RESOLUTION, RESOLUTION)
    assert torch.isfinite(distance).all()

    # The grid cell closest to the centre of the visual field.
    row, column = np.unravel_index(int(field.norm(dim=-1).argmin()), field.shape[:2])
    for env, (tilt, pan) in enumerate(later.tolist()):
        assert float(distance[env, row, column]) == pytest.approx(_ground_distance_along_gaze(tilt, pan), rel=0.01)

    # The image is upright: the top looks farther along the ground than the bottom.
    centred = distance[0]
    seen = centred > 0
    top = centred[: RESOLUTION // 2][seen[: RESOLUTION // 2]].mean()
    bottom = centred[RESOLUTION // 2 :][seen[RESOLUTION // 2 :]].mean()
    assert top > bottom + 0.25
    # The camera has no roll, so the left and right halves see the same distances.
    left = centred[:, : RESOLUTION // 2][seen[:, : RESOLUTION // 2]].mean()
    right = centred[:, RESOLUTION // 2 :][seen[:, RESOLUTION // 2 :]].mean()
    assert float(left) == pytest.approx(float(right), rel=0.01)

    # Looking right moves part of the retina past the edge of the physical camera, and those samples are zeroed.
    assert (distance[1] == 0).sum() > (distance[0] == 0).sum()


def gaze_stays_relative_to_moved_mounts(backend: tuple):
    """Moving a camera keeps its gaze relative to the new mount; moving only its position keeps its orientation, so the
    gaze is not applied a second time; and the camera that did not move is left alone."""
    gaze = torch.tensor([[0.0, -10.0]]).repeat(3, 1)
    turned_eye, turned_target = (0.3, -0.4, 3.2), (2.0, 0.3, 0.0)
    shifted_eye = (0.4, -0.3, 3.0)
    shifted_target = tuple(e + t - c for e, t, c in zip(shifted_eye, CAM_TARGET, CAM_EYE))
    distance, field, _, _ = _render(
        backend, gaze, moved_mounts={1: (turned_eye, turned_target), 2: (shifted_eye, None)}
    )

    row, column = np.unravel_index(int(field.norm(dim=-1).argmin()), field.shape[:2])
    for env, (eye, target) in enumerate(
        ((CAM_EYE, CAM_TARGET), (turned_eye, turned_target), (shifted_eye, shifted_target))
    ):
        expected = _ground_distance_along_gaze(0.0, -10.0, eye, target)
        assert float(distance[env, row, column]) == pytest.approx(expected, rel=0.01), env


def _relative_errors(distance: torch.Tensor, angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative error of the rendered distance of each seen cell from the ground along the cell's own ray.

    Args:
        distance: Rendered distance image [m], shape (H, W).
        angles: Angles (tilt, pan) [deg] of each cell's ray, shape (H, W, 2), with the gaze on the optical axis.

    Returns:
        The relative errors and the eccentricities [deg] of the rays of the cells the camera sees.
    """
    expected = torch.tensor([_ground_distance_along_gaze(tilt, pan) for tilt, pan in angles.view(-1, 2).tolist()])
    tilt, pan = angles.view(-1, 2).deg2rad().unbind(-1)
    eccentricity = torch.acos(tilt.cos() * pan.cos()).rad2deg()
    distance = distance.flatten()
    seen = torch.isfinite(distance) & (distance > 0)
    return ((distance - expected).abs() / expected)[seen], eccentricity[seen]


def foveated_camera_samples_densest_at_the_fixation(backend: tuple):
    """Samples are densest at the fixation and each renders along its own ray.

    The rendered distances match the ground along each sample's ray well off the centre, and not along the rays of an
    even grid with the same resolution and field of view, so a renderer that traced an even grid instead fails.
    """
    distance, field, angles, fov = _render(backend, torch.zeros(1, 2))
    steps = field[RESOLUTION // 2, :, 0].diff()
    assert float(steps[len(steps) // 2] / steps[-1]) < 0.2

    errors, eccentricity = _relative_errors(distance[0], angles)
    # Most of the grid is seen, and the checked rays reach far from the centre of the field.
    assert len(errors) > RESOLUTION**2 // 2
    assert (eccentricity > 10.0).sum() > RESOLUTION**2 // 4
    assert float(eccentricity.max()) > 20.0
    assert float(errors.median()) < 0.01
    assert float((errors < 0.03).float().mean()) > 0.98

    even = SamplingCoords(fov, 1000.0, RESOLUTION, style="warped_cartesian_as_grid", field_geometry="spherical")
    even_angles = _cell_angles(even.as_grid(even.cartesian, sample_dim=0).float(), fov)
    errors, _ = _relative_errors(distance[0], even_angles)
    assert float((errors < 0.03).float().mean()) < 0.5


def hybrid_gaze_turns_the_head_onto_the_fixation(backend: tuple):
    """Hybrid gaze on the pan-tilt mount: the retina turns at once, then the joints carry the camera onto the fixation.

    The policy pans left three full steps, then holds. The retina leads while the head turns; the head then ends on
    the commanded fixation, the retina back on the optical axis, and the camera's world axis turned with the head.
    A reset brings the joint targets and the gaze back to the rig's default pose.
    """
    from isaaclab_fovi.assets import PAN_TILT_CAMERA_OFFSET, PAN_TILT_CFG
    from isaaclab_fovi.mdp import GazeActionCfg, GazeRigCfg, gaze_state

    from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils.math import quat_apply

    physics_cfg, renderer_cfg, _ = backend

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        rig = PAN_TILT_CFG.replace(init_state=PAN_TILT_CFG.init_state.replace(pos=(0.0, 0.0, 0.5)))
        eye = FoveatedCameraCfg(
            prim_path="{ENV_REGEX_NS}/PanTilt/Geometry/world/pan_link/tilt_link/Eye",
            data_types=["distance_to_camera"],
            renderer_cfg=renderer_cfg(),
            spawn=sim_utils.PinholeCameraCfg(clipping_range=CLIPPING_RANGE),
            foveated_res=16,
            offset=PAN_TILT_CAMERA_OFFSET,
        )

    @configclass
    class ActionsCfg:
        gaze = GazeActionCfg(asset_name="eye", mode="hybrid", mount="tracked", rig=GazeRigCfg(asset_name="rig"))

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObservationGroupCfg):
            gaze = ObservationTermCfg(func=gaze_state)

        policy = PolicyCfg()

    @configclass
    class EnvCfg(ManagerBasedEnvCfg):
        decimation = 4
        scene = SceneCfg(num_envs=1, env_spacing=5.0)
        actions = ActionsCfg()
        observations = ObservationsCfg()
        ui_window_class_type = None
        sim = SimulationCfg(dt=1 / 120, physics=physics_cfg, device=DEVICE)

    sim_utils.create_new_stage()
    env = ManagerBasedEnv(EnvCfg())
    try:
        env.reset()
        term = env.action_manager.get_term("gaze")
        camera, rig = env.scene["eye"], env.scene["rig"]
        joints = rig.find_joints(["pan_joint", "tilt_joint"], preserve_order=True)[0]

        def pan_deg():
            """The head's pan, the retina's pan and the camera's world yaw, left of the start [deg]."""
            axis = quat_apply(camera.data.quat_w_ros.torch[:1], torch.tensor([[0.0, 0.0, 1.0]], device=env.device))
            yaw = math.degrees(math.atan2(-float(axis[0, 0]), float(axis[0, 1])))
            head = math.degrees(float(rig.data.joint_pos.torch[0, joints[0]]))
            return head, float(camera.gaze_angles.torch[0, 1]), yaw

        commanded = 3 * term.cfg.step_size * term.field_radius_deg
        pan_left = torch.tensor([[0.0, 1.0]], device=env.device)
        # Nothing reads the camera between steps, so it updates lazily: the gaze must read the head's pose itself.
        for _ in range(3):
            env.step(pan_left)
        head, retina, fixation = pan_deg()
        # The retina leads the head, and the fixation is already near the command.
        assert retina > head
        assert fixation == pytest.approx(commanded, abs=1.5)
        for _ in range(30):
            env.step(torch.zeros_like(pan_left))
        head, retina, fixation = pan_deg()
        assert head == pytest.approx(commanded, abs=0.2)
        assert retina == pytest.approx(0.0, abs=0.2)
        assert fixation == pytest.approx(commanded, abs=0.2)
        assert float(term.state[0, 1]) == pytest.approx(3 * term.cfg.step_size, abs=0.005)

        env.reset()
        assert not term.joint_targets.any()
        assert not camera.gaze_angles.torch.any()
        assert not term.state.any()
    finally:
        env.close()
