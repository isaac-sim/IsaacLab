# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render cases for the foveated camera, shared by the render tests of each renderer.

A camera looks down at the ground plane and renders distance. The ground distance along any camera
ray is known in closed form, so the images check the sampling geometry without reference images.

:func:`ground_scene` builds one scene per backend (the physics configuration, the renderer configuration class, and
the camera class the renderer selects), which every case renders in turn; each case sets the gaze and mounts it
needs. Import this module after the test has launched its simulation app, if it needs one.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import angular_directions
from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg
from isaaclab_fovi.sensors.gaze_controls import ViewerGazeControls

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
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


NUM_ENVS = 3
"""Environments of the shared scene: the moved-mounts case turns one camera and shifts another."""


class GroundScene:
    """A ground scene shared by the render cases of one backend: per environment, a camera with an explicit camera model
    and one derived from the spawned camera's OpenCV model, at the same mount."""

    def __init__(self, sim, scene: InteractiveScene):
        self.sim = sim
        self.scene = scene
        self.calibrated: FoveatedCamera = scene["camera"]
        self.spawned: FoveatedCamera = scene["spawned"]

    def reset_mounts(self, camera: FoveatedCamera) -> None:
        """Put every environment's camera back at the shared mount."""
        positions = self.scene.env_origins + torch.tensor([CAM_EYE], device=self.sim.device)
        orientations = torch.tensor([_look_at(CAM_EYE, CAM_TARGET)], device=self.sim.device).repeat(NUM_ENVS, 1)
        camera.set_world_poses(positions, orientations, convention="opengl")

    def move_mount(self, camera: FoveatedCamera, env: int, eye, target) -> None:
        """Move one environment's camera to ``eye`` looking at ``target``; a target of None moves only its position."""
        position = self.scene.env_origins[env : env + 1] + torch.tensor([eye], device=self.sim.device)
        orientation = None if target is None else torch.tensor([_look_at(eye, target)], device=self.sim.device)
        camera.set_world_poses(position, orientation, env_ids=[env], convention="opengl")

    def render(
        self, camera: FoveatedCamera, frames: int = WARMUP_STEPS
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Render ``frames`` frames and return the camera's ground distance.

        Returns:
            The distance images [m], shape (num_envs, H, W), the visual-field position of each grid cell, shape
            (H, W, 2), the angles (tilt, pan) [deg] of each grid cell's ray from the gaze, shape (H, W, 2), and the
            retina's field of view [deg].
        """
        for _ in range(frames):
            self.sim.step()
            camera.update(self.sim.get_physics_dt(), force_recompute=True)
        distance = camera.data.output["distance_to_camera"].torch[..., 0].detach().cpu().float().clone()
        coords = camera.sampling_coords
        field = coords.as_grid(coords.cartesian, sample_dim=0).detach().cpu().float().clone()
        return distance, field, _cell_angles(field, coords.fov), float(coords.fov)


@contextmanager
def ground_scene(backend: tuple) -> Iterator[GroundScene]:
    """Build the ground scene once for ``backend``, so its render cases share it instead of rebuilding it."""
    physics_cfg, renderer_cfg, camera_class = backend
    sim_cfg = SimulationCfg(physics=physics_cfg, device=DEVICE)
    with build_simulation_context(device=DEVICE, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = _GroundSceneCfg(num_envs=NUM_ENVS, env_spacing=20.0)
        scene_cfg.camera.renderer_cfg = renderer_cfg()
        rot = _look_at(CAM_EYE, CAM_TARGET)
        scene_cfg.camera.offset = FoveatedCameraCfg.OffsetCfg(pos=CAM_EYE, rot=rot, convention="opengl")
        scene_cfg.spawned = scene_cfg.camera.replace(prim_path="{ENV_REGEX_NS}/Spawned", **SPAWNED_CAMERA)
        scene = InteractiveScene(scene_cfg)
        for name in ("camera", "spawned"):
            assert isinstance(scene[name], camera_class)
        sim.reset()
        yield GroundScene(sim, scene)


def foveated_camera_follows_gaze(ground: GroundScene):
    """The retina's centre sees the ground along the gaze, from the first frame after the gaze changes.

    The environments render along the optical axis, then one turns 16 degrees to the right, as a gaze action does
    every step. The camera is the spawned camera's OpenCV model. The reference is the analytic ground distance along
    each gaze direction.
    """
    camera = ground.spawned
    camera.set_gaze_angles(torch.zeros(NUM_ENVS, 2, device=ground.sim.device))
    ground.render(camera)
    # The first frame rendered after the command must already show the new gaze.
    later = torch.tensor([[0.0, 0.0], [0.0, -16.0], [0.0, 0.0]])
    camera.set_gaze_angles(later.to(ground.sim.device))
    distance, field, _, _ = ground.render(camera, frames=1)

    assert distance.shape == (NUM_ENVS, RESOLUTION, RESOLUTION)
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


def gaze_stays_relative_to_moved_mounts(ground: GroundScene):
    """Moving a camera keeps its gaze relative to the new mount; moving only its position keeps its orientation, so the
    gaze is not applied a second time; and the camera that did not move is left alone."""
    camera = ground.calibrated
    ground.reset_mounts(camera)
    camera.set_gaze_angles(torch.tensor([[0.0, -10.0]], device=ground.sim.device).repeat(NUM_ENVS, 1))
    turned_eye, turned_target = (0.3, -0.4, 3.2), (2.0, 0.3, 0.0)
    shifted_eye = (0.4, -0.3, 3.0)
    shifted_target = tuple(e + t - c for e, t, c in zip(shifted_eye, CAM_TARGET, CAM_EYE))
    ground.move_mount(camera, 1, turned_eye, turned_target)
    ground.move_mount(camera, 2, shifted_eye, None)
    distance, field, _, _ = ground.render(camera)

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


def foveated_camera_samples_densest_at_the_fixation(ground: GroundScene):
    """Samples are densest at the fixation and each renders along its own ray.

    The rendered distances match the ground along each sample's ray well off the centre, and not along the rays of an
    even grid with the same resolution and field of view, so a renderer that traced an even grid instead fails.
    """
    camera = ground.calibrated
    ground.reset_mounts(camera)
    camera.set_gaze_angles(torch.zeros(NUM_ENVS, 2, device=ground.sim.device))
    distance, field, angles, fov = ground.render(camera)
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


def cmf_a_switches_the_retina_at_runtime(ground: GroundScene):
    """Newton re-samples the retina for a new CMF constant in place: each sample renders along its new ray, the outputs
    keep their shape, and switching back restores the first image. Building the camera added its viewer controls."""
    camera = ground.calibrated
    controls = [
        callback.__self__
        for callback in SimulationContext._reset_callbacks.values()
        if isinstance(getattr(callback, "__self__", None), ViewerGazeControls)
    ]
    assert any(control._camera() is camera for control in controls)
    ground.reset_mounts(camera)
    camera.set_gaze_angles(torch.zeros(NUM_ENVS, 2, device=ground.sim.device))
    first = camera.cfg.cmf_a_deg
    before, _, first_angles, _ = ground.render(camera)
    camera.set_cmf_a(10.0)
    try:
        after, _, angles, _ = ground.render(camera)
        assert after.shape == before.shape
        errors, _ = _relative_errors(after[0], angles)
        assert float((errors < 0.03).float().mean()) > 0.98
        # The new image does not follow the first retina's rays.
        errors, _ = _relative_errors(after[0], first_angles)
        assert float((errors < 0.03).float().mean()) < 0.9
    finally:
        camera.set_cmf_a(first)
    again, _, _, _ = ground.render(camera)
    assert torch.equal(again, before)
