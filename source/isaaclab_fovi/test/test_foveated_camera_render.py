# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render tests for the foveated camera on the Newton Warp and Isaac RTX renderers.

A camera looks down at the ground plane and renders distance. The ground distance along any camera
ray is known in closed form, so the images check the sampling geometry without reference images.
"""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import math

import numpy as np
import pytest
import torch
from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg, FoveatedNewtonCamera, FoveatedRTXCamera
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

# Each backend pairs a physics engine with the renderer under test and the implementation it selects.
BACKENDS = {
    "newton_warp": (NewtonCfg(solver_cfg=MJWarpSolverCfg()), NewtonWarpRendererCfg, FoveatedNewtonCamera),
    "isaac_rtx": (PhysxCfg(), IsaacRtxRendererCfg, FoveatedRTXCamera),
}

DEVICE = "cuda:0"
WARMUP_STEPS = 4
RESOLUTION = 32

# The physical camera the retina looks through: a 640x480 pinhole.
WIDTH, HEIGHT = 640, 480
FOCAL, CX, CY = 550.0, 319.5, 239.5
CLIPPING_RANGE = (0.001, 20.0)
CAMERA_MODEL = {"model": "pinhole", "image_size": (HEIGHT, WIDTH), "intrinsics": (FOCAL, FOCAL, CX, CY)}

# The same camera given in either supported way: as fovi camera-model fields, or as the spawned
# camera's OpenCV model, from which the foveated camera derives them.
CAMERA_SOURCES = {
    "camera_model": dict(camera_model=CAMERA_MODEL),
    "spawn": dict(
        camera_model=None,
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=CLIPPING_RANGE,
            distortion=sim_utils.OpenCvPinholeDistortionCfg(
                fx=FOCAL, fy=FOCAL, cx=CX, cy=CY, image_size=(WIDTH, HEIGHT)
            ),
        ),
    ),
}

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


def _render(
    backend: str,
    gaze: torch.Tensor,
    moved_mount: tuple | None = None,
    later_gaze: torch.Tensor | None = None,
    **camera_overrides,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render ground distance for one environment per row of gaze angles (tilt, pan) [deg].

    With ``moved_mount`` (eye, target), the last environment's camera is moved there after its gaze is set.
    With ``later_gaze``, the gaze changes to it after the first frames, and the camera renders again.

    Returns:
        The distance images [m], shape (num_envs, H, W), and the visual-field position of each grid
        cell, shape (H, W, 2).
    """
    physics_cfg, renderer_cfg, camera_class = BACKENDS[backend]
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
        if moved_mount is not None:
            last = len(gaze) - 1
            eye, target = moved_mount
            position = scene.env_origins[last : last + 1] + torch.tensor([eye], device=sim.device)
            orientation = torch.tensor([_look_at(eye, target)], device=sim.device)
            camera.set_world_poses(position, orientation, env_ids=[last], convention="opengl")
        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(sim.get_physics_dt(), force_recompute=True)
        if later_gaze is not None:
            camera.set_gaze_angles(later_gaze.to(sim.device))
            for _ in range(WARMUP_STEPS):
                sim.step()
                camera.update(sim.get_physics_dt(), force_recompute=True)
        distance = camera.data.output["distance_to_camera"].torch[..., 0].detach().cpu().float().clone()
        coords = camera.sampling_coords
        field = coords.as_grid(coords.cartesian, sample_dim=0).detach().cpu().float().clone()
        return distance, field


@pytest.mark.parametrize("source", CAMERA_SOURCES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_foveated_camera_follows_gaze(backend, source):
    """The retina's centre sees the ground along the gaze direction.

    One environment looks along the optical axis and one 16 degrees to the right. The reference is
    the analytic ground distance along each gaze direction.
    """
    gaze = torch.tensor([[0.0, 0.0], [0.0, -16.0]])
    distance, field = _render(backend, gaze, **CAMERA_SOURCES[source])

    assert distance.shape == (2, RESOLUTION, RESOLUTION)
    assert torch.isfinite(distance).all()

    # The grid cell closest to the centre of the visual field.
    row, column = np.unravel_index(int(field.norm(dim=-1).argmin()), field.shape[:2])
    for env, (tilt, pan) in enumerate(gaze.tolist()):
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaze_change_reaches_the_next_frame(backend):
    """A gaze set after the camera has rendered moves the retina, as a gaze action does every step."""
    later = torch.tensor([[0.0, -16.0], [8.0, 10.0]])
    distance, field = _render(backend, torch.zeros(2, 2), later_gaze=later)

    row, column = np.unravel_index(int(field.norm(dim=-1).argmin()), field.shape[:2])
    for env, (tilt, pan) in enumerate(later.tolist()):
        assert float(distance[env, row, column]) == pytest.approx(_ground_distance_along_gaze(tilt, pan), rel=0.01)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaze_stays_relative_to_a_moved_mount(backend):
    """Moving one environment's camera keeps its gaze relative to the new mount and leaves the other alone."""
    gaze = torch.tensor([[0.0, -10.0], [0.0, -10.0]])
    eye, target = (0.3, -0.4, 3.2), (2.0, 0.3, 0.0)
    distance, field = _render(backend, gaze, moved_mount=(eye, target))

    row, column = np.unravel_index(int(field.norm(dim=-1).argmin()), field.shape[:2])
    assert float(distance[0, row, column]) == pytest.approx(_ground_distance_along_gaze(0.0, -10.0), rel=0.01)
    assert float(distance[1, row, column]) == pytest.approx(
        _ground_distance_along_gaze(0.0, -10.0, eye, target), rel=0.01
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_foveated_camera_samples_densest_at_the_fixation(backend):
    """Neighbouring samples are closer together at the fixation, and nearly even without foveation."""

    def central_to_edge_spacing(field: torch.Tensor) -> float:
        steps = field[RESOLUTION // 2, :, 0].diff()
        return float(steps[len(steps) // 2] / steps[-1])

    gaze = torch.zeros(1, 2)
    foveated_distance, foveated_field = _render(backend, gaze)
    uniform_distance, uniform_field = _render(backend, gaze, cmf_a_deg=1000.0)

    assert central_to_edge_spacing(foveated_field) < 0.2
    assert central_to_edge_spacing(uniform_field) > 0.9
    # Both cover the same field of view, so they see the same range of ground distances.
    for distance in (foveated_distance, uniform_distance):
        assert torch.isfinite(distance).all()
    assert float(foveated_distance.max()) == pytest.approx(float(uniform_distance.max()), rel=0.05)
