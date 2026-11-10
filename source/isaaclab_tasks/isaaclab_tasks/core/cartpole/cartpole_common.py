# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations shared by the direct and manager-based cartpole environments."""

from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg, SteelSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import CameraCfg, LidarCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

LIGHT_ORIENTATION: tuple[float, float, float, float] = (
    -0.14644663035869598,
    -0.3535534143447876,
    -0.3535534143447876,
    0.8535533547401428,
)
"""Distant light orientation as an ``(x, y, z, w)`` quaternion for roll, pitch, yaw = (0, -45, -45) degrees."""


@configclass
class CartpolePhysicsCfg(PresetCfg):
    """Physics backend presets for the cartpole environments."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=5,
            nconmax=3,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_steel: NewtonCfg = NewtonCfg(
        solver_cfg=SteelSolverCfg(
            enable_joint_limits=True, iterations=4, articulation_constraint_max=8, free_body_constraint_max=64
        ),
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class CartpoleTiledCameraCfg(PresetCfg):
    """Tiled-camera presets, one per rendered data type.

    Each variant selects its rendering backend (RTX, OmniverseRTX, Newton + Warp) through the
    nested :attr:`~BaseCartpoleTiledCameraCfg.renderer_cfg` preset, so a single ``presets=`` selector
    can pick both the data type and the backend.
    """

    @configclass
    class BaseCartpoleTiledCameraCfg(CameraCfg):
        """Camera looking at the cartpole from the side."""

        prim_path: str = "{ENV_REGEX_NS}/Camera"
        offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
            pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        )
        width: int = 96
        height: int = 96
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()

    default = BaseCartpoleTiledCameraCfg(data_types=["rgb"])
    depth = BaseCartpoleTiledCameraCfg(data_types=["depth"])
    albedo = BaseCartpoleTiledCameraCfg(data_types=["albedo"])
    semantic_segmentation = BaseCartpoleTiledCameraCfg(data_types=["semantic_segmentation"])
    simple_shading_constant_diffuse = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_full_mdl"])
    rgb = default


def cartpole_lidar_pattern() -> tuple[list[float], list[float], list[int], list[int]]:
    """Build one solid-state grid aimed at the cartpole from the side camera.

    Azimuth runs along the rail. Elevation is symmetric about the horizon because the pole can hang
    below the slider as well as stand above it. Both angles use a 0.5° step, from -40° to 40° in
    azimuth and from -12° to 12° in elevation.

    Returns:
        Azimuth degrees, elevation degrees, firing times in nanoseconds, and channel ids. Each
        sequence has one entry per ray.
    """
    azimuths = [round(-40.0 + index * 0.5, 1) for index in range(161)]
    elevations = [round(-12.0 + index * 0.5, 1) for index in range(49)]
    azimuth_deg = [azimuth for elevation in elevations for azimuth in azimuths]
    elevation_deg = [elevation for elevation in elevations for _azimuth in azimuths]
    fire_time_ns = [index * 1000 for index in range(len(azimuth_deg))]
    channel_id = [index + 1 for index in range(len(azimuth_deg))]
    return azimuth_deg, elevation_deg, fire_time_ns, channel_id


_CARTPOLE_LIDAR_AZIMUTH, _CARTPOLE_LIDAR_ELEVATION, _CARTPOLE_LIDAR_TIME, _CARTPOLE_LIDAR_CHANNEL = (
    cartpole_lidar_pattern()
)
CARTPOLE_LIDAR_RAY_COUNT = len(_CARTPOLE_LIDAR_AZIMUTH)
"""Number of rays in :func:`cartpole_lidar_pattern`."""


def cartpole_lidar_camera_cfg() -> CameraCfg:
    """Return the side RGB camera pinned to the lidar's OVRTX renderer.

    The cartpole camera presets keep a selectable renderer. This concrete camera keeps the lidar
    preset from falling through to the Newton Warp renderer.

    Returns:
        Camera configuration whose renderer matches :func:`cartpole_lidar_cfg`.
    """
    return CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
        data_types=["rgb"],
        renderer_cfg=OVRTXRendererCfg(motion_bvh="auto"),
    )


def cartpole_lidar_cfg() -> LidarCfg:
    """Return one solid-state lidar aimed at the cartpole from the side camera.

    The position matches the side camera. The quaternion aims the boresight along world +X, so
    azimuth 0 and elevation 0 look at the rail center.

    Returns:
        Lidar configuration pinned to OVRTX with motion BVH enabled.
    """
    return LidarCfg(
        prim_path="{ENV_REGEX_NS}/Lidar",
        offset=LidarCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.5, -0.5, -0.5, 0.5)),
        azimuth_deg=_CARTPOLE_LIDAR_AZIMUTH,
        elevation_deg=_CARTPOLE_LIDAR_ELEVATION,
        fire_time_ns=_CARTPOLE_LIDAR_TIME,
        channel_id=_CARTPOLE_LIDAR_CHANNEL,
        renderer_cfg=OVRTXRendererCfg(motion_bvh="auto"),
    )
