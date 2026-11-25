# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations shared by the direct and manager-based cartpole environments."""

import torch
from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg, SteelSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXLidarCfg

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sensors import CameraCfg
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


# 0.5° steps. Elevation is the slow axis and azimuth is the fast axis of the ray grid.
_CARTPOLE_LIDAR_AZIMUTH_AXIS = [round(-40.0 + index * 0.5, 1) for index in range(161)]
_CARTPOLE_LIDAR_ELEVATION_AXIS = [round(-12.0 + index * 0.5, 1) for index in range(49)]
CARTPOLE_LIDAR_AZIMUTH_COUNT = len(_CARTPOLE_LIDAR_AZIMUTH_AXIS)
"""Width of the lidar range image, one column per azimuth."""
CARTPOLE_LIDAR_ELEVATION_COUNT = len(_CARTPOLE_LIDAR_ELEVATION_AXIS)
"""Height of the lidar range image, one row per elevation."""


CARTPOLE_LIDAR_SUBSAMPLING = 2
"""``LidarWpm`` tiled stride. Values above 1 publish one return out of this many authored rays."""

CARTPOLE_LIDAR_FAR_RANGE = 12.0
"""Maximum lidar range, in meters.

The sensor is 5 m from the rail and the cart can travel 3 m, so the local robot is within about 8 m.
The next environment is 20 m away. A 12 m range includes the local cart and pole and excludes the neighbors.
"""


def cartpole_lidar_range_image(points: torch.Tensor) -> torch.Tensor:
    """Pack one scan into a range image on the lidar's elevation-azimuth grid.

    Each pixel is the hit distance divided by :data:`CARTPOLE_LIDAR_FAR_RANGE`. Misses are non-finite
    on the sensor and become 0. Neighboring rays stay neighboring pixels, so a one-ray-wide pole
    remains a column instead of disappearing into a flat coordinate list.

    A subsampled scan still uses that grid. ``LidarWpm`` publishes authored rays ``0, stride, 2 * stride, ...``
    and drops the remainder. Unpublished pixels stay 0.

    Args:
        points: Sensor-frame returns, in meters. Shape is ``(num_envs, num_rays, 3)``, either one return
            per authored ray or the shorter published scan.

    Returns:
        Range image in ``[0, 1]``. Shape is ``(num_envs, elevation, azimuth)``.

    Raises:
        ValueError: If ``points`` is neither the authored grid nor the subsampled scan.
    """
    height = CARTPOLE_LIDAR_ELEVATION_COUNT
    width = CARTPOLE_LIDAR_AZIMUTH_COUNT
    authored = height * width
    stride = max(1, CARTPOLE_LIDAR_SUBSAMPLING)
    published = authored // stride
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] not in (authored, published):
        raise ValueError(
            f"Expected lidar points of shape (num_envs, {authored}, 3) or (num_envs, {published}, 3),"
            f" got {tuple(points.shape)}."
        )

    if points.shape[1] == authored:
        grid = points.reshape(points.shape[0], height, width, 3)
        distance = torch.linalg.vector_norm(grid, dim=-1)
    else:
        samples = torch.linalg.vector_norm(points, dim=-1)
        distance = samples.new_zeros((samples.shape[0], height, width))
        distance.flatten(1)[:, : published * stride : stride] = samples
    distance = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    return distance.div_(CARTPOLE_LIDAR_FAR_RANGE).clamp_(0.0, 1.0)


def cartpole_lidar_cfg() -> OVRTXLidarCfg:
    """Return a ``LidarWpm`` solid-state scan aimed at the cartpole from the side camera.

    The scan fires the full elevation-azimuth grid once. Azimuth runs from -40° to 40° and elevation
    from -12° to 12°, both at 0.5°. Elevation is the slow axis. :data:`CARTPOLE_LIDAR_SUBSAMPLING`
    keeps one published slot out of that many authored rays.
    """
    azimuths = _CARTPOLE_LIDAR_AZIMUTH_AXIS
    elevations = _CARTPOLE_LIDAR_ELEVATION_AXIS
    azimuth_deg = [azimuth for _elevation in elevations for azimuth in azimuths]
    elevation_deg = [elevation for elevation in elevations for _azimuth in azimuths]
    fire_time_ns = [index * 1000 for index in range(len(azimuth_deg))]
    channel_id = [index + 1 for index in range(len(azimuth_deg))]
    return OVRTXLidarCfg(
        prim_path="{ENV_REGEX_NS}/Lidar",
        scan_type="solidState",
        offset=OVRTXLidarCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(0.5, -0.5, -0.5, 0.5)),
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        fire_time_ns=fire_time_ns,
        channel_id=channel_id,
        output_frame_of_reference="SENSOR",
        elements_coords_type="CARTESIAN",
        far_range_m=CARTPOLE_LIDAR_FAR_RANGE,
        tiled_subsampling=CARTPOLE_LIDAR_SUBSAMPLING,
        max_returns=1,
        instant_lidar=True,
        include_invalid_points=True,
    )
