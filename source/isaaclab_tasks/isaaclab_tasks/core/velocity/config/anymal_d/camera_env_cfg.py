# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-D flat locomotion with a per-environment tiled camera.

This variant is meant for validating RTX per-environment scene partitioning on a
robot with a *moving base*. Enable partitioning with
``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` and, to preview the new
Kit feature, pass ``--kit_args "--/rtx/scenePartitioning/showPartitionsInBackground=true"``.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import MySceneCfg, ObservationsCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .flat_env_cfg import AnymalDFlatEnvCfg

##
# Camera presets
##


@configclass
class AnymalDTiledCameraCfg(PresetCfg):
    """Per-environment tiled-camera presets, selectable via ``presets=``.

    The camera prim lives under each env root (``/World/envs/env_.*/Camera``), so the
    env-root ``primvars:omni:scenePartition`` token (authored only when
    ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1``) propagates to it and the
    render is culled to its own environment.
    """

    @configclass
    class BaseAnymalDTiledCameraCfg(CameraCfg):
        prim_path: str = "/World/envs/env_.*/Camera"
        # Placed behind and slightly above each env origin, looking down +x toward the
        # robot's spawn. The robot walks around, so the view intentionally shows the base
        # moving within (and potentially out of) its own partition volume.
        offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
            pos=(-5.0, 0.0, 1.2), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 40.0)
        )
        width: int = 128
        height: int = 128
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()

    default = BaseAnymalDTiledCameraCfg(data_types=["rgb"])
    rgb64 = BaseAnymalDTiledCameraCfg(data_types=["rgb"], width=64, height=64)
    rgb128 = BaseAnymalDTiledCameraCfg(data_types=["rgb"], width=128, height=128)
    rgb256 = BaseAnymalDTiledCameraCfg(data_types=["rgb"], width=256, height=256)
    depth128 = BaseAnymalDTiledCameraCfg(data_types=["depth"], width=128, height=128)
    rgb = default


##
# Scene + observations
##


@configclass
class AnymalDCameraSceneCfg(MySceneCfg):
    """Anymal-D locomotion scene with a selectable per-env tiled camera."""

    tiled_camera: AnymalDTiledCameraCfg = AnymalDTiledCameraCfg()


@configclass
class AnymalDCameraObservationsCfg(ObservationsCfg):
    """Adds a camera observation group so the tiled camera renders every step.

    rsl_rl consumes only the ``policy`` group (``obs_groups = {"actor": ["policy"],
    "critic": ["policy"]}``), so this extra group does not alter the locomotion policy;
    it only forces the sensor to be read (and thus rendered) each step.
    """

    @configclass
    class ImageObsCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    camera: ObsGroup = ImageObsCfg()


##
# Environment configuration
##


@configclass
class AnymalDFlatCameraEnvCfg(AnymalDFlatEnvCfg):
    """Anymal-D flat locomotion with a per-environment tiled camera."""

    scene: AnymalDCameraSceneCfg = AnymalDCameraSceneCfg(num_envs=4096, env_spacing=10.0)
    observations: AnymalDCameraObservationsCfg = AnymalDCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Widen env spacing so each env's partition bounding volume stays clear of its
        # neighbours' as the base moves (Steven Bloemer's layout constraint). Lower this
        # (e.g. back to 2.5) to observe the limitation when partitions overlap.
        self.scene.env_spacing = 10.0
