# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-throughput benchmark variant of the MicroDuck flat walking task with a head camera."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    MicroDuckSceneCfg,
    MicroDuckVelocityFlatEnvCfg,
    ObservationsCfg,
)
from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/neck/neck_pitch_1/yaw_roll_motion/jaw_soft/Camera",
    # The camera rides on the head, so the renderer needs its pose refreshed every capture.
    update_latest_camera_pose=True,
    # The robot USD's head_camera site, moved 10 mm along the optical axis to clear the lens and head shell.
    offset=CameraCfg.OffsetCfg(pos=(0.0155, 0.0, -0.0833), rot=(0.0, 0.7071, 0.0, 0.7071), convention="world"),
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 20.0)),
    data_types=["rgb"],
    width=64,
    height=64,
    renderer_cfg=MultiBackendRendererCfg(),
)

# Head camera outputs selectable with ``presets``, each with the observation term that reads it.
_HEAD_CAMERA_OUTPUTS = {
    "rgb": base_mdp.image_rgb,
    "depth": base_mdp.image_depth,
    "simple_shading_diffuse_mdl": base_mdp.image_rgb,
}

_HEAD_IMAGE_TERMS = {
    data_type: ObsTerm(
        func=func, params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": data_type, "normalize": False}
    )
    for data_type, func in _HEAD_CAMERA_OUTPUTS.items()
}


@configclass
class MicroDuckCameraSceneCfg(MicroDuckSceneCfg):
    """MicroDuck walking scene with a camera on each robot's head."""

    head_camera = preset(
        default=_HEAD_CAMERA_CFG,
        **{data_type: _HEAD_CAMERA_CFG.replace(data_types=[data_type]) for data_type in _HEAD_CAMERA_OUTPUTS},
    )


@configclass
class MicroDuckCameraObservationsCfg(ObservationsCfg):
    """The walking task's observations plus the head camera image."""

    @configclass
    class HeadCameraCfg(ObsGroup):
        """Head camera image, unnormalized. Reading it renders one frame per camera each step."""

        image = preset(default=_HEAD_IMAGE_TERMS["rgb"], **_HEAD_IMAGE_TERMS)

    head_camera: HeadCameraCfg = HeadCameraCfg()


@configclass
class MicroDuckCameraBenchmarkEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """MicroDuck flat walking with a head camera on every robot, for rendering-throughput benchmarks.

    The renderer backend and the camera output are selected with ``presets``::

        presets = ovrtx  # RGB with the OVRTX renderer
        presets = ovrtx, simple_shading_diffuse_mdl  # OVRTX Minimal mode
        presets = depth  # depth with the default Newton Warp renderer
    """

    scene: MicroDuckCameraSceneCfg = MicroDuckCameraSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: MicroDuckCameraObservationsCfg = MicroDuckCameraObservationsCfg()
