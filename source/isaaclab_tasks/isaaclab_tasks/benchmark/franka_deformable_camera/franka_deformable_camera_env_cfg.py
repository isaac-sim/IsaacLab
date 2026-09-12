# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the Franka deformable camera benchmark tasks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka_soft.franka_cable_env_cfg import FrankaCableEnvCfg, FrankaCableSceneCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg, FrankaClothSceneCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

FRANKA_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.85, -0.55, 0.42),
        rot=(0.5080, 0.2114, 0.318, 0.7720),
        convention="opengl",
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),
    width=128,
    height=128,
    renderer_cfg=MultiBackendRendererCfg(),
)


@configclass
class _FrankaSoftCameraSceneCfg(_FrankaSoftSceneCfg):
    """Franka soft scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class FrankaCameraObservationsCfg:
    """Observation groups for visual deformable lifting."""

    @configclass
    class PolicyCfg(ObsGroup):
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "deformable_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PerceptionCfg(ObsGroup):
        deformable_sampled_points = ObsTerm(
            func=mdp.DeformableSampledPointsInRobotRootFrame,
            params={"asset_cfg": SceneEntityCfg("deformable"), "num_points": 20},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BaseImageCfg(ObsGroup):
        image = ObsTerm(
            func=env_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "normalize": True,
                "permute": True,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    perception: PerceptionCfg = PerceptionCfg()
    base_image: BaseImageCfg = BaseImageCfg()


@configclass
class FrankaSoftCameraSceneCfg(PresetCfg):
    """Scene presets for visual Franka soft lifting."""

    newton_mjwarp_vbd_proxy: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(
        num_envs=128, env_spacing=2.0, replicate_physics=True
    )
    physx: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaSoftCameraEnvCfg(FrankaSoftEnvCfg):
    """Visual Franka volume-deformable lifting environment."""

    scene: FrankaSoftCameraSceneCfg = FrankaSoftCameraSceneCfg()
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class FrankaClothCameraScenePresetCfg(PresetCfg):
    """Scene presets for visual Franka cloth lifting."""

    newton_mjwarp_vbd_proxy: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )
    physx: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=False)
    isaacsim_physx = physx
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothCameraEnvCfg(FrankaClothEnvCfg):
    """Visual Franka surface-deformable lifting environment."""

    scene: FrankaClothCameraScenePresetCfg = FrankaClothCameraScenePresetCfg()
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2


@configclass
class FrankaCableCameraSceneCfg(FrankaCableSceneCfg):
    """Franka cable scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class FrankaCableCameraObservationsCfg(FrankaCameraObservationsCfg):
    """Observation groups for visual cable lifting."""

    @configclass
    class PolicyCfg(FrankaCameraObservationsCfg.PolicyCfg):
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "cable_pose"})

    @configclass
    class PerceptionCfg(ObsGroup):
        cable_segment_positions = ObsTerm(
            func=mdp.cable_segment_positions_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cable")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    perception: PerceptionCfg = PerceptionCfg()


@configclass
class FrankaCableCameraEnvCfg(FrankaCableEnvCfg):
    """Visual Franka cable lifting environment."""

    scene: FrankaCableCameraSceneCfg = FrankaCableCameraSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
    observations: FrankaCableCameraObservationsCfg = FrankaCableCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.num_rerenders_on_reset = 2
