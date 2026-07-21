# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera variants of the Franka deformable lifting environments."""

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .franka_cloth_env_cfg import FrankaClothEnvCfg, FrankaClothSceneCfg
from .franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg


FRANKA_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Camera",
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
class _FrankaSoftCameraSceneCfg(_FrankaSoftSceneCfg):
    """Franka soft scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class FrankaSoftCameraSceneCfg(PresetCfg):
    """Scene presets for visual Franka soft lifting."""

    newton_mjwarp_vbd: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )
    physx: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=False
    )
    newton_mjwarp_vbd_proxy = newton_mjwarp_vbd
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class FrankaSoftCameraEnvCfg(FrankaSoftEnvCfg):
    """Visual Franka volume-deformable lifting environment."""

    scene: FrankaSoftCameraSceneCfg = FrankaSoftCameraSceneCfg()
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()


@configclass
class FrankaClothCameraEnvCfg(FrankaClothEnvCfg):
    """Visual Franka surface-deformable lifting environment."""

    scene: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()
