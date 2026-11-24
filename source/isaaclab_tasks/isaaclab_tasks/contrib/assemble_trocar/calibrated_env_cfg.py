# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in rl2real calibration and packaged assets, independent of image generation."""

import os

from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_isaacsim.renderers import IsaacRtxRendererCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.sim import MassCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg, set_isaac_rtx_global_settings

from .config.camera_config import CameraPresets
from .g129_dex3_env_cfg import (
    AssembleTrocarSceneCfg,
    EventCfgFixTrayRotation,
    G1AssembleTrocarEnvCfg,
    PhysicsCfg,
)

# Mean reset pose from 117 real trocar episodes.
_CALIBRATED_JOINT_POS = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": -0.699552,
    "left_shoulder_roll_joint": 0.208284,
    "left_shoulder_yaw_joint": 0.23312,
    "left_elbow_joint": -0.186277,
    "left_wrist_roll_joint": -0.206097,
    "left_wrist_pitch_joint": 0.883695,
    "left_wrist_yaw_joint": 0.010174,
    "right_shoulder_pitch_joint": -0.945077,
    "right_shoulder_roll_joint": 0.022149,
    "right_shoulder_yaw_joint": -0.040477,
    "right_elbow_joint": 0.023875,
    "right_wrist_roll_joint": 0.035349,
    "right_wrist_pitch_joint": 0.890212,
    "right_wrist_yaw_joint": -0.075485,
    "left_hand_index_0_joint": -1.053897,
    "left_hand_middle_0_joint": -1.06179,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_index_1_joint": -0.696634,
    "left_hand_middle_1_joint": -0.704563,
    "left_hand_thumb_1_joint": 0.016303,
    "left_hand_thumb_2_joint": 0.0,
    "right_hand_index_0_joint": 1.029162,
    "right_hand_middle_0_joint": 1.11628,
    "right_hand_thumb_0_joint": 0.008241,
    "right_hand_index_1_joint": 0.673514,
    "right_hand_middle_1_joint": 0.424304,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
}


@configclass
class CalibratedAssembleTrocarSceneCfg(AssembleTrocarSceneCfg):
    """Packaged room and instruments with the camera and reset calibration used by rl2real."""

    def __post_init__(self):
        root = os.environ.get("ASSET_LOCAL", "/workspace/assets").rstrip("/")
        props = f"{root}/Props/LightWheel"
        self.robot.spawn.usd_path = f"{root}/Robots/g1_29dof_with_dex3_base_fix02.usd"
        self.robot.init_state.pos = (-1.86719, 1.94, 0.81168)
        self.robot.init_state.joint_pos.update(_CALIBRATED_JOINT_POS)
        self.scene.spawn.usd_path = f"{props}/scene03cell.usd"
        self.trocar_1.spawn.usd_path = f"{props}/Assets/Trocar002/Trocar004_nodeform.usd"
        self.trocar_1.init_state.pos = (-1.60338, 1.91313, 0.86879)
        self.trocar_1.init_state.rot = (-0.04351, 0.70001, 0.71218, 0.02998)
        self.trocar_2.spawn.usd_path = (
            f"{props}/Assets/DisposableLaparoscopicPunctureDevice001/DisposableLaparoscopicPunctureDevice006.usd"
        )
        self.trocar_2.init_state.pos = (-1.507, 1.90732, 0.85542)
        self.trocar_2.init_state.rot = (-0.65889, -0.05555, 0.01346, 0.75007)
        self.tray.spawn.usd_path = f"{props}/Assets/SurgicalTray001/SurgicalTray001_FixedBoxProp.usd"
        self.tray.spawn.mass_props = {"/[^/]+": [MassCfg(mass=1000.0)]}
        self.front_camera = CameraPresets.g1_front_camera(
            height=240, width=320, focal_length=8.7, horizontal_aperture=9.0
        )
        self.left_wrist_camera = CameraPresets.left_dex3_wrist_camera(
            height=240,
            width=320,
            focal_length=11.4,
            horizontal_aperture=19.2,
            clipping_range=(0.05, 1.0e5),
            pos_offset=(-0.01, -0.025, 0.1),
            rot_offset=(0.93721, -0.15211, 0.31385, 0.00166),
        )
        self.right_wrist_camera = CameraPresets.right_dex3_wrist_camera(
            height=240,
            width=320,
            focal_length=11.4,
            horizontal_aperture=19.2,
            clipping_range=(0.05, 1.0e5),
            pos_offset=(-0.01, 0.025, 0.1),
            rot_offset=(0.93536, 0.13055, 0.32871, 0.00283),
        )
        for camera in (self.front_camera, self.left_wrist_camera, self.right_wrist_camera):
            camera.renderer_cfg = MultiBackendRendererCfg(default=IsaacRtxRendererCfg())


@configclass
class CalibratedPhysicsCfg(PhysicsCfg):
    """Preserve the rl2real PhysX force application convention."""

    isaacsim_physx = PhysxCfg(bounce_threshold_velocity=0.01, enable_external_forces_every_iteration=False)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)


@configclass
class G1AssembleTrocarCalibratedEnvCfg(G1AssembleTrocarEnvCfg):
    """30 Hz task for the packaged rl2real assets; legacy task defaults stay unchanged."""

    scene: CalibratedAssembleTrocarSceneCfg = CalibratedAssembleTrocarSceneCfg(num_envs=1, env_spacing=6.0)

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 8
        self.episode_length_s = 25.0
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation
        self.sim.physics = CalibratedPhysicsCfg()
        self.sim.use_newton_actuators = preset(default=False, newton_mjwarp=True)
        for asset in (self.scene.robot, self.scene.scene, self.scene.trocar_1, self.scene.trocar_2, self.scene.tray):
            asset.cloning_contexts = preset(default=("isaaclab.cloner:UsdReplicateContext",), newton_mjwarp=None)
        for term in (self.rewards.update_stage, self.rewards.lift_trocars):
            term.params.update(lift_threshold=0.08, use_com_for_lift=True)
        for camera in (self.scene.front_camera, self.scene.left_wrist_camera, self.scene.right_wrist_camera):
            set_isaac_rtx_global_settings(
                camera.renderer_cfg,
                enable_translucency=True,
                carb_settings={
                    "rtx.raytracing.fractionalCutoutOpacity": True,
                    "renderer.scenePartitioning.enabled": False,
                },
                antialiasing_mode="DLAA",
            )


@configclass
class G1AssembleTrocarCalibratedEvalEnvCfg(G1AssembleTrocarCalibratedEnvCfg):
    """Calibrated task with deterministic per-environment tray rotations."""

    events: EventCfgFixTrayRotation = EventCfgFixTrayRotation()
