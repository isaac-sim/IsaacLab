# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm cube-stack environment with front + wrist cameras for visuomotor recording."""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp_core
from . import stack_ik_abs_env_cfg


@configclass
class ObservationsCfg:
    """Camera-only policy observations (concatenate_terms=False so each cam is separate)."""

    @configclass
    class PolicyCfg(ObsGroup):
        front_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        body_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class OpenarmCubeStackVisuomotorEnvCfg(stack_ik_abs_env_cfg.OpenarmCubeStackEnvCfg):
    """OpenArm stack env with two cameras for visuomotor data collection.

    Cameras:
      front_cam — fixed table-side view (640×480)
      wrist_cam — mounted on openarm_left_ee_tcp (640×480)

    Record with:
      ./isaaclab.sh -p scripts/tools/record_demos.py \\
          --task Isaac-Stack-Cube-OpenArm-IK-Abs-Visuomotor-v0 \\
          --dataset_file logs/demos/openarm_visuomotor.hdf5
    """

    def __post_init__(self):
        super().__post_init__()  # runs parent (sets empty policy, IK action, etc.)

        # Override policy observations with camera terms
        self.observations.policy = ObservationsCfg.PolicyCfg()

        # ── Front / table-side camera ─────────────────────────────────────
        # Positioned ~1 m in front of the robot, looking down at the workspace.
        # Adjust pos/rot to match your real camera mount if needed.
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 4.0),
            ),
            # ROS convention (w, x, y, z): points toward robot from front-right, slightly down
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.5),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        # ── Left wrist camera (attached to openarm_left_ee_tcp) ──────────
        # Position tuned manually in Isaac Sim:
        #   Isaac Sim stage Translate = (0.06156, 0.00072, -0.20867)
        #   Isaac Sim stage Orient    = (-180, 6.001, -90) degrees
        # The translation maps directly to pos; the orientation is unchanged
        # from the original (±180° on X is the same rotation).
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp/wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.06156, 0.00072, -0.20867),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # ── Right wrist camera (attached to openarm_right_ee_tcp) ─────────
        # Same offset as the left wrist camera.
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp/right_wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.06156, 0.00072, -0.20867),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        # ── Body camera (attached to openarm_body_link, front of torso at shoulder height) ─
        # Positioned at the front face of the central body pillar, ~65 cm above the base,
        # matching the real OpenArm robot's chest-mounted camera location.
        # pos=(0.12, 0, 0.65) relative to openarm_body_link; looks toward workspace center.
        # Rotation: camera looks forward (+X world) and 60° downward toward the workspace.
        # To retune: adjust pos/rot manually in Isaac Sim, then copy values here.
        self.scene.body_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/body_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.01, 0.0, 0.65),
                rot=(0.17853, -0.68420, 0.68420, -0.17853),
                convention="ros",
            ),
        )

        # Re-render after reset so the first frames are clean
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        # Tells the converter which obs keys are cameras
        self.image_obs_list = ["front_cam", "wrist_cam", "right_wrist_cam", "body_cam"]
