# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-policy variant of the reset-oriented Franka stack task."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import tensor_processors
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import mdp
from ...stack_env_cfg import ObjectTableSceneCfg
from .stack_rl_env_cfg import EventCfg, FrankaCubeStackRLEnvCfg, FrankaStackStateObservationCfg


@configclass
class FrankaStackCameraSceneCfg(ObjectTableSceneCfg):
    """Franka stack scene with one fixed RGB camera."""

    # The oblique view resolves both table-plane axes and keeps the complete
    # 0.40-0.56 m by -0.18-0.18 m reset workspace in frame. The quaternion is
    # the OpenGL look-at rotation from ``pos`` to (0.48, 0.0, 0.08).
    base_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        # Keep cube edges and fingers visible to the CNN across the workspace.
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.90, -0.55, 0.48),
            rot=(0.4734479, 0.1600998, 0.2774602, 0.8205066),
            convention="opengl",
        ),
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class CameraEventCfg(EventCfg):
    """Physical reset table plus fixed-per-environment camera calibration error."""

    camera_calibration = EventTerm(
        func=mdp.randomize_camera_calibration,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("base_camera"),
            "eye": (0.90, -0.55, 0.48),
            "lookat": (0.48, 0.0, 0.08),
            "eye_position_noise": (0.020, 0.020, 0.015),
            "lookat_position_noise": (0.015, 0.015, 0.010),
        },
    )


@configclass
class CameraObservationsCfg:
    """RGB and proprioception for the actor, full state for the critic and teacher."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception and action history for the visual policy."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_target = ObsTerm(
            func=mdp.joint_position_target,
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        gripper_pos = ObsTerm(
            func=mdp.gripper_pos,
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class BaseImageCfg(ObsGroup):
        """Single-frame, channel-first RGB input for the visual encoder.

        Scale each frame by ``1 / 255`` before applying photometric noise in
        normalized image units.
        """

        rgb = ObsTerm(
            func=mdp.image_rgb,
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "normalize": False,
                "channel_first": True,
            },
            tensor_processors=[
                tensor_processors.TensorProcessorCfg(func=tensor_processors.scale, params={"multiplier": 1.0 / 255.0})
            ],
            noise=mdp.EpisodeCameraNoiseCfg(),
            clip=(0.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: FrankaStackStateObservationCfg = FrankaStackStateObservationCfg()
    base_image: BaseImageCfg = BaseImageCfg()


@configclass
class FrankaCubeStackCameraRLEnvCfg(FrankaCubeStackRLEnvCfg):
    """Stack from RGB and proprioception with no actor-side object state."""

    scene: FrankaStackCameraSceneCfg = FrankaStackCameraSceneCfg(
        # The 32-step rollout stores about 1.5 GiB of float RGB observations.
        num_envs=256,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: CameraObservationsCfg = CameraObservationsCfg()
    events: CameraEventCfg = CameraEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Assign the teacher's role order to fixed cube colors so the visual
        # student can identify each role. Any physical stack order can succeed.
        self.events.reset_from_state_buffer.params["fixed_role_permutation"] = 0

        # The first action after a reset must see the new robot and cube poses.
        self.num_rerenders_on_reset = 1

    def validate_config(self) -> None:
        """Require a fresh RGB frame at each policy step and after a reset."""
        super().validate_config()
        if (
            self.scene.base_camera.update_period != 0.0
            or self.sim.render_interval != self.decimation
            or self.num_rerenders_on_reset < 1
        ):
            raise ValueError("Stack camera observations must refresh at every policy step and after scene resets.")
