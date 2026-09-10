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
from isaaclab.utils import modifiers
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.stack_env_cfg import ObjectTableSceneCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .stack_rl_env_cfg import EventCfg, FrankaCubeStackRLEnvCfg, FrankaStackStateObservationCfg


@configclass
class FrankaStackCameraSceneCfg(ObjectTableSceneCfg):
    """Franka stack scene with one fixed, deployment-compatible RGB camera."""

    # The oblique view resolves both table-plane axes and keeps the complete
    # 0.40-0.56 m by -0.18-0.18 m reset workspace in frame. The quaternion is
    # the OpenGL look-at rotation from ``pos`` to (0.48, 0.0, 0.08).
    base_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        # Four-centimeter cubes project to only 3--6 pixels at 64 px from this
        # deployment camera. At 128 px they retain enough edge and finger
        # detail for the CNN encoder's final 12 x 12 feature map.
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
    """Deployment-compatible actor observations and a full-state training critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Camera-policy signals available directly from the robot and policy runtime."""

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

        The fixed ``uint8 / 255`` transform is independent of other
        environments and matches the preprocessing expected at deployment.
        Episode-consistent photometric variation models camera calibration and
        lighting changes without inventing temporal flicker.
        """

        rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "normalize": False,
                "permute": True,
            },
            modifiers=[modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 1.0 / 255.0})],
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
        # A 32-step rollout at this resolution already stores roughly 1.5 GiB
        # of float RGB before activations, optimizer state, and critic inputs.
        # Scale the batch from the CLI only after profiling the target GPU.
        num_envs=256,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The Franka demonstration parent reinstalls its own reset events.
        # Restore the camera variant while sharing the state task's physics,
        # actions, rewards, reset rows, curriculum, and success logic.
        self.events = CameraEventCfg()

        # The state teacher receives an abstract base/first/second role order.
        # A camera actor cannot observe the reset-time permutation that used to
        # assign colored cube assets to those roles. Tie the roles to the visible
        # blue/red/green assets so imitation has one identifiable target while
        # the physical success condition remains fully order-invariant.
        self.events.reset_from_state_buffer.params["fixed_role_permutation"] = 0

        # A reset changes the robot and all three cubes after the regular
        # render for that step. Refresh once so the first action of every
        # episode sees the new state instead of another environment's final
        # frame.
        self.num_rerenders_on_reset = 1
