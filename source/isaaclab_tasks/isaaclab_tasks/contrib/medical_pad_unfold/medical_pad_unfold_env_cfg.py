# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H2 medical-pad unfolding, with Cosmos3 imagining the camera view instead of a physics engine.

The task is written like any manager-based task: an articulation, a camera, joint-position actions, joint and image
observations and a classifier reward. The ``physics=`` preset selects the world-model backend that supplies the joint
state and the camera frames; no physics engine runs.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg, image_rgb, joint_pos, time_out
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import configclass

from isaaclab_neural.mdp import at_chunk_boundary
from isaaclab_neural.models.cosmos3 import Cosmos3Cfg
from isaaclab_neural.physics import NeuralCfg
from isaaclab_neural.renderers import NeuralRendererCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .h2_action_codec import H2JointMappingCfg
from .h2_robot import H2_JOINT_NAMES, recorded_joints_urdf

ASSETS = Path(__file__).with_name("assets")
MODEL_DIR = ".pretrained_checkpoints/medical_pad_unfold"
"""Where ``prepare_medical_pad.py`` puts the checkpoints, relative to the working directory."""

CHUNK = 16
"""Frames the world model generates, and joint targets it consumes, per model step."""

FPS = 30
IMAGE_SIZE = (480, 640)

CLASSIFIER = {
    "checkpoint": f"{MODEL_DIR}/classifier/reward_model.pt",
    "config_file": str(ASSETS / "classifier.yaml"),
    "success_class": 4,
    "failure_class": 0,
    "streak_frames": 30,
}
"""The reward model and the judgement of an episode, shared by the reward and the terminations.

The classes are ordered from worst to best (rewards -2 to 2). An episode succeeds (fails) at the frame that completes
``streak_frames`` consecutive frames of the best (worst) class.
"""


@configclass
class MedicalPadPhysicsCfg(PresetCfg):
    """Backends of the task. Only the world model exists for this task, so it is also the default.

    The Cosmos3 service, started with ``isaaclab-cosmos-world-model-server --checkpoint {MODEL_DIR}/world_model``,
    serves :data:`CHUNK` frames of :data:`IMAGE_SIZE` at :data:`FPS` per step.
    """

    default: NeuralCfg = NeuralCfg(
        model_cfg=Cosmos3Cfg(
            joint_mapping=H2JointMappingCfg(
                global_stats_path=f"{MODEL_DIR}/world_model/resources/global_asinh_v1_backward_anchored_gripper_metric.json"
            )
        ),
        reset_manifest=str(ASSETS / "resets.json"),
    )


@configclass
class MedicalPadSceneCfg(InteractiveSceneCfg):
    """The H2 robot and the head camera whose view the world model imagines."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(asset_path=str(recorded_joints_urdf()), fix_base=True, merge_fixed_joints=False),
        joint_ordering=H2_JOINT_NAMES,
        actuators={},
    )
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=IMAGE_SIZE[0],
        width=IMAGE_SIZE[1],
        data_types=["rgb"],
        frame_history_length=CHUNK,
        spawn=PinholeCameraCfg(focal_length=13.6, clipping_range=(0.001, 20.0)),
        renderer_cfg=NeuralRendererCfg(),
    )


@configclass
class ActionsCfg:
    """One absolute joint-position target [rad] per recorded joint and per step."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot", joint_names=H2_JOINT_NAMES, scale=1.0, use_default_offset=False, preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Joint positions and the head camera's image, in separate groups so an RL library can encode the image."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False

    @configclass
    class ImageCfg(ObsGroup):
        rgb = ObsTerm(
            func=image_rgb,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "channel_first": True},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    image: ImageCfg = ImageCfg()


@configclass
class EventCfg:
    """No reset event: the world-model articulation starts each episode from a reset record when it resets.

    This replaces the default ``reset_scene_to_default``, which also resets the root pose, and the world model has
    none."""


@configclass
class RewardsCfg:
    progress = RewTerm(
        func=mdp.classifier_progress,
        # The reward manager multiplies by the step duration; this weight cancels it, so the return equals the sum of
        # the classifier's per-frame progress.
        weight=FPS,
        params={"classifier": CLASSIFIER, "sensor_cfg": SceneEntityCfg("camera")},
    )


@configclass
class TerminationsCfg:
    # Ends at the first chunk boundary after the time limit, so random first episode lengths work too.
    time_out = DoneTerm(func=at_chunk_boundary, params={"term": time_out}, time_out=True)
    # Judged frame by frame inside a chunk and reported when the chunk completes, where environments can reset.
    success = DoneTerm(func=mdp.classifier_success, params={"classifier": CLASSIFIER})
    failure = DoneTerm(func=mdp.classifier_failure, params={"classifier": CLASSIFIER})


@configclass
class MedicalPadUnfoldEnvCfg(ManagerBasedRLEnvCfg):
    """Unfold a medical nursing pad with both H2 hands."""

    scene: MedicalPadSceneCfg = MedicalPadSceneCfg(num_envs=2, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # One environment step is one frame and one joint target; the model runs once per chunk of frames.
        self.decimation = 1
        self.sim.dt = 1.0 / FPS
        # 30 chunks.
        self.episode_length_s = 30 * CHUNK / FPS
        self.sim.physics = MedicalPadPhysicsCfg()
