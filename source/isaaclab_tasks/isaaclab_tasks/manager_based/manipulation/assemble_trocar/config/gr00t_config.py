# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T data configuration for IsaacLab tasks.

This module defines customizable GR00T data configurations for different
embodiments. Users can create their own data config classes by subclassing
BaseDataConfig or copying/modifying the examples here.

Example usage in run.sh:
    export RLINF_DATA_CONFIG="policy.gr00t_config"
    export RLINF_DATA_CONFIG_CLASS="policy.gr00t_config:IsaacLabDataConfig"
"""

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionSinCosTransform, StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import DATA_CONFIG_MAP, BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class IsaacLabDataConfig(BaseDataConfig):
    """Generic GR00T data config for IsaacLab tasks with G1 + Dex3."""

    # Video modality keys (from gr00t_mapping.video in RLINF_OBS_MAP_JSON)
    video_keys = [
        "video.left_wrist_view",
        "video.right_wrist_view",
        "video.room_view",
    ]

    # State modality keys (from gr00t_mapping.state in RLINF_OBS_MAP_JSON)
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]

    # Action modality keys (output from GR00T model)
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]

    # Language annotation key
    language_keys = ["annotation.human.task_description"]

    # Observation and action indices
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Define modality configurations for video, state, action, and language."""
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self):
        """Define the transform pipeline for processing observations and actions."""
        transforms = [
            # Video transforms
            VideoToTensor(apply_to=self.video_keys),
            # Disabled: camera already outputs 224×224 via TiledCameraCfg.
            # To avoid VideoToTensor size-check errors, either:
            #   1. Disable input size validation in VideoToTensor, OR
            #   2. Set modality meta height/width to 224 to match actual input.
            # Re-enable VideoCrop/VideoResize if camera resolution changes.
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(
            #     apply_to=self.video_keys,
            #     height=224,
            #     width=224,
            #     interpolation="linear",
            # ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # State transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # Action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # Concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # Model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# --------------------------------------------------------------------------
# Register data configs into GR00T's DATA_CONFIG_MAP
# --------------------------------------------------------------------------

# This allows load_data_config("policy.gr00t_config:IsaacLabDataConfig") to work
DATA_CONFIG_MAP["isaaclab_g1_dex3"] = IsaacLabDataConfig()
