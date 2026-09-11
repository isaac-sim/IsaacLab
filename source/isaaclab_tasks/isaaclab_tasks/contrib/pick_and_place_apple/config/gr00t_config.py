# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.5 data config for Unitree H2 + Sharpa Wave."""

from __future__ import annotations

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP, BaseDataConfig
from gr00t.model.transforms import GR00TTransform

from isaaclab_tasks.contrib.h2_sharpa.metadata import (
    ACTION_HORIZON_N15_INDICES,
    MODALITY_ACTION_KEYS,
    MODALITY_LANGUAGE_KEYS,
    MODALITY_STATE_KEYS,
    MODALITY_VIDEO_KEYS,
    OBSERVATION_DELTA_INDICES,
)


class H2SharpaPnpAppleDataConfig(BaseDataConfig):
    """H2 + Sharpa pnp-apple: 58-DoF policy, 3 fisheye cams, 16-step horizon."""

    video_keys = MODALITY_VIDEO_KEYS
    state_keys = MODALITY_STATE_KEYS
    action_keys = MODALITY_ACTION_KEYS
    language_keys = MODALITY_LANGUAGE_KEYS
    observation_indices = OBSERVATION_DELTA_INDICES
    action_indices = ACTION_HORIZON_N15_INDICES

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self) -> ModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                # VideoCrop(apply_to=self.video_keys, scale=0.95),
                # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: "min_max" for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: "min_max" for k in self.action_keys},
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    max_state_dim=64,
                    # Preserve all 58 action dimensions.
                    max_action_dim=64,
                ),
            ]
        )


# Allows load_data_config("gr00t_config:H2SharpaPnpAppleDataConfig") to resolve.
DATA_CONFIG_MAP["isaaclab_h2_sharpa"] = H2SharpaPnpAppleDataConfig()
