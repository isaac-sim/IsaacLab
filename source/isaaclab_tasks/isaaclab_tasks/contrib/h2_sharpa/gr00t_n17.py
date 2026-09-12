# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.7 modality configuration for the Unitree H2 + Sharpa Wave embodiment.

Importing this module registers ``h2_sharpa_config`` for ``EmbodimentTag.NEW_EMBODIMENT``. GR00T
N1.7 checkpoints carry their own processor and take no per-task data config, so the RLinf extension
imports this module (``env.train.isaaclab.modality_config_module``) before the model is built and
uses :func:`convert_gr00t_to_isaaclab_action` to map policy actions back onto the articulation.

Requires the GR00T N1.7 release; N1.5 environments must not import it.
"""

from __future__ import annotations

import numpy as np
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

from .metadata import (
    ACTION_HORIZON_N17_INDICES,
    H2_ACTION_JOINT_ORDER,
    MODALITY_ACTION_KEYS_BARE,
    MODALITY_LANGUAGE_KEYS,
    MODALITY_STATE_KEYS_BARE,
    MODALITY_VIDEO_KEYS_BARE,
    OBSERVATION_DELTA_INDICES,
    POLICY_58_ORDER,
)

_POLICY_PARTS = ("left_arm", "right_arm", "left_hand", "right_hand")
#: Slot in ``H2_ACTION_JOINT_ORDER`` of every joint the policy predicts, in ``POLICY_58_ORDER``.
#: Scatters both the policy's 58 actions and its 58-D joint state into the full action vector;
#: the RLinf extension reads it to build a "hold the current pose" action.
POLICY_STATE_TO_ACTION_INDICES: list[int] = [H2_ACTION_JOINT_ORDER.index(name) for name in POLICY_58_ORDER]


def convert_gr00t_to_isaaclab_action(action_chunk: dict, chunk_size: int = 1) -> np.ndarray:
    """Map GR00T's 58-D policy order into H2's 75-D action-manager order."""
    ordered_parts = []
    for bare_key in _POLICY_PARTS:
        value = action_chunk.get(bare_key)
        if value is None:
            value = action_chunk.get(f"action.{bare_key}")
        if value is None:
            raise KeyError(f"Missing H2 policy action key {bare_key!r}; got {list(action_chunk)}")
        ordered_parts.append(value[:, :chunk_size, :])

    policy_action = np.concatenate(ordered_parts, axis=-1)
    if policy_action.shape[-1] != len(POLICY_58_ORDER):
        raise ValueError(f"Expected {len(POLICY_58_ORDER)} H2 policy actions, got {policy_action.shape[-1]}")

    full_action = np.zeros(
        (*policy_action.shape[:-1], len(H2_ACTION_JOINT_ORDER)),
        dtype=policy_action.dtype,
    )
    full_action[..., POLICY_STATE_TO_ACTION_INDICES] = policy_action
    return full_action


_ABSOLUTE_JOINT_ACTION = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

h2_sharpa_config: dict[str, ModalityConfig] = {
    "video": ModalityConfig(
        delta_indices=OBSERVATION_DELTA_INDICES,
        modality_keys=list(MODALITY_VIDEO_KEYS_BARE),
    ),
    "state": ModalityConfig(
        delta_indices=OBSERVATION_DELTA_INDICES,
        modality_keys=list(MODALITY_STATE_KEYS_BARE),
    ),
    "action": ModalityConfig(
        delta_indices=list(ACTION_HORIZON_N17_INDICES),
        modality_keys=list(MODALITY_ACTION_KEYS_BARE),
        action_configs=[_ABSOLUTE_JOINT_ACTION] * len(MODALITY_ACTION_KEYS_BARE),
    ),
    "language": ModalityConfig(
        delta_indices=OBSERVATION_DELTA_INDICES,
        modality_keys=list(MODALITY_LANGUAGE_KEYS),
    ),
}

register_modality_config(h2_sharpa_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
