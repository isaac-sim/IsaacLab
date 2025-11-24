# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch.nn as nn  # noqa: F401
from typing import Any

from stable_baselines3.common.utils import constant_fn


def process_sb3_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components."""

    def update_dict(hyperparams: dict[str, Any]) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in ["policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"]:
                    hyperparams[key] = eval(value)
                elif key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    return update_dict(cfg)
