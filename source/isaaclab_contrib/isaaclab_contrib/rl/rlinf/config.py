# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf recipe validation that does not import RLinf or load model weights."""

from collections.abc import Mapping


def validate_image_config(cfg: Mapping) -> None:
    """Reject invalid image sizes on the driver before worker extension errors can be caught by RLinf."""
    env = cfg.get("env", {})
    training = env.get("train", {}).get("isaaclab", {})
    evaluation = {**training, **env.get("eval", {}).get("isaaclab", {})}
    keys = ("gpu_resize_width", "gpu_resize_height")
    for split, options in (("train", training), ("eval", evaluation)):
        if any(key in options for key in ("gpu_crop_scale", "gpu_crop_random")):
            raise ValueError(
                f"env.{split}.isaaclab: configure crop augmentation in task observation tensor_processors, "
                "not gpu_crop_scale/gpu_crop_random."
            )
        for key in keys:
            value = options.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"env.{split}.isaaclab.{key} must be a positive integer, got {value!r}.")
    model_type = cfg.get("actor", {}).get("model", {}).get("model_type", "gr00t")
    if model_type == "gr00t_n1d7" or not training.get("data_config_class"):
        return
    for split, options in (("train", training), ("eval", evaluation)):
        if (options.get(keys[0]) is None) != (options.get(keys[1]) is None):
            raise ValueError(f"GR00T N1.5 requires both gpu_resize_width and gpu_resize_height in {split}.")
    training_size = tuple(training.get(key) for key in keys)
    evaluation_size = tuple(evaluation.get(key) for key in keys)
    if training_size != evaluation_size:
        raise ValueError(
            "GR00T N1.5 requires matching train/eval image sizes after GPU resizing "
            f"(width, height): train={training_size}, eval={evaluation_size}."
        )
