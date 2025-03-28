
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlSymmetryCfg:
    """Configuration for the symmetry-augmentation in the training.

    For more information, please check the work from :cite:`mittal2024symmetry`.
    """

    use_data_augmentation: bool = False
    """Whether to use symmetry-based data augmentation. Default is False."""

    use_mirror_loss: bool = False
    """Whether to use the symmetry-augmentation loss. Default is False."""

    data_augmentation_func: callable = MISSING
    """The symmetry data augmentation function.

    The function signature should be as follows:

    Args:

        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (torch.Tensor | None): The observation tensor. If None, the observation is not used.
        action (torch.Tensor | None): The action tensor. If None, the action is not used.
        is_critic (bool): Whether the observation is for the critic network. Default is False.

    Returns:
        A tuple containing the augmented observation and action tensors. The tensors can be None,
        if their respective inputs are None.
    """

    mirror_loss_coeff: float = 0.0
    """The weight for the symmetry-mirror loss. Default is 0.0."""
