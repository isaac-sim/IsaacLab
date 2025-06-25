# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlRndCfg:
    """Configuration for the Random Network Distillation (RND) module.

    For more information, please check the work from :cite:`schwarke2023curiosity`.
    """

    @configclass
    class WeightScheduleCfg:
        """Configuration for the weight schedule."""

        mode: str = "constant"
        """The type of weight schedule. Default is "constant"."""

    @configclass
    class LinearWeightScheduleCfg(WeightScheduleCfg):
        """Configuration for the linear weight schedule.

        This schedule decays the weight linearly from the initial value to the final value
        between :attr:`initial_step` and before :attr:`final_step`.
        """

        mode: str = "linear"

        final_value: float = MISSING
        """The final value of the weight parameter."""

        initial_step: int = MISSING
        """The initial step of the weight schedule.

        For steps before this step, the weight is the initial value specified in :attr:`RslRlRndCfg.weight`.
        """

        final_step: int = MISSING
        """The final step of the weight schedule.

        For steps after this step, the weight is the final value specified in :attr:`final_value`.
        """

    @configclass
    class StepWeightScheduleCfg(WeightScheduleCfg):
        """Configuration for the step weight schedule.

        This schedule sets the weight to the value specified in :attr:`final_value` at step :attr:`final_step`.
        """

        mode: str = "step"

        final_step: int = MISSING
        """The final step of the weight schedule.

        For steps after this step, the weight is the value specified in :attr:`final_value`.
        """

        final_value: float = MISSING
        """The final value of the weight parameter."""

    weight: float = 0.0
    """The weight for the RND reward (also known as intrinsic reward). Default is 0.0.

    Similar to other reward terms, the RND reward is scaled by this weight.
    """

    weight_schedule: WeightScheduleCfg | None = None
    """The weight schedule for the RND reward. Default is None, which means the weight is constant."""

    reward_normalization: bool = False
    """Whether to normalize the RND reward. Default is False."""

    state_normalization: bool = False
    """Whether to normalize the RND state. Default is False."""

    learning_rate: float = 1e-3
    """The learning rate for the RND module. Default is 1e-3."""

    num_outputs: int = 1
    """The number of outputs for the RND module. Default is 1."""

    predictor_hidden_dims: list[int] = [-1]
    """The hidden dimensions for the RND predictor network. Default is [-1].

    If the list contains -1, then the hidden dimensions are the same as the input dimensions.
    """

    target_hidden_dims: list[int] = [-1]
    """The hidden dimensions for the RND target network. Default is [-1].

    If the list contains -1, then the hidden dimensions are the same as the input dimensions.
    """
