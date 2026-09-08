# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`Sampler`."""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.utils.sampling.sampler import Sampler


@configclass
class SamplerCfg:
    """Blueprint for a :class:`Sampler`.

    Attributes:
        class_type: Runtime sampler class to instantiate.
        strategies: Weighted sampling-strategy cfgs composed by the sampler.
        eps: Soft floor on per-item probability.
    """

    class_type: type[Sampler] | str = "{DIR}.sampler:Sampler"
    """Runtime sampler class."""
    strategies: list = field(default_factory=list)
    """Weighted sampling-strategy cfgs composed by the sampler."""
    eps: float = 1e-3
    """Soft floor on per-item probability before normalization."""
