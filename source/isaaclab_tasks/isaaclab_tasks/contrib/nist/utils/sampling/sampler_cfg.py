# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`Sampler`."""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils.configclass import configclass

from .sampler import Sampler


@configclass
class SamplerCfg:
    """Blueprint for a :class:`Sampler`.

    Attributes:
        class_type: Runtime sampler class to instantiate.
        strategies: Weighted sampling-strategy cfgs composed by the sampler.
        eps: Soft floor on per-item probability.
        warp: Whether to use the preallocated Warp backend for score/probability/sample kernels.
            In Warp mode, :meth:`Sampler.probabilities_and_sample` is CUDA-graph captured.
        seed: Base seed for Warp categorical sampling.
        max_samples: Preallocated sample-output capacity for Warp mode. Set to the
            maximum number of samples passed to :meth:`Sampler.sample` in one call;
            Warp mode raises if a call exceeds this capacity.
    """

    class_type: type[Sampler] | str = "{DIR}.sampler:Sampler"
    """Runtime sampler class."""
    strategies: list = field(default_factory=list)
    """Weighted sampling-strategy cfgs composed by the sampler."""
    eps: float = 1e-3
    """Soft floor on per-item probability before normalization."""
    warp: bool = False
    """Whether to use the Warp backend."""
    seed: int = 17
    """Base seed for Warp categorical sampling."""
    max_samples: int | None = None
    """Maximum number of indices sampled in one Warp call."""
