# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing different noise models implementations.

The noise models are implemented as functions that take in a tensor and a configuration and return a tensor
with the noise applied. These functions are then used in the :class:`NoiseCfg` configuration class.

Usage:

.. code-block:: python

    import torch
    from omni.isaac.orbit.utils.noise import AdditiveGaussianNoiseCfg

    # create a random tensor
    my_tensor = torch.rand(128, 128, device="cuda")

    # create a noise configuration
    cfg = AdditiveGaussianNoiseCfg(mean=0.0, std=1.0)

    # apply the noise
    my_noisified_tensor = cfg.func(my_tensor, cfg)

"""
from .noise_cfg import NoiseCfg  # noqa: F401
from .noise_cfg import AdditiveGaussianNoiseCfg, AdditiveUniformNoiseCfg, ConstantBiasNoiseCfg
