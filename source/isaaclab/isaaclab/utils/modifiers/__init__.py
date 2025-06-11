# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing different modifiers implementations.

Modifiers are used to apply stateful or stateless modifications to tensor data. They take
in a tensor and a configuration and return a tensor with the modification applied. This way users
can define custom operations to apply to a tensor. For instance, a modifier can be used to normalize
the input data or to apply a rolling average.

They are primarily used to apply custom operations in the :class:`~isaaclab.managers.ObservationManager`
as an alternative to the built-in noise, clip and scale post-processing operations. For more details, see
the :class:`~isaaclab.managers.ObservationTermCfg` class.

Usage with a function modifier:

.. code-block:: python

    import torch
    from isaaclab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a modifier configuration
    cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.0, torch.inf)})

    # apply the modifier
    my_modified_tensor = cfg.func(my_tensor, cfg)


Usage with a class modifier:

.. code-block:: python

    import torch
    from isaaclab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a modifier configuration
    # a digital filter with a simple delay of 1 timestep
    cfg = modifiers.DigitalFilterCfg(A=[0.0], B=[0.0, 1.0])

    # create the modifier instance
    my_modifier = modifiers.DigitalFilter(cfg, my_tensor.shape, "cuda")

    # apply the modifier as a callable object
    my_modified_tensor = my_modifier(my_tensor)

"""

# isort: off
from .modifier_cfg import ModifierCfg
from .modifier_base import ModifierBase
from .modifier import DigitalFilter
from .modifier_cfg import DigitalFilterCfg
from .modifier import Integrator
from .modifier_cfg import IntegratorCfg

# isort: on
from .modifier import bias, clip, scale
