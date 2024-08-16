# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing different modifiers implementations.

Modifiers are used to apply stateful or stateless modifications to tensor data. They take
in a tensor and a configuration and return a tensor with the modification applied. This way users
can define custom operations to apply to a tensor. For instance, a modifier can be used to normalize
the input data or to apply a rolling average.

They are primarily used to apply custom operations in the :class:`~omni.isaac.lab.managers.ObservationManager`
as an alternative to the built-in noise, clip and scale post-processing operations. For more details, see
the :class:`~omni.isaac.lab.managers.ObservationTermCfg` class.

Usage with a pre-existing modifier configuration:

.. code-block:: python

    import torch
    from omni.isaac.lab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(128, 128, device="cuda")

    # create a modifier configuration
    cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.0, torch.inf)})

    # apply the modifier
    my_modified_tensor = cfg.func(my_tensor, cfg)


Usage with custom modifier configuration:

.. code-block:: python

    import torch
    from omni.isaac.lab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(128, 128, device="cuda")

    # create a modifier configuration
    cfg = modifiers.ModifierCfg(func=torch.nn.functional.relu)

    # apply the modifier
    my_modified_tensor = cfg.func(my_tensor, cfg)

"""

from .modifier import *
from .modifier_cfg import *
