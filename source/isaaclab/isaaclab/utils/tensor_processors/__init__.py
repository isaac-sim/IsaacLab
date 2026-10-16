# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing tensor processor implementations.

Tensor processors are used to apply stateful or stateless modifications to tensor data. They take
in a tensor and a configuration and return a tensor with the modification applied. This way users
can define custom operations to apply to a tensor. For instance, a processor can be used to normalize
the input data or to apply a rolling average.

They are primarily used to apply custom operations in the :class:`~isaaclab.managers.ObservationManager`
as an alternative to the built-in noise, clip and scale post-processing operations. For more details, see
the :class:`~isaaclab.managers.ObservationTermCfg` class.

Usage with a function processor:

.. code-block:: python

    import torch
    from isaaclab.utils import tensor_processors

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a processor configuration
    cfg = tensor_processors.TensorProcessorCfg(func=tensor_processors.clip, params={"bounds": (0.0, torch.inf)})

    # apply the processor
    my_modified_tensor = cfg.func(my_tensor, **cfg.params)


Usage with a class processor:

.. code-block:: python

    import torch
    from isaaclab.utils import tensor_processors

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a processor configuration
    # a digital filter with a simple delay of 1 timestep
    cfg = tensor_processors.DigitalFilterCfg(A=[0.0], B=[0.0, 1.0])

    # create the processor instance
    my_processor = tensor_processors.DigitalFilter(cfg, my_tensor.shape, "cuda")

    # apply the processor as a callable object
    my_modified_tensor = my_processor(my_tensor)

"""

from ..module import lazy_export

lazy_export()
