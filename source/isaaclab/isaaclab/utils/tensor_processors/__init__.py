# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful and stateless processing for tensors and dictionaries of named tensors.

Single-tensor functions and :class:`TensorProcessor` subclasses take and return one tensor.
They apply custom operations in the :class:`~isaaclab.managers.ObservationManager`, such as
normalization or rolling averages. Configure them through
:attr:`~isaaclab.managers.ObservationTermCfg.tensor_processors`.

:class:`DictTensorProcessor` subclasses take and return dictionaries of named
:class:`~isaaclab.utils.warp.ProxyArray` tensors. They own persistent outputs and state for camera
pipelines and processed sensor observation terms. :class:`DictTensorProcessorAdapter` applies
these dictionary processors to an incoming observation tensor.

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
