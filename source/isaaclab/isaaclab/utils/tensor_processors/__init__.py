# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful and stateless processing for tensors and nested tensor collections.

Functions and :class:`TensorProcessor` subclasses share :class:`TensorProcessorCfg`.
``TensorProcessorData`` describes :class:`torch.Tensor`, :class:`~isaaclab.utils.warp.ProxyArray`,
and recursively nested lists, tuples, and dictionaries of those values. Each implementation
defines the types and structures it accepts; the base class does not traverse collections.

Inputs belong to the caller and are read-only. Structured processors can inspect input metadata
and allocate outputs during construction, without reading input values or advancing state.
Subclasses own any persistent output buffers they allocate; their contents remain valid until
the next call, so consumers must copy results they need to retain. Warp operations use the
caller's current Warp stream. Override ``reset`` or ``close`` when a subclass owns state or
external resources; both methods do nothing by default.

The :class:`~isaaclab.managers.ObservationManager` uses shape-preserving single-tensor processors,
configured through :attr:`~isaaclab.managers.ObservationTermCfg.tensor_processors`. Built-in
filters also operate on single tensors. Construct structured processors directly using their
implementation's constructor. Configuration ``inputs`` and ``outputs`` fields describe named
keys only; they do not select values or route data between processors.

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
