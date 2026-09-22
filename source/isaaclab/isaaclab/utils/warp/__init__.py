# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing operations based on warp."""

import warnings

import warp as wp

from ..module import lazy_export
from .proxy_array import ProxyArray

wp.config.quiet = True
wp.init()

lazy_export()

_WP_TO_TORCH_ORIGINAL = wp.to_torch
_WP_TO_TORCH_WARNED = False


def _wp_to_torch_with_proxyarray(a, requires_grad=None):
    """Return a ProxyArray's cached tensor with a one-time deprecation warning.

    Other inputs retain the original :func:`warp.to_torch` behavior.
    """
    global _WP_TO_TORCH_WARNED
    if isinstance(a, ProxyArray):
        if not _WP_TO_TORCH_WARNED:
            _WP_TO_TORCH_WARNED = True
            warnings.warn(
                "wp.to_torch(<ProxyArray>) is deprecated; use the `.torch` accessor on"
                " the ProxyArray directly (e.g. `asset.data.joint_pos.torch`).",
                DeprecationWarning,
                stacklevel=2,
            )
        return a.torch
    return _WP_TO_TORCH_ORIGINAL(a, requires_grad=requires_grad)


# Patch at both the top-level ``warp`` namespace and the underlying module so
# callers using ``import warp as wp`` and rare ``from warp._src.torch import
# to_torch`` patterns both pick up the shim.
wp.to_torch = _wp_to_torch_with_proxyarray
wp._src.torch.to_torch = _wp_to_torch_with_proxyarray  # noqa: SLF001
