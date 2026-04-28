# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing operations based on warp."""

import warnings

import warp as wp

wp.config.quiet = True
wp.init()

from isaaclab.utils.module import lazy_export

lazy_export()

# Avoid a circular import at module load: `.proxy_array` imports warp, which is
# already loaded above. Importing it here ensures the class is available inside
# the shim.
from .proxy_array import ProxyArray  # noqa: E402

_WP_TO_TORCH_ORIGINAL = wp.to_torch
_WP_TO_TORCH_WARNED = False


def _wp_to_torch_with_proxyarray(a, requires_grad=None):
    """Shim for :func:`warp.to_torch` that gracefully handles :class:`ProxyArray`.

    Without this shim, ``wp.to_torch(proxy)`` would fail with
    ``AttributeError: 'ProxyArray' object has no attribute 'requires_grad'``
    because :class:`ProxyArray` intentionally doesn't replicate the full
    ``wp.array`` attribute surface. Users and third-party code that still use
    ``wp.to_torch(asset.data.<field>)`` from before the ProxyArray migration
    would break hard.

    The shim routes :class:`ProxyArray` arguments to their cached ``.torch``
    view (a zero-copy :class:`torch.Tensor` of the same underlying memory) and
    emits a one-shot :class:`DeprecationWarning`. For any other input type,
    the original :func:`warp.to_torch` handles the call as before.
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
