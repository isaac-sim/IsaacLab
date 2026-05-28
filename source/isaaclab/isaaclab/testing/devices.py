# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-selection helper for parametrizing tests over visible devices.

Intended use:

    from isaaclab.testing import cuda_test_devices

    @pytest.mark.parametrize("device", cuda_test_devices())
    def test_foo(device): ...

The helper resolves a mask (from the ``ISAACLAB_TEST_DEVICES`` env var, or an
explicit ``mask=`` argument) into a list of device strings the host can
satisfy. Single-GPU CI runs yield ``[cpu, cuda:0]``; multi-GPU CI (with
``ISAACLAB_TEST_DEVICES=001``) yields ``[cuda:1]``. Same test, different
runtime parametrization.

Mask grammar
------------
Per position, one character: ``0`` (exclude) or ``1`` (include). An optional
trailing ``X`` expands to "all remaining positions are 1". Anything past the
mask end (with no ``X``) is treated as ``0``.

Position 0 maps to ``cpu``; position ``k`` (k >= 1) maps to ``cuda:{k-1}``.

The common case is 3 positions (cpu, cuda:0, cuda:1), but the grammar accepts
any length so larger multi-GPU pools work too.

Examples (host with 4 visible devices: ``cpu, cuda:0, cuda:1, cuda:2``):

==========  ====================================================
Mask        Resolves to
==========  ====================================================
``110``     ``[cpu, cuda:0]``                  single-GPU CI default
``001``     ``[cuda:1]``                       multi-GPU CI minimum
``00X``     ``[cuda:1, cuda:2]``               multi-GPU CI exhaustive
``X``       ``[cpu, cuda:0, cuda:1, cuda:2]``  everything visible
==========  ====================================================

Strict vs non-strict
--------------------
``strict=True`` (the default) raises ``ValueError`` when the mask asks for
devices the host doesn't have, or when the resolved list is empty. This is
the right mode for env-driven calls in CI: misconfigured runners surface
immediately rather than silently zero-ing test coverage.

``strict=False`` truncates silently. The empty-list path lets pytest's
``parametrize`` mechanism skip a test cleanly when a multi-GPU-only test
runs on a single-GPU host, replacing the older
``@pytest.mark.skipif(not os.environ.get("ISAACLAB_TEST_MULTI_GPU"))`` plus
hardcoded ``parametrize("device", ["cuda:1"])`` pattern.

Local runs
----------
The env var is settable from the shell::

    ISAACLAB_TEST_DEVICES=001 ./isaaclab.sh -p -m pytest path/to/test.py

This makes the local invocation produce the same device set as the
multi-GPU CI runner would.
"""

from __future__ import annotations

import os

import torch


_DEFAULT_MASK = "110"
"""Default mask when ``ISAACLAB_TEST_DEVICES`` is unset.

``110`` means cpu + cuda:0, matching the historical single-GPU CI device list
so adopting this helper has zero impact on single-GPU runs.
"""

_ENV_VAR = "ISAACLAB_TEST_DEVICES"
"""Name of the environment variable that overrides the default mask."""


def cuda_test_devices(*, mask: str | None = None, strict: bool = True) -> list[str]:
    """Resolve a device-selection mask to a list of device strings.

    Args:
        mask: Optional explicit mask string following the grammar in the
            module docstring. When ``None``, the helper reads the mask from
            the ``ISAACLAB_TEST_DEVICES`` environment variable, defaulting
            to ``"110"`` if the variable is unset.
        strict: When ``True`` (the default), raise ``ValueError`` if the
            mask requests devices the host does not have or if the resolved
            list would be empty. When ``False``, silently truncate to what
            the host can satisfy - callers using this in
            ``pytest.mark.parametrize`` get a clean "no tests collected for
            this function" skip when the resolved list is empty.

    Returns:
        Ordered list of device strings (``"cpu"`` and/or ``"cuda:N"``)
        produced by applying the mask to the host's visible device list.

    Raises:
        ValueError: When the mask is syntactically invalid (``X`` not at the
            end, or any character outside ``{0, 1, X}``); or when ``strict``
            is true and the mask requests an unavailable device or resolves
            to an empty list.
    """
    if mask is None:
        mask = os.environ.get(_ENV_VAR, _DEFAULT_MASK)
    available = _list_available_devices()
    flags = _expand_mask(mask, len(available), strict=strict)
    devices = [device for device, keep in zip(available, flags) if keep]
    if strict and not devices:
        raise ValueError(
            f"Mask {mask!r} resolves to empty device list (available: {available})"
        )
    return devices


def _list_available_devices() -> list[str]:
    """Return the host's visible device list in the order the mask grammar uses.

    The order is ``[cpu]`` followed by ``cuda:0, cuda:1, ...`` for each CUDA
    device torch reports as visible. ``cpu`` is always included even when
    torch has no CUDA backend; downstream callers can filter via the mask.

    Returns:
        Ordered list of device strings as torch would address them.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    return devices


def _expand_mask(mask: str, length: int, *, strict: bool) -> list[bool]:
    """Expand a mask string into a list of exactly ``length`` include flags.

    Three concrete steps:

      1. Peel off an optional trailing ``X``. ``wildcard`` becomes the fill
         value for positions past the body.
      2. Validate the body characters (must be ``0`` or ``1``).
      3. Reconcile body length against ``length``: truncate if longer, pad
         with ``wildcard`` if shorter. In strict mode, raise instead of
         truncating when the surplus contains a ``1``.

    Args:
        mask: Raw mask string. May be empty.
        length: Number of include flags to produce. When called from
            :func:`cuda_test_devices` this equals the host's visible-device
            count.
        strict: When ``True``, raise instead of silently truncating when
            the mask body requests positions past ``length``.

    Returns:
        List of booleans of length ``length`` where each entry indicates
        whether the corresponding device position should be included.

    Raises:
        ValueError: When the mask contains ``X`` anywhere other than as a
            trailing character; when the mask contains a character outside
            ``{0, 1, X}``; or when ``strict`` and the mask body requests
            positions past ``length`` via a surplus ``1``.
    """
    if mask.endswith("X"):
        body, wildcard = mask[:-1], True
    else:
        body, wildcard = mask, False

    if "X" in body:
        raise ValueError(f"Invalid mask {mask!r}: 'X' must be the last character")
    for c in body:
        if c not in "01":
            raise ValueError(f"Invalid mask {mask!r}: char {c!r} not in {{0, 1, X}}")

    body_flags = [c == "1" for c in body]

    if len(body_flags) > length:
        if strict:
            surplus = body_flags[length:]
            if any(surplus):
                pos = length + surplus.index(True)
                raise ValueError(
                    f"Mask {mask!r} requires a device at position {pos} but the "
                    f"host only has {length} devices"
                )
        return body_flags[:length]

    body_flags.extend([wildcard] * (length - len(body_flags)))
    return body_flags
