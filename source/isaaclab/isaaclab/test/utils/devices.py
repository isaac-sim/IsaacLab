# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device selection for parametrizing tests over cpu / cuda devices.

Intended use::

    from isaaclab.test.utils import DeviceScope, test_devices

    @pytest.mark.parametrize("device", test_devices())        # cpu + cuda:0 + a non-default GPU
    @pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
    def test_foo(device): ...

Two inputs, two roles
---------------------
A test runs on ``scope ∩ runtime``:

* **scope** — a :class:`DeviceScope` or mask declaring the devices the *test*
  is valid on. The author owns this; defaults to :attr:`DeviceScope.ALL`.
* **runtime** — the ``ISAACLAB_TEST_DEVICES`` env var: the devices the *run*
  may use. The operator / CI owns this; defaults to ``"110"`` (cpu + cuda:0).
  The multi-GPU workflow sets it to one device per shard, matching
  ``ISAACLAB_TEST_SIM_DEVICE`` (the explicit AppLauncher boot device — not read here).

A test author never names the shard's GPU; the operator never inspects a
test's device support. The helper intersects the two.

Mask grammar
------------
Positions left to right: ``0`` = cpu, ``1`` = cuda:0 (the default GPU),
``2`` = cuda:1, ``3`` = cuda:2, ... Each position is ``0`` (exclude) or ``1``
(include). A trailing ``X`` means "include all the remaining devices".

cpu, cuda:0, and a non-default GPU stay distinct by position — running on
cuda:0 says nothing about cuda:1+, which is the whole reason this exists.

Common scopes
-------------
====================================  =========  =============================================
Named scope                           Mask       Meaning
====================================  =========  =============================================
``DeviceScope.ALL``                   ``11X``    cpu + every GPU
``DeviceScope.CUDA``                  ``01X``    every GPU
``DeviceScope.CPU_AND_DEFAULT_CUDA``  ``110``    cpu + cuda:0
``DeviceScope.NON_DEFAULT_CUDA``      ``00X``    cuda:1 and above
``DeviceScope.CPU``                   ``100``    cpu only
====================================  =========  =============================================

Named scopes cover the common cases; string masks remain supported for custom
device combinations.

Worked example — a ``scope="11X"`` (default) test:

* single-GPU CI (runtime unset ⇒ ``"110"``) ⇒ ``[cpu, cuda:0]``.
* a multi-GPU shard (runtime ``"0001"``, one device) ⇒ ``[cuda:2]``.

So argless tests run on cpu + cuda:0 in single-GPU CI and on exactly one
non-default GPU per shard in multi-GPU CI (run-once is a property of the
one-device-per-shard runtime plus round-robin file assignment).

An empty result means the test is cleanly skipped for this run (e.g. a
``"00X"`` test on a single-GPU host, or a ``"110"`` test on a non-default-GPU
shard). But if the run *explicitly* set ``ISAACLAB_TEST_DEVICES`` to devices
the host does not have, that is a misconfigured run and raises instead of
skipping everything and reporting a vacuous green.

Local runs
----------
Set the runtime from the shell to opt a run into non-default GPUs::

    ISAACLAB_TEST_DEVICES=0001 ISAACLAB_TEST_SIM_DEVICE=cuda:2 \\
        ./isaaclab.sh -p -m pytest path/to/test.py
"""

from __future__ import annotations

import os
from enum import StrEnum

_RUNTIME_DEVICES_ENV_VAR = "ISAACLAB_TEST_DEVICES"
"""Env var naming the run's devices: the devices a run may use (see module docstring)."""

_DEFAULT_RUNTIME = "110"
"""Runtime devices when :data:`_RUNTIME_DEVICES_ENV_VAR` is unset: cpu + cuda:0,
i.e. the historical single-GPU device set, so non-default GPUs are opt-in per run."""


class DeviceScope(StrEnum):
    """Named device scopes for common test parametrization cases.

    String masks remain accepted by :func:`test_devices` for custom device
    combinations that are not represented here.
    """

    ALL = "11X"
    CUDA = "01X"
    CPU_AND_DEFAULT_CUDA = "110"
    NON_DEFAULT_CUDA = "00X"
    CPU = "100"


def test_devices(scope: str | DeviceScope = DeviceScope.ALL, *, skip: dict[str, str] | None = None) -> list:
    """Resolve the device list to parametrize a test over, as ``scope ∩ runtime``.

    ``scope`` is this argument; the runtime comes from the
    ``ISAACLAB_TEST_DEVICES`` env var (see the module docstring for the grammar
    and the scope / runtime split).

    Args:
        scope: Named scope or device mask (e.g. ``"11X"``) the test is valid
            on. Defaults to :attr:`DeviceScope.ALL`, so device-agnostic call
            sites can use ``test_devices()`` with no argument.
        skip: Optional ``{device: reason}``; in-scope devices listed here are
            wrapped in :func:`pytest.mark.skip` so they collect as SKIPPED with
            the reason instead of being dropped. Use to gate a device variant
            that is known broken while keeping it visible.

    Returns:
        Ordered device entries (``"cpu"`` / ``"cuda:N"`` strings, or
        :func:`pytest.param` for skipped ones) for the second argument to
        :func:`pytest.mark.parametrize`. Empty means the test is skipped for
        this run.

    Raises:
        ValueError: When the run explicitly set ``ISAACLAB_TEST_DEVICES`` to
            devices that are not available on this host (a misconfigured run that
            would otherwise skip everything and pass vacuously). A scope that
            merely does not intersect this run's devices skips, it does not raise.
    """
    requested = os.environ.get(_RUNTIME_DEVICES_ENV_VAR)
    available = _list_available_devices()
    runtime_keep = _expand(requested or _DEFAULT_RUNTIME, len(available))
    # A run that explicitly named devices the host cannot provide (e.g. a shard
    # pinned to cuda:2 on a 2-GPU host) is misconfigured: fail loudly rather than
    # skip every test and report a vacuous green. A scope that simply does not
    # intersect this run's devices (a "110" test on a non-default-GPU shard) is
    # not an error — it skips.
    if requested is not None and not any(runtime_keep):
        raise ValueError(
            f"{_RUNTIME_DEVICES_ENV_VAR}={requested!r} names no device available on this host (available: {available})"
        )
    scope_keep = _expand(scope, len(available))
    devices = [
        device for device, in_scope, in_runtime in zip(available, scope_keep, runtime_keep) if in_scope and in_runtime
    ]
    if not skip:
        return devices
    import pytest

    return [
        pytest.param(device, marks=pytest.mark.skip(reason=skip[device])) if device in skip else device
        for device in devices
    ]


# Prevent pytest from collecting this imported ``test_``-prefixed helper.
test_devices.__test__ = False


def _expand(mask: str, count: int) -> list[bool]:
    """Expand a mask to ``count`` include-flags; a trailing ``X`` fills the rest with ``True``.

    Args:
        mask: A mask string (positions plus an optional trailing ``X``).
        count: The number of available devices to expand to.

    Returns:
        A list of ``count`` booleans, one per device position.
    """
    body, fill = (mask[:-1], True) if mask.endswith("X") else (mask, False)
    return ([char == "1" for char in body] + [fill] * count)[:count]


def _list_available_devices() -> list[str]:
    """Return the host's visible devices in mask order: ``cpu`` then ``cuda:0, cuda:1, ...``.

    Returns:
        Ordered list of device strings as torch addresses them.
    """
    # Keep the import local so importing this test helper before AppLauncher
    # does not import torch (and its transitive native libraries) before Kit.
    import torch

    devices = ["cpu"]
    # torch.cuda (not Warp) is deliberate: warp.get_cuda_devices() initializes
    # the Warp runtime, but non-default-GPU runs must select cuda:N before
    # wp.init(). Enumeration here must not perturb the initialization order that
    # this suite validates (#5132).
    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    return devices
