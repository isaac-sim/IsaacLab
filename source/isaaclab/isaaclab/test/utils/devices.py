# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device selection for parametrizing tests over cpu / cuda devices.

Intended use::

    from isaaclab.test.utils import test_devices

    @pytest.mark.parametrize("device", test_devices("11X"))
    def test_foo(device): ...

Two masks, two roles
--------------------
The set a test actually runs on is ``scope ∩ runtime_devices``:

* **scope** — passed at the call site, fixed. The devices the *test* is valid
  on. The author owns this.
* **runtime_devices** — the ``ISAACLAB_TEST_DEVICES`` env var, per run. The
  devices the *run* is allowed to use. The operator / CI owns this; defaults
  to ``"110"``.

A test author never has to know which device a shard holds; the operator never
has to know which devices a test supports. The helper intersects the two.

Mask grammar
------------
Positions left to right: ``0`` = cpu, ``1`` = cuda:0 (the default GPU),
``2`` = cuda:1, ``3`` = cuda:2, ... Each position is ``0`` (exclude) or ``1``
(include). An optional trailing ``X`` means "any **one** of the remaining
non-default GPUs" and resolves to a single device — the one named by
``ISAACLAB_SIM_DEVICE`` if it is a non-default GPU, else the lowest-index
available non-default GPU. (``X`` is *any one*, not *all*; exhaustive
"every non-default GPU" coverage is intentionally not supported yet.)

cpu, cuda:0, and a non-default GPU stay distinct by position — running on
cuda:0 says nothing about cuda:1+, which is the whole reason this exists.

Common masks
------------
======  ===================================================================
Mask    Meaning
======  ===================================================================
``110`` cpu + cuda:0 (the default scope and the default runtime devices)
``11X`` cpu + cuda:0 + any one non-default GPU (device-agnostic test)
``00X`` a non-default GPU only (validates non-default-device behavior)
``100`` cpu only (pure logic)
``001`` cuda:1 specifically (only when a device must be pinned, rare)
======  ===================================================================

Worked example — a ``scope="11X"`` test:

* single-GPU CI (runtime devices unset ⇒ ``"110"``) ⇒ ``[cpu, cuda:0]``.
* a multi-GPU shard (runtime devices ``"00X"``, ``ISAACLAB_SIM_DEVICE=cuda:2``)
  ⇒ ``[cuda:2]``.

An empty result means the test is cleanly skipped for this run (e.g. a
``"00X"`` test on a single-GPU host).

Local runs
----------
Set the runtime devices from the shell to opt a run into non-default GPUs::

    ISAACLAB_TEST_DEVICES=11X ISAACLAB_SIM_DEVICE=cuda:1 \\
        ./isaaclab.sh -p -m pytest path/to/test.py
"""

from __future__ import annotations

import os

import torch

_RUNTIME_DEVICES_ENV_VAR = "ISAACLAB_TEST_DEVICES"
"""Env var naming the run's devices: the devices a run may use (see module docstring)."""

_DEFAULT_RUNTIME_DEVICES = "110"
"""Runtime devices when :data:`_RUNTIME_DEVICES_ENV_VAR` is unset: cpu + cuda:0,
i.e. the historical single-GPU device set, so non-default GPUs are opt-in per run."""


def test_devices(scope: str | list[str] = "110", *, require_available: bool = False) -> list[str]:
    """Resolve the device list to parametrize a test over.

    The result is ``scope ∩ runtime_devices``, where ``scope`` is this argument
    and the runtime devices come from the ``ISAACLAB_TEST_DEVICES`` env var (see
    the module docstring for the grammar and the scope / runtime-devices split).

    Args:
        scope: Device mask (e.g. ``"11X"``) or an explicit device list (e.g.
            ``["cpu", "cuda:0"]``) the test is valid on. A list is treated as
            those exact devices, with no ``X`` wildcard.
        require_available: When ``True``, raise ``ValueError`` if the resolved
            list is empty (the run cannot satisfy the test on any in-scope
            device) instead of letting pytest skip it. Use for CI calls that
            must hard-fail when a runner has fewer GPUs than expected.

    Returns:
        Ordered list of device strings (``"cpu"`` and/or ``"cuda:N"``) suitable
        as the second argument to :func:`pytest.mark.parametrize`. Empty means
        the test is skipped for this run.

    Raises:
        ValueError: When a mask is syntactically invalid (``X`` not trailing,
            or a character outside ``{0, 1, X}``), or when ``require_available``
            is set and the resolved list is empty.
    """
    available = _list_available_devices()
    target = _target_nondefault(available)
    scope_set = _select(scope, available, target)
    runtime_set = _select(os.environ.get(_RUNTIME_DEVICES_ENV_VAR, _DEFAULT_RUNTIME_DEVICES), available, target)
    devices = [d for d in available if d in scope_set and d in runtime_set]
    if require_available and not devices:
        raise ValueError(f"scope {scope!r} ∩ runtime devices resolves to no device (available: {available})")
    return devices


def _list_available_devices() -> list[str]:
    """Return the host's visible devices in mask order: ``cpu`` then ``cuda:0, cuda:1, ...``.

    Returns:
        Ordered list of device strings as torch addresses them.
    """
    devices = ["cpu"]
    # torch.cuda (not warp) is deliberate: this runs at pytest collection time,
    # before AppLauncher boots Kit. torch.cuda.device_count() enumerates without
    # creating a CUDA context, whereas warp.get_cuda_devices() initializes the
    # warp runtime — doing that at collection, ahead of Kit's device setup,
    # risks the non-default-GPU init-order fragility this suite targets (#5132).
    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    return devices


def _target_nondefault(available: list[str]) -> str | None:
    """Pick the single non-default GPU that a trailing ``X`` resolves to.

    Prefers the device named by ``ISAACLAB_SIM_DEVICE`` when it is a visible
    non-default GPU (so a shard's ``X`` lands on the device Kit booted on),
    otherwise the lowest-index available non-default GPU (deterministic).

    Returns:
        A ``"cuda:N"`` (N >= 1) string, or ``None`` when the host has no
        non-default GPU.
    """
    nondefault = [d for d in available if d.startswith("cuda:") and d != "cuda:0"]
    if not nondefault:
        return None
    env_device = os.environ.get("ISAACLAB_SIM_DEVICE", "")
    return env_device if env_device in nondefault else nondefault[0]


def _select(mask: str | list[str], available: list[str], target: str | None) -> set[str]:
    """Resolve one mask (or explicit list) to the set of devices it selects.

    Args:
        mask: A mask string (positions plus optional trailing ``X``) or an
            explicit device list.
        available: The host's visible devices, in mask order.
        target: The single non-default GPU a trailing ``X`` resolves to (from
            :func:`_target_nondefault`), or ``None`` if the host has none.

    Returns:
        The subset of ``available`` the mask selects.

    Raises:
        ValueError: When ``X`` is not the trailing character, or a character is
            outside ``{0, 1, X}``.
    """
    if isinstance(mask, (list, tuple)):
        return {d for d in mask if d in available}

    body, wildcard = (mask[:-1], True) if mask.endswith("X") else (mask, False)
    if "X" in body:
        raise ValueError(f"Invalid mask {mask!r}: 'X' must be the trailing character")
    for c in body:
        if c not in "01":
            raise ValueError(f"Invalid mask {mask!r}: char {c!r} not in {{0, 1, X}}")

    selected = {available[i] for i, c in enumerate(body) if c == "1" and i < len(available)}
    if wildcard and target is not None:
        selected.add(target)  # "any one" non-default GPU for this run
    return selected
