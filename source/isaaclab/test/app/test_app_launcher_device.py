# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mapping CUDA device indices to the physical indices the renderer expects."""

from isaaclab.app.app_launcher import AppLauncher


def test_physical_device_id_identity_mask_is_noop(monkeypatch):
    """Return the index unchanged when the mask already starts at zero."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    assert AppLauncher._physical_device_id(2) == 2


def test_physical_device_id_maps_through_offset_mask(monkeypatch):
    """Resolve against the mask so ``cuda:1`` becomes the second visible device."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    assert AppLauncher._physical_device_id(1) == 5


def test_physical_device_id_ignores_mask_order(monkeypatch):
    """Follow the order given in the mask rather than sorting it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4")

    assert AppLauncher._physical_device_id(0) == 7


def test_physical_device_id_without_mask(monkeypatch):
    """Return the index unchanged when no mask is set."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert AppLauncher._physical_device_id(3) == 3


def test_physical_device_id_falls_back_for_uuid_mask(monkeypatch):
    """Fall back to the CUDA index when the mask names devices by UUID.

    ``CUDA_VISIBLE_DEVICES`` also accepts ``GPU-<uuid>`` and MIG identifiers, for which no physical
    index can be derived. Those runs keep the previous behavior instead of failing.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0d1e2f3a,GPU-4b5c6d7e")

    assert AppLauncher._physical_device_id(1) == 1


def test_physical_device_id_falls_back_when_index_exceeds_mask(monkeypatch):
    """Fall back when the index is outside the mask rather than raising."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert AppLauncher._physical_device_id(5) == 5
