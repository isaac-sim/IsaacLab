# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python ctypes binding for ``omni::fabric::IFabricUsd::setEnableChangeNotifies``.

Acquires the ``omni::fabric::IFabricUsd`` carb interface directly from the Carbonite
framework so cloning can suspend Fabric's USD notice listener without depending on
``isaacsim.core.simulation_manager``.

Mirrors the in-tree pattern in :mod:`isaaclab_newton.physics._cubric` for
``omni::cubric::IAdapter`` — same problem (base-Kit Carbonite interface with no
Python binding), same solution. When Kit exposes this from Python, replace this
module with a one-line import.
"""

from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)

# Offsets are valid on Linux 64-bit (x86_64 and aarch64). Function pointers are
# 8 bytes on both, no padding between same-size fields, so the vtable layout is
# architecture-independent. Windows/macOS would need a different shared-library
# name and are out of scope here (same constraint as
# :mod:`isaaclab_newton.physics._cubric`).

# carb::Framework vtable (carb/Framework.h)
#   0: loadPluginsEx, 8: unloadAllPlugins, 16: acquireInterfaceWithClient,
#  24: tryAcquireInterfaceWithClient  ← used here
_FW_OFF_TRY_ACQUIRE = 24

# omni::fabric::IFabricUsd vtable (omni/fabric/usd/interface/IFabricUsd.h)
#  0..88: prefetch / export / type-conversion entry points
#  96: setEnableChangeNotifies(FabricId, bool)
# 104: getEnableChangeNotifies(FabricId) -> bool
_IFU_OFF_SET_ENABLE = 96
_IFU_OFF_GET_ENABLE = 104


class _Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _InterfaceDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("version", _Version)]


def _read_u64(addr: int) -> int:
    return ctypes.c_uint64.from_address(addr).value


class FabricNoticeBindings:
    """Typed wrappers around ``omni::fabric::IFabricUsd``'s notice toggle."""

    def __init__(self) -> None:
        self._iface_ptr: int = 0
        self._set_fn = None
        self._get_fn = None

    def initialize(self) -> bool:
        """Acquire the ``IFabricUsd`` interface. Returns False if unavailable."""
        try:
            libcarb = ctypes.CDLL("libcarb.so")
        except OSError:
            return False

        libcarb.acquireFramework.restype = ctypes.c_void_p
        libcarb.acquireFramework.argtypes = [ctypes.c_char_p, _Version]
        fw_ptr = libcarb.acquireFramework(b"isaaclab.cloner", _Version(0, 0))
        if not fw_ptr:
            return False

        try_acquire_addr = _read_u64(fw_ptr + _FW_OFF_TRY_ACQUIRE)
        if not try_acquire_addr:
            return False

        try_acquire = ctypes.CFUNCTYPE(
            ctypes.c_void_p,  # IFabricUsd*
            ctypes.c_char_p,  # clientName
            _InterfaceDesc,  # desc (by value)
            ctypes.c_char_p,  # pluginName
        )(try_acquire_addr)

        desc = _InterfaceDesc(name=b"omni::fabric::IFabricUsd", version=_Version(1, 0))

        # clientName varies across Kit configurations — same fallback chain as _cubric.py
        ptr = try_acquire(b"carb.scripting-python.plugin", desc, None) or try_acquire(None, desc, None)
        if not ptr:
            return False
        self._iface_ptr = ptr

        set_addr = _read_u64(ptr + _IFU_OFF_SET_ENABLE)
        get_addr = _read_u64(ptr + _IFU_OFF_GET_ENABLE)
        if not (set_addr and get_addr):
            return False

        # FabricId is uint64; CARB_ABI uses the platform's standard C calling convention
        self._set_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_bool)(set_addr)
        self._get_fn = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint64)(get_addr)
        return True

    @property
    def available(self) -> bool:
        return self._iface_ptr != 0

    def set_enable(self, fabric_id: int, enable: bool) -> None:
        if self._set_fn is not None:
            self._set_fn(ctypes.c_uint64(fabric_id), ctypes.c_bool(enable))

    def is_enabled(self, fabric_id: int) -> bool:
        if self._get_fn is None:
            return False
        return bool(self._get_fn(ctypes.c_uint64(fabric_id)))


_BINDINGS: FabricNoticeBindings | None = None
_INIT_TRIED: bool = False


def get_bindings() -> FabricNoticeBindings | None:
    """Return the lazily-initialised bindings, or ``None`` if Kit/Carbonite is unavailable."""
    global _BINDINGS, _INIT_TRIED
    if _BINDINGS is not None:
        return _BINDINGS
    if _INIT_TRIED:
        return None
    _INIT_TRIED = True
    b = FabricNoticeBindings()
    if not b.initialize():
        logger.debug("omni::fabric::IFabricUsd not available — Fabric notice suspension disabled")
        return None
    _BINDINGS = b
    return _BINDINGS
