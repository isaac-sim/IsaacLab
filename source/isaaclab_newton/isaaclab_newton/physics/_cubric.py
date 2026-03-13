# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python ctypes bindings for the cubric GPU transform-hierarchy API.

Acquires the ``omni::cubric::IAdapter`` carb interface directly from the
Carbonite framework and wraps its function-pointer methods so that Newton
can call cubric's GPU transform propagation without C++ pybind11 changes.

The flow mirrors PhysX's ``DirectGpuHelper::updateXForms_GPU()``:

1. ``IAdapter::create``     → allocate a cubric adapter ID
2. ``IAdapter::bindToStage`` → bind to the current Fabric stage
3. ``IAdapter::compute``     → GPU kernel: propagate world transforms
4. ``IAdapter::release``     → free the adapter

When cubric is unavailable (e.g. CPU-only machine, plugin not loaded), the
caller falls back to the CPU ``update_world_xforms()`` path.
"""

from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Carb Framework struct layout (CARB_ABI function-pointer offsets, x86_64)
# ---------------------------------------------------------------------------
# Counting only CARB_ABI fields from the top of ``struct Framework``:
#   0: loadPluginsEx
#   8: unloadAllPlugins
#  16: acquireInterfaceWithClient
#  24: tryAcquireInterfaceWithClient  ← we use this one
_FW_OFF_TRY_ACQUIRE = 24

# ---------------------------------------------------------------------------
#  IAdapter struct layout  (from omni/cubric/IAdapter.h)
# ---------------------------------------------------------------------------
#   0: getAttribute
#   8: create(AdapterId*)
#  16: refcount
#  24: retain
#  32: release(AdapterId)
#  40: bindToStage(AdapterId, const FabricId&)
#  48: unbind
#  56: compute(AdapterId, options, dirtyMode, outFlags*)
_IA_OFF_CREATE = 8
_IA_OFF_RELEASE = 32
_IA_OFF_BIND = 40
_IA_OFF_COMPUTE = 56

# AdapterId sentinel
_INVALID_ADAPTER_ID = ctypes.c_uint64(~0).value

# AdapterComputeOptions flags  (from IAdapter.h)
_OPT_FORCE_UPDATE = 1 << 0              # Force update, ignoring invalidation status
_OPT_FORCE_STATE_RECONSTRUCTION = 1 << 1  # Force full rebuild of internal accel structures
_OPT_SKIP_ISOLATED = 1 << 2             # Skip prims with connectivity degree 0
_OPT_RIGID_BODY = 1 << 3                # Use PhysicsRigidBodyAPI tag for inverse propagation

# Newton prims get tagged with PhysicsRigidBodyAPI at init time so
# cubric's eRigidBody mode can distinguish rigid-body buckets
# (Inverse: preserve world matrix written by Newton, derive local)
# from non-rigid-body buckets (Forward: propagate to children).
# eForceUpdate is ORed in to bypass the change-listener check.
_OPT_DEFAULT = _OPT_RIGID_BODY | _OPT_FORCE_UPDATE

# AdapterDirtyMode
_DIRTY_ALL = 0     # eAll — dirty all prims in the stage
_DIRTY_COARSE = 1  # eCoarse — dirty all prims in visited buckets


# ---------------------------------------------------------------------------
#  ctypes struct mirrors
# ---------------------------------------------------------------------------
class _Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _InterfaceDesc(ctypes.Structure):
    """``carb::InterfaceDesc`` — {const char* name, Version version}."""
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("version", _Version),
    ]


def _read_u64(addr: int) -> int:
    return ctypes.c_uint64.from_address(addr).value


def _dump_fn_ptrs(base: int, names: list[str], label: str) -> None:
    """Log function pointer values at 8-byte intervals from *base*."""
    for i, name in enumerate(names):
        addr = _read_u64(base + i * 8)
        tag = "OK" if addr else "NULL"
        logger.info("  %s+%d (%s) = 0x%016x  [%s]", label, i * 8, name, addr, tag)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
class CubricBindings:
    """Typed wrappers around the cubric ``IAdapter`` API.

    Call :meth:`initialize` once; if it returns ``True``, the four adapter
    methods are available.
    """

    def __init__(self) -> None:
        self._ia_ptr: int = 0
        self._create_fn = None
        self._release_fn = None
        self._bind_fn = None
        self._compute_fn = None
        self._log_count = 0

    # -- lifecycle -----------------------------------------------------------

    def initialize(self) -> bool:
        """Acquire the cubric ``IAdapter`` from the carb framework."""
        # Ensure the omni.cubric extension (native carb plugin) is loaded.
        try:
            import omni.kit.app

            ext_mgr = omni.kit.app.get_app().get_extension_manager()
            cubric_enabled = ext_mgr.is_extension_enabled("omni.cubric")
            logger.info("omni.cubric extension enabled: %s", cubric_enabled)
            if not cubric_enabled:
                logger.info("Enabling omni.cubric extension")
                ext_mgr.set_extension_enabled_immediate("omni.cubric", True)
                cubric_enabled = ext_mgr.is_extension_enabled("omni.cubric")
                logger.info("omni.cubric after enable: %s", cubric_enabled)
        except Exception as exc:
            logger.warning("Cannot enable omni.cubric: %s", exc)
            return False

        # Get Framework* via libcarb.so acquireFramework (singleton).
        try:
            libcarb = ctypes.CDLL("libcarb.so")
        except OSError:
            logger.warning("Could not load libcarb.so")
            return False

        # Check which symbols libcarb exports for framework access
        for sym_name in ("acquireFramework", "carbGetSdkVersion", "isFrameworkValid"):
            try:
                sym = getattr(libcarb, sym_name, None)
                logger.info("libcarb.%s: %s", sym_name, "found" if sym else "missing")
            except Exception:
                logger.info("libcarb.%s: not accessible", sym_name)

        # Verify framework is alive
        try:
            libcarb.isFrameworkValid.restype = ctypes.c_bool
            libcarb.isFrameworkValid.argtypes = []
            fw_valid = libcarb.isFrameworkValid()
            logger.info("isFrameworkValid() = %s", fw_valid)
        except Exception as exc:
            logger.warning("isFrameworkValid() failed: %s", exc)

        # Get SDK version for diagnostics
        try:
            libcarb.carbGetSdkVersion.restype = ctypes.c_char_p
            libcarb.carbGetSdkVersion.argtypes = []
            sdk_ver = libcarb.carbGetSdkVersion()
            logger.info("carbGetSdkVersion() = %s", sdk_ver)
        except Exception as exc:
            logger.info("carbGetSdkVersion() failed: %s", exc)

        libcarb.acquireFramework.restype = ctypes.c_void_p
        libcarb.acquireFramework.argtypes = [ctypes.c_char_p, _Version]
        fw_ptr = libcarb.acquireFramework(b"isaaclab.cubric", _Version(0, 0))
        if not fw_ptr:
            logger.warning("acquireFramework returned null")
            return False
        logger.info("carb Framework* = 0x%016x", fw_ptr)

        # Dump first several framework function pointers for diagnosis
        fw_fn_names = [
            "loadPluginsEx",
            "unloadAllPlugins",
            "acquireInterfaceWithClient",
            "tryAcquireInterfaceWithClient",
        ]
        _dump_fn_ptrs(fw_ptr, fw_fn_names, "Framework")

        # Read tryAcquireInterfaceWithClient fn-ptr from Framework.
        try_acquire_addr = _read_u64(fw_ptr + _FW_OFF_TRY_ACQUIRE)
        if try_acquire_addr == 0:
            logger.warning("tryAcquireInterfaceWithClient is null in Framework")
            return False
        logger.info("tryAcquireInterfaceWithClient addr = 0x%016x", try_acquire_addr)

        try_acquire_fn = ctypes.CFUNCTYPE(
            ctypes.c_void_p,   # return: void* (IAdapter*)
            ctypes.c_char_p,   # clientName
            _InterfaceDesc,    # desc (by value)
            ctypes.c_char_p,   # pluginName
        )(try_acquire_addr)

        desc = _InterfaceDesc(
            name=b"omni::cubric::IAdapter",
            version=_Version(0, 1),
        )
        logger.info(
            "Calling tryAcquireInterfaceWithClient("
            "client=%r, iface=%r, ver=%d.%d, plugin=%r)",
            b"carb.scripting-python.plugin",
            desc.name,
            desc.version.major,
            desc.version.minor,
            None,
        )
        ia_ptr = try_acquire_fn(b"carb.scripting-python.plugin", desc, None)
        if not ia_ptr:
            # Try without client name restriction
            logger.info("First attempt returned null; retrying with client=None")
            ia_ptr = try_acquire_fn(None, desc, None)
        if not ia_ptr:
            # Try acquireInterfaceWithClient (offset 16) which logs errors
            logger.info("tryAcquire returned null; trying acquireInterfaceWithClient")
            acquire_addr = _read_u64(fw_ptr + 16)
            if acquire_addr:
                acquire_fn = ctypes.CFUNCTYPE(
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    _InterfaceDesc,
                    ctypes.c_char_p,
                )(acquire_addr)
                ia_ptr = acquire_fn(b"isaaclab.cubric", desc, None)
        if not ia_ptr:
            logger.warning(
                "All IAdapter acquisition attempts returned null — "
                "cubric plugin may not be registered or interface mismatch"
            )
            return False
        self._ia_ptr = ia_ptr
        logger.info("IAdapter* = 0x%016x", ia_ptr)

        # Dump all IAdapter function pointers
        ia_fn_names = [
            "getAttribute", "create", "refcount", "retain",
            "release", "bindToStage", "unbind", "compute",
        ]
        _dump_fn_ptrs(ia_ptr, ia_fn_names, "IAdapter")

        # Wrap the four IAdapter function pointers we need.
        create_addr = _read_u64(ia_ptr + _IA_OFF_CREATE)
        release_addr = _read_u64(ia_ptr + _IA_OFF_RELEASE)
        bind_addr = _read_u64(ia_ptr + _IA_OFF_BIND)
        compute_addr = _read_u64(ia_ptr + _IA_OFF_COMPUTE)

        if not all([create_addr, release_addr, bind_addr, compute_addr]):
            logger.warning("One or more IAdapter function pointers are null")
            return False

        # create(AdapterId* out) -> bool
        self._create_fn = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.POINTER(ctypes.c_uint64),
        )(create_addr)

        # release(AdapterId) -> bool
        self._release_fn = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_uint64,
        )(release_addr)

        # bindToStage(AdapterId, const FabricId&) -> bool
        # FabricId is uint64, passed by const-ref → pointer on x86_64
        self._bind_fn = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
        )(bind_addr)

        # compute(AdapterId, options, dirtyMode, outAccountFlags*) -> bool
        self._compute_fn = ctypes.CFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_uint64,   # adapterId
            ctypes.c_uint32,   # options  (AdapterComputeOptions)
            ctypes.c_int32,    # dirtyMode (AdapterDirtyMode)
            ctypes.c_void_p,   # outAccountFlags* (nullable)
        )(compute_addr)

        logger.info(
            "cubric IAdapter bindings ready (opts=0x%x [eRigidBody|eForceUpdate], dirty=%d [eAll])",
            _OPT_DEFAULT,
            _DIRTY_ALL,
        )
        return True

    @property
    def available(self) -> bool:
        return self._ia_ptr != 0

    # -- cubric adapter methods ----------------------------------------------

    def create_adapter(self) -> int | None:
        """Create a cubric adapter. Returns an adapter ID or ``None``."""
        if not self._create_fn:
            return None
        adapter_id = ctypes.c_uint64(_INVALID_ADAPTER_ID)
        ok = self._create_fn(ctypes.byref(adapter_id))
        if not ok or adapter_id.value == _INVALID_ADAPTER_ID:
            logger.warning("IAdapter::create failed")
            return None
        logger.info("cubric adapter created (id=%d)", adapter_id.value)
        return adapter_id.value

    def bind_to_stage(self, adapter_id: int, fabric_id: int) -> bool:
        """Bind the adapter to a Fabric stage."""
        if not self._bind_fn:
            return False
        fid = ctypes.c_uint64(fabric_id)
        ok = self._bind_fn(adapter_id, ctypes.byref(fid))
        if not ok:
            logger.warning("IAdapter::bindToStage failed (adapter=%d, fabricId=%d)", adapter_id, fabric_id)
        elif self._log_count < 3:
            logger.info("IAdapter::bindToStage ok (adapter=%d, fabricId=%d)", adapter_id, fabric_id)
        return ok

    def compute(self, adapter_id: int) -> bool:
        """Run the GPU transform-hierarchy compute pass.

        Uses ``eRigidBody | eForceUpdate`` with ``eAll`` dirty mode.
        ``eRigidBody`` makes cubric apply Inverse propagation on buckets
        tagged with ``PhysicsRigidBodyAPI`` (keeps Newton's world transforms,
        derives local) and Forward on everything else (propagates to children).
        ``eForceUpdate`` bypasses the change-listener dirty check.
        """
        if not self._compute_fn:
            return False
        flags = ctypes.c_uint32(0)
        ok = self._compute_fn(adapter_id, _OPT_DEFAULT, _DIRTY_ALL, ctypes.byref(flags))
        if not ok:
            logger.warning("IAdapter::compute returned false (flags=0x%x)", flags.value)
        return ok

    def release_adapter(self, adapter_id: int) -> None:
        """Release an adapter."""
        if not adapter_id or not self._release_fn:
            return
        self._release_fn(adapter_id)
