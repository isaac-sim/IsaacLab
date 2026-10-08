# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU memory and events shared between the camera and the Cosmos service on one machine.

The two processes use different PyTorch builds, so sharing goes through the CUDA driver API, which only
depends on the installed driver: one process allocates buffers and interprocess events and exports their
handles; the other opens them. Buffers appear in both processes as zero-copy uint8 tensors, and events
order the producer's and consumer's CUDA streams without synchronizing the host. CUDA IPC is available
on Linux; on other platforms :func:`available` returns False and images go through the socket.
"""

from __future__ import annotations

import base64
import ctypes
import math
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_HANDLE_BYTES = 64
_EVENT_FLAGS = 0x2 | 0x4  # CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS
_LAZY_PEER_ACCESS = 0x1  # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS


class _Handle(ctypes.Structure):
    # Raw bytes: a c_char array would read as text and stop at the first zero byte.
    _fields_ = [("reserved", ctypes.c_ubyte * _HANDLE_BYTES)]


_driver: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _driver
    if _driver is None:
        driver = ctypes.CDLL("libcuda.so.1")
        _check(driver, driver.cuInit(0))
        _driver = driver
    return _driver


def _check(driver: ctypes.CDLL, result: int) -> None:
    if result != 0:
        message = ctypes.c_char_p()
        driver.cuGetErrorString(result, ctypes.byref(message))
        name = message.value.decode() if message.value else f"error {result}"
        raise RuntimeError(f"CUDA driver call failed: {name}.")


def available() -> bool:
    """Return whether this process can share GPU memory with another process through CUDA IPC."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        _lib()
    except (OSError, RuntimeError):
        return False
    return True


def _retain_primary_context(device_index: int) -> tuple[ctypes.c_int, ctypes.c_void_p]:
    """Retain the device's primary context, the context PyTorch uses, creating it if needed."""
    driver = _lib()
    device = ctypes.c_int()
    _check(driver, driver.cuDeviceGet(ctypes.byref(device), device_index))
    context = ctypes.c_void_p()
    _check(driver, driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
    return device, context


class _Context:
    """Make the device's primary context current for driver calls."""

    def __init__(self, device_index: int):
        self.device_index = device_index

    def __enter__(self) -> ctypes.CDLL:
        self._device, context = _retain_primary_context(self.device_index)
        driver = _lib()
        _check(driver, driver.cuCtxPushCurrent_v2(context))
        return driver

    def __exit__(self, *exc) -> None:
        driver = _lib()
        popped = ctypes.c_void_p()
        driver.cuCtxPopCurrent_v2(ctypes.byref(popped))
        driver.cuDevicePrimaryCtxRelease_v2(self._device)


class _ContextHold:
    """Keep the primary context alive while a shared resource lives in it.

    Releasing the last reference destroys the context with its memory and events. That happens in a process
    where PyTorch has not used the device yet; PyTorch would then create a new context without the shared mapping.
    """

    def __init__(self, device_index: int):
        self._device, _ = _retain_primary_context(device_index)

    def release(self) -> None:
        if self._device is not None:
            _lib().cuDevicePrimaryCtxRelease_v2(self._device)
            self._device = None


def pci_bus_id(device_index: int) -> str:
    """Return the PCI bus ID of a CUDA device as seen by this process, for matching across processes."""
    with _Context(device_index) as driver:
        device = ctypes.c_int()
        _check(driver, driver.cuDeviceGet(ctypes.byref(device), device_index))
        buffer = ctypes.create_string_buffer(32)
        _check(driver, driver.cuDeviceGetPCIBusId(buffer, 32, device))
        return buffer.value.decode().lower()


class _CudaArray:
    """Expose a device pointer through the CUDA array interface so PyTorch can view it without a copy."""

    def __init__(self, pointer: int, shape: tuple[int, ...]):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


def _encode(handle: _Handle) -> str:
    return base64.b64encode(bytes(handle.reserved)).decode("ascii")


def _decode(text: str) -> _Handle:
    raw = base64.b64decode(text)
    if len(raw) != _HANDLE_BYTES:
        raise ValueError("A CUDA IPC handle must contain 64 bytes.")
    return _Handle.from_buffer_copy(raw)


def _stream_handle(device: torch.device) -> ctypes.c_void_p:
    import torch

    return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)


class SharedBuffer:
    """A uint8 GPU buffer allocated here and opened by another process, or opened from a handle."""

    def __init__(self, device: torch.device, shape: tuple[int, ...], handle: str | None = None):
        import torch

        self.device = torch.device(device)
        self.shape = tuple(shape)
        self._owner = handle is None
        self._pointer = ctypes.c_uint64()
        self.tensor = None
        self._hold = _ContextHold(self.device.index or 0)
        try:
            with _Context(self.device.index or 0) as driver:
                if self._owner:
                    size = ctypes.c_size_t(math.prod(shape))
                    _check(driver, driver.cuMemAlloc_v2(ctypes.byref(self._pointer), size))
                    exported = _Handle()
                    _check(driver, driver.cuIpcGetMemHandle(ctypes.byref(exported), self._pointer))
                    self.handle = _encode(exported)
                else:
                    _check(
                        driver,
                        driver.cuIpcOpenMemHandle_v2(ctypes.byref(self._pointer), _decode(handle), _LAZY_PEER_ACCESS),
                    )
                    self.handle = handle
            self.tensor = torch.as_tensor(_CudaArray(self._pointer.value, self.shape), device=self.device)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release the memory, or this process's mapping of it. Repeated calls are safe."""
        self.tensor = None
        if self._pointer.value:
            with _Context(self.device.index or 0) as driver:
                if self._owner:
                    driver.cuMemFree_v2(self._pointer)
                else:
                    driver.cuIpcCloseMemHandle(self._pointer)
            self._pointer = ctypes.c_uint64()
        self._hold.release()


class SharedEvent:
    """An interprocess CUDA event created here and opened by another process, or opened from a handle."""

    def __init__(self, device: torch.device, handle: str | None = None):
        import torch

        self.device = torch.device(device)
        self._event = ctypes.c_void_p()
        self._hold = _ContextHold(self.device.index or 0)
        try:
            with _Context(self.device.index or 0) as driver:
                if handle is None:
                    _check(driver, driver.cuEventCreate(ctypes.byref(self._event), _EVENT_FLAGS))
                    exported = _Handle()
                    _check(driver, driver.cuIpcGetEventHandle(ctypes.byref(exported), self._event))
                    self.handle = _encode(exported)
                else:
                    _check(driver, driver.cuIpcOpenEventHandle(ctypes.byref(self._event), _decode(handle)))
                    self.handle = handle
        except Exception:
            self.close()
            raise

    def record(self) -> None:
        """Mark the current point of this device's current PyTorch stream."""
        with _Context(self.device.index or 0) as driver:
            _check(driver, driver.cuEventRecord(self._event, _stream_handle(self.device)))

    def wait(self) -> None:
        """Make this device's current PyTorch stream wait for the last recorded point, without blocking the host."""
        with _Context(self.device.index or 0) as driver:
            _check(driver, driver.cuStreamWaitEvent(_stream_handle(self.device), self._event, 0))

    def close(self) -> None:
        """Destroy this process's event. Repeated calls are safe."""
        if self._event.value:
            with _Context(self.device.index or 0) as driver:
                driver.cuEventDestroy_v2(self._event)
            self._event = ctypes.c_void_p()
        self._hold.release()


class SharedChannel:
    """The buffers and events of one camera session: controls in, generated images out."""

    def __init__(self, device: torch.device, shape: tuple[int, ...], handles: dict | None = None):
        """Create the session's buffers and events, or open them from the other process's handles.

        Args:
            device: CUDA device of this process holding the memory.
            shape: Largest chunk ``(frames, height, width, 3)``.
            handles: Handles from :attr:`handles` of the creating process. Defaults to None, which creates them.
        """
        handles = handles or {}
        self.control = SharedBuffer(device, shape, handles.get("control"))
        self.output = SharedBuffer(device, shape, handles.get("output"))
        self.control_ready = SharedEvent(device, handles.get("control_ready"))
        self.output_ready = SharedEvent(device, handles.get("output_ready"))

    @property
    def handles(self) -> dict:
        """Handles the other process opens."""
        return {
            "control": self.control.handle,
            "output": self.output.handle,
            "control_ready": self.control_ready.handle,
            "output_ready": self.output_ready.handle,
        }

    def close(self) -> None:
        """Release every buffer and event. Repeated calls are safe."""
        for resource in (self.control, self.output, self.control_ready, self.output_ready):
            resource.close()
