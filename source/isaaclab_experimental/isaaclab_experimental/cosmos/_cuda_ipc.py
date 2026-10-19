# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU memory and events shared between the camera and the Cosmos service on one machine.

The two processes use different PyTorch builds, so sharing goes through the CUDA driver API, which only
depends on the installed driver: each process allocates the buffer it writes and the interprocess event it
records on its own GPU and exports their handles; the other process maps the buffer on that GPU and imports the
event on its own GPU, whose stream waits for it. Buffers appear in both processes as zero-copy uint8 tensors, and
events order the producer's and consumer's CUDA streams without synchronizing the host. The processes can use
different GPUs that access each other's memory directly, over NVLink or PCIe; the reader then copies between the
GPUs without host memory. CUDA IPC is available on Linux; on other platforms :func:`available` returns False and
images go through the socket.
"""

from __future__ import annotations

import base64
import ctypes
import math
import sys
from contextlib import ExitStack
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


def host_id() -> str | None:
    """Return this machine's boot ID, the same in every process on it, or None if unknown.

    A local endpoint can be a tunnel to another machine, whose GPU can have the same PCI bus ID.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id") as file:
            return file.read().strip() or None
    except OSError:
        return None


def pci_bus_id(device_index: int) -> str:
    """Return the PCI bus ID of a CUDA device as seen by this process, for matching across processes."""
    with _Context(device_index) as driver:
        device = ctypes.c_int()
        _check(driver, driver.cuDeviceGet(ctypes.byref(device), device_index))
        # A PCI bus ID such as "00000000:01:00.0" is 16 characters; 32 bytes leave room for the null terminator.
        buffer = ctypes.create_string_buffer(32)
        _check(driver, driver.cuDeviceGetPCIBusId(buffer, 32, device))
        return buffer.value.decode().lower()


def device_by_bus_id(bus_id: str | None) -> int | None:
    """Return this process's index of the CUDA device with PCI bus ID ``bus_id``, or None if it cannot see it."""
    if not isinstance(bus_id, str):
        return None
    device = ctypes.c_int()
    if _lib().cuDeviceGetByPCIBusId(ctypes.byref(device), bus_id.encode()) != 0:
        return None
    return device.value


def can_access_peer(index: int, peer_index: int) -> bool:
    """Return whether two CUDA devices of this process can read each other's memory directly, over NVLink or PCIe.

    A device is its own peer. Between devices without peer access, CUDA would stage copies through host memory.
    """
    if index == peer_index:
        return True
    driver = _lib()
    for first, second in ((index, peer_index), (peer_index, index)):
        allowed = ctypes.c_int()
        _check(driver, driver.cuDeviceCanAccessPeer(ctypes.byref(allowed), first, second))
        if not allowed.value:
            return False
    return True


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


_KEPT: list[ExitStack] = []
"""Buffers and events kept until this process exits because the other process may still have them open."""


class SharedChannel:
    """The buffers and events of one camera session: controls in, generated images out.

    Each process creates the buffer it writes and the event it records on its own GPU: the camera creates
    ``control`` and ``control_ready``, the service ``output`` and ``output_ready``. Neither process writes memory the
    other can free. CUDA leaves freeing memory that another process still has open undefined, so each process
    releases the other's buffer first and frees its own only after the other confirms it released it. CUDA records
    an event only on the device that created it, so this split also lets the two processes use different GPUs. Each
    process maps the other's buffer on the GPU that holds it and imports the other's event on its own GPU, whose
    stream waits for it.
    """

    def __init__(self, device: torch.device, shape: tuple[int, ...], writes: str):
        """Create the buffer and event this process writes; :meth:`open` adds the other process's.

        Args:
            device: CUDA device of this process that holds them.
            shape: Largest chunk of every view, ``(views, frames, height, width, 3)``.
            writes: ``"control"`` for the camera, ``"output"`` for the service.

        Raises:
            ValueError: If ``writes`` names neither buffer.
        """
        if writes not in ("control", "output"):
            raise ValueError("A CUDA IPC channel writes either control or output.")
        self._writes = writes
        self._device = device
        self._shape = tuple(shape)
        self.control = self.control_ready = self.output = self.output_ready = None
        self._opened = ExitStack()
        with ExitStack() as created:
            buffer = SharedBuffer(device, self._shape)
            created.callback(buffer.close)
            event = SharedEvent(device)
            created.callback(event.close)
            self._created = created.pop_all()
        self._own = (buffer, event)
        self._assign(writes, buffer, event)

    @property
    def handles(self) -> dict[str, str]:
        """Handles of the buffer and event this process created, for the other process to open."""
        buffer, event = self._own
        return {self._writes: buffer.handle, f"{self._writes}_ready": event.handle}

    def open(self, device: torch.device, handles: dict) -> None:
        """Map the other process's buffer on ``device`` and import its event on this channel's GPU.

        ``device`` is this process's index of the GPU that holds the buffer. The event is imported on the GPU whose
        stream waits for it: waiting on an event imported on the other GPU crashed the CUDA driver.

        Raises:
            ValueError: If the handles do not name the other process's buffer and event.
        """
        reads = "output" if self._writes == "control" else "control"
        if (
            not isinstance(handles, dict)
            or set(handles) != {reads, f"{reads}_ready"}
            or not all(isinstance(value, str) for value in handles.values())
        ):
            raise ValueError(f"CUDA IPC needs the other process's {reads} and {reads}_ready handles.")
        with ExitStack() as opened:
            buffer = SharedBuffer(device, self._shape, handles[reads])
            opened.callback(buffer.close)
            event = SharedEvent(self._device, handles[f"{reads}_ready"])
            opened.callback(event.close)
            self._opened = opened.pop_all()
        self._assign(reads, buffer, event)

    def close_opened(self) -> None:
        """Release the other process's buffer and event before that process frees them. Repeated calls are safe."""
        self._opened.close()

    def close(self, *, peer_released: bool = True) -> None:
        """Release the other process's buffer and event, then free this process's own. Repeated calls are safe.

        Args:
            peer_released: Whether the other process confirmed it released this process's buffer and event, or
                never received their handles. Otherwise they stay allocated until this process exits.
        """
        try:
            self._opened.close()
        finally:
            if peer_released:
                self._created.close()
            else:
                _KEPT.append(self._created.pop_all())

    def _assign(self, name: str, buffer: SharedBuffer, event: SharedEvent) -> None:
        if name == "control":
            self.control, self.control_ready = buffer, event
        else:
            self.output, self.output_ready = buffer, event
