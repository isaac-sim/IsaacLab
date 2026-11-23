# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Client side of the model bridge: one generation session with a model service."""

from __future__ import annotations

import logging
import socket
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from .protocol import ProtocolError, check_reply, connect, receive_message, send_message

if TYPE_CHECKING:
    import numpy as np
    import torch

_LOGGER = logging.getLogger(__name__)


class SentStep(NamedTuple):
    """A step sent to the service, whose reply is outstanding."""

    frames: list[int]
    """Frames each view sent."""

    device: torch.device
    """Device of the controls, which the images return to."""


class BridgeSession:
    """One generation session with a model service: its connection, transport, and shared GPU buffers.

    Controls are one uint8 THWC chunk per view, and the generated images come back with the same shapes on the
    controls' device. With CUDA IPC they stay on the GPU: the session writes the controls into a buffer it shares
    with the service, reads the images from the service's buffer, and interprocess events order the two processes'
    streams. Otherwise they travel through host memory in the socket messages. The model's settings pass through
    the session unread.

    A failed exchange leaves the service's generation state unknown, so the caller releases the session instead of
    retrying it.
    """

    def __init__(self, endpoint: str, timeout: float, transport: str = "auto"):
        """Prepare the session; :meth:`open` connects.

        Args:
            endpoint: The service's ``unix:///path`` or ``tcp://host:port`` endpoint.
            timeout: Wait per network operation [s].
            transport: ``"auto"`` for CUDA IPC where the service can share this GPU's memory and the socket otherwise,
                ``"cuda_ipc"`` to require CUDA IPC, or ``"socket"``.
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self._requested_transport = transport
        self.transport: str | None = None
        """The transport chosen when the session opened, ``"cuda_ipc"`` or ``"socket"``."""
        self._socket: socket.socket | None = None
        self._channel = None
        self._service_device: torch.device | None = None
        self._max_chunk_frames = 1
        self._shape: tuple[int, int, int] | None = None

    @property
    def is_open(self) -> bool:
        """Whether the session is connected to its service."""
        return self._socket is not None

    @property
    def image_shape(self) -> tuple[int, int, int] | None:
        """The session's image shape ``(height, width, 3)``, once it opened."""
        return self._shape

    def open(
        self,
        device: torch.device,
        num_views: int,
        seeds: tuple[int, ...],
        image_shape: tuple[int, int, int],
        settings: dict,
    ) -> None:
        """Connect, choose the transport, and open the service's session with fresh views.

        Args:
            device: Device of the controls.
            num_views: Number of views the session generates as one batch.
            seeds: Opening seed of each view.
            image_shape: Control image shape ``(height, width, 3)``.
            settings: The model's settings, which the service passes to the model.

        Raises:
            ConnectionError: If the service does not answer.
            RuntimeError: If the service rejects the session, or the requested transport is unavailable.
        """
        self._socket = connect(self.endpoint, self.timeout)
        message = {
            "op": "open",
            "num_views": num_views,
            "seeds": tuple(seeds),
            "height": image_shape[0],
            "width": image_shape[1],
            "settings": settings,
        }
        self.transport = self._select_transport(device)
        if self.transport == "cuda_ipc":
            from . import ipc

            shape = (num_views, self._max_chunk_frames, *image_shape)
            self._channel = ipc.SharedChannel(device, shape, writes="control")
            # The service opens the controls on this GPU, which it finds by its UUID.
            handles = {**self._channel.handles, "gpu_uuid": ipc.gpu_uuid(device.index or 0)}
            message.update(transport="cuda_ipc", ipc=handles)
        try:
            reply, arrays = self.exchange(message)
        except RuntimeError:
            # The service answered and rejected the session, so it holds none of this session's buffers.
            if self._channel is not None:
                self._channel.close(peer_released=True)
                self._channel = None
            raise
        if arrays:
            raise ProtocolError("An open reply must not contain image arrays.")
        if self._channel is not None:
            # The service writes generated images into a buffer on its own GPU.
            self._channel.open(self._service_device, reply.get("ipc"))
        self._shape = tuple(image_shape)

    def send(
        self,
        controls: list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episode: dict | None = None,
    ) -> SentStep:
        """Send one chunk per view without waiting for the reply; :meth:`receive` returns the images.

        Args:
            controls: One uint8 THWC chunk per view, on one device, at the session's image size.
            reset_rows: Views starting a new episode, in increasing order.
            seeds: The seed of each view in ``reset_rows``.
            episode: The model's settings for the views that reset, such as a new prompt.

        Raises:
            ValueError: If a CUDA IPC chunk holds more frames than the model declared.
        """
        frames = [chunk.shape[0] for chunk in controls]
        message = {"op": "step", "reset_rows": reset_rows, "seeds": seeds}
        if episode is not None:
            message["episode"] = episode
        arrays = []
        if self._channel is not None:
            if max(frames) > self._max_chunk_frames:
                raise ValueError(f"The model takes at most {self._max_chunk_frames} frames per view and step.")
            # Controls and images stay on the GPU, one buffer row per view; events order the processes.
            for view, (chunk, count) in enumerate(zip(controls, frames)):
                self._channel.control.tensor[view, :count].copy_(chunk)
            self._channel.control_ready.record()
            message["frames"] = frames
        else:
            # Images cross the process boundary through host memory and the socket.
            arrays = [chunk.detach().cpu().numpy() for chunk in controls]
        send_message(self._socket, message, arrays)
        return SentStep(frames, controls[0].device)

    def receive(self, sent: SentStep) -> list[torch.Tensor]:
        """Wait for the reply to a step from :meth:`send` and return its images on the controls' device."""
        import torch

        reply, generated = receive_message(self._socket)
        check_reply(reply)
        if self._channel is not None:
            if generated:
                raise ProtocolError("The service returned images through the socket in a CUDA IPC session.")
            self._channel.output_ready.wait()
            # From the service's GPU, the images copy directly to this one.
            return [
                self._channel.output.tensor[view, :count].to(sent.device, copy=True)
                for view, count in enumerate(sent.frames)
            ]
        if len(generated) != len(sent.frames) or any(
            images.shape != (count, *self._shape) for images, count in zip(generated, sent.frames)
        ):
            raise ProtocolError("The service returned images that do not match the control chunks.")
        return [torch.from_numpy(view).to(device=sent.device) for view in generated]

    def exchange(self, metadata: dict, arrays: Sequence[np.ndarray] = ()) -> tuple[dict, list[np.ndarray]]:
        """Send a request on the session's connection and return the service's successful reply."""
        if self._socket is None:
            raise RuntimeError("The session has no service connection.")
        send_message(self._socket, metadata, arrays)
        reply, images = receive_message(self._socket)
        check_reply(reply)
        return reply, images

    def release(self, *, notify: bool) -> None:
        """Close the session and its connection, then free the shared buffers. Repeated calls are safe.

        Args:
            notify: Close the service's session with a message first. Without its confirmation, the service may
                still read the control buffer, which then stays allocated until this process exits.
        """
        released = False
        try:
            if self._socket is not None:
                try:
                    if self._channel is not None:
                        # Release the service's output buffer before the service frees it when the session closes.
                        self._channel.close_opened()
                    if notify:
                        self.exchange({"op": "close"})
                        # The service releases the control buffer before it confirms the close.
                        released = True
                except (OSError, ValueError, RuntimeError):
                    _LOGGER.debug("The session was already unavailable during close.", exc_info=True)
                finally:
                    self._socket.close()
                    self._socket = None
        finally:
            if self._channel is not None:
                self._channel.close(peer_released=released)
                self._channel = None

    def _select_transport(self, device: torch.device) -> str:
        """Choose CUDA IPC or the socket, as requested.

        CUDA IPC needs the service on this machine, on this GPU or on another one this process can see and that
        can access this GPU's memory directly, and the service must see this GPU. GPU UUIDs are unique across
        machines, so a service that sees this GPU runs on this machine. Choosing it records this process's index of
        the service's GPU and the model's chunk limit, which sizes the shared buffers.
        """
        import torch

        if self._requested_transport == "socket":
            return "socket"
        reason = None
        if device.type != "cuda":
            reason = "the images are not on a CUDA device"
        else:
            from . import ipc

            capabilities = self.exchange({"op": "status"})[0].get("capabilities", {})
            if not ipc.available():
                reason = "CUDA IPC is not available on this platform"
            elif "cuda_ipc" not in capabilities.get("transports", ()):
                reason = "the service does not offer CUDA IPC"
            elif ipc.gpu_uuid(device.index or 0) not in capabilities.get("visible_gpus", ()):
                # A local endpoint can be a tunnel to another machine, and the service opens the controls on this
                # GPU, so it must see this GPU.
                reason = "the service runs on another machine or cannot see this GPU"
            else:
                service_gpu = ipc.device_by_uuid(capabilities.get("gpu_uuid"))
                if service_gpu is None:
                    reason = "this process cannot see the service's GPU"
                elif not ipc.can_access_peer(device.index or 0, service_gpu):
                    reason = "this GPU and the service's GPU cannot access each other's memory directly"
                else:
                    self._service_device = torch.device("cuda", service_gpu)
                    self._max_chunk_frames = capabilities.get("max_chunk_frames", 1)
        if reason is None:
            _LOGGER.info("Images stay on the GPU with CUDA IPC; the service uses %s.", self._service_device)
            return "cuda_ipc"
        if self._requested_transport == "cuda_ipc":
            raise RuntimeError(f"Transport cuda_ipc is unavailable: {reason}.")
        _LOGGER.info("Images go through the socket because %s.", reason)
        return "socket"
