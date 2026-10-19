# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-transfer backend connecting cameras to independently started Cosmos services."""

from __future__ import annotations

import logging
import math
import socket
import threading
from collections.abc import Sequence
from contextlib import ExitStack
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

from .._protocol import (
    MAX_CHUNK_FRAMES,
    ProtocolError,
    check_reply,
    connect,
    parse_endpoint,
    receive_message,
    send_message,
)

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .cosmos_model_cfg import CosmosModelCfg

_LOGGER = logging.getLogger(__name__)


class CosmosModel:
    """Client model resource; generation state belongs to its camera streams.

    The services keep model weights resident. Closing this client closes its camera sessions and
    leaves the services running, so a later Isaac Lab command can connect to the same models.
    """

    def __init__(self, cfg: CosmosModelCfg):
        """Validate client settings without loading Cosmos or opening a camera session."""
        if not isinstance(cfg.endpoint, (str, list, tuple)):
            raise ValueError("Cosmos endpoint must be a string, or a list with one endpoint per service.")
        endpoints = service_endpoints(cfg.endpoint)
        if not endpoints:
            raise ValueError("Cosmos needs at least one service endpoint.")
        for endpoint in endpoints:
            parse_endpoint(endpoint)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("Cosmos endpoints must name distinct services.")
        if not math.isfinite(cfg.timeout) or cfg.timeout <= 0:
            raise ValueError("Cosmos timeout must be finite and positive.")
        if isinstance(cfg.prompt, (list, tuple)):
            if not cfg.prompt or any(not isinstance(prompt, str) or not prompt.strip() for prompt in cfg.prompt):
                raise ValueError("A Cosmos prompt list must be nonempty and contain nonempty strings.")
        elif cfg.prompt is not None and not isinstance(cfg.prompt, str):
            raise ValueError("Cosmos prompt must be a string, a list of strings, or None.")
        if cfg.modality not in ("edge", "depth", "seg"):
            raise ValueError("Cosmos modality must be edge, depth, or seg.")
        if type(cfg.max_episode_frames) is not int or cfg.max_episode_frames < 1 or (cfg.max_episode_frames - 1) % 4:
            raise ValueError(
                "Set CosmosModelCfg.max_episode_frames to the episode's camera-frame budget (1 + 4*k), "
                "or use apply_cosmos to derive it from the task."
            )
        if cfg.transport not in ("auto", "cuda_ipc", "socket"):
            raise ValueError("Cosmos transport must be auto, cuda_ipc, or socket.")
        self._cfg = cfg
        self._lock = threading.Lock()
        self._streams: set[_CosmosStream] = set()
        self._closed = False

    def open_stream(self, num_views: int, seeds: tuple[int, ...]) -> _CosmosStream | _ShardedStream:
        """Create one camera stream; connect on its first step when image dimensions are known.

        With several service endpoints, the views are split into contiguous shares of near-equal size, one per
        service in endpoint order (see :func:`split_views`).

        Args:
            num_views: Number of camera views. Each service generates its share as one batch, within GPU memory and
                transport limits.
            seeds: Initial integer seed in ``[0, 2**31)`` for each view.

        Raises:
            ValueError: If the view count or seeds are invalid, or there are fewer views than services.
            RuntimeError: If the client model has already closed.
        """
        if type(num_views) is not int or num_views < 1:
            raise ValueError("Cosmos needs a positive number of camera views.")
        _validate_seeds(seeds, num_views)
        endpoints = service_endpoints(self._cfg.endpoint)
        shares = split_views(num_views, len(endpoints))
        with self._lock:
            if self._closed:
                raise RuntimeError("Cosmos client model is closed.")
            streams = [
                _CosmosStream(self, self._cfg, endpoint, seeds[share.start : share.stop], share, batched=num_views > 1)
                for endpoint, share in zip(endpoints, shares)
            ]
            self._streams.update(streams)
        return streams[0] if len(streams) == 1 else _ShardedStream(streams, shares)

    def close(self) -> None:
        """Close this client's camera sessions without shutting down the Cosmos services."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            streams = tuple(self._streams)
        for stream in streams:
            stream.close()

    def _discard_stream(self, stream: _CosmosStream) -> None:
        with self._lock:
            self._streams.discard(stream)


def service_endpoints(endpoint: str | Sequence[str]) -> list[str]:
    """Return the service endpoints of :attr:`CosmosModelCfg.endpoint`: one, or one per service."""
    return [endpoint] if isinstance(endpoint, str) else list(endpoint)


def split_views(num_views: int, num_services: int) -> list[range]:
    """Split a camera's views into contiguous shares of near-equal size, one per service in order.

    For example, 8 views on 3 services become views 0-2, 3-5, and 6-7. A view stays on its service, which keeps its
    generation history.

    Raises:
        ValueError: If there are fewer views than services.
    """
    if num_views < num_services:
        raise ValueError(f"Cosmos needs at least one environment per service, got {num_views} for {num_services}.")
    size, extra = divmod(num_views, num_services)
    bounds = [0]
    for service in range(num_services):
        bounds.append(bounds[-1] + size + (service < extra))
    return [range(start, stop) for start, stop in pairwise(bounds)]


def _validate_seeds(seeds: tuple[int, ...], count: int) -> None:
    if (
        not isinstance(seeds, tuple)
        or len(seeds) != count
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
    ):
        raise ValueError("Cosmos requires one integer seed in [0, 2**31) for each opened or reset view.")


class _SentStep(NamedTuple):
    """A step sent to a service, whose reply is outstanding."""

    frames: list[int]
    device: torch.device
    episode: int
    reset: bool


class _CosmosStream:
    """One service session generating a contiguous share of a camera's views as one batch."""

    def __init__(
        self,
        model: CosmosModel,
        cfg: CosmosModelCfg,
        endpoint: str,
        seeds: tuple[int, ...],
        views: range,
        *,
        batched: bool,
    ):
        """Prepare the session; it connects on its first step.

        Args:
            model: Client model that tracks this session.
            cfg: Client settings.
            endpoint: Endpoint of this session's service.
            seeds: Initial seed of each view of this session.
            views: The camera's view indices this session generates; they select the views' prompts.
            batched: Whether the camera has several views. Each view then keeps one prompt for all its episodes.
        """
        self._model = model
        self._cfg = cfg
        self._endpoint = endpoint
        self._seeds = seeds
        self._views = views
        self._batched = batched
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._shape: tuple[int, int, int] | None = None
        self._closed = False
        self._episode = 0
        self._episode_frames = 0
        self._channel = None
        self._service_device: torch.device | None = None
        self.transport: str | None = None

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor]:
        """Send one control chunk per view and return matching uint8 RGB on the controls' original device.

        Views starting or restarting an episode send one frame, the others four. A transport, model, or validation
        failure closes the session. The same chunk cannot safely be retried because the service may already have
        advanced its temporal generation state.
        """
        with self._lock:
            return self._receive(self._send(controls, reset_rows, seeds))

    def _send(self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]) -> _SentStep:
        """Validate and send one step without waiting for its reply, opening the session on first use.

        Call with this stream's lock held, then :meth:`_receive`. A failure closes the session.
        """
        import torch

        if self._closed:
            raise RuntimeError("Cosmos stream is closed or failed; recreate the camera stream.")
        try:
            num_views = len(self._seeds)
            if (
                not isinstance(controls, list)
                or len(controls) != num_views
                or any(
                    not isinstance(control, torch.Tensor)
                    or control.dtype != torch.uint8
                    or control.ndim != 4
                    or control.shape[-1] != 3
                    or any(size <= 0 for size in control.shape)
                    for control in controls
                )
            ):
                raise ValueError(
                    "Cosmos controls must hold one nonempty uint8 THWC tensor with three channels per view."
                )
            if (
                not isinstance(reset_rows, tuple)
                or any(type(row) is not int or not 0 <= row < num_views for row in reset_rows)
                or list(reset_rows) != sorted(set(reset_rows))
            ):
                raise ValueError("Cosmos resets name distinct views in increasing order.")
            _validate_seeds(seeds, len(reset_rows))
            control = controls[0]
            image_shape = tuple(control.shape[1:])
            if any(tuple(view.shape[1:]) != image_shape or view.device != control.device for view in controls):
                raise ValueError("Cosmos views must share one image size and device.")
            if self._shape is not None and image_shape != self._shape:
                raise ValueError("Cosmos image size cannot change within a camera stream.")
            if self._socket is None:
                self._socket = connect(self._endpoint, self._cfg.timeout)
                request = {
                    "op": "open",
                    "num_views": num_views,
                    "seeds": self._seeds,
                    "prompt": self._session_prompt(),
                    "modality": self._cfg.modality,
                    "height": image_shape[0],
                    "width": image_shape[1],
                    "max_episode_frames": self._cfg.max_episode_frames,
                }
                self.transport = self._select_transport(control.device)
                if self.transport == "cuda_ipc":
                    from .. import _cuda_ipc

                    self._channel = _cuda_ipc.SharedChannel(
                        control.device, (num_views, MAX_CHUNK_FRAMES, *image_shape), writes="control"
                    )
                    # The service opens the controls on this GPU, which it finds by its PCI bus ID.
                    ipc = {**self._channel.handles, "pci_bus_id": _cuda_ipc.pci_bus_id(control.device.index or 0)}
                    request.update(transport="cuda_ipc", ipc=ipc)
                reply, arrays = self._exchange(request)
                if arrays:
                    raise ProtocolError("Cosmos open reply must not contain image arrays.")
                if self._channel is not None:
                    # The service writes generated images into a buffer on its own GPU.
                    self._channel.open(self._service_device, reply.get("ipc"))
                self._shape = image_shape
            frames = [view.shape[0] for view in controls]
            if max(frames) > MAX_CHUNK_FRAMES:
                raise ValueError(f"Cosmos chunks hold at most {MAX_CHUNK_FRAMES} frames.")
            request = {"op": "step", "reset_rows": reset_rows, "seeds": seeds}
            episode = self._episode
            if not self._batched and reset_rows and isinstance(self._cfg.prompt, (list, tuple)):
                # A reset starts the next prompt's episode, unless the ending episode never got past its
                # first frame, such as the capture taken before the environment's initial reset.
                episode += 1 if self._episode_frames > 1 else 0
                request["prompt"] = self._episode_prompt(episode)
            arrays = []
            if self._channel is not None:
                # Controls and images stay on the GPU, one buffer row per view; events order the processes.
                for view, (chunk, count) in enumerate(zip(controls, frames)):
                    self._channel.control.tensor[view, :count].copy_(chunk)
                self._channel.control_ready.record()
                request["frames"] = frames
            else:
                # Images cross the process boundary through host memory and the socket.
                arrays = [chunk.detach().cpu().numpy() for chunk in controls]
            send_message(self._socket, request, arrays)
            return _SentStep(frames, control.device, episode, bool(reset_rows))
        except Exception:
            self._release(notify=False)
            raise

    def _receive(self, sent: _SentStep) -> list[torch.Tensor]:
        """Wait for the reply to a step from :meth:`_send` and return its images on the controls' device.

        Call with this stream's lock held. A failure closes the session.
        """
        import torch

        try:
            reply, generated = receive_message(self._socket)
            check_reply(reply)
            if self._channel is not None:
                if generated:
                    raise ProtocolError("Cosmos returned images through the socket in a CUDA IPC session.")
                self._channel.output_ready.wait()
                # From the service's GPU, the images copy directly to this one.
                images = [
                    self._channel.output.tensor[view, :count].to(sent.device, copy=True)
                    for view, count in enumerate(sent.frames)
                ]
            else:
                if len(generated) != len(sent.frames) or any(
                    images.shape != (count, *self._shape) for images, count in zip(generated, sent.frames)
                ):
                    raise ProtocolError("Cosmos returned images that do not match the control chunks.")
                images = [torch.from_numpy(view).to(device=sent.device) for view in generated]
            if not self._batched:
                if sent.reset:
                    self._episode, self._episode_frames = sent.episode, 0
                self._episode_frames += sent.frames[0]
            return images
        except Exception:
            self._release(notify=False)
            raise

    def close(self) -> None:
        """Close this camera's session. Repeated calls are safe."""
        with self._lock:
            self._release(notify=True)

    def _session_prompt(self) -> str | None | list[str | None]:
        """Opening prompt: one view starts its prompt list; several views each take ``prompt[view % len]``."""
        prompt = self._cfg.prompt
        if not self._batched or not isinstance(prompt, (list, tuple)):
            return self._episode_prompt(self._episode)
        return [prompt[view % len(prompt)] for view in self._views]

    def _episode_prompt(self, episode: int) -> str | None:
        prompt = self._cfg.prompt
        return prompt[episode % len(prompt)] if isinstance(prompt, (list, tuple)) else prompt

    def _select_transport(self, device: torch.device) -> str:
        """Choose CUDA IPC or the socket, per the configuration.

        CUDA IPC needs the service on this machine, on this GPU or on another one this process can see and that
        can access this GPU's memory directly. Choosing it records this process's index of the service's GPU.
        """
        import torch

        if self._cfg.transport == "socket":
            return "socket"
        reason = None
        family, address = parse_endpoint(self._endpoint)
        if device.type != "cuda":
            reason = "the camera images are not on a CUDA device"
        elif family == socket.AF_INET and address[0] not in _LOOPBACK_HOSTS:
            reason = f"the service at {address[0]} is not on this machine"
        else:
            from .. import _cuda_ipc

            capabilities = self._exchange({"op": "status"})[0].get("capabilities", {})
            if not _cuda_ipc.available():
                reason = "CUDA IPC is not available on this platform"
            elif "cuda_ipc" not in capabilities.get("transports", ()):
                reason = "the service does not offer CUDA IPC"
            elif capabilities.get("host_id") is None or capabilities["host_id"] != _cuda_ipc.host_id():
                # A local endpoint can be a tunnel; the other machine's GPU can have a PCI bus ID this one uses.
                reason = "the service runs on another machine"
            else:
                service_gpu = _cuda_ipc.device_by_bus_id(capabilities.get("pci_bus_id"))
                if service_gpu is None:
                    reason = "this process cannot see the service's GPU"
                elif not _cuda_ipc.can_access_peer(device.index or 0, service_gpu):
                    reason = "this GPU and the service's GPU cannot access each other's memory directly"
                else:
                    self._service_device = torch.device("cuda", service_gpu)
        if reason is None:
            _LOGGER.info("Cosmos keeps images on the GPU with CUDA IPC; the service uses %s.", self._service_device)
            return "cuda_ipc"
        if self._cfg.transport == "cuda_ipc":
            raise RuntimeError(f"Cosmos transport cuda_ipc is unavailable: {reason}.")
        _LOGGER.info("Cosmos sends images through the socket because %s.", reason)
        return "socket"

    def _exchange(self, metadata: dict, arrays: Sequence[np.ndarray] = ()) -> tuple[dict, list[np.ndarray]]:
        if self._socket is None:
            raise RuntimeError("Cosmos stream has no service connection.")
        send_message(self._socket, metadata, arrays)
        reply, images = receive_message(self._socket)
        check_reply(reply)
        return reply, images

    def _release(self, *, notify: bool) -> None:
        if self._closed:
            return
        self._closed = True
        released = False
        try:
            if self._socket is not None:
                try:
                    if self._channel is not None:
                        # Release the service's output buffer before the service frees it when the session closes.
                        self._channel.close_opened()
                    if notify:
                        self._exchange({"op": "close"})
                        # The service releases the control buffer before it confirms the close.
                        released = True
                except (OSError, ValueError, RuntimeError):
                    _LOGGER.debug("Cosmos session was already unavailable during close.", exc_info=True)
                finally:
                    self._socket.close()
                    self._socket = None
        finally:
            if self._channel is not None:
                # Without that confirmation the service may still read the control buffer, which then stays
                # allocated until this process exits.
                self._channel.close(peer_released=released)
                self._channel = None
            self._model._discard_stream(self)


class _ShardedStream:
    """A camera's views split across several Cosmos services, one contiguous share per service.

    Each step sends every service its share before waiting for any reply, so the services generate at the same
    time and a step lasts as long as the slowest one. A view always goes to the same service, which keeps its
    generation history.
    """

    def __init__(self, streams: list[_CosmosStream], shares: list[range]):
        self._streams = streams
        self._shares = shares

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor]:
        """Send each service its views' control chunks, then return every view's images in view order.

        A failure on any service closes every session, because the others may already have advanced their
        generation state.
        """
        with ExitStack() as locks:
            for stream in self._streams:
                locks.enter_context(stream._lock)
            sent: list[_SentStep] = []
            received = 0
            try:
                num_views = self._shares[-1].stop
                if (
                    not isinstance(controls, list)
                    or len(controls) != num_views
                    or not isinstance(reset_rows, tuple)
                    or any(type(row) is not int or not 0 <= row < num_views for row in reset_rows)
                    or list(reset_rows) != sorted(set(reset_rows))
                ):
                    raise ValueError("Cosmos needs one control chunk per view and distinct reset views in order.")
                _validate_seeds(seeds, len(reset_rows))
                for stream, share in zip(self._streams, self._shares):
                    resets = [index for index, row in enumerate(reset_rows) if row in share]
                    sent.append(
                        stream._send(
                            controls[share.start : share.stop],
                            tuple(reset_rows[index] - share.start for index in resets),
                            tuple(seeds[index] for index in resets),
                        )
                    )
                images = []
                for stream, pending in zip(self._streams, sent):
                    images.extend(stream._receive(pending))
                    received += 1
                return images
            except Exception:
                for index, stream in enumerate(self._streams):
                    # A session still awaiting its reply cannot exchange a close message; its service closes the
                    # session when the connection drops, and the camera keeps the controls that service may still read.
                    awaiting_reply = received <= index < len(sent)
                    stream._release(notify=not awaiting_reply)
                raise

    def close(self) -> None:
        """Close every service's session. Repeated calls are safe."""
        for stream in self._streams:
            stream.close()
