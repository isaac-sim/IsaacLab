# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-transfer backend connecting cameras to an independently started Cosmos service."""

from __future__ import annotations

import logging
import math
import socket
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

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

    The service keeps model weights resident. Closing this client closes its camera sessions and
    leaves the service running, so a later Isaac Lab command can connect to the same model.
    """

    def __init__(self, cfg: CosmosModelCfg):
        """Validate client settings without loading Cosmos or opening a camera session."""
        parse_endpoint(cfg.endpoint)
        if not math.isfinite(cfg.timeout) or cfg.timeout <= 0:
            raise ValueError("Cosmos timeout must be finite and positive.")
        if isinstance(cfg.prompt, (list, tuple)):
            if not cfg.prompt or any(not isinstance(prompt, str) or not prompt.strip() for prompt in cfg.prompt):
                raise ValueError("A Cosmos prompt list must be nonempty and contain nonempty strings.")
        elif cfg.prompt is not None and not isinstance(cfg.prompt, str):
            raise ValueError("Cosmos prompt must be a string, a list of strings, or None.")
        if cfg.modality not in ("edge", "depth", "seg"):
            raise ValueError("Cosmos modality must be edge, depth, or seg.")
        if type(cfg.max_episode_frames) is not int or cfg.max_episode_frames <= 0:
            raise ValueError("Cosmos max_episode_frames must be a positive integer.")
        if cfg.transport not in ("auto", "cuda_ipc", "socket"):
            raise ValueError("Cosmos transport must be auto, cuda_ipc, or socket.")
        self._cfg = cfg
        self._lock = threading.Lock()
        self._streams: set[_CosmosStream] = set()
        self._closed = False

    def open_stream(self, num_views: int, seeds: tuple[int, ...]) -> _CosmosStream:
        """Create one camera stream; connect on its first step when image dimensions are known.

        Args:
            num_views: Number of camera views, generated as one batch; the service sets the maximum.
            seeds: Initial integer seed in ``[0, 2**31)`` for each view.

        Raises:
            ValueError: If the view count or seeds are invalid.
            RuntimeError: If the client model has already closed.
        """
        if type(num_views) is not int or num_views < 1:
            raise ValueError("Cosmos needs a positive number of camera views.")
        _validate_seeds(seeds, num_views)
        with self._lock:
            if self._closed:
                raise RuntimeError("Cosmos client model is closed.")
            stream = _CosmosStream(self, self._cfg, seeds)
            self._streams.add(stream)
            return stream

    def close(self) -> None:
        """Close this client's camera sessions without shutting down the Cosmos service."""
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


def _validate_seeds(seeds: tuple[int, ...], count: int) -> None:
    if (
        not isinstance(seeds, tuple)
        or len(seeds) != count
        or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
    ):
        raise ValueError("Cosmos requires one integer seed in [0, 2**31) for each opened or reset view.")


class _CosmosStream:
    def __init__(self, model: CosmosModel, cfg: CosmosModelCfg, seeds: tuple[int, ...]):
        self._model = model
        self._cfg = cfg
        self._seeds = seeds
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._shape: tuple[int, int, int] | None = None
        self._closed = False
        self._episode = 0
        self._episode_frames = 0
        self._channel = None
        self.transport: str | None = None

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor]:
        """Send one control chunk per view and return matching uint8 RGB on the controls' original device.

        Views starting or restarting an episode send one frame, the others four. A transport, model, or validation
        failure closes the session. The same chunk cannot safely be retried because the service may already have
        advanced its temporal generation state.
        """
        import torch

        with self._lock:
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
                    self._socket = connect(self._cfg.endpoint, self._cfg.timeout)
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
                        from .._cuda_ipc import SharedChannel

                        self._channel = SharedChannel(control.device, (num_views, MAX_CHUNK_FRAMES, *image_shape))
                        request.update(transport="cuda_ipc", ipc=self._channel.handles)
                    _, arrays = self._exchange(request)
                    if arrays:
                        raise ProtocolError("Cosmos open reply must not contain image arrays.")
                    self._shape = image_shape
                frames = [view.shape[0] for view in controls]
                if max(frames) > MAX_CHUNK_FRAMES:
                    raise ValueError(f"Cosmos chunks hold at most {MAX_CHUNK_FRAMES} frames.")
                request = {"op": "step", "reset_rows": reset_rows, "seeds": seeds}
                episode = self._episode
                if num_views == 1 and reset_rows and isinstance(self._cfg.prompt, (list, tuple)):
                    # A reset starts the next prompt's episode, unless the ending episode never got past its
                    # first frame, such as the capture taken before the environment's initial reset.
                    episode += 1 if self._episode_frames > 1 else 0
                    request["prompt"] = self._episode_prompt(episode)
                if self._channel is not None:
                    # Controls and images stay on the GPU, one buffer row per view; events order the processes.
                    for view, (chunk, count) in enumerate(zip(controls, frames)):
                        self._channel.control.tensor[view, :count].copy_(chunk)
                    self._channel.control_ready.record()
                    request["frames"] = frames
                    reply, generated = self._exchange(request)
                    if generated or reply.get("frames") != frames:
                        raise ProtocolError("Cosmos returned images that do not match the control chunks.")
                    self._channel.output_ready.wait()
                    images = [self._channel.output.tensor[view, :count].clone() for view, count in enumerate(frames)]
                else:
                    # Images cross the process boundary through host memory and the socket.
                    _, generated = self._exchange(request, [chunk.detach().cpu().numpy() for chunk in controls])
                    if len(generated) != num_views or any(
                        images.shape != tuple(chunk.shape) for images, chunk in zip(generated, controls)
                    ):
                        raise ProtocolError("Cosmos returned images that do not match the control chunks.")
                    images = [torch.from_numpy(view).to(device=control.device) for view in generated]
                if num_views == 1:
                    if reset_rows:
                        self._episode, self._episode_frames = episode, 0
                    self._episode_frames += frames[0]
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
        if len(self._seeds) == 1 or not isinstance(prompt, (list, tuple)):
            return self._episode_prompt(self._episode)
        return [prompt[view % len(prompt)] for view in range(len(self._seeds))]

    def _episode_prompt(self, episode: int) -> str | None:
        prompt = self._cfg.prompt
        return prompt[episode % len(prompt)] if isinstance(prompt, (list, tuple)) else prompt

    def _select_transport(self, device: torch.device) -> str:
        """Choose CUDA IPC when the service shares this GPU on this machine, else the socket, per the configuration."""
        if self._cfg.transport == "socket":
            return "socket"
        reason = None
        family, address = parse_endpoint(self._cfg.endpoint)
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
            elif capabilities.get("pci_bus_id") != _cuda_ipc.pci_bus_id(device.index or 0):
                reason = "the service uses a different GPU"
        if reason is None:
            _LOGGER.info("Cosmos keeps images on the GPU with CUDA IPC.")
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
        try:
            if self._socket is not None:
                try:
                    if notify:
                        self._exchange({"op": "close"})
                except (OSError, ValueError, RuntimeError):
                    _LOGGER.debug("Cosmos session was already unavailable during close.", exc_info=True)
                finally:
                    self._socket.close()
                    self._socket = None
        finally:
            # The service unmaps the shared memory when the session closes; then this process frees it.
            if self._channel is not None:
                self._channel.close()
                self._channel = None
            self._model._discard_stream(self)
