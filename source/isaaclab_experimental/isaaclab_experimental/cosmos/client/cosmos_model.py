# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-transfer backend connecting cameras to independently started Cosmos services."""

from __future__ import annotations

import hashlib
import logging
import math
import socket
import threading
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

from .._protocol import (
    MAX_CHUNK_FRAMES,
    SEED_LIMIT,
    ProtocolError,
    check_reply,
    connect,
    parse_endpoint,
    receive_message,
    request,
    send_message,
)
from .prompt_files import read_prompt_files

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
        if cfg.prompt_files:
            files = [cfg.prompt_files] if isinstance(cfg.prompt_files, str) else cfg.prompt_files
            cfg = cfg.replace(prompt=read_prompt_files(files), prompt_files=None)
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
        if not math.isfinite(cfg.fps) or cfg.fps <= 0:
            raise ValueError("Cosmos fps must be finite and positive.")
        if isinstance(cfg.prompt, (list, tuple)):
            if not cfg.prompt or any(not isinstance(prompt, str) or not prompt.strip() for prompt in cfg.prompt):
                raise ValueError("A Cosmos prompt list must be nonempty and contain nonempty strings.")
        elif cfg.prompt is not None and not isinstance(cfg.prompt, str):
            raise ValueError("Cosmos prompt must be a string, a list of strings, or None.")
        if cfg.generation_mode not in ("autoregressive", "single_frame"):
            raise ValueError("Cosmos generation_mode must be autoregressive or single_frame.")
        if type(cfg.decision_interval) is not int or cfg.decision_interval < 1:
            raise ValueError("Cosmos decision_interval must be a positive integer.")
        if not 0 <= cfg.augmentation_probability <= 1:
            raise ValueError("Cosmos augmentation_probability must lie in [0, 1].")
        if cfg.generation_mode == "autoregressive" and (cfg.decision_interval, cfg.augmentation_probability) != (1, 1):
            raise ValueError("Cosmos decision_interval and augmentation_probability need single_frame generation.")
        if cfg.generation_mode == "single_frame" and cfg.cameras:
            raise ValueError("Cameras sharing a Cosmos session need autoregressive generation.")
        if cfg.generation_mode == "autoregressive" and (
            type(cfg.max_episode_frames) is not int
            or cfg.max_episode_frames < 1
            or (cfg.max_episode_frames - 1) % MAX_CHUNK_FRAMES
        ):
            raise ValueError(
                "Set CosmosModelCfg.max_episode_frames to the episode's camera-frame budget (1 + 4*k), "
                "for example episode_frame_budget(env_cfg, camera.update_period)[2]."
            )
        if cfg.transport not in ("auto", "cuda_ipc", "socket"):
            raise ValueError("Cosmos transport must be auto, cuda_ipc, or socket.")
        if (
            not isinstance(cfg.cameras, (list, tuple))
            or any(not isinstance(name, str) or not name for name in cfg.cameras)
            or len(set(cfg.cameras)) != len(cfg.cameras)
        ):
            raise ValueError("Cosmos cameras sharing a session must be distinct scene camera names.")
        self._cfg = cfg
        self._lock = threading.Lock()
        self._streams: set[_CosmosStream] = set()
        self._group: _CameraGroup | None = None
        self._closed = False

    def open_stream(self, num_views: int, seeds: tuple[int, ...]) -> _CosmosStream | _ShardedStream | _CameraStream:
        """Create one camera stream; connect on its first step when image dimensions are known.

        With several service endpoints, the views are split into contiguous shares of near-equal size, one per
        service in endpoint order (see :func:`split_views`). With :attr:`CosmosModelCfg.cameras`, the stream instead
        joins the session those cameras share; the camera's processor binds it to its camera.

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
        if self._cfg.cameras:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Cosmos client model is closed.")
                # A closed group keeps its cameras' old streams; rebuilt cameras start a new session.
                if self._group is None or self._group._closed:
                    self._group = _CameraGroup(self, tuple(self._cfg.cameras))
                group = self._group
            return group.join(num_views, seeds)
        return self._open_session(num_views, seeds)

    def _open_session(self, num_views: int, seeds: tuple[int, ...]) -> _CosmosStream | _ShardedStream:
        """Create one session's stream, split across the services; see :meth:`open_stream`."""
        endpoints = service_endpoints(self._cfg.endpoint)
        shares = split_views(num_views, len(endpoints))
        with self._lock:
            if self._closed:
                raise RuntimeError("Cosmos client model is closed.")
            streams = [
                _CosmosStream(self, endpoint, seeds[share.start : share.stop], share, batched=num_views > 1)
                for endpoint, share in zip(endpoints, shares)
            ]
            self._streams.update(streams)
        return streams[0] if len(streams) == 1 else _ShardedStream(streams, shares)

    def offload(self) -> None:
        """Move every service's weights to host memory, for example while the policy trains between rollouts.

        Call it between camera captures and :meth:`activate` before the next one. Single-frame camera sessions stay
        open. Autoregressive sessions would lose their generation history, so they must be closed first. Each change
        moves the weights between host and GPU memory, and the first generations after activation compile again.

        Raises:
            RuntimeError: If the client is closed, an autoregressive session is open, or a service fails.
        """
        self._change_residency("offload")

    def activate(self) -> None:
        """Move every service's weights back to its GPU after :meth:`offload`."""
        self._change_residency("activate")

    def _change_residency(self, operation: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cosmos client model is closed.")
            streams = tuple(stream for stream in self._streams if stream._socket is not None)
        if operation == "offload" and streams and self._cfg.generation_mode != "single_frame":
            raise RuntimeError("Close autoregressive Cosmos sessions before offloading; they would lose their history.")
        for endpoint in service_endpoints(self._cfg.endpoint):
            session = next((stream for stream in streams if stream._endpoint == endpoint), None)
            if session is None:
                request(endpoint, {"op": operation}, timeout=self._cfg.timeout)
            else:
                # The service accepts the change only from the connection that owns its session.
                with session._lock:
                    session._exchange({"op": operation})

    def close(self) -> None:
        """Close this client's camera sessions without shutting down the Cosmos services."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            streams = tuple(self._streams)
            group = self._group
        if group is not None:
            group.close()
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
        or any(type(seed) is not int or not 0 <= seed < SEED_LIMIT for seed in seeds)
    ):
        raise ValueError("Cosmos requires one integer seed in [0, 2**31) for each opened or reset view.")


def _validate_resets(
    reset_rows: tuple[int, ...], seeds: tuple[int, ...], episodes: tuple[int, ...] | None, num_views: int
) -> None:
    if (
        not isinstance(reset_rows, tuple)
        or any(type(row) is not int or not 0 <= row < num_views for row in reset_rows)
        or list(reset_rows) != sorted(set(reset_rows))
    ):
        raise ValueError("Cosmos resets name distinct views in increasing order.")
    _validate_seeds(seeds, len(reset_rows))
    _validate_episodes(episodes, len(reset_rows))


def _validate_episodes(episodes: tuple[int, ...] | None, count: int) -> None:
    if episodes is not None and (
        not isinstance(episodes, tuple)
        or len(episodes) != count
        or any(type(episode) is not int or episode < 0 for episode in episodes)
    ):
        raise ValueError("Cosmos episode numbers must be one nonnegative integer per reset view.")


def _image_seed(seed: int, count: int) -> int:
    """Return the seed of a view's image ``count`` (from 0) in the episode that starts from ``seed``.

    Episode seeds of neighbouring views and episodes differ by small integers, so ``seed + count`` would give a view
    the noise of another view's or episode's image, and identical images for identical controls. Later images take
    a hash of the episode seed and the image count instead.
    """
    if count == 0:
        return seed
    digest = hashlib.blake2b(f"{seed}:{count}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % SEED_LIMIT


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
        endpoint: str,
        seeds: tuple[int, ...],
        views: range,
        *,
        batched: bool,
    ):
        """Prepare the session; it connects on its first step.

        Args:
            model: Client model that tracks this session and holds its settings.
            endpoint: Endpoint of this session's service.
            seeds: Initial seed of each view of this session.
            views: The camera's view indices this session generates; they select the views' prompts.
            batched: Whether the camera has several views. Each view then keeps one prompt for all its episodes.
        """
        self._model = model
        self._cfg = model._cfg
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
        # Single-frame generation, per view: episode seed, images generated in the episode, resets, prompt offset, and
        # the caller's episode number at the latest reset.
        self._view_seeds = list(seeds)
        self._view_frames = [0] * len(seeds)
        self._view_resets = [0] * len(seeds)
        self._view_episodes = [0] * len(seeds)
        self._view_episode_numbers = [0] * len(seeds)
        self._channel = None
        self._service_device: torch.device | None = None
        self.transport: str | None = None

    def step(
        self,
        controls: list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episodes: tuple[int, ...] | None = None,
    ) -> list[torch.Tensor]:
        """Send one control chunk per view and return matching uint8 RGB on the controls' original device.

        Views starting or restarting an episode send one frame, the others four. ``episodes`` gives each reset view's
        episode number, so a single-frame view that reset several times since its last image moves as many prompts;
        without it, each reset counts as one episode. A transport, model, or validation failure closes the session.
        The same chunk cannot safely be retried because the service may already have advanced its temporal
        generation state.
        """
        with self._lock:
            return self._receive(self._send(controls, reset_rows, seeds, episodes))

    def _send(
        self,
        controls: list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episodes: tuple[int, ...] | None = None,
    ) -> _SentStep:
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
            _validate_resets(reset_rows, seeds, episodes, num_views)
            control = controls[0]
            image_shape = tuple(control.shape[1:])
            if any(tuple(view.shape[1:]) != image_shape or view.device != control.device for view in controls):
                raise ValueError("Cosmos views must share one image size and device.")
            if self._shape is not None and image_shape != self._shape:
                raise ValueError("Cosmos image size cannot change within a camera stream.")
            frames = [view.shape[0] for view in controls]
            if max(frames) > MAX_CHUNK_FRAMES:
                raise ValueError(f"Cosmos chunks hold at most {MAX_CHUNK_FRAMES} frames.")
            message = {"op": "step", "reset_rows": reset_rows, "seeds": seeds}
            episode = self._episode
            if self._single_frame:
                message.update(self._single_frame_step(frames, reset_rows, seeds, episodes))
            elif not self._batched and reset_rows and isinstance(self._cfg.prompt, (list, tuple)):
                # A reset starts the next prompt's episode, unless the ending episode never got past its
                # first frame, such as the capture taken before the environment's initial reset.
                episode += 1 if self._episode_frames > 1 else 0
                message["prompt"] = self._episode_prompt(episode)
            if self._socket is None:
                self._open(control.device, image_shape, message)
            arrays = []
            if self._channel is not None:
                # Controls and images stay on the GPU, one buffer row per view; events order the processes.
                for view, (chunk, count) in enumerate(zip(controls, frames)):
                    self._channel.control.tensor[view, :count].copy_(chunk)
                self._channel.control_ready.record()
                message["frames"] = frames
            else:
                # Images cross the process boundary through host memory and the socket.
                arrays = [chunk.detach().cpu().numpy() for chunk in controls]
            send_message(self._socket, message, arrays)
            return _SentStep(frames, control.device, episode, bool(reset_rows))
        except Exception:
            self._release(notify=False)
            raise

    def _open(self, device: torch.device, image_shape: tuple[int, int, int], step: dict) -> None:
        """Connect and open the session for the first step, whose resets become the session's opening state.

        The service starts every view fresh, so the views ``step`` resets open with its seeds and prompts, and the
        step itself is sent without resets.

        Args:
            device: Device of the controls.
            image_shape: Control image shape ``(height, width, 3)``.
            step: The first step's message; its ``reset_rows``, ``seeds`` and ``prompt`` are consumed.
        """
        num_views = len(self._seeds)
        seeds = list(self._seeds)
        prompt = step.pop("prompt", self._session_prompt())
        for row, seed in zip(step["reset_rows"], step["seeds"]):
            seeds[row] = seed
        step.update(reset_rows=(), seeds=())
        self._socket = connect(self._endpoint, self._cfg.timeout)
        message = {
            "op": "open",
            "num_views": num_views,
            "seeds": tuple(seeds),
            "prompt": prompt,
            "height": image_shape[0],
            "width": image_shape[1],
            # A single-frame image is a one-frame episode.
            "max_episode_frames": 1 if self._single_frame else self._cfg.max_episode_frames,
            "fps": self._cfg.fps,
        }
        self.transport = self._select_transport(device)
        if self.transport == "cuda_ipc":
            from .. import _cuda_ipc

            self._channel = _cuda_ipc.SharedChannel(
                device, (num_views, MAX_CHUNK_FRAMES, *image_shape), writes="control"
            )
            # The service opens the controls on this GPU, which it finds by its PCI bus ID.
            ipc = {**self._channel.handles, "pci_bus_id": _cuda_ipc.pci_bus_id(device.index or 0)}
            message.update(transport="cuda_ipc", ipc=ipc)
        try:
            reply, arrays = self._exchange(message)
        except RuntimeError:
            # The service answered and rejected the session, so it holds none of this camera's buffers.
            if self._channel is not None:
                self._channel.close(peer_released=True)
                self._channel = None
            raise
        if arrays:
            raise ProtocolError("Cosmos open reply must not contain image arrays.")
        if self._channel is not None:
            # The service writes generated images into a buffer on its own GPU.
            self._channel.open(self._service_device, reply.get("ipc"))
        self._shape = image_shape

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
            if not self._batched and not self._single_frame:
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

    @property
    def _single_frame(self) -> bool:
        return self._cfg.generation_mode == "single_frame"

    def _single_frame_step(
        self,
        frames: list[int],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episodes: tuple[int, ...] | None = None,
    ) -> dict:
        """Return a single-frame step's service request fields: every view restarts with one frame.

        ``reset_rows`` names the views starting a new environment episode, with its seed and, from ``episodes``, its
        episode number. A view's first image of the episode uses that seed and later images a seed derived from it
        (see :func:`_image_seed`). A view moves one prompt for each episode that ended since its last image, also for
        episodes that generated no image and episodes before the first image, except at the view's first reset while
        it has at most one image: the environment's initial reset, which can follow one capture.
        """
        if any(count != 1 for count in frames):
            raise ValueError("Single-frame Cosmos sends one frame per view.")
        prompt = self._cfg.prompt
        new_prompt = False
        for index, (row, seed) in enumerate(zip(reset_rows, seeds)):
            ended = 1 if episodes is None else episodes[index] - self._view_episode_numbers[row]
            if not self._view_resets[row] and self._view_frames[row] <= 1:
                ended -= 1
            if isinstance(prompt, (list, tuple)) and ended > 0:
                self._view_episodes[row] += ended
                new_prompt = True
            if episodes is not None:
                self._view_episode_numbers[row] = episodes[index]
            self._view_resets[row] += 1
            self._view_seeds[row], self._view_frames[row] = seed, 0
        # Every step restarts every view. On the first step the restarts open the session, so episodes that ended
        # before it also select its seeds and prompts.
        fields = {
            "reset_rows": tuple(range(len(frames))),
            "seeds": tuple(_image_seed(seed, count) for seed, count in zip(self._view_seeds, self._view_frames)),
        }
        if new_prompt:
            prompts = [
                prompt[(view + episode) % len(prompt)] for view, episode in zip(self._views, self._view_episodes)
            ]
            # The service takes one text for a one-view session, also for one view of a split camera.
            fields["prompt"] = prompts if len(self._views) > 1 else prompts[0]
        self._view_frames = [count + 1 for count in self._view_frames]
        return fields

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
        can access this GPU's memory directly, and the service must see this GPU. Choosing it records this process's
        index of the service's GPU.
        """
        import torch

        if self._cfg.transport == "socket":
            return "socket"
        reason = None
        if device.type != "cuda":
            reason = "the camera images are not on a CUDA device"
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
            elif _cuda_ipc.pci_bus_id(device.index or 0) not in capabilities.get("visible_gpus", ()):
                # The service opens the camera's buffers on the camera's GPU, so it must see that GPU too.
                reason = "the service cannot see this camera's GPU"
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
        self,
        controls: list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episodes: tuple[int, ...] | None = None,
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
                if not isinstance(controls, list) or len(controls) != num_views:
                    raise ValueError("Cosmos needs one control chunk per view.")
                _validate_resets(reset_rows, seeds, episodes, num_views)
                for stream, share in zip(self._streams, self._shares):
                    resets = [index for index, row in enumerate(reset_rows) if row in share]
                    sent.append(
                        stream._send(
                            controls[share.start : share.stop],
                            tuple(reset_rows[index] - share.start for index in resets),
                            tuple(seeds[index] for index in resets),
                            None if episodes is None else tuple(episodes[index] for index in resets),
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


def _scene_cameras() -> dict[str, Any]:
    """Return the scene's cameras by name, through the simulation's scene data provider."""
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    return sim.get_scene_data_provider().get_camera_sensors() if sim is not None else {}


class _CameraGroup:
    """Cameras that share one session; every capture generates all their environments as one batch.

    Each camera's processor queues its own controls and steps its :class:`_CameraStream` once its chunk is complete.
    The camera that adds the last chunk of a capture runs the batched step, and every camera gets its own views of
    it. A camera that steps before the others reads their data, which renders them and runs their processors, unless
    they were already rendered for this capture, as an eager scene update renders every camera before running any
    processor: their processors then add their chunks next, and the last one publishes the earlier cameras' images.
    All cameras thus publish images of the same capture, so they must capture at the same steps. A camera whose
    images are published later must have its Cosmos processor last, since its own later processors already ran. A
    failure closes the session for every camera.
    """

    def __init__(self, model: CosmosModel, cameras: tuple[str, ...]):
        self._model = model
        self._cameras = cameras
        self._num_views: int | None = None
        self._streams: list[_CameraStream | None] = [None] * len(cameras)
        self._session: _CosmosStream | _ShardedStream | None = None
        self._chunks: dict[int, tuple[list[torch.Tensor], tuple[int, ...], tuple[int, ...]]] = {}
        self._images: dict[int, list[torch.Tensor]] | None = None
        # Cameras that stepped before the cameras rendered with them added their chunks; they get their images later.
        self._waiting: set[int] = set()
        self._closed = False

    def join(self, num_views: int, seeds: tuple[int, ...]) -> _CameraStream:
        """Return a camera's stream into the shared session; its camera binds it later."""
        if self._num_views is None:
            self._num_views = num_views
        elif num_views != self._num_views:
            raise ValueError("Cameras sharing a Cosmos session need the same number of environments.")
        return _CameraStream(self, seeds)

    def bind(self, stream: _CameraStream, name: str) -> None:
        """Assign ``stream`` the views of the scene camera ``name``."""
        if name not in self._cameras:
            raise ValueError(
                f"Camera {name!r} is not one of the cameras sharing this Cosmos session {list(self._cameras)}."
            )
        index = self._cameras.index(name)
        if self._streams[index] not in (None, stream):
            raise ValueError(f"Camera {name!r} already joined its Cosmos session.")
        self._streams[index] = stream
        stream.index = index

    def step(
        self, stream: _CameraStream, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor] | None:
        """Add a camera's chunk and return this camera's images, or None if they are published later.

        Cameras that have not added their chunk are updated by reading their data, unless they were rendered with
        this camera; this camera then waits, and the camera adding the last chunk publishes its images.
        """
        if self._closed:
            raise RuntimeError("The Cosmos session shared by these cameras is closed or failed.")
        if stream.index is None:
            raise RuntimeError("A camera sharing a Cosmos session must be bound to its camera before it steps.")
        if stream.index in self._chunks:
            raise RuntimeError("A camera stepped twice before every camera of its Cosmos session added its chunk.")
        self._chunks[stream.index] = (controls, reset_rows, seeds)
        try:
            cameras = _scene_cameras()
            if self._session is None:
                periods = {name: cameras[name].cfg.update_period for name in self._cameras if name in cameras}
                if len(set(periods.values())) > 1:
                    raise ValueError(f"Cameras sharing a Cosmos session need the same update period, got {periods}.")
            for index, name in enumerate(self._cameras):
                if self._images is not None or index in self._chunks:
                    continue
                camera = cameras.get(name)
                if camera is None:
                    raise ValueError(f"Scene camera {name!r} of the Cosmos session does not exist.")
                if stream.publish is not None and self._rendered_together(cameras, stream.index, camera):
                    # Its processors add its chunk after this camera's; reading its data would render it again.
                    continue
                # Reading the camera's data updates it; its processor then adds the camera's chunk to this one.
                camera.data
                if self._images is None and index not in self._chunks:
                    raise RuntimeError(
                        f"Camera {name!r} did not capture with the other cameras of its Cosmos session; cameras "
                        "sharing a session need the same update period, and each must be read every capture."
                    )
            if self._images is None:
                if len(self._chunks) < len(self._cameras):
                    name = self._cameras[stream.index]
                    if getattr(cameras[name].cfg.tensor_processors[-1], "camera", None) != name:
                        raise ValueError(
                            f"Camera {name!r} runs tensor processors after its Cosmos processor, but its images are "
                            "published when a later camera of the capture generates. Put the Cosmos processor last "
                            "or use lazy sensor updates (scene.lazy_sensor_update=True)."
                        )
                    # This camera's processor reuses its control queue before the session generates.
                    self._chunks[stream.index] = ([chunk.clone() for chunk in controls], reset_rows, seeds)
                    self._waiting.add(stream.index)
                    return None
                self._images = self._generate()
                for index in sorted(self._waiting):
                    self._streams[index].publish(self._images.pop(index))
                self._waiting.clear()
            images = self._images.pop(stream.index)
            if not self._images:
                self._images = None
                self._chunks.clear()
            return images
        except Exception:
            self.close()
            raise

    def _rendered_together(self, cameras: dict[str, Any], index: int, camera: Any) -> bool:
        """Whether ``camera`` was already rendered for the capture camera ``index`` is processing.

        Preparing a capture advances a camera's frame counters, so cameras rendered for the same capture have equal
        counters. This reads them on the host, once per chunk, where the session's generation waits anyway.
        """
        import torch

        own = cameras.get(self._cameras[index])
        return own is not None and torch.equal(own.frame.torch, camera.frame.torch)

    def close(self) -> None:
        """Close the shared session; every camera's stream then fails. Repeated calls are safe."""
        self._closed = True
        self._chunks.clear()
        self._images = None
        self._waiting.clear()
        if self._session is not None:
            self._session.close()

    def _generate(self) -> dict[int, list[torch.Tensor]]:
        """Step the shared session with every camera's chunk; camera ``k`` has rows ``k * n`` up to ``k * n + n``."""
        num_views = self._num_views
        if self._session is None:
            seeds = tuple(seed for stream in self._streams for seed in stream.seeds)
            self._session = self._model._open_session(num_views * len(self._cameras), seeds)
        controls, reset_rows, reset_seeds = [], [], []
        for index in range(len(self._cameras)):
            camera_controls, camera_rows, camera_seeds = self._chunks[index]
            controls.extend(camera_controls)
            reset_rows.extend(index * num_views + row for row in camera_rows)
            reset_seeds.extend(camera_seeds)
        images = self._session.step(controls, tuple(reset_rows), tuple(reset_seeds))
        return {index: images[index * num_views : (index + 1) * num_views] for index in range(len(self._cameras))}


class _CameraStream:
    """One camera's stream into the session it shares with other cameras; see :class:`_CameraGroup`."""

    def __init__(self, group: _CameraGroup, seeds: tuple[int, ...]):
        self._group = group
        self.seeds = seeds
        self.index: int | None = None
        self.publish: Callable[[list[torch.Tensor]], None] | None = None

    def bind(self, camera: str, publish: Callable[[list[torch.Tensor]], None] | None = None) -> None:
        """Take the views of the scene camera ``camera``, whose processor owns this stream.

        Args:
            camera: Name of the scene camera.
            publish: Called with this camera's images when the session generates them after :meth:`step` returned
                None. Without it, a step always reads the other cameras' data.
        """
        self._group.bind(self, camera)
        self.publish = publish

    def step(
        self, controls: list[torch.Tensor], reset_rows: tuple[int, ...], seeds: tuple[int, ...]
    ) -> list[torch.Tensor] | None:
        """Generate this camera's views with the other cameras of its session; see :meth:`_CameraGroup.step`."""
        return self._group.step(self, controls, reset_rows, seeds)

    def close(self) -> None:
        """Close the shared session. Repeated calls are safe."""
        self._group.close()
