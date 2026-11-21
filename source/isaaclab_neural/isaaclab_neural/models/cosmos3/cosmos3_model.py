# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 behind the :class:`~isaaclab_neural.world_model.WorldModel` interface, served by separate processes.

The Cosmos service (``isaaclab-cosmos-world-model-server``) loads the Cosmos Framework in its own environment, so
Isaac Lab does not install it. This client maps joint targets to controls with the task's
:class:`~.cosmos3_cfg.JointMappingCfg` and exchanges controls and frames with one service, or with several that each
generate a share of the environments.
"""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_experimental.cosmos._protocol import (
    check_reply,
    connect,
    parse_endpoint,
    receive_message,
    request,
    send_message,
)
from isaaclab_experimental.cosmos.client.cosmos_model import service_endpoints, split_views

from isaaclab.utils.configclass import instantiate

from ...world_model import WorldModel, WorldModelAttribute, WorldModelSpec

if TYPE_CHECKING:
    from .cosmos3_cfg import Cosmos3Cfg

_LOGGER = logging.getLogger(__name__)


class Cosmos3Model(WorldModel):
    """Client of one or more Cosmos3 world-model services; the model predicts RGB of one view.

    With several services (:attr:`~.Cosmos3Cfg.endpoint` is a list), the environments are split into contiguous
    shares of near-equal size, one per service in list order. Each service generates its share as one batch, all
    services at the same time. On a service's machine, chunks arrive through a GPU buffer the service writes (CUDA IPC)
    and are copied to :attr:`device` without the host; otherwise they arrive through the socket.
    """

    cfg: Cosmos3Cfg

    def __init__(self, cfg: Cosmos3Cfg, num_envs: int, device: str):
        endpoints = service_endpoints(cfg.endpoint)
        if not endpoints:
            raise ValueError("Cosmos3 needs at least one service endpoint.")
        for endpoint in endpoints:
            parse_endpoint(endpoint)
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("Cosmos3 endpoints must name distinct services.")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.mapping = None
        self._services = [
            _Cosmos3Service(endpoint, share, cfg, device)
            for endpoint, share in zip(endpoints, split_views(num_envs, len(endpoints)))
        ]
        self._spec: WorldModelSpec | None = None
        self._inputs: dict[str, torch.Tensor] = {}

    @property
    def transports(self) -> list[str | None]:
        """``"cuda_ipc"`` or ``"socket"`` for each service in endpoint order, once the session is open."""
        return [service.transport for service in self._services]

    def spec(self) -> WorldModelSpec:
        """Read the chunk length, frame rate and image size from the services, which must agree."""
        if self._spec is None:
            capabilities = [service.status() for service in self._services]
            keys = ("chunk_length", "fps", "image_size")
            if any([c[key] for key in keys] != [capabilities[0][key] for key in keys] for c in capabilities):
                raise RuntimeError("The Cosmos3 services differ in chunk length, frame rate or image size.")
            limits = [c["max_envs"] for c in capabilities]
            self._spec = WorldModelSpec(
                chunk_length=capabilities[0]["chunk_length"],
                fps=capabilities[0]["fps"],
                image_size=tuple(capabilities[0]["image_size"]),
                max_envs=None if None in limits else sum(limits),
            )
        return self._spec

    def reset(
        self, env_ids: list[int], images: dict[str, dict[str, torch.Tensor]], prompts: list[str], seeds: list[int]
    ) -> None:
        """Send the start images and prompts of the selected environments to their services."""
        if self._services[0].connection is None:
            self._open()
        (outputs,) = images.values()
        start = outputs["rgb"].cpu().numpy()
        requests = []
        for service in self._services:
            rows = [row for row, env in enumerate(env_ids) if env in service.share]
            if rows:
                message = {
                    "op": "reset",
                    "env_ids": [env_ids[row] - service.share.start for row in rows],
                    "prompts": [prompts[row] for row in rows],
                    "seeds": [seeds[row] for row in rows],
                }
                requests.append((service, message, start[rows], None))
        self._exchange(requests)

    def step_controls(self, controls: np.ndarray) -> torch.Tensor:
        """Predict the next chunk from normalized ``unified_v1`` controls shaped ``(N, T, 59)``.

        Replay of recorded controls uses this directly; the manager goes through :meth:`step`.
        """
        spec = self.spec()
        controls = np.ascontiguousarray(controls, dtype=np.float32)
        frames = torch.empty(
            (self.num_envs, spec.chunk_length, *spec.image_size, 3), dtype=torch.uint8, device=self.device
        )
        requests = []
        for service in self._services:
            share = slice(service.share.start, service.share.stop)
            requests.append((service, {"op": "step"}, controls[share], frames[share]))
        self._exchange(requests)
        return frames

    def write(self, name: str, value: torch.Tensor) -> None:
        """Keep the chunk's joint targets or the joint state before it for :meth:`step`."""
        self._inputs[name] = value

    def step(self) -> dict[str, dict[str, torch.Tensor]]:
        """Convert the written joint targets to controls with the task's joint mapping, then predict the chunk.

        The mapping runs on the host and the controls cross to the services there, so the targets are read once per
        chunk.
        """
        if self.mapping is None:
            self.mapping = instantiate(self.cfg.joint_mapping)
        targets = self._inputs[WorldModelAttribute.JOINT_TARGETS].cpu().numpy()
        start = self._inputs[WorldModelAttribute.START_JOINT_STATE].cpu().numpy()
        controls = np.stack([self.mapping.encode(targets[i], start[i]) for i in range(self.num_envs)])
        (view,) = self.spec().views
        return {view: {"rgb": self.step_controls(controls)}}

    def close(self) -> None:
        """Close the sessions; the services keep the weights loaded for the next client."""
        for service in self._services:
            service.release(notify=True)

    def _open(self) -> None:
        spec = self.spec()
        try:
            for service in self._services:
                service.open(spec)
        except Exception:
            # Disconnecting without a close tells each service that this process holds none of its buffers.
            for service in self._services:
                service.release(notify=False)
            raise

    def _exchange(self, requests: list[tuple[_Cosmos3Service, dict, np.ndarray, torch.Tensor | None]]) -> None:
        """Send every request before reading any reply, so the services work at the same time; each reply's frames
        fill the request's output, if it has one.

        A failure closes every session: the services may already have advanced their environments.
        """
        if any(service.connection is None for service in self._services):
            raise RuntimeError("The Cosmos world-model session is closed or failed; reset the environments again.")
        awaiting = set()
        try:
            for service, message, array, _ in requests:
                service.send(message, array)
                awaiting.add(service)
            for service, _, _, output in requests:
                service.receive(output)
                awaiting.discard(service)
        except Exception:
            for service in self._services:
                # A service still generating cannot take a close message; it closes the session when the connection
                # drops.
                service.release(notify=service not in awaiting)
            raise


class _Cosmos3Service:
    """The session of one service: a contiguous share of the environments, its connection and, with CUDA IPC, the
    service's output buffer."""

    def __init__(self, endpoint: str, share: range, cfg: Cosmos3Cfg, device: str):
        self.endpoint = endpoint
        self.share = share
        self.cfg = cfg
        self.device = device
        self.transport: str | None = None
        self.capabilities: dict = {}
        self.connection: socket.socket | None = None
        self._output = None
        self._output_ready = None
        # Recorded after each copy out of the service's buffer, which the next request may overwrite.
        self._output_copied: torch.cuda.Event | None = None

    def status(self) -> dict:
        """Read the service's capabilities."""
        reply, _ = request(self.endpoint, {"op": "status"}, timeout=self.cfg.timeout)
        capabilities = reply.get("capabilities", {})
        if capabilities.get("kind") != "world_model":
            raise RuntimeError(f"The Cosmos service at {self.endpoint} does not serve a world model.")
        self.capabilities = capabilities
        return capabilities

    def open(self, spec: WorldModelSpec) -> None:
        """Connect and open a session for this share, mapping the service's output buffer for CUDA IPC."""
        self.connection = connect(self.endpoint, self.cfg.timeout)
        self.transport, service_gpu = self._select_transport()
        num_envs = len(self.share)
        send_message(self.connection, {"op": "open", "num_envs": num_envs, "transport": self.transport})
        reply = receive_message(self.connection)[0]
        check_reply(reply)
        if self.transport == "cuda_ipc":
            from isaaclab_experimental.cosmos import _cuda_ipc

            handles = reply.get("ipc")
            if not isinstance(handles, dict) or set(handles) != {"output", "output_ready"}:
                raise RuntimeError("The Cosmos world model did not return its output buffer and event.")
            shape = (num_envs, spec.chunk_length, *spec.image_size, 3)
            # The buffer stays on the service's GPU; the event is imported on this one, whose stream waits for it.
            self._output = _cuda_ipc.SharedBuffer(service_gpu, shape, handles["output"])
            self._output_ready = _cuda_ipc.SharedEvent(torch.device(self.device), handles["output_ready"])

    def send(self, message: dict, array: np.ndarray) -> None:
        """Send one request with one array, once the last copy out of the service's buffer finished."""
        if self._output_copied is not None:
            self._output_copied.synchronize()
        send_message(self.connection, message, [array])

    def receive(self, output: torch.Tensor | None) -> None:
        """Read the reply and copy its frames into ``output`` on :attr:`device`, if given.

        With CUDA IPC, the frames are read from the service's GPU buffer instead of the reply.
        """
        reply, arrays = receive_message(self.connection)
        check_reply(reply)
        if output is None:
            return
        if self._output is not None:
            if arrays:
                raise RuntimeError("The Cosmos world model sent frames through the socket in a CUDA IPC session.")
            self._output_ready.wait()
            # From the service's GPU the frames copy directly to this one; the buffer is reused for the next chunk.
            output.copy_(self._output.tensor)
            if self._output_copied is None:
                self._output_copied = torch.cuda.Event()
            self._output_copied.record(torch.cuda.current_stream(self.device))
            return
        if len(arrays) != 1 or arrays[0].dtype != np.uint8 or arrays[0].shape != tuple(output.shape):
            raise RuntimeError("The Cosmos world model must reply with one uint8 frame array for its environments.")
        output.copy_(torch.from_numpy(arrays[0]))

    def release(self, *, notify: bool) -> None:
        """Release the service's buffer and event, then close the session (``notify``) and the connection.

        The mapping goes first, so the service can free the buffer once the session closes or the connection ends.
        """
        if self._output_copied is not None:
            self._output_copied.synchronize()
            self._output_copied = None
        for shared in (self._output, self._output_ready):
            if shared is not None:
                shared.close()
        self._output = self._output_ready = None
        if self.connection is None:
            return
        try:
            if notify:
                send_message(self.connection, {"op": "close"})
                receive_message(self.connection)
        except (OSError, ValueError):
            _LOGGER.debug("The Cosmos world-model session was already unavailable during close.", exc_info=True)
        finally:
            self.connection.close()
            self.connection = None

    def _select_transport(self) -> tuple[str, torch.device | None]:
        """Choose CUDA IPC or the socket, per :attr:`~.Cosmos3Cfg.transport`, and the service's GPU for CUDA IPC.

        CUDA IPC needs the service on this machine, on this GPU or on another one this process can see and that can
        access this GPU's memory directly.
        """
        if self.cfg.transport == "socket":
            return "socket", None
        capabilities = self.capabilities
        device = torch.device(self.device)
        reason, service_gpu = None, None
        if device.type != "cuda":
            reason = "the simulation does not run on a CUDA device"
        else:
            from isaaclab_experimental.cosmos import _cuda_ipc

            if not _cuda_ipc.available():
                reason = "CUDA IPC is not available on this platform"
            elif "cuda_ipc" not in capabilities.get("transports", ()):
                reason = "the service does not offer CUDA IPC"
            elif capabilities.get("host_id") is None or capabilities["host_id"] != _cuda_ipc.host_id():
                # A local endpoint can be a tunnel; the other machine's GPU can have a PCI bus ID this one uses.
                reason = "the service runs on another machine"
            else:
                index = _cuda_ipc.device_by_bus_id(capabilities.get("pci_bus_id"))
                if index is None:
                    reason = "this process cannot see the service's GPU"
                elif not _cuda_ipc.can_access_peer(device.index or 0, index):
                    reason = "this GPU and the service's GPU cannot access each other's memory directly"
                else:
                    service_gpu = torch.device("cuda", index)
        if reason is None:
            return "cuda_ipc", service_gpu
        if self.cfg.transport == "cuda_ipc":
            raise RuntimeError(f"Cosmos transport cuda_ipc is unavailable: {reason}.")
        _LOGGER.info("The Cosmos world model sends frames through the socket because %s.", reason)
        return "socket", None
