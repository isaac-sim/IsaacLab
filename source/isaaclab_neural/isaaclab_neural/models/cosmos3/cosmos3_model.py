# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 behind the :class:`~isaaclab_neural.world_model.WorldModel` interface, served by a separate process.

The Cosmos service (``isaaclab-cosmos-world-model-server``) loads the Cosmos Framework in its own environment, so
Isaac Lab does not install it. This client maps joint targets to controls with the task's
:class:`~.cosmos3_cfg.JointMappingCfg` and exchanges controls and frames with the service.
"""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab_experimental.cosmos._protocol import check_reply, connect, receive_message, request, send_message

from isaaclab.utils.configclass import instantiate

from ...world_model import WorldModel, WorldModelSpec

if TYPE_CHECKING:
    from .cosmos3_cfg import Cosmos3Cfg

_LOGGER = logging.getLogger(__name__)


class Cosmos3Model(WorldModel):
    """Client of a Cosmos3 world-model service; one session serves all environments, and the model predicts one view.

    On the service's machine, chunks arrive through a GPU buffer the service writes (CUDA IPC) and are copied to
    :attr:`device` without the host; otherwise they arrive through the socket.
    """

    cfg: Cosmos3Cfg

    def __init__(self, cfg: Cosmos3Cfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.mapping = None
        self.transport: str | None = None
        """``"cuda_ipc"`` or ``"socket"``, once the session is open."""
        self._capabilities: dict = {}
        self._spec: WorldModelSpec | None = None
        self._socket: socket.socket | None = None
        self._output = None
        self._output_ready = None

    def spec(self) -> WorldModelSpec:
        """Read the chunk length, frame rate and image size from the service."""
        if self._spec is None:
            reply, _ = request(self.cfg.endpoint, {"op": "status"}, timeout=self.cfg.timeout)
            capabilities = reply.get("capabilities", {})
            if capabilities.get("kind") != "world_model":
                raise RuntimeError(f"The Cosmos service at {self.cfg.endpoint} does not serve a world model.")
            self._capabilities = capabilities
            self._spec = WorldModelSpec(
                chunk_length=capabilities["chunk_length"],
                fps=capabilities["fps"],
                image_size=tuple(capabilities["image_size"]),
                max_envs=capabilities["max_envs"],
            )
        return self._spec

    def reset(self, env_ids: list[int], images: dict[str, torch.Tensor], prompts: list[str], seeds: list[int]) -> None:
        """Send the start images and prompts of the selected environments."""
        if self._socket is None:
            self._open()
        (start,) = images.values()
        message = {"op": "reset", "env_ids": list(env_ids), "prompts": list(prompts), "seeds": list(seeds)}
        self._exchange(message, start.cpu().numpy(), frames=False)

    def step_controls(self, controls: np.ndarray) -> torch.Tensor:
        """Predict the next chunk from normalized ``unified_v1`` controls shaped ``(N, T, 59)``.

        Replay of recorded controls uses this directly; the manager goes through :meth:`step`.
        """
        return self._exchange({"op": "step"}, np.ascontiguousarray(controls, dtype=np.float32), shared=True)

    def step(self, joint_targets: torch.Tensor, start_joint_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert joint targets to controls with the task's joint mapping, then predict the chunk.

        The mapping runs on the host and the controls cross to the service there, so the targets are read once per
        chunk.
        """
        if self.mapping is None:
            self.mapping = instantiate(self.cfg.joint_mapping)
        targets, start = joint_targets.cpu().numpy(), start_joint_state.cpu().numpy()
        controls = np.stack([self.mapping.encode(targets[i], start[i]) for i in range(self.num_envs)])
        (view,) = self.spec().views
        return {view: self.step_controls(controls)}

    def close(self) -> None:
        """Close the session; the service keeps the weights loaded for the next client."""
        self._release(notify=True)

    def _open(self) -> None:
        spec = self.spec()
        self._socket = connect(self.cfg.endpoint, self.cfg.timeout)
        try:
            self.transport, service_gpu = self._select_transport()
            send_message(self._socket, {"op": "open", "num_envs": self.num_envs, "transport": self.transport})
            reply = receive_message(self._socket)[0]
            check_reply(reply)
            if self.transport == "cuda_ipc":
                from isaaclab_experimental.cosmos import _cuda_ipc

                handles = reply.get("ipc")
                if not isinstance(handles, dict) or set(handles) != {"output", "output_ready"}:
                    raise RuntimeError("The Cosmos world model did not return its output buffer and event.")
                shape = (self.num_envs, spec.chunk_length, *spec.image_size, 3)
                # The buffer stays on the service's GPU; the event is imported on this one, whose stream waits for it.
                self._output = _cuda_ipc.SharedBuffer(service_gpu, shape, handles["output"])
                self._output_ready = _cuda_ipc.SharedEvent(torch.device(self.device), handles["output_ready"])
        except Exception:
            # Disconnecting without a close tells the service that this process holds none of its buffers.
            self._release(notify=False)
            raise

    def _select_transport(self) -> tuple[str, torch.device | None]:
        """Choose CUDA IPC or the socket, per :attr:`~.Cosmos3Cfg.transport`, and the service's GPU for CUDA IPC.

        CUDA IPC needs the service on this machine, on this GPU or on another one this process can see and that can
        access this GPU's memory directly.
        """
        if self.cfg.transport == "socket":
            return "socket", None
        capabilities = self._capabilities
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

    def _exchange(
        self, message: dict, array: np.ndarray, *, shared: bool = False, frames: bool = True
    ) -> torch.Tensor | None:
        """Send one request with one array and return the frames of the reply on :attr:`device`, if it has ``frames``.

        With ``shared`` and CUDA IPC, the frames are read from the service's GPU buffer instead of the reply. A
        failure closes the session: the service may already have advanced the model's state.
        """
        if self._socket is None:
            raise RuntimeError("The Cosmos world-model session is closed or failed; reset the environments again.")
        try:
            send_message(self._socket, message, [array])
            reply, arrays = receive_message(self._socket)
            check_reply(reply)
            if not frames:
                return None
            if shared and self._output is not None:
                if arrays:
                    raise RuntimeError("The Cosmos world model sent frames through the socket in a CUDA IPC session.")
                self._output_ready.wait()
                # From the service's GPU the frames copy directly to this one; the buffer is reused for the next chunk.
                return self._output.tensor.to(self.device, copy=True)
            if len(arrays) != 1 or arrays[0].dtype != np.uint8:
                raise RuntimeError("The Cosmos world model must reply with one uint8 frame array.")
            return torch.from_numpy(arrays[0]).to(self.device)
        except Exception:
            self._release(notify=False)
            raise

    def _release(self, *, notify: bool) -> None:
        """Release the service's buffer and event, then close the session (``notify``) and the connection.

        The mapping goes first, so the service can free the buffer once the session closes or the connection ends.
        """
        for shared in (self._output, self._output_ready):
            if shared is not None:
                shared.close()
        self._output = self._output_ready = None
        if self._socket is None:
            return
        try:
            if notify:
                send_message(self._socket, {"op": "close"})
                receive_message(self._socket)
        except (OSError, ValueError):
            _LOGGER.debug("The Cosmos world-model session was already unavailable during close.", exc_info=True)
        finally:
            self._socket.close()
            self._socket = None
