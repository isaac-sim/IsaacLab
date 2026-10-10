# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local, bounded RPC messages shared by the isolated model and simulation processes."""

from __future__ import annotations

import io
import pickle
import socket
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class RunConfig:
    """Configuration written by the launcher and consumed by both processes."""

    model_path: str
    backbone_path: str
    run_dir: str
    socket_path: str
    task: str = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor"
    seed: int = 42
    timesteps: int = 2
    rollouts: int = 2
    learning_epochs: int = 2
    generation_steps: int = 2
    sigma: float = 0.05
    learning_rate: float = 1e-6
    max_tokens: int = 1024
    episode_steps: int | None = None
    resume: str | None = None
    mode: Literal["train", "inference"] = "train"
    rpc_timeout: float = 300.0
    instruction: str = "Stack the red cube on the blue cube, then stack the green cube on the red cube."

    def validate(self) -> None:
        """Reject invalid configurations before starting either process."""
        if min(self.timesteps, self.rollouts, self.learning_epochs, self.generation_steps, self.max_tokens) < 1:
            raise ValueError("Step counts and token capacity must be positive")
        if self.rollouts < 2 or self.timesteps % self.rollouts:
            raise ValueError("Use complete rollouts with at least two transitions for native GAE normalization")
        if not self.sigma > 0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite and positive")
        if not self.learning_rate > 0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.rpc_timeout <= 0 or self.episode_steps is not None and self.episode_steps < 1:
            raise ValueError("Timeout and episode length must be positive")
        if "nvidia/Cosmos-Reason2" not in self.backbone_path:
            raise ValueError("Keep the nvidia/Cosmos-Reason2 substring in the backbone path")


@dataclass
class Observation:
    """Two RGB images and XYZ [m], rotation vector [rad], finger positions [m]."""

    table_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray

    def validate(self) -> None:
        """Check the wire contract before using an observation."""
        if self.state.shape != (8,) or not np.isfinite(self.state).all():
            raise ValueError("Expected eight finite state components")
        for rgb in (self.table_rgb, self.wrist_rgb):
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Expected H x W x 3 uint8 RGB images")


@dataclass
class Request:
    """Simulation command, with seven control values for a step."""

    command: Literal["reset", "step", "close"]
    action: np.ndarray | None = None

    def validate(self) -> None:
        """Reject malformed commands and non-finite robot controls."""
        if self.command not in ("reset", "step", "close"):
            raise ValueError(f"Unknown command: {self.command}")
        if self.command == "step":
            if self.action is None or self.action.shape != (7,) or not np.isfinite(self.action).all():
                raise ValueError("Expected exactly seven finite robot controls")
            if np.any(np.abs(self.action[:6]) > 0.100001) or self.action[6] not in (-1, 1):
                raise ValueError("Arm controls must be in [-0.1, 0.1] and gripper must be binary")


@dataclass
class Transition:
    """Reset-aware reply; final_state is captured before automatic reset."""

    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    final_state: np.ndarray | None = None
    executed_action: np.ndarray | None = None
    motion: float = 0.0
    """End-effector displacement during the physics step [m], before any reset."""


class RpcConnection:
    """Length-framed RPC over a private Unix socket; all reads and writes have deadlines.

    Pickle is used only between launcher-owned local processes in a private directory.
    """

    def __init__(self, connection: socket.socket, timeout: float):
        self.connection = connection
        self.connection.settimeout(timeout)

    def send(self, message: Request | Transition | str | None) -> None:
        """Send one bounded message."""
        buffer = io.BytesIO()
        pickler = pickle.Pickler(buffer, protocol=5)
        pickler.dispatch_table = {np.ndarray: _reduce_array}
        pickler.dump(message)
        payload = buffer.getvalue()
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("RPC message exceeds 64 MiB")
        self.connection.sendall(struct.pack("!I", len(payload)) + payload)

    def receive(self) -> Request | Transition | str | None:
        """Receive a complete message, or raise on timeout/disconnection."""
        length = struct.unpack("!I", self._receive_exactly(4))[0]
        if length > 64 * 1024 * 1024:
            raise ValueError("RPC message exceeds 64 MiB")
        return pickle.loads(self._receive_exactly(length))

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    def _receive_exactly(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.connection.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError("Peer disconnected")
            chunks.extend(chunk)
        return bytes(chunks)


def _reduce_array(array: np.ndarray) -> tuple:
    # NumPy 2's default pickle refers to numpy._core, which is absent in the model's NumPy 1.26.
    if array.dtype.hasobject:
        raise ValueError("RPC does not accept object arrays")
    return _restore_array, (array.tobytes(), array.shape, array.dtype.str)


def _restore_array(data: bytes, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
