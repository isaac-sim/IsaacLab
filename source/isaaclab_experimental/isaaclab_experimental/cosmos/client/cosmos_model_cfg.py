# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos image-transfer backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

from .._protocol import DEFAULT_ENDPOINT, DEFAULT_MAX_EPISODE_FRAMES

if TYPE_CHECKING:
    from .cosmos_model import CosmosModel


@configclass
class CosmosModelCfg(BackendCfg):
    """Settings for a camera's resident Cosmos service connection.

    Use with :class:`~isaaclab_experimental.image_transfer.ImageTransferModifierCfg` after a control
    preparation modifier. Cosmos generates an initial one-frame chunk, then four-frame chunks.
    Configure that modifier with ``initial_frames=1`` and ``update_frames=4``.
    """

    class_type: type[CosmosModel] | str = "{DIR}.cosmos_model:CosmosModel"
    """Client resource constructor. The model framework runs in the service's environment."""

    endpoint: str = DEFAULT_ENDPOINT
    """Endpoint of the independently started Cosmos service: ``unix:///path`` on the same machine, or
    ``tcp://host:port``. Defaults to the service's default, a Unix socket private to this user on Linux and
    ``tcp://127.0.0.1:5555`` on Windows."""

    prompt: str | list[str] | None = None
    """Appearance prompt. A string applies to every episode. A list varies the appearance per episode, for
    visual randomization: episode ``k`` of a camera stream uses ``prompt[k % len(prompt)]``. An episode reset
    before the first generated update (for example the environment's initial reset right after its first capture)
    keeps the current prompt. None omits an appearance description."""

    modality: Literal["edge", "depth", "seg"] = "edge"
    """Type of uint8 three-channel guidance prepared by the preceding camera modifier."""

    max_episode_frames: int = DEFAULT_MAX_EPISODE_FRAMES
    """Image-frame budget per episode, including the initial frame: ``1 + 4*k``. It must be within the service's
    cap, set with ``isaaclab-cosmos-server --max-episode-frames`` (default 201, 0 for no cap)."""

    transport: Literal["auto", "cuda_ipc", "socket"] = "auto"
    """How images move between the camera and the service. ``"cuda_ipc"`` keeps them on the GPU through shared
    device memory and interprocess events, with no host copies or synchronization; it needs the service on the same
    Linux machine and GPU. ``"socket"`` sends them through host memory in the endpoint's messages, which also works
    across machines. ``"auto"`` (default) uses CUDA IPC when the service supports it and shares the camera's GPU,
    and the socket otherwise. The endpoint carries the small step messages in both cases."""

    timeout: float = 600.0
    """Maximum wait per network operation [s], including the first generation."""
