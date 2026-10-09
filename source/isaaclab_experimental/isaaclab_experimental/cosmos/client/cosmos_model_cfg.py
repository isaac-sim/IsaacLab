# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos image-transfer backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

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

    endpoint: str = "tcp://127.0.0.1:5555"
    """Endpoint of the independently started Cosmos service."""

    prompt: str | list[str] | None = None
    """Appearance prompt. A string applies to every episode. A list varies the appearance, for visual
    randomization. With one view, episode ``k`` of the camera stream uses ``prompt[k % len(prompt)]``; an episode
    reset before the first generated update (for example the environment's initial reset right after its first
    capture) keeps the current prompt. With several views (environments), view ``v`` uses
    ``prompt[v % len(prompt)]`` for all its episodes, because a batched session keeps each view's prompt across
    resets. None omits an appearance description."""

    modality: Literal["edge", "blur", "depth", "seg"] = "edge"
    """Type of uint8 three-channel guidance. The preceding camera modifier prepares edge, depth, and seg controls;
    for blur the camera sends its RGB and the service applies the Framework's own blur filter."""

    max_episode_frames: int | None = None
    """Image-frame budget per episode, including the initial frame: ``1 + 4*k``. The
    Shadow Hand preset derives it from the task duration and capture rate. Low-level callers must set it before
    creating the client. It must fit any explicit server ``--max-episode-frames`` cap; the server has none by default.
    """

    timeout: float = 600.0
    """Maximum wait per network operation [s], including the first generation."""
