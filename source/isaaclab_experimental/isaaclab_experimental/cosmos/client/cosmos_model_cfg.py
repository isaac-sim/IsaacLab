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
    """Appearance prompt. A string applies to every episode. A list varies the appearance per episode, for
    visual randomization: episode ``k`` of a camera stream uses ``prompt[k % len(prompt)]``. An episode reset
    before the first generated update (for example the environment's initial reset right after its first capture)
    keeps the current prompt. None omits an appearance description."""

    modality: Literal["edge", "depth", "seg"] = "edge"
    """Type of uint8 three-channel guidance prepared by the preceding camera modifier."""

    max_episode_frames: int = 201
    """Image-frame budget per episode, including the initial frame."""

    timeout: float = 600.0
    """Maximum wait per network operation [s], including the first generation."""
