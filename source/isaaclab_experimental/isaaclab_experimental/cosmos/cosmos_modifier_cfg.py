# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos camera modifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_experimental.image_transfer import ImageTransferModifierCfg

if TYPE_CHECKING:
    from .cosmos_modifier import CosmosTransferModifier


@configclass
class CosmosTransferModifierCfg(ImageTransferModifierCfg):
    """Generate one initial frame, then four frames per Cosmos Sim-Transfer chunk."""

    func: type[CosmosTransferModifier] | str = "{DIR}.cosmos_modifier:CosmosTransferModifier"
    """The Cosmos camera modifier."""

    initial_frames: int = 1
    """Captures required for the first chunk of an episode."""

    update_frames: int = 4
    """Captures required for subsequent chunks."""

    seed: int = 42
    """Initial generation seed; full camera resets advance it deterministically."""
