# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime visual domain randomization. Cosmos is imported only when configured."""

from .backends import DRBackend, DRFrame, DRRequest, PassthroughBackend
from .cfg import CameraDRCfg, CosmosBackendCfg, DRBackendCfg, PromptBankCfg, RemoteCosmosBackendCfg, VisualDRCfg
from .observations import image_runtime_dr, preserve_mask
from .runtime import VisualDRRuntime

__all__ = [
    "CameraDRCfg",
    "CosmosBackendCfg",
    "DRBackend",
    "DRBackendCfg",
    "DRFrame",
    "DRRequest",
    "PassthroughBackend",
    "PromptBankCfg",
    "RemoteCosmosBackendCfg",
    "VisualDRCfg",
    "VisualDRRuntime",
    "image_runtime_dr",
    "preserve_mask",
]
