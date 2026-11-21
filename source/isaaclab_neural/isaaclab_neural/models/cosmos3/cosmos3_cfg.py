# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the Cosmos3 world model."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab_experimental.cosmos._protocol import DEFAULT_TIMEOUT_S
from isaaclab_experimental.cosmos.server.world_model import DEFAULT_WORLD_MODEL_ENDPOINT

from isaaclab.utils import configclass

from ...world_model import ModelCfg


@configclass
class JointMappingCfg:
    """Robot-specific conversion of joint targets to Cosmos3 controls.

    ``class_type`` is built as ``class_type(cfg)`` and provides ``encode(joint_targets, start_joint_state)``, which
    maps joint targets [rad], shape ``(T, J)``, and the joint positions at the start of the chunk, shape ``(J,)``, to
    normalized ``unified_v1`` controls, shape ``(T, 59)``. The mapping belongs to the task, which owns the robot
    description and the normalization data.
    """

    class_type: type | str = MISSING


@configclass
class Cosmos3Cfg(ModelCfg):
    """Cosmos3 served by ``isaaclab-cosmos-world-model-server`` in its own process and environment.

    The service loads the checkpoint and fixes the chunk length, frame rate and image size; the client reads them
    from the service.
    """

    class_type: type | str = "{DIR}.cosmos3_model:Cosmos3Model"

    endpoint: str = DEFAULT_WORLD_MODEL_ENDPOINT
    """Service endpoint: ``unix:///path`` on this machine or ``tcp://host:port``."""

    timeout: float = DEFAULT_TIMEOUT_S
    """Wait per service request [s]. A window-mode chunk of many environments can take minutes."""

    transport: Literal["auto", "cuda_ipc", "socket"] = "auto"
    """How chunks reach the simulation: ``"cuda_ipc"`` keeps them on the GPU, for a service on this machine whose GPU
    this process can access directly; ``"socket"`` sends them through host memory; ``"auto"`` picks CUDA IPC when it
    is available."""

    joint_mapping: JointMappingCfg | None = None
    """Joint mapping of the robot. Needed to drive the model with joint targets; replaying recorded controls does
    not need it."""
