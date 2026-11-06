# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Capture input for an existing task action controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from isaaclab.managers import ActionInputCfg
from isaaclab.utils import configclass

from .source_cfg import TeleopSourceCfg

if TYPE_CHECKING:
    import torch

    from .input import TeleopInput


@configclass
class TeleopInputCfg(ActionInputCfg):
    """Add Capture input to an existing task action's ``input`` field.

    The controller retains its configuration, type, and public API. Playback
    activates this input after the full environment is ready; ordinary task
    construction and training continue to use policy actions.
    """

    class_type: type[TeleopInput] | str = "{DIR}.input:TeleopInput"

    source: TeleopSourceCfg | None = None
    """Input resource declared by exactly one of the task's teleop inputs.

    Playback constructs this source once after the environment is ready. Other
    ``TeleopInputCfg`` declarations for the same environment row leave this unset
    and consume the same input sample. Each controller selects its own asset.
    The declaring input need not belong to the first action term or control a
    particular asset.
    """

    env_ids: list[int] = [0]
    """Controlled environment rows, in source-output order."""

    action_adapter: Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor] | None = None
    """Optional conversion called with ``(source_actions, env, env_ids)``.

    Every term receives the full Capture action vector, so its adapter selects
    and converts the inputs needed by that controller. Adapters receive their
    own copy and cannot change the sample consumed by another term.
    Return controller input with shape ``(selected rows, controller.action_dim)``.
    A vector is also accepted for one selected row. Units, scale, and frames must
    match the controller's raw input. Conversion must not step or reset the scene.
    """

    def validate_config(self) -> None:
        """Reject invalid row selections before Capture is activated."""
        if not self.env_ids or any(type(index) is not int or index < 0 for index in self.env_ids):
            raise ValueError("TeleopInputCfg.env_ids must contain non-negative integer environment indices.")
        if len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("TeleopInputCfg.env_ids must not contain duplicates.")
