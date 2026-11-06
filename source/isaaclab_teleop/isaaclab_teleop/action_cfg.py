# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task action configuration for optional Capture input."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

from isaaclab.managers import ActionTermCfg, InteractiveActionTermCfg
from isaaclab.utils import configclass

from .source_cfg import TeleopSourceCfg

if TYPE_CHECKING:
    import torch

    from .action import TeleopAction


@configclass
class TeleopActionCfg(InteractiveActionTermCfg):
    """Compose optional teleoperation with a task's existing controller.

    Declare this term inside the task's ``Actions`` configuration. Its action
    dimension and processing remain those of ``controller``. Capture resources
    are activated by interactive playback; constructing or training the task
    alone retains ordinary policy control.
    """

    class_type: type[TeleopAction] | str = "{DIR}.action:TeleopAction"

    controller: ActionTermCfg = MISSING
    """Controller owning asset selection, scaling, clipping, and visualization."""

    source: TeleopSourceCfg | None = None
    """Input resource declared by exactly one of the robot's teleop action terms.

    Playback constructs this source once after the environment is ready. Other
    ``TeleopActionCfg`` terms for the same robot and environment row leave this
    unset and consume the same input sample. The declaring term need not be the
    first action term or represent a particular part of the robot.
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

    # These inherited fields describe the nested controller, not a second asset
    # or transformation. Keep them out of this composite's constructor.
    asset_name: str = field(default="", init=False, repr=False)
    debug_vis: bool = field(default=False, init=False, repr=False)
    clip: dict[str, tuple] | None = field(default=None, init=False, repr=False)

    def validate_config(self) -> None:
        """Reject ambiguous input ownership before Capture is activated."""
        if not isinstance(self.controller, ActionTermCfg) or isinstance(self.controller, TeleopActionCfg):
            raise TypeError("TeleopActionCfg.controller must be a non-teleop ActionTermCfg.")
        if not self.env_ids or any(type(index) is not int or index < 0 for index in self.env_ids):
            raise ValueError("TeleopActionCfg.env_ids must contain non-negative integer environment indices.")
        if len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("TeleopActionCfg.env_ids must not contain duplicates.")
        if self.asset_name or self.debug_vis or self.clip is not None:
            raise ValueError("Configure asset_name, debug_vis, and clip on TeleopActionCfg.controller.")
