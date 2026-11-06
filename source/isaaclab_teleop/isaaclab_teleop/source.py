# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Input lifecycle shared by task action terms and interactive playback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class TeleopSample:
    """One non-blocking source poll in the task action term's conventions.

    ``active=False`` or ``actions=None`` yields control to the policy.
    ``reset=True`` requests an episode reset through the playback owner;
    the action input discards that sample's actions.
    """

    actions: torch.Tensor | None = None
    active: bool = True
    reset: bool = False


class TeleopSource(ABC):
    """An input resource whose lifecycle is independent of simulation.

    Playback constructs sources after the environment and activates the task's
    action inputs. Sources may borrow scene and visualizer resources but release
    those resources on :meth:`close`, including after partially failed startup.
    Polling must not step, render, or reset the environment.
    """

    @abstractmethod
    def advance(self) -> TeleopSample:
        """Read the latest actions and control events without advancing simulation."""

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset source state after the specified controlled environments reset."""

    @abstractmethod
    def close(self) -> None:
        """Release input resources. Repeated successful calls must be harmless."""
