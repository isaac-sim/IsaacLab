# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific capability protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class NewtonState(Protocol):
    """Direct read-only access to the active Newton ``Model`` and ``State``.

    Used by Newton-flavoured consumers (NewtonWarp renderer, Newton/Rerun/
    Viser visualizers, Newton GL video) that pass the full ``State`` object
    to a Newton-aware viewer or sensor and cannot work from a transform
    buffer alone.

    A provider exposes this capability when it has a Newton state to share
    — natively in :class:`NewtonSceneDataProvider`, or via the PhysX→Newton
    sync bridge in :class:`PhysxSceneDataProvider` when at least one
    consumer requires it.
    """

    def get_state(self) -> Any:
        """Return the current Newton ``State`` object.

        The returned object is read-only; consumers must not mutate it.
        """
        ...

    def get_model(self) -> Any:
        """Return the Newton ``Model`` object.

        The model is structural (joint topology, link names, shape data)
        and stable across frames.
        """
        ...
