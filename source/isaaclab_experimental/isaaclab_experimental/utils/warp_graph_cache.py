# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp CUDA graph capture-or-replay utility."""

from collections.abc import Callable
from typing import Any

import warp as wp


class WarpGraphCache:
    """Caches Warp CUDA graphs by stage name: captures on first call, replays after.

    Usage::

        cache = WarpGraphCache()
        cache.capture_or_replay("my_stage", my_warp_function)
        # uncaptured work here ...
        cache.capture_or_replay("my_stage_post", my_other_function)
    """

    def __init__(self):
        self._graphs: dict[str, Any] = {}

    def capture_or_replay(self, stage: str, fn: Callable[[], Any]) -> None:
        """Capture *fn* into a CUDA graph on the first call, then replay."""
        graph = self._graphs.get(stage)
        if graph is None:
            with wp.ScopedCapture() as capture:
                fn()
            self._graphs[stage] = capture.graph
        else:
            wp.capture_launch(graph)

    def invalidate(self, stage: str | None = None) -> None:
        """Drop cached graph(s). If *stage* is ``None``, drop all."""
        if stage is None:
            self._graphs.clear()
        else:
            self._graphs.pop(stage, None)
