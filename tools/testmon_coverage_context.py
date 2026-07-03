# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coverage plugin that tags child-process lines with the active pytest node id."""

from __future__ import annotations

import os
from types import FrameType

from coverage import CoveragePlugin


class NodeIdContextPlugin(CoveragePlugin):
    """Report the active pytest node id as coverage's dynamic context."""

    def dynamic_context(self, frame: FrameType) -> str | None:
        return os.environ.get("COVERAGE_CONTEXT") or None


def coverage_init(reg, options) -> None:
    reg.add_dynamic_context(NodeIdContextPlugin())
