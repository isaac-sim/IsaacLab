# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderer_cfg import RendererCfg


class RendererBase(ABC):
    """Base class for renderers.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: RendererCfg):
        self.cfg = cfg
        self._height = cfg.height
        self._width = cfg.width
        # List of data types to use for rendering, e.g. ["rgb", "depth", "semantic_segmentation"]
        self._data_types = []

        # output buffer format is a ditc, where the keys is the data type and the value is a list of buffers for each camera
        self._output_data_buffers = dict()

    def initialize(self):
        """Initialize the renderer."""
        # Step 1: Initialize the corresponding output data type
        # Step 2: initialize output buffers
        raise NotImplementedError("initialize() is not implemented.")

    def step(self):
        """Step the renderer."""
        raise NotImplementedError("step() is not implemented.")

    def initialize_output(self):
        """Initialize the output of the renderer."""
        raise NotImplementedError("initialize_output() is not implemented.")

    def get_output(self):
        return self._output_data_buffers

    def close(self):
        """Close the renderer."""
        raise NotImplementedError("close() is not implemented.")
