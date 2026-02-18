# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC

from .renderer_cfg import RendererCfg


class RendererBase(ABC):
    """Base class for renderers.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: RendererCfg):
        self.cfg = cfg
        self._height = cfg.height
        self._width = cfg.width
        self._num_envs = cfg.num_envs
        self._num_cameras = 1  # TODO: currently only supports 1 camera per environment
        # List of data types to use for rendering, e.g. ["rgb", "depth", "semantic_segmentation"]
        self._data_types = []

        # output buffer format is a dict, where the keys is the data type and the value is a list of buffers for each camera
        # TODO: Document the standard format of the output data buffers. Need discussion.
        self._output_data_buffers = dict()

        # TODO: share the same renderer for different cameras/rendering jobs.

    def initialize(self):
        """Initialize the renderer."""
        # Step 1: Initialize the corresponding output data type
        # Step 2: initialize output buffers
        raise NotImplementedError("initialize() is not implemented.")

    def step(self):
        """Step the renderer."""
        raise NotImplementedError("step() is not implemented.")

    def reset(self):
        """Reset the renderer."""
        raise NotImplementedError("reset() is not implemented.")

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        raise NotImplementedError("initialize_output() is not implemented.")

    def get_output(self):
        return self._output_data_buffers

    def clone(self, cameras):
        """TODO: Clone the camera in renderer."""
        raise NotImplementedError("clone() is not implemented.")
