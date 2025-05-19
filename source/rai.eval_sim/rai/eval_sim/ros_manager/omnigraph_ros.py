# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.graph.core as og
import usdrt.Sdf
from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from . import omnigraph_ros_cfg


class OmniGraphTerm(ABC):
    def __init__(self, cfg: omnigraph_ros_cfg.OmniGraphTermCfg, env: ManagerBasedEnv, env_idx: int = 0) -> None:
        """Initializes the Omnigraph term.

        Args:
            cfg: The configuration object
            env: The orbit environment
            env_indx: The index of the environment, used when multiple environments are managed. Defaults to 0.
        """
        # store inputs
        self._cfg = cfg
        self._env = env
        self._env_idx = env_idx

        # resolve scene entity
        self.cfg.asset_cfg.resolve(env.scene)

        self._create_omnigraph()

    def close(self):
        del self._env

    @property
    def cfg(self) -> omnigraph_ros_cfg.OmniGraphTermCfg:
        return self._cfg

    @abstractmethod
    def _create_omnigraph(self):
        """Create Omnigraph."""
        raise NotImplementedError


class OmniGraphCameraTerm(OmniGraphTerm):
    def __init__(self, cfg: omnigraph_ros_cfg.OmniGraphCameraTermCfg, env: ManagerBasedEnv, env_idx: int = 0):
        super().__init__(cfg, env, env_idx)
        """Creates an Omnigraph graph to publish camera data.

        It does this by reading in a OmniGraphCameraTermCfg that contains enables and
        parameters for different camera data modes. Currently supported outputs are RBG,
        Pseudo-depth, and camera info. When creating the OmniGraphCameraTermCfg take care to
        set the asset_cfg to a Camera created using a CameraCfg. When creating the CameraCfg
        be sure to set data_types=[]. This prevents duplicate RenderProducts used in the
        Synthetic Data Pipeline. No error will occur if data_types is non-empty, but
        performance will be slower.

        The Camera publish rate is controlled by the rendering rate of the simulation.
        """

    def _create_omnigraph(self):
        """Creates the configurable Camera Publisher Omnigraph"""

        prim_path = self._env.scene[self.cfg.asset_cfg.name].cfg.prim_path
        camera_prim = prim_path.replace("env_.*", "env_" + str(self._env_idx))

        camera_name = camera_prim.split("/")[-1]
        height = self._env.scene[self.cfg.asset_cfg.name].cfg.height
        width = self._env.scene[self.cfg.asset_cfg.name].cfg.width

        # define graph prim name
        if self.cfg.graph_name is None:
            self.graph_name = "/ActionGraph_" + camera_name
        else:
            self.graph_name = self.cfg.graph_name

        # initialize nodes, connections, and values lists
        nodes = list()
        connect = list()
        values = list()

        # setup execution via Physics step NOTE: this does not control rate of execution for rendering products
        # setup Rendering Product basics
        nodes.append(("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"))
        nodes.append(("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"))

        connect.append(("OnPhysicsStep.outputs:step", "RenderProduct.inputs:execIn"))

        values.append(("RenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim)]))
        values.append(("RenderProduct.inputs:height", height))
        values.append(("RenderProduct.inputs:width", width))

        # RGB publishing
        if self.cfg.enable_rgb:
            nodes.append(("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"))
            connect.append(("RenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"))
            connect.append(("RenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"))
            values.append(("cameraHelperRgb.inputs:frameId", camera_name))
            values.append(("cameraHelperRgb.inputs:topicName", self.cfg.rgb_topic))
            values.append(("cameraHelperRgb.inputs:type", "rgb"))
            values.append(("cameraHelperRgb.inputs:nodeNamespace", self.cfg.namespace))
            values.append(("cameraHelperRgb.inputs:useSystemTime", True))
        # Pseudo-Depth publishing
        if self.cfg.enable_depth:
            nodes.append(("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"))
            connect.append(("RenderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"))
            connect.append(("RenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"))
            values.append(("cameraHelperDepth.inputs:frameId", camera_name))
            values.append(("cameraHelperDepth.inputs:topicName", self.cfg.depth_topic))
            values.append(("cameraHelperDepth.inputs:type", "depth"))
            values.append(("cameraHelperDepth.inputs:nodeNamespace", self.cfg.namespace))
            values.append(("cameraHelperDepth.inputs:useSystemTime", True))
        # Camera Info Publishing
        if self.cfg.enable_info:
            nodes.append(("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"))
            connect.append(("RenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"))
            connect.append(("RenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"))
            values.append(("cameraHelperInfo.inputs:frameId", camera_name))
            values.append(("cameraHelperInfo.inputs:topicName", self.cfg.info_topic))
            values.append(("cameraHelperInfo.inputs:nodeNamespace", self.cfg.namespace))
            values.append(("cameraHelperInfo.inputs:useSystemTime", True))

        # create omnigraph
        keys = og.Controller.Keys
        _ = og.Controller.edit(
            {
                "graph_path": self.graph_name,
                "evaluator_name": "push",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: nodes,
                keys.CONNECT: connect,
                keys.SET_VALUES: values,
            },
        )

    def close(self):
        """Deletes omnigraph prim to clear all callbacks and OgnIsaacCreateRenderProductInternalStates."""
        prim_utils.delete_prim(self.graph_name)
        super().close()
