# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import numpy as np
import torch
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.kit.app
import omni.timeline

from isaaclab.assets.articulation.articulation import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedEnv, ViewerCfg


class ViewportCameraController:
    """This class handles controlling the camera associated with a viewport in the simulator.

    It can be used to set the viewpoint camera to track different origin types:

    - **world**: the center of the world (static)
    - **env**: the center of an environment (static)
    - **asset_root**: the root of an asset in the scene (e.g. tracking a robot moving in the scene)

    On creation, the camera is set to track the origin type specified in the configuration.

    For the :attr:`asset_root` origin type, the camera is updated at each rendering step to track the asset's
    root position. For this, it registers a callback to the post update event stream from the simulation app.
    """

    def __init__(self, env: ManagerBasedEnv | DirectRLEnv, cfg: ViewerCfg):
        """Initialize the ViewportCameraController.

        Args:
            env: The environment.
            cfg: The configuration for the viewport camera controller.

        Raises:
            ValueError: If origin type is configured to be "env" but :attr:`cfg.env_index` is out of bounds.
            ValueError: If origin type is configured to be "asset_root" but :attr:`cfg.asset_name` is unset.

        """
        # store inputs
        self._env = env
        self._cfg = copy.deepcopy(cfg)
        # cast viewer eye and look-at to numpy arrays
        self.default_cam_eye = np.array(self._cfg.eye)
        self.default_cam_lookat = np.array(self._cfg.lookat)

        # set the camera origins
        if self.cfg.origin_type == "env":
            # check that the env_index is within bounds
            self.set_view_env_index(self.cfg.env_index)
            # set the camera origin to the center of the environment
            self.update_view_to_env()
        elif self.cfg.origin_type == "asset_root" or self.cfg.origin_type == "asset_body":
            # note: we do not yet update camera for tracking an asset origin, as the asset may not yet be
            # in the scene when this is called. Instead, we subscribe to the post update event to update the camera
            # at each rendering step.
            if self.cfg.asset_name is None:
                raise ValueError(f"No asset name provided for viewer with origin type: '{self.cfg.origin_type}'.")
            if self.cfg.origin_type == "asset_body":
                if self.cfg.body_name is None:
                    raise ValueError(f"No body name provided for viewer with origin type: '{self.cfg.origin_type}'.")
        else:
            # set the camera origin to the center of the world
            self.update_view_to_world()

        # subscribe to post update event so that camera view can be updated at each rendering step
        app_interface = omni.kit.app.get_app_interface()
        app_event_stream = app_interface.get_post_update_event_stream()
        self._viewport_camera_update_handle = app_event_stream.create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._update_tracking_callback(event)
        )

    def __del__(self):
        """Unsubscribe from the callback."""
        # use hasattr to handle case where __init__ has not completed before __del__ is called
        if hasattr(self, "_viewport_camera_update_handle") and self._viewport_camera_update_handle is not None:
            self._viewport_camera_update_handle.unsubscribe()
            self._viewport_camera_update_handle = None

    """
    Properties
    """

    @property
    def cfg(self) -> ViewerCfg:
        """The configuration for the viewer."""
        return self._cfg

    """
    Public Functions
    """

    def set_view_env_index(self, env_index: int):
        """Sets the environment index for the camera view.

        Args:
            env_index: The index of the environment to set the camera view to.

        Raises:
            ValueError: If the environment index is out of bounds. It should be between 0 and num_envs - 1.
        """
        # check that the env_index is within bounds
        if env_index < 0 or env_index >= self._env.num_envs:
            raise ValueError(
                f"Out of range value for attribute 'env_index': {env_index}."
                f" Expected a value between 0 and {self._env.num_envs - 1} for the current environment."
            )
        # update the environment index
        self.cfg.env_index = env_index
        # update the camera view if the origin is set to env type (since, the camera view is static)
        # note: for assets, the camera view is updated at each rendering step
        if self.cfg.origin_type == "env":
            self.update_view_to_env()

    def update_view_to_world(self):
        """Updates the viewer's origin to the origin of the world which is (0, 0, 0)."""
        # set origin type to world
        self.cfg.origin_type = "world"
        # update the camera origins
        self.viewer_origin = torch.zeros(3)
        # update the camera view
        self.update_view_location()

    def update_view_to_env(self):
        """Updates the viewer's origin to the origin of the selected environment."""
        # set origin type to world
        self.cfg.origin_type = "env"
        # update the camera origins
        self.viewer_origin = self._env.scene.env_origins[self.cfg.env_index]
        # update the camera view
        self.update_view_location()

    def update_view_to_asset_root(self, asset_name: str):
        """Updates the viewer's origin based upon the root of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.

        Raises:
            ValueError: If the asset is not in the scene.
        """
        # check if the asset is in the scene
        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")
        # update the asset name
        self.cfg.asset_name = asset_name
        # set origin type to asset_root
        self.cfg.origin_type = "asset_root"
        # update the camera origins
        self.viewer_origin = self._env.scene[self.cfg.asset_name].data.root_pos_w[self.cfg.env_index]
        # update the camera view
        self.update_view_location()

    def update_view_to_asset_body(self, asset_name: str, body_name: str):
        """Updates the viewer's origin based upon the body of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.
            body_name: The name of the body in the asset.

        Raises:
            ValueError: If the asset is not in the scene or the body is not valid.
        """
        # check if the asset is in the scene
        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")
        # check if the body is in the asset
        asset: Articulation = self._env.scene[asset_name]
        if body_name not in asset.body_names:
            raise ValueError(
                f"'{body_name}' is not a body of Asset '{asset_name}'. Available bodies: {asset.body_names}."
            )
        # get the body index
        body_id, _ = asset.find_bodies(body_name)
        # update the asset name
        self.cfg.asset_name = asset_name
        # set origin type to asset_body
        self.cfg.origin_type = "asset_body"
        # update the camera origins
        self.viewer_origin = self._env.scene[self.cfg.asset_name].data.body_pos_w[self.cfg.env_index, body_id].view(3)
        # update the camera view
        self.update_view_location()

    def update_view_location(self, eye: Sequence[float] | None = None, lookat: Sequence[float] | None = None):
        """Updates the camera view pose based on the current viewer origin and the eye and lookat positions.

        Args:
            eye: The eye position of the camera. If None, the current eye position is used.
            lookat: The lookat position of the camera. If None, the current lookat position is used.
        """
        # store the camera view pose for later use
        if eye is not None:
            self.default_cam_eye = np.asarray(eye)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat)
        # set the camera locations
        viewer_origin = self.viewer_origin.detach().cpu().numpy()
        cam_eye = viewer_origin + self.default_cam_eye
        cam_target = viewer_origin + self.default_cam_lookat

        # set the camera view
        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)

    """
    Private Functions
    """

    def _update_tracking_callback(self, event):
        """Updates the camera view at each rendering step."""
        # update the camera view if the origin is set to asset_root
        # in other cases, the camera view is static and does not need to be updated continuously
        if self.cfg.origin_type == "asset_root" and self.cfg.asset_name is not None:
            self.update_view_to_asset_root(self.cfg.asset_name)
        if self.cfg.origin_type == "asset_body" and self.cfg.asset_name is not None and self.cfg.body_name is not None:
            self.update_view_to_asset_body(self.cfg.asset_name, self.cfg.body_name)
