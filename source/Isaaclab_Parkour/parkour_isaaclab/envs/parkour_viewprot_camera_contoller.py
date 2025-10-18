
from __future__ import annotations

import copy
import numpy as np
import weakref
from typing import TYPE_CHECKING
import omni.kit.app
import omni.timeline
import carb
from isaaclab.envs.ui import ViewportCameraController 

if TYPE_CHECKING:
    from parkour_isaaclab.envs import  ParkourManagerBasedEnv
    from isaaclab.envs import ViewerCfg

class ParkourViewportCameraController(ViewportCameraController):
    """
    Viewport Camera Controller with Keyboard 
    reference: 
        https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
    """
    def __init__(self, env: ParkourManagerBasedEnv, cfg: ViewerCfg):
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

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self.free_cam_trigger = False 
        self.is_free_cam = False

        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # self._mouse = self._appwindow.get_mouse()
        # subscribe to post update event so that camera view can be updated at each rendering step
        app_interface = omni.kit.app.get_app_interface()
        app_event_stream = app_interface.get_post_update_event_stream()
        self._viewport_camera_update_handle = app_event_stream.create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._update_tracking_callback(event)
        )

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "NUMPAD_7" and not self.is_free_cam:
                self.cfg.env_index  = (self.cfg.env_index - 1) % self._env.num_envs

            elif event.input.name in "NUMPAD_9" and not self.is_free_cam:
                self.cfg.env_index = (self.cfg.env_index + 1) % self._env.num_envs
        
            elif event.input.name in "NUMPAD_8" and not self.is_free_cam:
                self.default_cam_eye[0] += 0.2

            elif event.input.name in "NUMPAD_4" and not self.is_free_cam:
                self.default_cam_eye[1] += 0.2
        
            elif event.input.name in "NUMPAD_6" and not self.is_free_cam:
                self.default_cam_eye[2] += 0.2
        
            elif event.input.name in "NUMPAD_5" and not self.is_free_cam:
                self.default_cam_eye[0] -= 0.2

            elif event.input.name in "NUMPAD_2" and not self.is_free_cam:
                self.default_cam_eye[:] = self._cfg.eye
            
            if event.input.name in "NUMPAD_0":
                """go to free cam"""
                self.is_free_cam = True
                self.free_cam_trigger = True 

            elif event.input.name in "NUMPAD_1":
                """back to root cam"""
                self.is_free_cam = False
                self.free_cam_trigger = False 

    def _update_tracking_callback(self, event):
        if self.cfg.origin_type == "asset_root" and self.cfg.asset_name is not None and not self.is_free_cam:
            self.update_view_to_asset_root(self.cfg.asset_name)
        if self.cfg.origin_type == "asset_body" and self.cfg.asset_name is not None and self.cfg.body_name is not None and not self.is_free_cam:
            self.update_view_to_asset_body(self.cfg.asset_name, self.cfg.body_name)

        if self.is_free_cam and self.free_cam_trigger:
            self.free_cam_trigger = False 
            if self.cfg.origin_type == "asset_root" and self.cfg.asset_name is not None :
                self.update_view_to_asset_root(self.cfg.asset_name)
            if self.cfg.origin_type == "asset_body" and self.cfg.asset_name is not None and self.cfg.body_name is not None:
                self.update_view_to_asset_body(self.cfg.asset_name, self.cfg.body_name)
