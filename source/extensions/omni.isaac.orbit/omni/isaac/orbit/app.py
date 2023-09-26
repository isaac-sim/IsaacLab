# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility to configure the ``omni.isaac.kit.SimulationApp`` based on environment variables.

Based on the desired functionality, this class parses environment variables and input keyword arguments
to launch the simulator in various different modes. This includes with or without GUI, switching between
different Omniverse remote clients, and enabling particular ROS bridges. Some of these require the
extensions to be loaded in a specific order, otherwise a segmentation fault occurs.
The launched `SimulationApp`_ instance is accessible via the :attr:`AppLauncher.app` property.

Available modes
---------------

The following details the behavior of the class based on the environment variables:

* **Headless mode**: If the environment variable ``REMOTE_DEPLOYMENT>0``, then SimulationApp will be started in headless mode.

* **Livestreaming**: If the environment variable ``REMOTE_DEPLOYMENT={2,3,4}`` , then `livestream`_ is enabled.

  * ``REMOTE_DEPLOYMENT=1`` does not enable livestreaming, though it causes the app to run in headless mode.
  * ``REMOTE_DEPLOYMENT=2`` enables streaming via the Isaac `Native Livestream`_ extension. This allows users to
    connect through the Omniverse Streaming Client.
  * ``REMOTE_DEPLOYMENT=3`` enables streaming via the `Websocket Livestream` extension. This allows users to
    connect in a browser using the WebSocket protocol.
  * ``REMOTE_DEPLOYMENT=4`` enables streaming  via the `WebRTC Livestream` extension. This allows users to
    connect in a browser using the WebRTC protocol.

* **Viewport**: If the environment variable ``VIEWPORT_ENABLED`` is set to non-zero, then the following behavior happens:

  * ``VIEWPORT_ENABLED=1``: Ensures that the VIEWPORT member is set to true, to enable lightweight streaming
    when the full GUI is not needed (i.e. headless mode).

* **Loading ROS Bridge**: If the environment variable ``ROS_ENABLED`` is set to non-zero, then the
  following behavior happens:

  * ``ROS_ENABLED=1``: Enables the ROS1 Noetic bridge in Isaac Sim.
  * ``ROS_ENABLED=2``: Enables the ROS2 Foxy bridge in Isaac Sim.

  .. caution::
    Currently, in Isaac Sim 2022.2.1, loading ``omni.isaac.ros_bridge`` before ``omni.kit.livestream.native``
    causes a segfault. Thus, to work around this issue, we enable the ROS-bridge extensions after the
    livestreaming extensions.


Usage
-----

To set the environment variables, one can use the following command in the terminal:

.. code:: bash

    export REMOTE_DEPLOYMENT=3
    export VIEWPORT_ENABLED=1
    # run the python script
    ./orbit.sh -p source/standalone/demo/play_quadrupeds.py

Alternatively, one can set the environment variables to the python script directly:

.. code:: bash

    REMOTE_DEPLOYMENT=3 VIEWPORT_ENABLED=1 ./orbit.sh -p source/standalone/demo/play_quadrupeds.py


.. _SimulationApp: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
.. _livestream: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html
.. _`Native Livestream`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html#isaac-sim-setup-kit-remote
.. _`Websocket Livestream`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html#isaac-sim-setup-livestream-webrtc
.. _`WebRTC Livestream`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html#isaac-sim-setup-livestream-websocket

"""

from __future__ import annotations

import faulthandler
import os
import re
import sys
from typing import ClassVar

import carb
from omni.isaac.kit import SimulationApp

__all__ = ["AppLauncher"]


class AppLauncher:
    """A class to create the simulation application based on keyword arguments and environment variables."""

    RENDER: ClassVar[bool]
    """
    Whether or not to render the GUI associated with IsaacSim.

    This flag can be used in subsequent execution of the created simulator, i.e.

    .. code:: python

        from omni.isaac.core.simulation_context import SimulationContext

        SimulationContext.instance().step(render=AppLauncher.RENDER)

    Also, can be passed to :class:`IsaacEnv` instance using the :attr:`render` attribute, i.e.

    .. code:: python

        gym.make(render=app_launcher.RENDER, viewport=app_launcher.VIEWPORT)
    """

    VIEWPORT: ClassVar[bool]
    """
    Whether or not to render the lighter 'viewport' elements even when the application might be
    executing in headless (no-GUI) mode. This is useful for off-screen rendering to gather images and
    video more efficiently.

    Also, can be passed to :class:`IsaacEnv` instance using the :attr:`viewport` attribute, i.e.

    .. code:: python

        gym.make(render=app_launcher.RENDER, viewport=app_launcher.VIEWPORT)
    """

    def __init__(self, **kwargs):
        """Parses environments variables and keyword arguments to create a `SimulationApp`_ instance.

        If the keyword argument ``headless`` is set to True, then the SimulationApp will be started in headless mode.
        It will be given priority over the environment variable setting ``REMOTE_DEPLOYMENT=0``.

        Args:
            **kwargs: Keyword arguments passed to the :class:`SimulationApp` from Isaac Sim.
              A detailed description of the possible arguments is available in its `documentation`_.

        Raises:
            ValueError: If incompatible or undefined values are assigned to relevant environment values,
              such as ``REMOTE_DEPLOYMENT`` and ``ROS_ENABLED``

        .. _SimulationApp: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
        .. _documentation: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
        """
        # Enable call-stack on crash
        faulthandler.enable()

        # Headless is always true for remote deployment
        remote_deployment = int(os.environ.get("REMOTE_DEPLOYMENT", 0))
        # resolve headless execution of simulation app
        headless = kwargs.get("headless", False) or remote_deployment
        kwargs.update({"headless": headless})
        # hack sys module to make sure that the SimulationApp is initialized correctly
        # this is to avoid the warnings from the simulation app about not ok modules
        r = re.compile(".*orbit.*")
        found_modules = list(filter(r.match, list(sys.modules.keys())))
        found_modules += ["omni.isaac.kit.app_framework"]
        # remove orbit modules from sys.modules
        hacked_modules = dict()
        for key in found_modules:
            hacked_modules[key] = sys.modules[key]
            del sys.modules[key]
        # launch simulation app
        self._app = SimulationApp(kwargs)
        # add orbit modules back to sys.modules
        for key, value in hacked_modules.items():
            sys.modules[key] = value

        # These have to be loaded after SimulationApp is initialized
        from omni.isaac.core.utils.extensions import enable_extension

        # Retrieve carb settings for modification
        carb_settings_iface = carb.settings.get_settings()

        if remote_deployment >= 2:
            # Set carb settings to allow for livestreaming
            carb_settings_iface.set_bool("/app/livestream/enabled", True)
            carb_settings_iface.set_bool("/app/window/drawMouse", True)
            carb_settings_iface.set_bool("/ngx/enabled", False)
            carb_settings_iface.set_string("/app/livestream/proto", "ws")
            carb_settings_iface.set_int("/app/livestream/websocket/framerate_limit", 120)
            # Note: Only one livestream extension can be enabled at a time
            if remote_deployment == 2:
                # Enable Native Livestream extension
                # Default App: Streaming Client from the Omniverse Launcher
                enable_extension("omni.kit.livestream.native")
                enable_extension("omni.services.streaming.manager")
            elif remote_deployment == 3:
                # Enable WebSocket Livestream extension
                # Default URL: http://localhost:8211/streaming/client/
                enable_extension("omni.services.streamclient.websocket")
            elif remote_deployment == 4:
                # Enable WebRTC Livestream extension
                # Default URL: http://localhost:8211/streaming/webrtc-client/
                enable_extension("omni.services.streamclient.webrtc")
            else:
                raise ValueError(
                    f"Invalid assignment for env variable `REMOTE_DEPLOYMENT`: {remote_deployment}. Expected 1, 2, 3, 4."
                )

        # As of IsaacSim 2022.1.1, the ros extension has to be loaded
        # after the streaming extension or it will cause a segfault
        ros = int(os.environ.get("ROS_ENABLED", 0))
        # Note: Only one ROS bridge extension can be enabled at a time
        if ros > 0:
            if ros == 1:
                enable_extension("omni.isaac.ros_bridge")
            elif ros == 2:
                enable_extension("omni.isaac.ros2_bridge")
            else:
                raise ValueError(f"Invalid assignment for env variable `ROS_ENABLED`: {ros}. Expected 1 or 2.")

        # off-screen rendering
        viewport = int(os.environ.get("VIEWPORT_ENABLED", 0))
        # enable extensions for off-screen rendering
        # note: depending on the app file, some extensions might not be available in it.
        #   Thus, we manually enable these extensions to make sure they are available.
        if viewport > 0 or not headless:
            # note: enabling extensions is order-sensitive. please do not change the order!
            # extension to enable UI buttons (otherwise we get attribute errors)
            enable_extension("omni.kit.window.toolbar")
            # extension to make RTX realtime and path-traced renderers
            enable_extension("omni.kit.viewport.rtx")
            # extension to make HydraDelegate renderers
            enable_extension("omni.kit.viewport.pxr")
            # enable viewport extension if full rendering is enabled
            enable_extension("omni.kit.viewport.bundle")
            # extension for window status bar
            enable_extension("omni.kit.window.status_bar")
        # enable isaac replicator extension
        # note: moved here since it requires to have the viewport extension to be enabled first.
        enable_extension("omni.replicator.isaac")
        # enable urdf importer
        enable_extension("omni.isaac.urdf")

        # update the global flags
        # TODO: Remove all these global flags. We don't need it anymore.
        # -- render GUI
        if headless and (remote_deployment < 2):
            self.RENDER = False
        else:
            self.RENDER = True
        # -- render viewport
        if not viewport:
            self.VIEWPORT = False
        else:
            self.VIEWPORT = True

    @property
    def app(self) -> SimulationApp:
        """The launched SimulationApp."""
        if self._app is not None:
            return self._app
        else:
            raise RuntimeError("The `AppLauncher.app` member cannot be retrieved until the class is initialized.")
