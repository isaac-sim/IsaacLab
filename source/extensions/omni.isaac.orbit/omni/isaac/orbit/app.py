# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with the utility class to configure the :class:`omni.isaac.kit.SimulationApp`.

Based on the desired functionality, this class parses environment variables and input CLI arguments
to launch the simulator in various different modes. This includes with or without GUI and switching between
different Omniverse remote clients. Some of these require the
extensions to be loaded in a specific order, otherwise a segmentation fault occurs.
The launched `SimulationApp`_ instance is accessible via the :attr:`AppLauncher.app` property.

Environment variables
---------------------

The following details the behavior of the class based on the environment variables:

* **Headless mode**: If the environment variable ``HEADLESS=1``, then SimulationApp will be started in headless mode.
  If ``LIVESTREAM={1,2,3}``, then it will supersede the ``HEADLESS`` envvar and force headlessness.

  * ``HEADLESS=1`` causes the app to run in headless mode.

* **Livestreaming**: If the environment variable ``LIVESTREAM={1,2,3}`` , then `livestream`_ is enabled. Any
  of the livestream modes being true forces the app to run in headless mode.

  * ``LIVESTREAM=1`` enables streaming via the Isaac `Native Livestream`_ extension. This allows users to
    connect through the Omniverse Streaming Client.
  * ``LIVESTREAM=2`` enables streaming via the `Websocket Livestream`_ extension. This allows users to
    connect in a browser using the WebSocket protocol.
  * ``LIVESTREAM=3`` enables streaming  via the `WebRTC Livestream`_ extension. This allows users to
    connect in a browser using the WebRTC protocol.

* **Offscreen Render**: If the environment variable ``OFFSCREEN_RENDER`` is set to 1, then the
  offscreen-render pipeline is enabled. This is useful for running the simulator without a GUI but
  still rendering the viewport and camera images.

  * ``OFFSCREEN_RENDER=1``: Enables the offscreen-render pipeline which allows users to render
    the scene without launching a GUI.

  .. note::

      The off-screen rendering pipeline only works when used in conjunction with the
      :class:`omni.isaac.orbit.sim.SimulationContext` class. This is because the off-screen rendering
      pipeline enables flags that are internally used by the SimulationContext class.


To set the environment variables, one can use the following command in the terminal:

.. code:: bash

    export REMOTE_DEPLOYMENT=3
    export OFFSCREEN_RENDER=1
    # run the python script
    ./orbit.sh -p source/standalone/demo/play_quadrupeds.py

Alternatively, one can set the environment variables to the python script directly:

.. code:: bash

    REMOTE_DEPLOYMENT=3 OFFSCREEN_RENDER=1 ./orbit.sh -p source/standalone/demo/play_quadrupeds.py


Overriding the environment variables
------------------------------------

The environment variables can be overridden in the python script itself using the :class:`AppLauncher`.
These can be passed as a dictionary, a :class:`argparse.Namespace` object or as keyword arguments.
When the passed arguments are not the default values, then they override the environment variables.

The following snippet shows how use the :class:`AppLauncher` in different ways:

.. code:: python

    import argparser

    from omni.isaac.orbit.app import AppLauncher

    # add argparse arguments
    parser = argparse.ArgumentParser()
    # add your own arguments
    # ....
    # add app launcher arguments for cli
    AppLauncher.add_app_launcher_args(parser)
    # parse arguments
    args = parser.parse_args()

    # launch omniverse isaac-sim app
    # -- Option 1: Pass the settings as a Namespace object
    app_launcher = AppLauncher(args).app
    # -- Option 2: Pass the settings as keywords arguments
    app_launcher = AppLauncher(headless=args.headless, livestream=args.livestream)
    # -- Option 3: Pass the settings as a dictionary
    app_launcher = AppLauncher(vars(args))
    # -- Option 4: Pass no settings
    app_launcher = AppLauncher()

    # obtain the launched app
    simulation_app = app_launcher.app


.. _SimulationApp: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
.. _livestream: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html
.. _`Native Livestream`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html#isaac-sim-setup-kit-remote
.. _`Websocket Livestream`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html#isaac-sim-setup-livestream-webrtc
.. _`WebRTC Livestream`: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/manual_livestream_clients.html#isaac-sim-setup-livestream-websocket

"""

from __future__ import annotations

import argparse
import faulthandler
import os
import re
import sys
from typing import Any, Literal

from omni.isaac.kit import SimulationApp


class AppLauncher:
    """A utility class to launch Isaac Sim application based on command-line arguments and environment variables.

    The class resolves the simulation app settings that appear through environments variables,
    command-line arguments (CLI) or as input keyword arguments. Based on these settings, it launches the
    simulation app and configures the extensions to load (as a part of post-launch setup).

    The input arguments provided to the class are given higher priority than the values set
    from the corresponding environment variables. This provides flexibility to deal with different
    users' preferences.

    .. note::
        Explicitly defined arguments are only given priority when their value is set to something outside
        their default configuration. For example, the ``livestream`` argument is -1 by default. It only
        overrides the ``LIVESTREAM`` environment variable when ``livestream`` argument is set to a
        value >-1. In other words, if ``livestream=-1``, then the value from the environment variable
        ``LIVESTREAM`` is used.

    """

    def __init__(self, launcher_args: argparse.Namespace | dict = None, **kwargs):
        """Create a `SimulationApp`_ instance based on the input settings.

        Args:
            launcher_args: Input arguments to parse using the AppLauncher and set into the SimulationApp.
                Defaults to None, which is equivalent to passing an empty dictionary. A detailed description of
                the possible arguments is available in the `SimulationApp`_ documentation.
            **kwargs : Additional keyword arguments that will be merged into :attr:`launcher_args`.
                They serve as a convenience for those who want to pass some arguments using the argparse
                interface and others directly into the AppLauncher. Duplicated arguments with
                the :attr:`launcher_args` will raise a ValueError.

        Raises:
            ValueError: If there are common/duplicated arguments between ``launcher_args`` and ``kwargs``.
            ValueError: If combination of ``launcher_args`` and ``kwargs`` are missing the necessary arguments
                that are needed by the AppLauncher to resolve the desired app configuration.
            ValueError: If incompatible or undefined values are assigned to relevant environment values,
                such as ``LIVESTREAM``.

        .. _argparse.Namespace: https://docs.python.org/3/library/argparse.html?highlight=namespace#argparse.Namespace
        .. _SimulationApp: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
        """
        # Enable call-stack on crash
        faulthandler.enable()

        # We allow users to pass either a dict or an argparse.Namespace into
        # __init__, anticipating that these will be all of the argparse arguments
        # used by the calling script. Those which we appended via add_app_launcher_args
        # will be used to control extension loading logic. Additional arguments are allowed,
        # and will be passed directly to the SimulationApp initialization.
        #
        # We could potentially require users to enter each argument they want passed here
        # as a kwarg, but this would require them to pass livestream, headless, and
        # any other options we choose to add here explicitly, and with the correct keywords.
        #
        # @hunter: I feel that this is cumbersome and could introduce error, and would prefer to do
        # some sanity checking in the add_app_launcher_args function
        if launcher_args is None:
            launcher_args = {}
        elif isinstance(launcher_args, argparse.Namespace):
            launcher_args = launcher_args.__dict__

        # Check that arguments are unique
        if len(kwargs) > 0:
            if not set(kwargs.keys()).isdisjoint(launcher_args.keys()):
                overlapping_args = set(kwargs.keys()).intersection(launcher_args.keys())
                raise ValueError(
                    f"Input `launcher_args` and `kwargs` both provided common attributes: {overlapping_args}."
                    " Please ensure that each argument is supplied to only one of them, as the AppLauncher cannot"
                    " discern priority between them."
                )
            launcher_args.update(kwargs)

        # Define config members that are read from env-vars or keyword args
        self._headless: bool  # 0: GUI, 1: Headless
        self._livestream: Literal[0, 1, 2, 3]  # 0: Disabled, 1: Native, 2: Websocket, 3: WebRTC
        self._offscreen_render: bool  # 0: Disabled, 1: Enabled

        # Integrate env-vars and input keyword args into simulation app config
        self._config_resolution(launcher_args)
        # Create SimulationApp, passing the resolved self._config to it for initialization
        self._create_app()
        # Load IsaacSim extensions
        self._load_extensions()
        # Hide the stop button in the toolbar
        self._hide_stop_button()

    """
    Properties.
    """

    @property
    def app(self) -> SimulationApp:
        """The launched SimulationApp."""
        if self._app is not None:
            return self._app
        else:
            raise RuntimeError("The `AppLauncher.app` member cannot be retrieved until the class is initialized.")

    """
    Operations.
    """

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        """Utility function to configure AppLauncher arguments with an existing argument parser object.

        This function takes an ``argparse.ArgumentParser`` object and does some sanity checking on the existing
        arguments for ingestion by the SimulationApp. It then appends custom command-line arguments relevant
        to the SimulationApp to the input :class:`argparse.ArgumentParser` instance. This allows overriding the
        environment variables using command-line arguments.

        Currently, it adds the following parameters to the argparser object:

        * ``headless`` (bool): If True, the app will be launched in headless (no-gui) mode. The values map the same
          as that for the ``HEADLESS`` environment variable. If False, then headless mode is determined by the
          ``HEADLESS`` environment variable.
        * ``livestream`` (int): If one of {0, 1, 2, 3}, then livestreaming and headless mode is enabled. The values
          map the same as that for the ``LIVESTREAM`` environment variable. If :obj:`-1`, then livestreaming is
          determined by the ``LIVESTREAM`` environment variable.
        * ``offscreen_render`` (bool): If True, the app will be launched in offscreen-render mode. The values
          map the same as that for the ``OFFSCREEN_RENDER`` environment variable. If False, then offscreen-render
          mode is determined by the ``OFFSCREEN_RENDER`` environment variable.

        Args:
            parser: An argument parser instance to be extended with the AppLauncher specific options.
        """
        # If the passed parser has an existing _HelpAction when passed,
        # we here remove the options which would invoke it,
        # to be added back after the additional AppLauncher args
        # have been added. This is equivalent to
        # initially constructing the ArgParser with add_help=False,
        # but this means we don't have to require that behavior
        # in users and can handle it on our end.
        # We do this because calling parse_known_args() will handle
        # any -h/--help options being passed and then exit immediately,
        # before the additional arguments can be added to the help readout.
        parser_help = None
        if len(parser._actions) > 0 and isinstance(parser._actions[0], argparse._HelpAction):  # type: ignore
            parser_help = parser._actions[0]
            parser._option_string_actions.pop("-h")
            parser._option_string_actions.pop("--help")

        # Parse known args for potential name collisions/type mismatches
        # between the config fields SimulationApp expects and the ArgParse
        # arguments that the user passed.
        known, _ = parser.parse_known_args()
        config = vars(known)
        if len(config) == 0:
            print(
                "[Warn][AppLauncher]: There are no arguments attached to the ArgumentParser object."
                " If you have your own arguments, please load your own arguments before calling the"
                " `AppLauncher.add_app_launcher_args` method. This allows the method to check the validity"
                " of the arguments and perform checks for argument names."
            )
        else:
            AppLauncher._check_argparser_config_params(config)

        # Add custom arguments to the parser
        arg_group = parser.add_argument_group("app_launcher arguments")
        arg_group.add_argument(
            "--headless",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["headless"][1],
            help="Force display off at all times.",
        )
        arg_group.add_argument(
            "--livestream",
            type=int,
            default=AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1],
            choices={0, 1, 2, 3},
            help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.",
        )
        arg_group.add_argument(
            "--offscreen_render",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["offscreen_render"][1],
            help="Enable offscreen rendering when running without a GUI.",
        )

        # Corresponding to the beginning of the function,
        # if we have removed -h/--help handling, we add it back.
        if parser_help is not None:
            parser._option_string_actions["-h"] = parser_help
            parser._option_string_actions["--help"] = parser_help

    """
    Internal functions.
    """

    _APPLAUNCHER_CFG_INFO: dict[str, tuple[list[type], Any]] = {
        "headless": ([bool], False),
        "livestream": ([int], -1),
        "offscreen_render": ([bool], False),
    }
    """A dictionary of arguments added manually by the :meth:`AppLauncher.add_app_launcher_args` method.

    The values are a tuple of the expected type and default value. This is used to check against name collisions
    for arguments passed to the :class:`AppLauncher` class as well as for type checking.

    They have corresponding environment variables as detailed in the documentation.
    """

    # TODO: Find some internally managed NVIDIA list of these types.
    # SimulationApp.DEFAULT_LAUNCHER_CONFIG almost works, except that
    # it is ambiguous where the default types are None
    _SIMULATIONAPP_CFG_TYPES: dict[str, list[type]] = {
        "headless": [bool],
        "active_gpu": [int, type(None)],
        "physics_gpu": [int],
        "multi_gpu": [bool],
        "sync_loads": [bool],
        "width": [int],
        "height": [int],
        "window_width": [int],
        "window_height": [int],
        "display_options": [int],
        "subdiv_refinement_level": [int],
        "renderer": [str],
        "anti_aliasing": [int],
        "samples_per_pixel_per_frame": [int],
        "denoiser": [bool],
        "max_bounces": [int],
        "max_specular_transmission_bounces": [int],
        "max_volume_bounces": [int],
        "open_usd": [str, type(None)],
        "livesync_usd": [str, type(None)],
        "fast_shutdown": [bool],
        "experience": [str],
    }
    """A dictionary containing the type of arguments passed to SimulationApp.

    This is used to check against name collisions for arguments passed to the :class:`AppLauncher` class
    as well as for type checking. It corresponds closely to the :attr:`SimulationApp.DEFAULT_LAUNCHER_CONFIG`,
    but specifically denotes where None types are allowed.
    """

    @staticmethod
    def _check_argparser_config_params(config: dict) -> None:
        """Checks that input argparser object has parameters with valid settings with no name conflicts.

        First, we inspect the dictionary to ensure that the passed ArgParser object is not attempting to add arguments
        which should be assigned by calling :meth:`AppLauncher.add_app_launcher_args`.

        Then, we check that if the key corresponds to a config setting expected by SimulationApp, then the type of
        that key's value corresponds to the type expected by the SimulationApp. If it passes the check, the function
        prints out that the setting with be passed to the SimulationApp. Otherwise, we raise a ValueError exception.

        Args:
            config: A configuration parameters which will be passed to the SimulationApp constructor.

        Raises:
            ValueError: If a key is an already existing field in the configuration parameters but
                should be added by calling the :meth:`AppLauncher.add_app_launcher_args.
            ValueError: If keys corresponding to those used to initialize SimulationApp
                (as found in :attr:`_SIMULATIONAPP_CFG_TYPES`) are of the wrong value type.
        """
        # check that no config key conflicts with AppLauncher config names
        applauncher_keys = set(AppLauncher._APPLAUNCHER_CFG_INFO.keys())
        for key, value in config.items():
            if key in applauncher_keys:
                raise ValueError(
                    f"The passed ArgParser object already has the field '{key}'. This field will be added by"
                    " `AppLauncher.add_app_launcher_args()`, and should not be added directly. Please remove the"
                    " argument or rename it to a non-conflicting name."
                )
        # check that type of the passed keys are valid
        simulationapp_keys = set(AppLauncher._SIMULATIONAPP_CFG_TYPES.keys())
        for key, value in config.items():
            if key in simulationapp_keys:
                given_type = type(value)
                expected_types = AppLauncher._SIMULATIONAPP_CFG_TYPES[key]
                if type(value) not in set(expected_types):
                    raise ValueError(
                        f"Invalid value type for the argument '{key}': {given_type}. Expected one of {expected_types},"
                        " if intended to be ingested by the SimulationApp object. Please change the type if this"
                        " intended for the SimulationApp or change the name of the argument to avoid name conflicts."
                    )
                # Print out values which will be used
                print(f"[INFO][AppLauncher]: The argument '{key}' will be used to configure the SimulationApp.")

    def _config_resolution(self, launcher_args: dict):
        """Resolve the input arguments and environment variables.

        Args:
            launcher_args: A dictionary of all input arguments passed to the class object.
        """
        # Handle all control logic resolution

        # --LIVESTREAM logic--
        #
        livestream_env = int(os.environ.get("LIVESTREAM", 0))
        livestream_arg = launcher_args.pop("livestream", AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1])
        livestream_valid_vals = {0, 1, 2, 3}
        # Value checking on LIVESTREAM
        if livestream_env not in livestream_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `LIVESTREAM`: {livestream_env} ."
                f" Expected: {livestream_valid_vals}."
            )
        # We allow livestream kwarg to supersede LIVESTREAM envvar
        if livestream_arg >= 0:
            if livestream_arg in livestream_valid_vals:
                self._livestream = livestream_arg
                # print info that we overrode the env-var
                print(
                    f"[INFO][AppLauncher]: Input keyword argument `livestream={livestream_arg}` has overridden"
                    f" the environment variable `LIVESTREAM={livestream_env}`."
                )
            else:
                raise ValueError(
                    f"Invalid value for input keyword argument `livestream`: {livestream_arg} ."
                    f" Expected: {livestream_valid_vals}."
                )
        else:
            self._livestream = livestream_env

        # --HEADLESS logic--
        #
        # Resolve headless execution of simulation app
        # HEADLESS is initially passed as an int instead of
        # the bool of headless_arg to avoid messy string processing,
        headless_env = int(os.environ.get("HEADLESS", 0))
        headless_arg = launcher_args.pop("headless", AppLauncher._APPLAUNCHER_CFG_INFO["headless"][1])
        headless_valid_vals = {0, 1}
        # Value checking on HEADLESS
        if headless_env not in headless_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `HEADLESS`: {headless_env} . Expected: {headless_valid_vals}."
            )
        # We allow headless kwarg to supersede HEADLESS envvar if headless_arg does not have the default value
        # Note: Headless is always true when livestreaming
        if headless_arg is True:
            self._headless = headless_arg
        elif self._livestream in {1, 2, 3}:
            # we are always headless on the host machine
            self._headless = True
            # inform who has toggled the headless flag
            if self._livestream == livestream_arg:
                print(
                    f"[INFO][AppLauncher]: Input keyword argument `livestream={self._livestream}` has implicitly"
                    f" overridden the environment variable `HEADLESS={headless_env}` to True."
                )
            elif self._livestream == livestream_env:
                print(
                    f"[INFO][AppLauncher]: Environment variable `LIVESTREAM={self._livestream}` has implicitly"
                    f" overridden the environment variable `HEADLESS={headless_env}` to True."
                )
        else:
            # Headless needs to be a bool to be ingested by SimulationApp
            self._headless = bool(headless_env)
        # Headless needs to be passed to the SimulationApp so we keep it here
        launcher_args["headless"] = self._headless

        # --OFFSCREEN_RENDER logic--
        #
        # off-screen rendering
        offscreen_render_env = int(os.environ.get("OFFSCREEN_RENDER", 0))
        offscreen_render_arg = launcher_args.pop(
            "offscreen_render", AppLauncher._APPLAUNCHER_CFG_INFO["offscreen_render"][1]
        )
        offscreen_render_valid_vals = {0, 1}
        if offscreen_render_env not in offscreen_render_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `OFFSCREEN_RENDER`: {offscreen_render_env} ."
                f"Expected: {offscreen_render_valid_vals} ."
            )
        # We allow offscreen_render kwarg to supersede OFFSCREEN_RENDER envvar
        if offscreen_render_arg is True:
            self._offscreen_render = offscreen_render_arg
        else:
            self._offscreen_render = bool(offscreen_render_env)

        # Check if input keywords contain an 'experience' file setting
        # Note: since experience is taken as a separate argument by Simulation App, we store it separately
        self._simulationapp_experience = launcher_args.pop("experience", "")
        print(f"[INFO][AppLauncher]: Loading experience file: {self._simulationapp_experience} .")
        # Remove all values from input keyword args which are not meant for SimulationApp
        # Assign all the passed settings to a dictionary for the simulation app
        self._simulationapp_config = {
            key: launcher_args[key]
            for key in set(AppLauncher._SIMULATIONAPP_CFG_TYPES.keys()) & set(launcher_args.keys())
        }

    def _create_app(self):
        """Launch and create the SimulationApp based on the parsed simulation config."""
        # Initialize SimulationApp
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
        self._app = SimulationApp(self._simulationapp_config, experience=self._simulationapp_experience)
        # add orbit modules back to sys.modules
        for key, value in hacked_modules.items():
            sys.modules[key] = value

    def _load_extensions(self):
        """Load correct extensions based on AppLauncher's resolved config member variables."""
        # These have to be loaded after SimulationApp is initialized
        import carb
        from omni.isaac.core.utils.extensions import enable_extension

        # Retrieve carb settings for modification
        carb_settings_iface = carb.settings.get_settings()

        if self._livestream >= 1:
            # Ensure that a viewport exists in case an experience has been
            # loaded which does not load it by default
            enable_extension("omni.kit.viewport.window")
            # Set carb settings to allow for livestreaming
            carb_settings_iface.set_bool("/app/livestream/enabled", True)
            carb_settings_iface.set_bool("/app/window/drawMouse", True)
            carb_settings_iface.set_bool("/ngx/enabled", False)
            carb_settings_iface.set_string("/app/livestream/proto", "ws")
            carb_settings_iface.set_int("/app/livestream/websocket/framerate_limit", 120)
            # Note: Only one livestream extension can be enabled at a time
            if self._livestream == 1:
                # Enable Native Livestream extension
                # Default App: Streaming Client from the Omniverse Launcher
                enable_extension("omni.kit.livestream.native")
                enable_extension("omni.services.streaming.manager")
            elif self._livestream == 2:
                # Enable WebSocket Livestream extension
                # Default URL: http://localhost:8211/streaming/client/
                enable_extension("omni.services.streamclient.websocket")
            elif self._livestream == 3:
                # Enable WebRTC Livestream extension
                # Default URL: http://localhost:8211/streaming/webrtc-client/
                enable_extension("omni.services.streamclient.webrtc")
            else:
                raise ValueError(f"Invalid value for livestream: {self._livestream}. Expected: 1, 2, 3 .")
        else:
            carb_settings_iface.set_bool("/app/livestream/enabled", False)

        # set carb setting to indicate orbit's offscreen_render pipeline should be enabled
        # this flag is used by the SimulationContext class to enable the offscreen_render pipeline
        # when the render() method is called.
        carb_settings_iface.set_bool("/orbit/offscreen_render/enabled", self._offscreen_render)

        # enable extensions for off-screen rendering
        # Depending on the app file, some extensions might not be available in it.
        # Thus, we manually enable these extensions to make sure they are available.
        # note: enabling extensions is order-sensitive. please do not change the order!
        if self._offscreen_render or not self._headless or self._livestream >= 1:
            # extension to enable UI buttons (otherwise we get attribute errors)
            enable_extension("omni.kit.window.toolbar")

        if self._offscreen_render or not self._headless:
            # extension to make RTX realtime and path-traced renderers
            enable_extension("omni.kit.viewport.rtx")
            # extension to make HydraDelegate renderers
            enable_extension("omni.kit.viewport.pxr")
            # enable viewport extension if full rendering is enabled
            enable_extension("omni.kit.viewport.bundle")
            # extension for window status bar
            enable_extension("omni.kit.window.status_bar")
        # enable replicator extension
        # note: moved here since it requires to have the viewport extension to be enabled first.
        enable_extension("omni.replicator.core")
        # enable UI tools
        # note: we need to always import this even with headless to make
        #   the module for orbit.envs.ui work
        enable_extension("omni.isaac.ui")
        # enable animation recording extension
        enable_extension("omni.kit.stagerecorder.core")

        # set the nucleus directory manually to the latest published Nucleus
        # note: this is done to ensure prior versions of Isaac Sim still use the latest assets
        carb_settings_iface.set_string(
            "/persistent/isaac/asset_root/default",
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1",
        )
        carb_settings_iface.set_string(
            "/persistent/isaac/asset_root/nvidia",
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1",
        )

    def _hide_stop_button(self):
        """Hide the stop button in the toolbar.

        For standalone executions, having a stop button is confusing since it invalidates the whole simulation.
        Thus, we hide the button so that users don't accidentally click it.
        """
        # when we are truly headless, then we can't import the widget toolbar
        # thus, we only hide the stop button when we are not headless (i.e. GUI is enabled)
        if self._livestream >= 1 or not self._headless:
            import omni.kit.widget.toolbar

            # grey out the stop button because we don't want to stop the simulation manually in standalone mode
            toolbar = omni.kit.widget.toolbar.get_instance()
            play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore
            if play_button_group is not None:
                play_button_group._stop_button.visible = False  # type: ignore
                play_button_group._stop_button.enabled = False  # type: ignore
                play_button_group._stop_button = None  # type: ignore
