# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with the utility class to configure the :class:`isaacsim.simulation_app.SimulationApp`.

The :class:`AppLauncher` parses environment variables and input CLI arguments to launch the simulator in
various different modes. This includes with or without GUI and switching between different Omniverse remote
clients. Some of these require the extensions to be loaded in a specific order, otherwise a segmentation
fault occurs. The launched :class:`isaacsim.simulation_app.SimulationApp` instance is accessible via the
:attr:`AppLauncher.app` property.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import signal
import sys
from typing import Any, Literal

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from isaacsim import SimulationApp

from isaaclab.app.settings_manager import get_settings_manager, initialize_carb_settings

# import logger
logger = logging.getLogger(__name__)


class ExplicitAction(argparse.Action):
    """Custom action to track if an argument was explicitly passed by the user."""

    def __call__(self, parser, namespace, values, option_string=None):
        # Set the parameter value
        setattr(namespace, self.dest, values)
        # Set a flag indicating the parameter was explicitly passed
        setattr(namespace, f"{self.dest}_explicit", True)


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

    def __init__(self, launcher_args: argparse.Namespace | dict | None = None, **kwargs):
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
        .. _SimulationApp: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.simulation_app/docs/index.html#isaacsim.simulation_app.SimulationApp
        """
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
        self._livestream: Literal[0, 1, 2]  # 0: Disabled, 1: WebRTC public, 2: WebRTC private
        self._offscreen_render: bool  # 0: Disabled, 1: Enabled
        self._sim_experience_file: str  # Experience file to load

        # Exposed to train scripts
        self.device_id: int  # device ID for GPU simulation (defaults to 0)
        self.local_rank: int  # local rank of GPUs in the current node
        self.global_rank: int  # global rank for multi-node training

        # Integrate env-vars and input keyword args into simulation app config
        self._config_resolution(launcher_args)

        # Internal: Override SimulationApp._start_app method to apply patches after app has started.
        self.__patch_simulation_start_app(launcher_args)

        # Create SimulationApp, passing the resolved self._config to it for initialization
        self._create_app()
        # Load IsaacSim extensions
        self._load_extensions()
        # Hide the stop button in the toolbar
        self._hide_stop_button()
        # Set settings from the given rendering mode
        self._set_rendering_mode_settings(launcher_args)
        # Set animation recording settings
        self._set_animation_recording_settings(launcher_args)
        # Set visualizer settings (if requested)
        self._set_visualizer_settings(launcher_args)

        # Hide play button callback if the timeline is stopped
        import omni.timeline

        self._hide_play_button_callback = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP), lambda e: self._hide_play_button(True)
            )
        )
        self._unhide_play_button_callback = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.PLAY), lambda e: self._hide_play_button(False)
            )
        )
        # Set up signal handlers for graceful shutdown
        # -- during explicit `kill` commands
        signal.signal(signal.SIGTERM, self._abort_signal_handle_callback)
        # -- during segfaults
        signal.signal(signal.SIGABRT, self._abort_signal_handle_callback)
        signal.signal(signal.SIGSEGV, self._abort_signal_handle_callback)

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
        * ``livestream`` (int): If one of {1, 2}, then livestreaming and headless mode is enabled. The values
          map the same as that for the ``LIVESTREAM`` environment variable. If :obj:`-1`, then livestreaming is
          determined by the ``LIVESTREAM`` environment variable.
          Valid options are:

          - ``0``: Disabled
          - ``1``: `WebRTC`_ over public network
          - ``2``: `WebRTC`_ over local/private network

        * ``enable_cameras`` (bool): If True, the app will enable camera sensors and render them, even when in
          headless mode. This flag must be set to True if the environments contains any camera sensors.
          The values map the same as that for the ``ENABLE_CAMERAS`` environment variable.
          If False, then enable_cameras mode is determined by the ``ENABLE_CAMERAS`` environment variable.
        * ``device`` (str): The device to run the simulation on.
          Valid options are:

          - ``cpu``: Use CPU.
          - ``cuda``: Use GPU with device ID ``0``.
          - ``cuda:N``: Use GPU, where N is the device ID. For example, "cuda:0".

        * ``experience`` (str): The experience file to load when launching the SimulationApp. If a relative path
          is provided, it is resolved relative to the ``apps`` folder in Isaac Sim and Isaac Lab (in that order).

          If provided as an empty string, the experience file is determined based on the command-line flags:

          * If headless and enable_cameras are True, the experience file is set to
            ``isaaclab.python.headless.rendering.kit``.
          * If headless is False and enable_cameras is True, the experience file is set to
            ``isaaclab.python.rendering.kit``.
          * If headless and enable_cameras are False, the experience file is set to
            ``isaaclab.python.kit``.
          * If headless is True and enable_cameras is False, the experience file is set to
            ``isaaclab.python.headless.kit``.

        * ``kit_args`` (str): Optional command line arguments to be passed to Omniverse Kit directly.
          Arguments should be combined into a single string separated by space.
          Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"


        .. _`WebRTC`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client

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
                "[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object."
                " If you have your own arguments, please load your own arguments before calling the"
                " `AppLauncher.add_app_launcher_args` method. This allows the method to check the validity"
                " of the arguments and perform checks for argument names."
            )
        else:
            AppLauncher._check_argparser_config_params(config)

        # Add custom arguments to the parser
        arg_group = parser.add_argument_group(
            "app_launcher arguments",
            description="Arguments for the AppLauncher. For more details, please check the documentation.",
        )
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
            choices={0, 1, 2},
            help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.",
        )
        arg_group.add_argument(
            "--enable_cameras",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["enable_cameras"][1],
            help="Enable camera sensors and relevant extension dependencies.",
        )
        arg_group.add_argument(
            "--xr",
            action="store_true",
            default=AppLauncher._APPLAUNCHER_CFG_INFO["xr"][1],
            help="Enable XR mode for VR/AR applications.",
        )
        arg_group.add_argument(
            "--device",
            type=str,
            action=ExplicitAction,
            default=AppLauncher._APPLAUNCHER_CFG_INFO["device"][1],
            help='The device to run the simulation on. Can be "cpu", "cuda", "cuda:N", where N is the device ID',
        )
        arg_group.add_argument(
            "--visualizer",
            type=str,
            nargs="+",
            default=None,
            help="Visualizer backends to enable (e.g., kit, newton, rerun).",
        )
        # Add the deprecated cpu flag to raise an error if it is used
        arg_group.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
        arg_group.add_argument(
            "--verbose",  # Note: This is read by SimulationApp through sys.argv
            action="store_true",
            help="Enable verbose-level log output from the SimulationApp.",
        )
        arg_group.add_argument(
            "--info",  # Note: This is read by SimulationApp through sys.argv
            action="store_true",
            help="Enable info-level log output from the SimulationApp.",
        )
        arg_group.add_argument(
            "--experience",
            type=str,
            default="",
            help=(
                "The experience file to load when launching the SimulationApp. If an empty string is provided,"
                " the experience file is determined based on the headless flag. If a relative path is provided,"
                " it is resolved relative to the `apps` folder in Isaac Sim and Isaac Lab (in that order)."
            ),
        )
        arg_group.add_argument(
            "--rendering_mode",
            type=str,
            action=ExplicitAction,
            choices={"performance", "balanced", "quality"},
            help=(
                "Sets the rendering mode. Preset settings files can be found in apps/rendering_modes."
                ' Can be "performance", "balanced", or "quality".'
                " Individual settings can be overwritten by using the RenderCfg class."
            ),
        )
        arg_group.add_argument(
            "--kit_args",
            type=str,
            default="",
            help=(
                "Command line arguments for Omniverse Kit as a string separated by a space delimiter."
                ' Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"'
            ),
        )
        arg_group.add_argument(
            "--anim_recording_enabled",
            action="store_true",
            help="Enable recording time-sampled USD animations from IsaacLab PhysX simulations.",
        )
        arg_group.add_argument(
            "--anim_recording_start_time",
            type=float,
            default=0,
            help=(
                "Set time that animation recording begins playing. If not set, the recording will start from the"
                " beginning."
            ),
        )
        arg_group.add_argument(
            "--anim_recording_stop_time",
            type=float,
            default=10,
            help=(
                "Set time that animation recording stops playing. If the process is shutdown before the stop time is"
                " exceeded, then the animation is not recorded."
            ),
        )
        # special flag for backwards compatibility

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
        "enable_cameras": ([bool], False),
        "xr": ([bool], False),
        "device": ([str], "cuda:0"),
        "experience": ([str], ""),
        "rendering_mode": ([str], "balanced"),
    }
    """A dictionary of arguments added manually by the :meth:`AppLauncher.add_app_launcher_args` method.

    The values are a tuple of the expected type and default value. This is used to check against name collisions
    for arguments passed to the :class:`AppLauncher` class as well as for type checking.

    They have corresponding environment variables as detailed in the documentation.
    """

    # TODO: Find some internally managed NVIDIA list of these types.
    # SimulationApp.DEFAULT_LAUNCHER_CONFIG almost works, except that
    # it is ambiguous where the default types are None
    _SIM_APP_CFG_TYPES: dict[str, list[type]] = {
        "headless": [bool],
        "hide_ui": [bool, type(None)],
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
                (as found in :attr:`_SIM_APP_CFG_TYPES`) are of the wrong value type.
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
        simulationapp_keys = set(AppLauncher._SIM_APP_CFG_TYPES.keys())
        for key, value in config.items():
            if key in simulationapp_keys:
                given_type = type(value)
                expected_types = AppLauncher._SIM_APP_CFG_TYPES[key]
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
        # Handle core settings
        livestream_arg, livestream_env = self._resolve_livestream_settings(launcher_args)
        self._resolve_headless_settings(launcher_args, livestream_arg, livestream_env)
        self._resolve_camera_settings(launcher_args)
        self._resolve_xr_settings(launcher_args)
        self._resolve_viewport_settings(launcher_args)

        # Handle device and distributed settings
        self._resolve_device_settings(launcher_args)

        # Handle experience file settings
        self._resolve_experience_file(launcher_args)

        # Handle animation recording settings
        self._resolve_anim_recording_settings(launcher_args)

        # Handle additional arguments
        self._resolve_kit_args(launcher_args)

        # Prepare final simulation app config
        # Remove all values from input keyword args which are not meant for SimulationApp
        # Assign all the passed settings to a dictionary for the simulation app
        self._sim_app_config = {
            key: launcher_args[key] for key in set(AppLauncher._SIM_APP_CFG_TYPES.keys()) & set(launcher_args.keys())
        }

    def _resolve_livestream_settings(self, launcher_args: dict) -> tuple[int, int]:
        """Resolve livestream related settings."""
        livestream_env = int(os.environ.get("LIVESTREAM", 0))
        livestream_arg = launcher_args.pop("livestream", AppLauncher._APPLAUNCHER_CFG_INFO["livestream"][1])
        livestream_valid_vals = {0, 1, 2}
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

        # Set public IP address of a remote instance
        public_ip_env = os.environ.get("PUBLIC_IP", "127.0.0.1")

        # Process livestream here before launching kit because some of the extensions only work
        # when launched with the kit file
        self._livestream_args = []
        if self._livestream >= 1:
            # Note: Only one livestream extension can be enabled at a time
            if self._livestream == 1:
                # WebRTC public network
                self._livestream_args += [
                    f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={public_ip_env}",
                    "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100",
                    "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998",
                    "--enable",
                    "omni.kit.livestream.app",
                ]
            elif self._livestream == 2:
                # WebRTC private network
                self._livestream_args += [
                    "--enable",
                    "omni.kit.livestream.app",
                ]
            else:
                raise ValueError(f"Invalid value for livestream: {self._livestream}. Expected: 1, 2 .")
            sys.argv += self._livestream_args

        return livestream_arg, livestream_env

    def _resolve_headless_settings(self, launcher_args: dict, livestream_arg: int, livestream_env: int):
        """Resolve headless related settings."""
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
        elif self._livestream in {1, 2}:
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

        # If visualizers are explicitly requested and Kit viewport is not among them,
        # force headless mode so Isaac Sim GUI does not launch unnecessarily.
        visualizers_arg = launcher_args.get("visualizer")
        if visualizers_arg:
            requested_visualizers = {str(v).strip().lower() for v in visualizers_arg if str(v).strip()}
            if requested_visualizers and "kit" not in requested_visualizers and self._livestream == 0:
                if not self._headless:
                    print(
                        "[INFO][AppLauncher]: Forcing headless mode because '--visualizer' excludes "
                        "'kit' and livestream is disabled."
                    )
                self._headless = True
        # Headless needs to be passed to the SimulationApp so we keep it here
        launcher_args["headless"] = self._headless

    def _resolve_camera_settings(self, launcher_args: dict):
        """Resolve camera related settings."""
        enable_cameras_env = int(os.environ.get("ENABLE_CAMERAS", 0))
        enable_cameras_arg = launcher_args.get("enable_cameras", AppLauncher._APPLAUNCHER_CFG_INFO["enable_cameras"][1])
        enable_cameras_valid_vals = {0, 1}
        if enable_cameras_env not in enable_cameras_valid_vals:
            raise ValueError(
                f"Invalid value for environment variable `ENABLE_CAMERAS`: {enable_cameras_env} ."
                f"Expected: {enable_cameras_valid_vals} ."
            )
        # We allow enable_cameras kwarg to supersede ENABLE_CAMERAS envvar
        if enable_cameras_arg is True:
            self._enable_cameras = enable_cameras_arg
        else:
            self._enable_cameras = bool(enable_cameras_env)
        self._offscreen_render = False
        if self._enable_cameras and self._headless:
            self._offscreen_render = True

    def _resolve_xr_settings(self, launcher_args: dict):
        """Resolve XR related settings."""
        xr_env = int(os.environ.get("XR", 0))
        xr_arg = launcher_args.get("xr", AppLauncher._APPLAUNCHER_CFG_INFO["xr"][1])
        xr_valid_vals = {0, 1}
        if xr_env not in xr_valid_vals:
            raise ValueError(f"Invalid value for environment variable `XR`: {xr_env} .Expected: {xr_valid_vals} .")
        # We allow xr kwarg to supersede XR envvar
        if xr_arg is True:
            self._xr = xr_arg
        else:
            self._xr = bool(xr_env)

    def _resolve_viewport_settings(self, launcher_args: dict):
        """Resolve viewport related settings."""
        # Check if we can disable the viewport to improve performance
        #   This should only happen if we are running headless and do not require livestreaming or video recording
        #   This is different from offscreen_render because this only affects the default viewport and
        #   not other render-products in the scene
        self._render_viewport = True
        if self._headless and not self._livestream and not launcher_args.get("video", False):
            self._render_viewport = False

        # hide_ui flag
        launcher_args["hide_ui"] = False
        if self._headless and not self._livestream:
            launcher_args["hide_ui"] = True

        # avoid creating new stage at startup by default for performance reasons
        launcher_args["create_new_stage"] = False

    def _resolve_device_settings(self, launcher_args: dict):
        """Resolve simulation GPU device related settings."""
        self.device_id = 0
        device = launcher_args.get("device", AppLauncher._APPLAUNCHER_CFG_INFO["device"][1])

        device_explicitly_passed = launcher_args.pop("device_explicit", False)
        if self._xr and not device_explicitly_passed:
            # If no device is specified, default to the CPU device if we are running in XR
            device = "cpu"

            # Overwrite for downstream consumers
            launcher_args["device"] = "cpu"

        if "cuda" not in device and "cpu" not in device:
            raise ValueError(
                f"Invalid value for input keyword argument `device`: {device}."
                " Expected: a string with the format 'cuda', 'cuda:<device_id>', or 'cpu'."
            )

        if "cuda:" in device:
            self.device_id = int(device.split(":")[-1])

        # Raise an error for the deprecated cpu flag
        if launcher_args.get("cpu", False):
            raise ValueError("The `--cpu` flag is deprecated. Please use `--device cpu` instead.")

        if "distributed" in launcher_args and launcher_args["distributed"]:
            # local rank (GPU id) in a current multi-gpu mode
            self.local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))
            # global rank (GPU id) in multi-gpu multi-node mode
            self.global_rank = int(os.getenv("RANK", "0")) + int(os.getenv("JAX_RANK", "0"))

            self.device_id = self.local_rank
            device = "cuda:" + str(self.device_id)
            launcher_args["multi_gpu"] = False
            # limit CPU threads to minimize thread context switching
            # this ensures processes do not take up all available threads and fight for resources
            num_cpu_cores = os.cpu_count()
            num_threads_per_process = num_cpu_cores // int(os.getenv("WORLD_SIZE", 1))
            # set environment variables to limit CPU threads
            os.environ["PXR_WORK_THREAD_LIMIT"] = str(num_threads_per_process)
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads_per_process)
            # pass command line variable to kit
            sys.argv.append(f"--/plugins/carb.tasking.plugin/threadCount={num_threads_per_process}")

        # set rendering device. We do not need to set physics_gpu because it will automatically pick the same one
        # as the active_gpu device. Setting physics_gpu explicitly may result in a different device to be used.
        launcher_args["physics_gpu"] = self.device_id
        launcher_args["active_gpu"] = self.device_id

        print(f"[INFO][AppLauncher]: Using device: {device}")

    def _resolve_experience_file(self, launcher_args: dict):
        """Resolve experience file related settings."""
        # Check if input keywords contain an 'experience' file setting
        # Note: since experience is taken as a separate argument by Simulation App, we store it separately
        self._sim_experience_file = launcher_args.pop("experience", "")

        # If nothing is provided resolve the experience file based on the headless flag
        kit_app_exp_path = os.environ["EXP_PATH"]
        isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
        # For Isaac Sim 4.5 compatibility, we use the 4.5 app files in a different folder
        # if launcher_args.get("use_isaacsim_45", False):
        if self.is_isaac_sim_version_5():
            isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_5")

        if self._sim_experience_file == "":
            # check if the headless flag is set
            # xr rendering overrides camera rendering settings
            if self._enable_cameras and not self._xr:
                if self._headless and not self._livestream:
                    self._sim_experience_file = os.path.join(
                        isaaclab_app_exp_path, "isaaclab.python.headless.rendering.kit"
                    )
                else:
                    self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.rendering.kit")
            elif self._xr:
                if self._headless:
                    self._sim_experience_file = os.path.join(
                        isaaclab_app_exp_path, "isaaclab.python.xr.openxr.headless.kit"
                    )
                else:
                    self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.xr.openxr.kit")
            elif self._headless and not self._livestream:
                self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.headless.kit")
            else:
                self._sim_experience_file = os.path.join(isaaclab_app_exp_path, "isaaclab.python.kit")
        elif not os.path.isabs(self._sim_experience_file):
            option_1_app_exp_path = os.path.join(kit_app_exp_path, self._sim_experience_file)
            option_2_app_exp_path = os.path.join(isaaclab_app_exp_path, self._sim_experience_file)
            if os.path.exists(option_1_app_exp_path):
                self._sim_experience_file = option_1_app_exp_path
            elif os.path.exists(option_2_app_exp_path):
                self._sim_experience_file = option_2_app_exp_path
            else:
                raise FileNotFoundError(
                    f"Invalid value for input keyword argument `experience`: {self._sim_experience_file}."
                    "\n No such file exists in either the Kit or Isaac Lab experience paths. Checked paths:"
                    f"\n\t [1]: {option_1_app_exp_path}"
                    f"\n\t [2]: {option_2_app_exp_path}"
                )
        elif not os.path.exists(self._sim_experience_file):
            raise FileNotFoundError(
                f"Invalid value for input keyword argument `experience`: {self._sim_experience_file}."
                " The file does not exist."
            )

        # Resolve the absolute path of the experience file
        self._sim_experience_file = os.path.abspath(self._sim_experience_file)
        print(f"[INFO][AppLauncher]: Loading experience file: {self._sim_experience_file}")

    def _resolve_anim_recording_settings(self, launcher_args: dict):
        """Resolve animation recording settings."""

        # Enable omni.physx.pvd extension if recording is enabled
        recording_enabled = launcher_args.get("anim_recording_enabled", False)
        if recording_enabled:
            if self._headless:
                raise ValueError("Animation recording is not supported in headless mode.")
            sys.argv += ["--enable", "omni.physx.pvd"]

    def _resolve_kit_args(self, launcher_args: dict):
        """Resolve additional arguments passed to Kit."""
        # Resolve additional arguments passed to Kit
        self._kit_args = []
        if "kit_args" in launcher_args:
            self._kit_args = [arg for arg in launcher_args["kit_args"].split()]
            sys.argv += self._kit_args

    def _create_app(self):
        """Launch and create the SimulationApp based on the parsed simulation config."""
        # Initialize SimulationApp
        # hack sys module to make sure that the SimulationApp is initialized correctly
        # this is to avoid the warnings from the simulation app about not ok modules
        r = re.compile(".*lab.*")
        found_modules = list(filter(r.match, list(sys.modules.keys())))
        # remove Isaac Lab modules from sys.modules
        hacked_modules = dict()
        for key in found_modules:
            hacked_modules[key] = sys.modules[key]
            del sys.modules[key]

        # disable sys stdout and stderr to avoid printing the warning messages
        # this is mainly done to purge the print statements from the simulation app
        if "--verbose" not in sys.argv and "--info" not in sys.argv:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115

        # pytest may have left some things in sys.argv, this will check for some of those
        # do a mark and sweep to remove any -m pytest and -m isaacsim_ci and -c **/pyproject.toml
        indexes_to_remove = []
        for idx, arg in enumerate(sys.argv[:-1]):
            if arg == "-m":
                value_for_dash_m = sys.argv[idx + 1]
                if "pytest" in value_for_dash_m or "isaacsim_ci" in value_for_dash_m:
                    indexes_to_remove.append(idx)
                    indexes_to_remove.append(idx + 1)
            if arg.startswith("--config-file=") and "pyproject.toml" in arg:
                indexes_to_remove.append(idx)
            if arg == "--capture=no":
                indexes_to_remove.append(idx)
        for idx in sorted(indexes_to_remove, reverse=True):
            sys.argv = sys.argv[:idx] + sys.argv[idx + 1 :]

        # launch simulation app
        self._app = SimulationApp(self._sim_app_config, experience=self._sim_experience_file)
        # enable sys stdout and stderr
        sys.stdout = sys.__stdout__

        # add Isaac Lab modules back to sys.modules
        for key, value in hacked_modules.items():
            sys.modules[key] = value
        # remove the threadCount argument from sys.argv if it was added for distributed training
        pattern = r"--/plugins/carb\.tasking\.plugin/threadCount=\d+"
        sys.argv = [arg for arg in sys.argv if not re.match(pattern, arg)]

        # remove additional OV args from sys.argv
        if len(self._kit_args) > 0:
            sys.argv = [arg for arg in sys.argv if arg not in self._kit_args]
        if len(self._livestream_args) > 0:
            sys.argv = [arg for arg in sys.argv if arg not in self._livestream_args]

    def _rendering_enabled(self) -> bool:
        """Check if rendering is required by the app."""
        # Indicates whether rendering is required by the app.
        # Extensions required for rendering bring startup and simulation costs, so we do not
        # enable them if not required.
        return not self._headless or self._livestream >= 1 or self._enable_cameras or self._xr

    def _load_extensions(self):
        """Load correct extensions based on AppLauncher's resolved config member variables."""
        # These have to be loaded after SimulationApp is initialized.
        # Use SettingsManager (backs onto carb when in Omniverse after initialize_carb_settings).
        initialize_carb_settings()
        settings = get_settings_manager()

        # set setting to indicate Isaac Lab's offscreen_render pipeline should be enabled
        settings.set_bool("/isaaclab/render/offscreen", self._offscreen_render)

        # set setting to indicate Isaac Lab's render_viewport pipeline should be enabled
        settings.set_bool("/isaaclab/render/active_viewport", self._render_viewport)

        # set setting to indicate no RTX sensors are used (set to True when RTX sensor is created)
        settings.set_bool("/isaaclab/render/rtx_sensors", False)

        # set fabric update flag to disable updating transforms when rendering is disabled
        settings.set_bool("/physics/fabricUpdateTransformations", self._rendering_enabled())

        # use fixed time stepping disabled; custom loop runner from Isaac Sim is used instead
        settings.set_bool("/app/player/useFixedTimeStepping", False)

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

    def _set_rendering_mode_settings(self, launcher_args: dict) -> None:
        """Store RTX rendering mode in settings."""
        rendering_mode = launcher_args.get("rendering_mode")

        if rendering_mode is None:
            # use default kit rendering settings if cameras are disabled and a rendering mode is not selected
            if not self._enable_cameras:
                return
            rendering_mode = ""

        get_settings_manager().set_string("/isaaclab/rendering/rendering_mode", rendering_mode)

    def _set_animation_recording_settings(self, launcher_args: dict) -> None:
        """Store animation recording settings in settings."""
        recording_enabled = launcher_args.get("anim_recording_enabled", False)
        if not recording_enabled:
            return

        # arg checks
        if launcher_args.get("anim_recording_start_time") >= launcher_args.get("anim_recording_stop_time"):
            raise ValueError(
                f"'anim_recording_start_time' {launcher_args.get('anim_recording_start_time')} must be less than"
                f" 'anim_recording_stop_time' {launcher_args.get('anim_recording_stop_time')}"
            )

        start_time = launcher_args.get("anim_recording_start_time")
        stop_time = launcher_args.get("anim_recording_stop_time")

        settings = get_settings_manager()
        settings.set_bool("/isaaclab/anim_recording/enabled", recording_enabled)
        settings.set_float("/isaaclab/anim_recording/start_time", start_time)
        settings.set_float("/isaaclab/anim_recording/stop_time", stop_time)

    def _set_visualizer_settings(self, launcher_args: dict) -> None:
        """Store visualizer selection in settings."""
        visualizers = launcher_args.get("visualizer")
        if not visualizers:
            return
        with contextlib.suppress(Exception):
            visualizer_str = " ".join(visualizers)
            get_settings_manager().set_string("/isaaclab/visualizer", visualizer_str)

    def _interrupt_signal_handle_callback(self, signal, frame):
        """Handle the interrupt signal from the keyboard."""
        # close the app
        self._app.close()
        # raise the error for keyboard interrupt
        raise KeyboardInterrupt

    def is_isaac_sim_version_5(self) -> bool:
        if not hasattr(self, "_is_sim_ver_5"):
            # 1) Try to read the VERSION file (for manual / binary installs)
            version_path = os.path.abspath(os.path.join(os.path.dirname(isaacsim.__file__), "../../VERSION"))
            if os.path.isfile(version_path):
                with open(version_path) as f:
                    ver = f.readline().strip()
                    if ver.startswith("5"):
                        self._is_sim_ver_5 = True
                        return True

            # 2) Fall back to metadata (for pip installs)
            from importlib.metadata import version as pkg_version

            try:
                ver = pkg_version("isaacsim")
                if ver.startswith("5"):
                    self._is_sim_ver_5 = True
                else:
                    self._is_sim_ver_5 = False
            except Exception:
                self._is_sim_ver_5 = False
        return self._is_sim_ver_5

    def _hide_play_button(self, flag):
        """Hide/Unhide the play button in the toolbar.

        This is used if the timeline is stopped by a GUI action like "save as" to not allow the user to
        resume the timeline afterwards.
        """
        # when we are truly headless, then we can't import the widget toolbar
        # thus, we only hide the play button when we are not headless (i.e. GUI is enabled)
        if self._livestream >= 1 or not self._headless:
            import omni.kit.widget.toolbar

            toolbar = omni.kit.widget.toolbar.get_instance()
            play_button_group = toolbar._builtin_tools._play_button_group  # type: ignore
            if play_button_group is not None:
                play_button_group._play_button.visible = not flag  # type: ignore
                play_button_group._play_button.enabled = not flag  # type: ignore

    def _abort_signal_handle_callback(self, signal, frame):
        """Handle the abort/segmentation/kill signals."""
        # close the app
        self._app.close()

    def __patch_simulation_start_app(self, launcher_args: dict):
        if not launcher_args.get("enable_pinocchio", False):
            return

        if launcher_args.get("disable_pinocchio_patch", False):
            return

        original_start_app = SimulationApp._start_app

        def _start_app_patch(sim_app_instance, *args, **kwargs):
            original_start_app(sim_app_instance, *args, **kwargs)
            self.__patch_pxr_gf_matrix4d(launcher_args)

        SimulationApp._start_app = _start_app_patch

    def __patch_pxr_gf_matrix4d(self, launcher_args: dict):
        import traceback

        from pxr import Gf

        logger.warning(
            "Due to an issue with Pinocchio and pxr.Gf.Matrix4d, patching the Matrix4d constructor to convert arguments"
            " into a list of floats."
        )

        # Store the original Matrix4d constructor
        original_matrix4d = Gf.Matrix4d.__init__

        # Define a wrapper function to handle different input types
        def patch_matrix4d(self, *args, **kwargs):
            try:
                # Case 1: No arguments (identity matrix)
                if len(args) == 0:
                    original_matrix4d(self, *args, **kwargs)
                    return

                # Case 2: Single argument
                elif len(args) == 1:
                    arg = args[0]

                    # Case 2a: Already a Matrix4d
                    if isinstance(arg, Gf.Matrix4d):
                        original_matrix4d(self, arg)
                        return

                    # Case 2b: Tuple of tuples (4x4 matrix) OR List of lists (4x4 matrix)
                    elif (isinstance(arg, tuple) and len(arg) == 4 and all(isinstance(row, tuple) for row in arg)) or (
                        isinstance(arg, list) and len(arg) == 4 and all(isinstance(row, list) for row in arg)
                    ):
                        float_list = [float(item) for row in arg for item in row]
                        original_matrix4d(self, *float_list)
                        return

                    # Case 2c: Flat list of 16 elements
                    elif isinstance(arg, (list, tuple)) and len(arg) == 16:
                        float_list = [float(item) for item in arg]
                        original_matrix4d(self, *float_list)
                        return

                    # Case 2d: Another matrix-like object with elements accessible via indexing
                    elif hasattr(arg, "__getitem__") and hasattr(arg, "__len__"):
                        with contextlib.suppress(IndexError, TypeError):
                            if len(arg) == 16:
                                float_list = [float(arg[i]) for i in range(16)]
                                original_matrix4d(self, *float_list)
                                return
                            # Try to extract as 4x4 matrix
                            elif len(arg) == 4 and all(len(row) == 4 for row in arg):
                                float_list = [float(arg[i][j]) for i in range(4) for j in range(4)]
                                original_matrix4d(self, *float_list)
                                return

                # Case 3: 16 separate arguments (individual matrix elements)
                elif len(args) == 16:
                    float_list = [float(arg) for arg in args]
                    original_matrix4d(self, *float_list)
                    return

                # Default: Use original constructor
                original_matrix4d(self, *args, **kwargs)

            except Exception as e:
                logger.error(f"Matrix4d wrapper error: {e}")
                traceback.print_stack()
                # Fall back to original constructor as last resort
                try:
                    original_matrix4d(self, *args, **kwargs)
                except Exception as inner_e:
                    logger.error(f"Original Matrix4d constructor also failed: {inner_e}")
                    # Initialize as identity matrix if all else fails
                    original_matrix4d(self)

        Gf.Matrix4d.__init__ = patch_matrix4d
