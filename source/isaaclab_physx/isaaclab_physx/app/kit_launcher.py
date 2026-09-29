# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package with the utility class to configure the :class:`isaacsim.simulation_app.SimulationApp`.

The :class:`KitLauncher` parses environment variables and input CLI arguments to launch the simulator in
various different modes. This includes with or without GUI and switching between different Omniverse remote
clients. Some of these require the extensions to be loaded in a specific order, otherwise a segmentation
fault occurs.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.metadata
import importlib.util
import logging
import os
import re
import signal
import sys
from typing import Literal

try:
    import isaacsim  # noqa: F401
except ModuleNotFoundError:
    isaacsim = None

SimulationApp = getattr(isaacsim, "SimulationApp", None)

from isaaclab.app.loading_screen import report_activity
from isaaclab.app.logging_utils import apply_python_logging_level
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.app.sim_launcher import SimulationLauncher, _parse_visualizer_csv, fuse_kit_args
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils.device import set_cuda_device
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING

# import logger
logger = logging.getLogger(__name__)

_FABRIC_GPU_INTEROP_ENV = "ISAACLAB_FABRIC_USE_GPU_INTEROP"

_SIM_APP_CONFIG_KEYS = frozenset(
    ("headless", "hide_ui", "physics_gpu", "multi_gpu", "limit_cpu_threads", "width", "height")
)
"""Launcher arguments forwarded to the :class:`SimulationApp` config."""

# Suppress noisy debug-level logs from third-party libraries
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)


def _sanitize_sys_argv_for_kit(argv: list[str]) -> list[str]:
    """Remove pytest arguments that Kit would otherwise interpret."""
    if "pytest" not in sys.modules:
        return argv

    indexes_to_remove: set[int] = set()
    for index, argument in enumerate(argv):
        if argument == "-m" and index + 1 < len(argv):
            marker_expression = argv[index + 1]
            if any(marker in marker_expression for marker in ("pytest", "isaacsim_ci", "windows_ci", "arm_ci")):
                indexes_to_remove.update((index, index + 1))
        elif (
            (argument.startswith("--config-file=") and "pyproject.toml" in argument)
            or argument == "--capture=no"
            or re.fullmatch(r"-v+", argument)
        ):
            indexes_to_remove.add(index)

    return [argument for index, argument in enumerate(argv) if index not in indexes_to_remove]


class _StoreAndMarkExplicit(argparse.Action):
    """Store the value and set ``<dest>_explicit``, for options whose default must stay a real value.

    ``--device`` defaults to ``"cuda:0"`` because scripts read ``args_cli.device`` before launch, yet XR
    needs to know whether it was passed (see :meth:`KitLauncher._resolve_device_settings`).
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


class KitLauncher(SimulationLauncher):
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

    The ``ISAACLAB_FABRIC_USE_GPU_INTEROP`` environment variable optionally overrides the
    ``/physics/fabricUseGPUInterop`` Kit setting. Set it to ``1`` or ``0`` to enable or disable the setting.
    When unset, Kit's configured default is preserved.

    """

    @staticmethod
    def _ensure_isaaclab_info_stream_handler() -> None:
        """Add a stream handler for Isaac Lab INFO records hidden by Kit logging."""
        handler_name = "isaaclab_info_stream"
        root_logger = logging.getLogger()
        if any(getattr(handler, "name", None) == handler_name for handler in root_logger.handlers):
            return

        class _IsaacLabInfoFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.levelno == logging.INFO and record.name.startswith("isaaclab")

        handler = logging.StreamHandler(sys.stdout)
        handler.name = handler_name
        handler.setLevel(logging.INFO)
        handler.addFilter(_IsaacLabInfoFilter())
        handler.setFormatter(logging.Formatter("[INFO]: %(message)s"))
        root_logger.addHandler(handler)

    def __init__(self, launcher_args: argparse.Namespace | dict | None = None):
        """Create a `SimulationApp`_ instance based on the input settings.

        Args:
            launcher_args: Launcher arguments, as normalized by :func:`~isaaclab.app.launch_simulation`.
                Defaults to None, which is equivalent to passing an empty dictionary. Keys named like
                `SimulationApp`_ config fields (e.g. ``width``) are forwarded to it.

        Raises:
            SystemExit: If the full Isaac Sim runtime is unavailable.
            ValueError: If incompatible or undefined values are assigned to relevant environment values,
                such as ``HEADLESS``.

        .. _SimulationApp: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.simulation_app/docs/index.html#isaacsim.simulation_app.SimulationApp
        """
        from isaaclab.utils import has_kit

        self._app = None
        if has_kit():
            # a Kit app already runs in this process; it is owned (and closed) by whoever started it
            return
        _ensure_isaac_sim_available()

        if launcher_args is None:
            launcher_args = {}
        elif isinstance(launcher_args, argparse.Namespace):
            launcher_args = launcher_args.__dict__

        # ``launch_simulation`` applied the Python logging level; keep it for after Kit installs its logging bridge.
        self._python_logging_level = logging.getLogger().getEffectiveLevel()

        # Define config members that are read from env-vars or keyword args
        self._headless: bool  # 0: GUI, 1: Headless
        self._livestream: Literal[0, 1, 2]  # 0: Disabled, 1: WebRTC public, 2: WebRTC private
        self._offscreen_render: bool  # 0: Disabled, 1: Enabled
        self._sim_experience_file: str  # Experience file to load
        self._video_enabled: bool  # Whether --video recording is enabled
        self.device: str  # resolved device string (e.g. "cuda:0" or "cpu")
        self._deferred_cuda_device_id: int | None = None

        # Integrate env-vars and input keyword args into simulation app config
        self._config_resolution(launcher_args)

        # PyTorch may already have been imported while constructing simulation configs. Drain
        # its deferred CUDA capability checks before Kit changes the set of CUDA devices that
        # correspond to Vulkan-capable GPUs. Otherwise a queued check for a device that Kit
        # filters out fails during the post-Kit ``set_device`` call (for example, device=1 with
        # two CUDA GPUs but only GPU 0 attached to the display).
        self._initialize_preloaded_torch_cuda()

        # Create SimulationApp, passing the resolved self._config to it for initialization
        self._create_app()
        # back the core settings with the running app
        import carb

        get_settings_manager().set_backend(carb.settings.get_settings())
        # Isaac Sim's stage helpers load only with their extension, which may be enabled later.
        import omni.kit.app

        self._stage_context_hook = (
            omni.kit.app.get_app()
            .get_extension_manager()
            .subscribe_to_extension_enable(
                lambda _: _share_stage_context(),
                ext_name="isaacsim.core.experimental.utils",
                hook_name="isaaclab stage context",
            )
        )
        if self._deferred_cuda_device_id is not None:
            set_cuda_device(self._deferred_cuda_device_id)
        # Load IsaacSim extensions
        self._load_extensions()

        # Re-run path sanitization.  Kit and its extensions may have inserted
        # additional ``pip_prebundle`` or conflicting extension directories onto
        # ``sys.path`` during startup.  A second pass ensures pip-installed
        # packages still take priority over bundled copies.
        from isaaclab import _deprioritize_prebundle_paths

        _deprioritize_prebundle_paths()

        # Hide the stop button in the toolbar
        self._set_toolbar_button_visible("_stop_button", False)

        # Hide play button callback if the timeline is stopped
        import omni.timeline

        self._hide_play_button_callback = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.STOP),
                lambda e: self._set_toolbar_button_visible("_play_button", False),
            )
        )
        self._unhide_play_button_callback = (
            omni.timeline.get_timeline_interface()
            .get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.PLAY),
                lambda e: self._set_toolbar_button_visible("_play_button", True),
            )
        )
        # the CI runner greps this marker for hang detection; __stderr__ survives
        # stdout being redirected to /dev/null during app creation
        print("[ISAACLAB] KitLauncher initialization complete", file=sys.__stderr__, flush=True)
        # Kit exits 0 from inside close() and from its own SIGINT handler: close with the real
        # status at exit, and let Ctrl-C unwind user code. Other signals keep their default action.
        atexit.register(lambda: self._app.close(exit_code=1 if getattr(sys, "last_exc", None) is not None else 0))
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def close(self, exit_code: int = 0) -> None:
        """Close the Kit app this launcher started; Kit fast shutdown exits with *exit_code*."""
        if self._app is not None:
            previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                # let callbacks queued by the closing simulation run before shutdown
                self._app.update()
                self._app.close(exit_code=exit_code)
            finally:
                signal.signal(signal.SIGINT, previous_handler)

    """
    Operations.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the full Isaac Sim runtime is importable, i.e. Kit can be launched.

        This reports launchability, not running state: it is ``True`` in any process with a
        full Isaac Sim installation, before and after Kit starts. Use
        :func:`~isaaclab.utils.version.has_kit` to check whether Kit is currently running.
        """
        return SimulationApp is not None

    @staticmethod
    def add_launcher_args(parser: argparse.ArgumentParser) -> None:
        """Utility function to configure KitLauncher arguments with an existing argument parser object.

        This function appends the command-line arguments relevant to the SimulationApp to the input
        :class:`argparse.ArgumentParser` instance. This allows overriding the environment variables using
        command-line arguments.

        Currently, it adds the following parameters to the argparser object:

        * ``livestream`` (int): If one of {1, 2}, then livestreaming and headless mode is enabled. The values
          map the same as that for the ``LIVESTREAM`` environment variable. If :obj:`-1`, then livestreaming is
          determined by the ``LIVESTREAM`` environment variable.
          Valid options are:

          - ``0``: Disabled
          - ``1``: `WebRTC`_ over public network
          - ``2``: `WebRTC`_ over local/private network

        * ``device`` (str): The device to run the simulation on.
          Valid options are:

          - ``cpu``: Use CPU.
          - ``cuda``: Use GPU with device ID ``0``.
          - ``cuda:N``: Use GPU, where N is the device ID. For example, "cuda:0".

        * ``experience`` (str): The experience file to load when launching the SimulationApp. If a relative path
          is provided, it is resolved relative to the ``apps`` folder in Isaac Sim and Isaac Lab (in that order).

          If provided as an empty string, the experience file is selected from the resolved visualizer and XR
          settings. Rendering support is available by default, including in headless execution.

        * ``deterministic`` (bool): Publishes ``/isaaclab/render/deterministic`` for reproducible rendering.
          Does not change how the default experience file is chosen.

        * ``kit_args`` (str): Optional command line arguments to be passed to Omniverse Kit directly.
          Arguments should be combined into a single string separated by space.
          Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2"
          A single Kit argument works in both the space-separated and the ``=``-attached form
          (e.g. ``--kit_args "--ext-folder=/path/to/ext1"`` or ``--kit_args=--ext-folder=/path/to/ext1``).
          Isaac Lab experiences use one renderer GPU by default. Applications that need single-process
          multi-GPU rendering can override the ``renderer.multiGpu`` settings through this argument.

        * ``visualizer`` (str): Visualizer backends to enable.
          Valid options are:

          - ``rerun``: Use Rerun visualizer.
          - ``newton_gl``: Use Newton GL visualizer.
          - ``newton_rtx``: Use Newton RTX path-tracer visualizer (experimental).
          - ``viser``: Use Viser visualizer.
          - ``kit``: Use Omniverse Kit visualizer.
          - ``none``: Disable all visualizers explicitly.
          - Multiple visualizers can be specified as a comma-delimited list:
            ``--viz rerun,newton_gl,viser``.

          .. deprecated:: Use ``newton_gl`` instead of ``newton``.

        * ``max_visible_envs`` (int | None): Optional global override for partial visualization by capping
          how many environments are shown in the visualizers.
          More partial visualization configuration fields are available in the ``VisualizerCfg`` class.

        .. _`WebRTC`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client

        Args:
            parser: An argument parser instance to be extended with the KitLauncher specific options.
        """
        # argparse rejects an option-like value token after "--kit_args"; fuse the pair before
        # anything parses the command line so the space-separated form works on every entry point
        sys.argv[1:] = fuse_kit_args(sys.argv[1:])

        # Add custom arguments to the parser
        arg_group = parser.add_argument_group(
            "launcher arguments",
            description="Arguments for the KitLauncher. For more details, please check the documentation.",
        )
        arg_group.add_argument(
            "--livestream",
            type=int,
            default=-1,
            choices={0, 1, 2},
            help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.",
        )
        arg_group.add_argument(
            "--xr",
            action="store_true",
            default=False,
            help="Enable XR mode for VR/AR applications.",
        )
        arg_group.add_argument(
            "--device",
            type=str,
            action=_StoreAndMarkExplicit,
            default="cuda:0",
            help='The device to run the simulation on. Can be "cpu", "cuda", "cuda:N", where N is the device ID',
        )
        arg_group.add_argument(
            "--visualizer",
            "--viz",
            type=_parse_visualizer_csv,
            default=None,
            help="Visualizer backends to enable as CSV (e.g., kit,newton,rerun,viser).",
        )
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
                " the experience file is determined from the resolved visualizer and XR settings. If a relative"
                " path is provided,"
                " it is resolved relative to the `apps` folder in Isaac Sim and Isaac Lab (in that order)."
            ),
        )
        arg_group.add_argument(
            "--deterministic",
            action="store_true",
            default=False,
            help="Request reproducible rendering (see KitLauncher docs).",
        )
        arg_group.add_argument(
            "--kit_args",
            type=str,
            default="",
            help=(
                "Command line arguments for Omniverse Kit as a string separated by a space delimiter."
                ' Example usage: --kit_args "--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2".'
                ' A single Kit argument works in both forms: --kit_args "--ext-folder=/path/to/ext1"'
                " or --kit_args=--ext-folder=/path/to/ext1."
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
        arg_group.add_argument(
            "--max_visible_envs",
            type=int,
            default=argparse.SUPPRESS,
            help=("When set, caps the nums of envs shown in the launched visualizers."),
        )

    """
    Internal functions.
    """

    # Set by :meth:`_resolve_xr_settings`. Defaulted here so :meth:`_resolve_headless_settings`
    # stays independent of resolver call order and of whether XR was resolved at all.
    _xr_auto_start: bool = False

    def _config_resolution(self, launcher_args: dict):
        """Resolve the input arguments and environment variables.

        Args:
            launcher_args: A dictionary of all input arguments passed to the class object.
        """
        self._kit_visualizer = bool(launcher_args.get("kit_visualizer", False))
        self._resolve_livestream_settings(launcher_args)
        # XR must be resolved before headless so that XR can prevent
        # visualizer-intent-based headless forcing.
        self._resolve_xr_settings(launcher_args)
        self._resolve_headless_settings(launcher_args)
        self._resolve_camera_settings(launcher_args)
        self._resolve_viewport_settings(launcher_args)
        self._resolve_device_settings(launcher_args)
        self._resolve_experience_file(launcher_args)
        self._resolve_anim_recording_settings(launcher_args)
        self._resolve_kit_args(launcher_args)

        # Prepare final simulation app config
        # Remove all values from input keyword args which are not meant for SimulationApp
        # Assign all the passed settings to a dictionary for the simulation app
        self._sim_app_config = {key: launcher_args[key] for key in _SIM_APP_CONFIG_KEYS & launcher_args.keys()}

    def _resolve_livestream_settings(self, launcher_args: dict):
        """Resolve livestream related settings."""
        # the mode (CLI over ``LIVESTREAM``) is resolved and validated by ``launch_simulation``
        self._livestream = int(launcher_args.get("livestream", 0))

        # Process livestream here before launching kit because some of the extensions only work
        # when launched with the kit file. Only one livestream extension can be enabled at a time.
        self._livestream_args = []
        if self._livestream == 1:
            # WebRTC over a public network advertises the public IP address of a remote instance
            public_ip = os.environ.get("PUBLIC_IP", "127.0.0.1")
            self._livestream_args.append(f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={public_ip}")
        if self._livestream >= 1:
            # Signal/stream ports and allowDynamicResize must be set explicitly; without
            # them NVST cannot bind its server socket (NVST_R_INTERNAL_ERROR) and any
            # subsequent window resize after a client connects triggers NVST_R_BUSY.
            self._livestream_args += [
                "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100",
                "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998",
                "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true",
                "--/exts/omni.kit.livestream.app/primaryStream/streamType=webrtc",
                "--enable",
                "omni.kit.livestream.app",
            ]
            sys.argv += self._livestream_args

    def _resolve_headless_settings(self, launcher_args: dict):
        """Resolve headless related settings."""
        headless_env = int(os.environ.get("HEADLESS", 0))
        if headless_env not in {0, 1}:
            raise ValueError(f"Invalid value for environment variable `HEADLESS`: {headless_env}. Expected: 0 or 1.")
        # livestreaming always runs headless on the host machine
        self._headless = bool(launcher_args.get("headless", False)) or self._livestream > 0 or bool(headless_env)
        # only a Kit visualizer opens a window, and XR without an explicit one has no viewport to start from
        if not self._headless and (self._xr_auto_start or not self._kit_visualizer):
            logger.info("Running headless because '--viz kit' was not requested. Pass it to open a viewport.")
            self._headless = True
        # Headless needs to be passed to the SimulationApp so we keep it here
        launcher_args["headless"] = self._headless

    def _resolve_camera_settings(self, launcher_args: dict):
        """Resolve camera related settings."""
        self._enable_cameras = bool(launcher_args.get("enable_cameras", False))
        self._offscreen_render = False
        if self._enable_cameras and self._headless:
            self._offscreen_render = True

    def _resolve_xr_settings(self, launcher_args: dict):
        """Resolve XR related settings."""
        xr_env = int(os.environ.get("XR", 0))
        xr_arg = launcher_args.get("xr", False)
        xr_valid_vals = {0, 1}
        if xr_env not in xr_valid_vals:
            raise ValueError(f"Invalid value for environment variable `XR`: {xr_env} .Expected: {xr_valid_vals} .")
        # We allow xr kwarg to supersede XR envvar
        if xr_arg is True:
            self._xr = xr_arg
        else:
            self._xr = bool(xr_env)

        # Determine whether XR should auto-inject a KitVisualizer.
        # When XR is enabled but no Kit visualizer was explicitly requested via
        # CLI, we auto-inject one so that app.update() and forward() are pumped
        # each frame -- the XR runtime needs both to receive updated hand/joint
        # transforms.
        self._xr_auto_start = self._xr and "kit" not in (launcher_args.get("visualizer") or ())

    def _resolve_viewport_settings(self, launcher_args: dict):
        """Resolve viewport related settings."""
        self._video_enabled = bool(launcher_args.get("video", False))
        if self._video_enabled and any(
            importlib.util.find_spec(package) is None for package in ("moviepy", "imageio_ffmpeg")
        ):
            raise ModuleNotFoundError(
                "Video recording with `--video` requires MoviePy and its imageio-ffmpeg backend, "
                "which are not installed by default. "
                "Run uv commands with `uv run --extra video ...`, or install MoviePy into the "
                'legacy environment with `./isaaclab.sh -p -m pip install "moviepy>=1.0.3,<2.0.0.dev0"` '
                "(`isaaclab.bat -p -m pip install ...` on Windows), and retry."
            )
        # Check if we can disable the viewport to improve performance
        #   This should only happen if we are running headless and do not require livestreaming or video recording
        #   This is different from offscreen_render because this only affects the default viewport and
        #   not other render-products in the scene
        self._render_viewport = True
        if self._headless and not self._livestream and not self._video_enabled:
            self._render_viewport = False

        # hide_ui flag
        launcher_args["hide_ui"] = False
        if self._headless and not self._livestream:
            launcher_args["hide_ui"] = True

    def _resolve_device_settings(self, launcher_args: dict):
        """Resolve simulation GPU device related settings."""
        device = launcher_args.get("device", "cuda:0")
        distributed = launcher_args.get("distributed", False)
        if self._xr and not launcher_args.get("device_explicit", False) and not distributed:
            # If no device is specified, default to the CPU device if we are running in XR
            device = launcher_args["device"] = "cpu"

        if "cuda" not in device and "cpu" not in device:
            raise ValueError(
                f"Invalid value for input keyword argument `device`: {device}."
                " Expected: a string with the format 'cuda', 'cuda:<device_id>', or 'cpu'."
            )
        device_id = int(device.split(":")[-1]) if "cuda:" in device else 0

        if distributed:
            # ``launch_simulation`` already resolved the per-rank ``device``
            launcher_args["multi_gpu"] = False
            # limit CPU threads to minimize thread context switching
            # this ensures processes do not take up all available threads and fight for resources
            num_cpu_cores = os.cpu_count()
            num_threads_per_process = num_cpu_cores // int(os.getenv("WORLD_SIZE", 1))
            # set environment variables to limit CPU threads
            os.environ["PXR_WORK_THREAD_LIMIT"] = str(num_threads_per_process)
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads_per_process)
            sys.argv.append(f"--/plugins/carb.tasking.plugin/threadCount={num_threads_per_process}")

        # ``/physics/cudaDevice`` is resolved by CUDA, so the masked index is correct there.
        # ``activeGpu`` is deliberately left unset; the renderer device is selected in
        # :meth:`_resolve_kit_args` instead.
        launcher_args["physics_gpu"] = device_id

        # Defer importing torch until after SimulationApp starts.  Importing
        # torch can import NumPy/OpenBLAS, whose at-fork handlers can crash
        # Kit's platform-info fork during startup.
        if "cuda" in device:
            self._deferred_cuda_device_id = device_id

        # Store the resolved device string for downstream consumers (e.g. sim_launcher)
        self.device = device

        logger.info("Using device: %s", device)

    def _initialize_preloaded_torch_cuda(self) -> None:
        """Initialize CUDA before Kit when another import has already loaded PyTorch.

        Importing PyTorch schedules capability checks for every CUDA device it can see. Kit may
        subsequently restrict CUDA to the GPUs that its Vulkan backend can use, leaving those
        queued checks with stale device indices. Do not import PyTorch here: when it has not
        already been loaded, the normal post-Kit initialization path remains preferable.
        """
        if self._deferred_cuda_device_id is None:
            return

        torch = sys.modules.get("torch")
        if torch is not None and not torch.cuda.is_initialized():
            torch.cuda.init()

    def _resolve_experience_file(self, launcher_args: dict):
        """Resolve experience file related settings."""
        # Check if input keywords contain an 'experience' file setting
        # Note: since experience is taken as a separate argument by Simulation App, we store it separately
        self._sim_experience_file = launcher_args.get("experience", "")
        deterministic_mode = bool(launcher_args.get("deterministic", False))

        # If nothing is provided resolve the experience file based on the headless flag.
        # EXP_PATH is normally set by ``isaacsim.bootstrap_kernel()`` on first import.
        # If it is not set (e.g. on aarch64 where the bootstrap early-return triggered
        # under certain install layouts), derive it from the installed isaacsim package.
        kit_app_exp_path = os.environ.get("EXP_PATH")
        if not kit_app_exp_path:
            try:
                import isaacsim as _isaacsim_for_paths
            except ImportError as e:
                raise RuntimeError(
                    "EXP_PATH is not set and the 'isaacsim' package is not importable."
                    " Install Isaac Sim (`pip install isaacsim` or the binary distribution)"
                    " before launching KitLauncher."
                ) from e
            kit_app_exp_path = os.path.join(os.path.dirname(_isaacsim_for_paths.__file__), "apps")
            os.environ["EXP_PATH"] = kit_app_exp_path
        isaaclab_app_exp_path = str(ISAACLAB_ROOT / "apps")

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
                if self._headless and not self._livestream:
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
        # Detect a known incompatibility between Isaac Lab and Isaac Sim full-app experiences.
        if self._livestream in {1, 2}:
            with open(self._sim_experience_file, encoding="utf-8") as file:
                experience = file.read()
            if re.search(r'^\s*["\']isaacsim\.exp\.full["\']\s*=', experience, re.MULTILINE):
                raise ValueError(
                    "The experience file depends on 'isaacsim.exp.full', which is known to hang or invalidate PhysX "
                    "tensor views when launched through Isaac Lab with livestreaming enabled. Omit '--experience' so "
                    "KitLauncher can select an Isaac Lab experience file, or remove the 'isaacsim.exp.full' dependency."
                )
        self._deterministic_rendering = bool(deterministic_mode)
        logger.info("Loading experience file: %s", self._sim_experience_file)

    def _resolve_anim_recording_settings(self, launcher_args: dict):
        """Resolve animation recording settings; recording needs the ``omni.physx.pvd`` extension."""
        self._anim_recording = None
        if not launcher_args.get("anim_recording_enabled", False):
            return
        if self._headless:
            raise ValueError("Animation recording is not supported in headless mode.")
        start_time = launcher_args.get("anim_recording_start_time", 0)
        stop_time = launcher_args.get("anim_recording_stop_time", 10)
        if start_time >= stop_time:
            raise ValueError(
                f"'anim_recording_start_time' {start_time} must be less than 'anim_recording_stop_time' {stop_time}"
            )
        self._anim_recording = (start_time, stop_time)
        sys.argv += ["--enable", "omni.physx.pvd"]

    def _resolve_kit_args(self, launcher_args: dict):
        """Resolve additional arguments passed to Kit."""
        self._kit_args = launcher_args.get("kit_args", "").split()

        fabric_gpu_interop = os.environ.get(_FABRIC_GPU_INTEROP_ENV)
        if fabric_gpu_interop is not None:
            if fabric_gpu_interop not in {"0", "1"}:
                raise ValueError(
                    f"Invalid value for environment variable `{_FABRIC_GPU_INTEROP_ENV}`: {fabric_gpu_interop}."
                    " Expected: 0 or 1."
                )
            argument = f"--/physics/fabricUseGPUInterop={'true' if fabric_gpu_interop == '1' else 'false'}"
            setting = argument.partition("=")[0]
            if not any(arg.partition("=")[0] == setting for arg in sys.argv + self._kit_args):
                self._kit_args.append(argument)

        argument = "--/exts/isaacsim.core.simulation_manager/enable_default_callbacks=false"
        setting = argument.partition("=")[0]
        if not any(arg.partition("=")[0] == setting for arg in sys.argv + self._kit_args):
            self._kit_args.append(argument)

        # RTX allocates spectator support during startup, so visual output intent (which needs an
        # unpartitioned all-environment view) must become a Kit argument before SimulationApp is created.
        if self._kit_visualizer or self._render_viewport or self._video_enabled or self._livestream > 0 or self._xr:
            argument = f"--{ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING}=true"
            setting = argument.partition("=")[0]
            if not any(arg.partition("=")[0] == setting for arg in sys.argv + self._kit_args):
                self._kit_args.append(argument)

        # Select the renderer by CUDA index; the trailing comma keeps the setting string-typed.
        # XR streams a single stereo swapchain that the CloudXR compositor imports, so the
        # renderer has to stay on one known device there too -- otherwise the compositor and
        # the renderer can end up on different GPUs and the headset receives noise. This only
        # applies once a CUDA device has actually been selected: ``--xr`` on its own resolves
        # to ``cpu``, where there is no simulation GPU to align to, so Kit's own choice stands.
        if launcher_args.get("multi_gpu") is False or (self._xr and "cuda" in self.device):
            argument = f"--/renderer/multiGpu/activeCudaGpus={launcher_args['physics_gpu']},"
            setting = argument.partition("=")[0]
            if not any(arg.partition("=")[0] == setting for arg in sys.argv + self._kit_args):
                self._kit_args.append(argument)

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

        sys.argv = _sanitize_sys_argv_for_kit(sys.argv)

        report_activity("Starting Isaac Sim")
        self._app = SimulationApp(self._sim_app_config, experience=self._sim_experience_file)
        report_activity(None)

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

    def _load_extensions(self):
        """Load correct extensions based on KitLauncher's resolved config member variables."""
        # After SimulationApp starts, Kit installs its Python log bridge at DEBUG level.
        # Re-apply the intended Python logging level, then add a scoped stream handler for
        # Isaac Lab INFO records that Kit's bridge does not mirror to the console.
        apply_python_logging_level(self._python_logging_level)
        if self._python_logging_level <= logging.WARNING:
            KitLauncher._ensure_isaaclab_info_stream_handler()
            # At WARNING, let Isaac Lab INFO records reach the scoped handler while the other
            # root handlers stay at WARNING.
            logging.getLogger().setLevel(min(self._python_logging_level, logging.INFO))
        settings = get_settings_manager()

        # Publish whether Kit has an interactive GUI (local window, livestream, or XR).
        # SimulationContext and renderers consume this setting during their initialization.
        settings.set("/isaaclab/has_gui", not self._headless or self._livestream >= 1 or self._xr)
        settings.set("/isaaclab/render/offscreen", self._offscreen_render)
        settings.set("/isaaclab/xr/enabled", self._xr)
        # set setting to indicate XR auto-start mode -- when running headless
        # (no Kit GUI) the AR profile must be enabled programmatically so that
        # the OpenXR session starts without user interaction
        settings.set("/isaaclab/xr/auto_start", self._headless and self._xr)
        settings.set("/isaaclab/video/enabled", self._video_enabled)

        # publish the reproducible-rendering intent; rendering backends read this on initialization
        settings.set("/isaaclab/render/deterministic", self._deterministic_rendering)

        # use fixed time stepping disabled; custom loop runner from Isaac Sim is used instead
        settings.set("/app/player/useFixedTimeStepping", False)

        if self._anim_recording is not None:
            settings.set("/isaaclab/anim_recording/enabled", True)
            settings.set("/isaaclab/anim_recording/start_time", self._anim_recording[0])
            settings.set("/isaaclab/anim_recording/stop_time", self._anim_recording[1])

    def _set_toolbar_button_visible(self, button_name: str, visible: bool):
        """Show and enable, or hide and disable, a button of the toolbar's play button group.

        Standalone runs hide the stop button for good, since stopping invalidates the whole simulation. They hide
        the play button while a GUI action like "save as" has stopped the timeline, so the user cannot resume it.

        Args:
            button_name: Attribute of the play button group, ``"_stop_button"`` or ``"_play_button"``.
            visible: Whether to show and enable the button.
        """
        # a truly headless app has no toolbar widget to import
        if self._livestream < 1 and self._headless:
            return
        import omni.kit.widget.toolbar

        play_button_group = omni.kit.widget.toolbar.get_instance()._builtin_tools._play_button_group  # type: ignore
        if play_button_group is None:
            return
        button = getattr(play_button_group, button_name)
        button.visible = visible
        button.enabled = visible
        if button_name == "_stop_button":
            # detach the stop button so the group never shows it again
            setattr(play_button_group, button_name, None)


def _share_stage_context() -> None:
    """Point Isaac Sim's stage helpers at Isaac Lab's thread-local current-stage context."""
    try:
        # Do not enable ``isaacsim.core.experimental.utils`` here. Stage creation is used by
        # Newton tests before Newton imports Warp, and enabling Isaac Sim experimental utils can
        # make Kit's importer expose the bundled ``omni.warp.core`` package ahead of pip Warp.
        from isaacsim.core.experimental.utils import stage as sim_stage
    except ImportError:
        return
    from isaaclab.sim.utils import stage as stage_utils

    # Isaac Sim stage helpers read this singleton context.
    sim_stage._context = stage_utils._context


def _ensure_isaac_sim_available() -> None:
    """Raise ``SystemExit`` with an actionable hint when Isaac Sim / Kit is missing."""
    if KitLauncher.is_available():
        return

    isaaclab_path = os.environ.get("ISAACLAB_PATH")
    local_sim = os.path.join(isaaclab_path, "_isaac_sim") if isaaclab_path else None
    extra_hint = ""
    if local_sim and os.path.isdir(local_sim):
        launcher, source = ("isaaclab.bat", f'call "{local_sim}\\setup_conda_env.bat"')
        if sys.platform != "win32":
            launcher, source = ("./isaaclab.sh", f'source "{local_sim}/setup_conda_env.sh"')
        extra_hint = (
            f"  Found a local Isaac Sim at {local_sim} but its environment is not active.\n"
            f"  Either run via `{launcher} ...` (which sources the Isaac Sim env automatically),\n"
            f"  or in your current shell run:\n"
            f"    {source}\n"
        )

    try:
        installed_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None

    if installed_version:
        logger.error(
            f"\n[ERROR] Isaac Sim {installed_version} is installed, but its full runtime is unavailable.\n"
            "\n"
            "  This environment requires Isaac Sim and Omniverse Kit.\n"
            "    PhysX backend and Kit visualizer require Isaac Sim.\n"
            "\n"
            "  The current Python environment does not expose the SimulationApp API.\n"
            f"{extra_hint}"
            "  Install the full Isaac Sim runtime from the Isaac Lab directory by running:\n"
            "    uv run isaaclab -i isaacsim\n"
            "\n"
            "  See https://isaac-sim.github.io/IsaacLab/main/source/setup/installation for details.\n"
        )
        raise SystemExit(1)

    logger.error(
        "\n[ERROR] Isaac Sim is not installed or not found on PYTHONPATH.\n"
        "\n"
        "  This environment requires Isaac Sim and Omniverse Kit.\n"
        "    PhysX backend and Kit visualizer currently requires Isaac Sim.\n"
        "\n"
        f"{extra_hint}"
        "  To fix this, ensure Isaac Sim is installed and available in the current environment.\n"
        "\n"
        "  See https://isaac-sim.github.io/IsaacLab/main/source/setup/installation for details.\n"
    )
    raise SystemExit(1)
