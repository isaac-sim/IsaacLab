# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Capture-based teleoperation device for Isaac Lab."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.sim.utils.stage import get_current_stage_id
from isaaclab.utils import replace
from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .command_handler import CommandHandler
from .control_events import ControlEvents
from .isaac_teleop_cfg import IsaacTeleopCfg
from .session_lifecycle import TeleopSessionLifecycle
from .xr_anchor_manager import XrAnchorManager

if TYPE_CHECKING:
    from .haptic_feedback import HapticFeedbackCfg
    from .session_lifecycle import SupportsDLPack
    from .visualizers.controller_aim_visualizer import ControllerAimVisualizer
    from .visualizers.hand_joint_visualizer import HandJointVisualizer

logger = logging.getLogger(__name__)


class IsaacTeleopDevice:
    """An Isaac Capture-based teleoperation device for Isaac Lab.

    This device provides an interface between Isaac Capture's retargeting pipeline
    and Isaac Lab environments.  It composes three focused collaborators:

    * :class:`XrAnchorManager` -- XR anchor prim setup, synchronization,
      and coordinate-frame transform computation.
    * :class:`TeleopSessionLifecycle` -- pipeline building, OpenXR handle
      acquisition, session creation/destruction, and action-tensor extraction.
    * :class:`CommandHandler` -- callback registration for START / STOP / RESET
      commands, bridged from the pipeline-based control events.

    Together they manage:

    1. XR anchor configuration and synchronization
    2. Isaac Capture session lifecycle
    3. Action tensor generation from the retargeting pipeline

    The device uses Isaac Capture's TensorReorderer to flatten pipeline outputs
    into a single action tensor matching the environment's action space.

    Frame rebasing:
        By default, all output poses are expressed in the simulation world
        frame.  When an application needs poses in a different frame (e.g.
        robot base link for IK), there are two options:

        * **Config-driven** (recommended): set
          :attr:`~IsaacTeleopCfg.target_frame_prim_path` to the USD prim
          whose frame the output should be expressed in.  The device reads
          the prim's world transform each frame and applies the rebase
          automatically.
        * **Explicit**: pass a ``target_T_world`` matrix directly to
          :meth:`advance`.

        In both cases the device composes
        ``target_T_world @ world_T_anchor`` before feeding the matrix into
        the retargeting pipeline, so all resulting poses are expressed in the
        target frame.

    Teleop commands:
        The device supports callbacks for START, STOP, and RESET commands
        that can be triggered via the message-channel control pipeline or
        registered directly via :meth:`add_callback`.  Without a headset, the
        ``B`` / ``P`` / ``R`` keys typed into a visualizer window start, pause
        and reset teleoperation, and a SpaceMouse pipeline's right button
        resets it.

    Example:
        .. code-block:: python

            cfg = IsaacTeleopCfg(
                pipeline_builder=my_pipeline_builder,
                sim_device="cuda:0",
            )

            # Poses in world frame (default)
            with IsaacTeleopDevice(cfg, visualizers=env.sim.visualizers) as device:
                while running:
                    action = device.advance()
                    env.step(action.repeat(num_envs, 1))

            # Config-driven rebase into robot base frame
            cfg.target_frame_prim_path = "/World/Robot/base_link"
            with IsaacTeleopDevice(cfg, visualizers=env.sim.visualizers) as device:
                while running:
                    action = device.advance()
                    env.step(action.repeat(num_envs, 1))

            # Explicit rebase into robot base frame
            with IsaacTeleopDevice(cfg, visualizers=env.sim.visualizers) as device:
                while running:
                    robot_T_world = get_robot_base_transform()
                    action = device.advance(target_T_world=robot_T_world)
                    env.step(action.repeat(num_envs, 1))
    """

    _START_KEY: ClassVar[int] = 48  # evdev code of B
    _STOP_KEY: ClassVar[int] = 25  # P
    _RESET_KEY: ClassVar[int] = 19  # R
    _SPACEMOUSE_RIGHT_BUTTON: ClassVar[int] = 1

    def __init__(
        self,
        cfg: IsaacTeleopCfg,
        cloudxr_env_file: str | None = None,
        auto_launch_cloudxr: bool = True,
        use_kit_xr_bridge: bool = True,
        mcap_record_path: str | None = None,
        mcap_replay_path: str | None = None,
        enable_debug_visualization: bool = False,
        haptic_cfg: HapticFeedbackCfg | None = None,
        visualizers: Sequence[BaseVisualizer] = (),
    ):
        """Initialize the Isaac Capture device.

        Args:
            cfg: Configuration object for Isaac Capture settings.
            cloudxr_env_file: Optional path to a CloudXR ``.env`` file.
                When provided and *auto_launch_cloudxr* is ``True``, the
                CloudXR runtime is launched automatically during session
                start.  When ``None``, no CloudXR runtime is launched.
            auto_launch_cloudxr: Whether to auto-launch the CloudXR runtime
                when *cloudxr_env_file* is set.  Ignored when
                *cloudxr_env_file* is ``None``.
            use_kit_xr_bridge: Whether to source live OpenXR handles from Kit's
                XR bridge (``True``, the full XR rendering / anchor path) or run
                standalone (``False``) with ``isaacteleop`` owning its own OpenXR
                session through the CloudXR runtime -- teleop I/O with no Kit XR
                rendering.  Typically wired to the ``--xr`` CLI flag.
            mcap_record_path: Optional MCAP file path to record the live
                teleop session into.  Mutually exclusive with
                *mcap_replay_path*.  Debug-grade only -- the produced file
                has no per-episode segmentation, no world-frame anchor, and
                no public Python decoder.
            mcap_replay_path: Optional MCAP file path to replay.  When set,
                the device runs in :class:`SessionMode.REPLAY` with no live
                XR connection and feeds the recorded tracker stream
                through the pipeline.  Mutually exclusive with
                *mcap_record_path*.
            enable_debug_visualization: Whether tracking debug visualization
                (red sphere markers at each OpenXR hand joint, RGB axis
                markers at the controller aim poses) is enabled at session
                start.  When ``False`` (the default), the pipeline carries no
                visualization overhead.
            haptic_cfg: Optional haptic-feedback configuration.  When provided,
                the device renders per-hand output vectors pushed via
                :meth:`send_haptic` on the configured device (controller, glove,
                ...).  ``None`` disables haptics.
            visualizers: Live sequence of windows supplying keyboard input, usually ``sim.visualizers``.
                The default empty sequence disables window input.
        """
        self._cfg = cfg
        self._target_world_matrix = None

        self._anchor_manager = XrAnchorManager(cfg.xr_cfg)
        self._command_handler = CommandHandler()
        self._session_lifecycle = TeleopSessionLifecycle(
            cfg,
            cloudxr_env_file=cloudxr_env_file,
            auto_launch_cloudxr=auto_launch_cloudxr,
            use_kit_xr_bridge=use_kit_xr_bridge,
            mcap_record_path=mcap_record_path,
            mcap_replay_path=mcap_replay_path,
            enable_debug_visualization=enable_debug_visualization,
            haptic_cfg=haptic_cfg,
            visualizers=visualizers,
        )

        self._prev_right_a_pressed = False
        self._prev_spacemouse_right_pressed = False
        self._prev_control_is_active: bool | None = None

        # Each visualizer is created lazily on the first frame with matching
        # data; the failure latches stop retry spam if creation fails (e.g.
        # sim not ready).
        self._enable_debug_visualization = enable_debug_visualization
        self._hand_visualizer: HandJointVisualizer | None = None
        self._hand_visualizer_failed = False
        self._aim_visualizer: ControllerAimVisualizer | None = None
        self._aim_visualizer_failed = False

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, "_command_handler"):
            self._command_handler.cleanup()
        if hasattr(self, "_anchor_manager"):
            self._anchor_manager.cleanup()

    def __str__(self) -> str:
        """Returns a string containing information about the Isaac Capture device."""
        xr_cfg = self._cfg.xr_cfg
        msg = f"Isaac Capture Device: {self.__class__.__name__}\n"
        msg += f"\tAnchor Position: {xr_cfg.anchor_pos}\n"
        msg += f"\tAnchor Rotation: {xr_cfg.anchor_rot}\n"
        if xr_cfg.anchor_prim_path is not None:
            msg += f"\tAnchor Prim Path: {xr_cfg.anchor_prim_path} (Dynamic Anchoring)\n"
        else:
            msg += "\tAnchor Mode: Static (Root Level)\n"
        msg += f"\tSim Device: {self._cfg.sim_device}\n"
        msg += f"\tApp Name: {self._cfg.app_name}\n"

        msg += "\t----------------------------------------------\n"
        msg += "\tAvailable Commands:\n"
        callbacks = self._command_handler.callbacks
        start_avail = "START" in callbacks
        stop_avail = "STOP" in callbacks
        reset_avail = "RESET" in callbacks
        msg += f"\t\tStart Teleoperation: {'registered' if start_avail else 'not registered'}\n"
        msg += f"\t\tStop Teleoperation: {'registered' if stop_avail else 'not registered'}\n"
        msg += f"\t\tReset Environment: {'registered' if reset_avail else 'not registered'}\n"

        return msg

    def __enter__(self) -> IsaacTeleopDevice:
        """Enter the context manager and prepare the Isaac Capture session.

        Binds the configured target frame, builds the retargeting pipeline, and acquires OpenXR handles
        from Kit's XR bridge extension.  If the handles are not yet available
        (e.g. the user has not clicked "Start AR"), session creation is deferred
        and will be retried automatically on each :meth:`advance` call.

        Returns:
            Self for context manager protocol.

        Raises:
            ValueError: If the configured target frame does not exist.
        """
        if self._cfg.target_frame_prim_path is not None:
            import usdrt
            from usdrt import Rt

            stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
            prim = stage.GetPrimAtPath(self._cfg.target_frame_prim_path)
            if not prim.IsValid():
                raise ValueError(f"No target frame at {self._cfg.target_frame_prim_path!r}.")
            self._target_world_matrix = Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr()
        self._session_lifecycle.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up the Isaac Capture session."""
        self._target_world_matrix = None
        self._anchor_manager.cleanup()
        self._session_lifecycle.stop(exc_type, exc_val, exc_tb)
        return False

    def reset(self, pause: bool = False) -> None:
        """Reset the device state.

        Resets the XR anchor synchronizer and schedules a
        ``reset`` :class:`~isaacteleop.retargeting_engine.interface.execution_events.ExecutionEvents`
        for the next pipeline step so that all retargeters reinitialize
        their cross-step state.  Also clears any pending haptic force so a
        pulse in progress at reset time does not persist into the next episode.

        Args:
            pause: When ``True``, also pause a running session so teleop resumes
                from a paused state -- the behavior for an *operator* reset
                (e.g. keyboard ``R``). Defaults to ``False`` for a *host* reset
                (e.g. environment auto-reset after task success), which keeps the
                session running into the next episode.
        """
        self._anchor_manager.reset()
        self._session_lifecycle.request_reset(pause=pause)
        self._session_lifecycle.reset_haptics()

    def request_start(self) -> None:
        """Start teleoperation without an XR client.

        Drives the internal teleop state machine toward RUNNING (see
        :meth:`TeleopSessionLifecycle.request_start`). Useful for headless or
        keyboard-driven control when no headset UI is available to send START.
        No-op when no control channel is configured.
        """
        self._session_lifecycle.request_start()

    def request_stop(self) -> None:
        """Stop (pause) teleoperation without an XR client.

        Drives the internal teleop state machine to PAUSED (see
        :meth:`TeleopSessionLifecycle.request_stop`). No-op when no control
        channel is configured.
        """
        self._session_lifecycle.request_stop()

    @property
    def last_control_events(self) -> ControlEvents:
        """Control events from the most recent :meth:`advance`.

        Returns a :class:`ControlEvents` derived from the teleop control
        pipeline.  When no control channel is configured, returns a
        default (no-op) :class:`ControlEvents`.
        """
        return self._session_lifecycle.last_control_events

    def add_callback(self, key: str, func: Callable) -> None:
        """Add a callback function for teleop commands.

        Args:
            key: The command type to bind to. Valid values are "START", "STOP", "RESET", and "R".
            func: The function to call when the command is received. Should take no arguments.
        """
        self._command_handler.add_callback(key, func)

    def advance(self, target_T_world: np.ndarray | torch.Tensor | SupportsDLPack | None = None) -> torch.Tensor | None:
        """Process current device state and return control commands.

        If the Isaac Capture session has not been started yet (because the OpenXR
        handles were not available at ``__enter__`` time), this method will
        attempt to start it on each call.  Once the user clicks "Start AR" and
        the handles become available, the session is created transparently.

        Args:
            target_T_world: Optional 4x4 transform matrix that rebases all
                output poses into an arbitrary target coordinate frame.  When
                provided, the matrix sent to the retargeting pipeline becomes
                ``target_T_world @ world_T_anchor`` instead of just
                ``world_T_anchor``, so all resulting poses are expressed in
                the target frame rather than the simulation world frame.

                Typical use case: pass ``robot_base_T_world`` so that an IK
                controller receives end-effector poses in the robot's base
                link frame.

                Accepts any object supporting the DLPack buffer protocol
                (``__dlpack__``), including :class:`numpy.ndarray`,
                :class:`torch.Tensor`, and ``wp.array``.

                When ``None`` and
                :attr:`~IsaacTeleopCfg.target_frame_prim_path` is set, the
                transform is computed automatically by reading the prim's
                world matrix from Fabric and inverting it.

        Returns:
            A flattened action :class:`torch.Tensor` ready for the Isaac Lab
            environment, or ``None`` if the session has not started yet
            (e.g. still waiting for the user to start AR).

        Raises:
            RuntimeError: If called outside of a context manager.
        """
        if target_T_world is None and self._target_world_matrix is not None:
            matrix = self._target_world_matrix.Get()
            quaternion = matrix.ExtractRotationQuat()
            inverse_rotation = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix().T
            target_T_world = np.eye(4, dtype=np.float32)
            target_T_world[:3, :3] = inverse_rotation
            target_T_world[:3, 3] = -inverse_rotation @ np.asarray(matrix.ExtractTranslation())

        # Step the session (handles lazy start and action extraction)
        action = self._session_lifecycle.step(
            anchor_world_matrix_fn=self._anchor_manager.get_world_matrix,
            target_T_world=target_T_world,
        )

        if action is not None:
            # Poll controller buttons (e.g. toggle anchor rotation on right 'A' press)
            self._poll_buttons()
            if self._enable_debug_visualization:
                self._update_debug_visualization()

        self._handle_operator_input()
        self._dispatch_control_callbacks()

        return action

    def send_haptic(self, endpoint: str, values) -> None:
        """Render one frame of haptic output on a device endpoint.

        Implements the :class:`~isaaclab_teleop.HapticFeedbackReceiver` protocol.
        The vector is cached and injected into the haptic sink on the next
        :meth:`advance`.  This is a no-op unless the device was constructed with
        a ``haptic_cfg``.

        Args:
            endpoint: ``"left"`` or ``"right"`` (see
                :data:`~isaaclab_teleop.haptic_feedback.ENDPOINT_LEFT`).
            values: The per-hand output vector (one scalar for a rumble motor, one
                value per finger for a glove); an all-zero vector stops feedback.
        """
        self._session_lifecycle.push_haptic(endpoint, values)

    def send_client_message(self, message: dict) -> None:
        """Queue a JSON message for delivery to the connected XR client.

        Delivery is deferred until the control channel is connected, so a
        message queued before the headset connects still reaches the client.
        This is a no-op unless a control channel is configured.

        Args:
            message: A JSON-serializable dict carrying a ``"type"``
                discriminator the client recognizes, e.g.
                ``{"type": "system_notice", "message": {...}}``.
        """
        self._session_lifecycle.send_client_message(message)

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _update_debug_visualization(self) -> None:
        """Create and update the enabled tracking debug visualizers."""
        result = self._session_lifecycle.last_step_result
        if result is None:
            return
        world_T_anchor = self._anchor_manager.get_world_matrix()

        if TeleopSessionLifecycle.HAND_LEFT_KEY in result or TeleopSessionLifecycle.HAND_RIGHT_KEY in result:
            if self._hand_visualizer is None and not self._hand_visualizer_failed:
                try:
                    from .visualizers.hand_joint_visualizer import HandJointVisualizer

                    self._hand_visualizer = HandJointVisualizer()
                except Exception:
                    logger.debug("HandJointVisualizer creation failed; disabling hand debug", exc_info=True)
                    self._hand_visualizer_failed = True
            if self._hand_visualizer is not None:
                self._hand_visualizer.update(result, world_T_anchor)

        left_controller = self._session_lifecycle.last_left_controller
        right_controller = self._session_lifecycle.last_right_controller
        if left_controller is not None or right_controller is not None:
            if self._aim_visualizer is None and not self._aim_visualizer_failed:
                try:
                    from .visualizers.controller_aim_visualizer import ControllerAimVisualizer

                    self._aim_visualizer = ControllerAimVisualizer()
                except Exception:
                    logger.debug("ControllerAimVisualizer creation failed; disabling aim debug", exc_info=True)
                    self._aim_visualizer_failed = True
            if self._aim_visualizer is not None:
                self._aim_visualizer.update(left_controller, right_controller, world_T_anchor)

    # ------------------------------------------------------------------
    # Control event -> callback bridge
    # ------------------------------------------------------------------

    def _dispatch_control_callbacks(self) -> None:
        """Fire legacy callbacks when control events indicate a state change.

        This bridges the pipeline-based :class:`ControlEvents` with the
        callback-based :class:`CommandHandler` so that scripts which registered
        callbacks via :meth:`add_callback` still receive dispatches.

        Only fires START/STOP when ``is_active`` transitions between ``True``
        and ``False``; initial transitions from ``None`` are ignored to avoid
        spurious callbacks during ``DefaultTeleopStateManager``'s
        STOPPED -> PAUSED progression.
        """
        from .control_events import _NO_OP_EVENTS

        events = self._session_lifecycle.last_control_events
        if events is _NO_OP_EVENTS:
            return
        if events.should_reset:
            self._command_handler.fire("RESET")
            self._anchor_manager.reset()
        if events.is_active is not None:
            if self._prev_control_is_active is not None and events.is_active != self._prev_control_is_active:
                self._command_handler.fire("START" if events.is_active else "STOP")
            self._prev_control_is_active = events.is_active

    # ------------------------------------------------------------------
    # Operator keys and buttons
    # ------------------------------------------------------------------

    def _handle_operator_input(self) -> None:
        """Start, pause or reset on the operator's ``B`` / ``P`` / ``R`` keys and SpaceMouse right button.

        An operator reset also pauses, so teleop resumes on ``B``. The keys come from the visualizer
        windows, each press once whatever the retargeting mode.
        """
        for code in self._session_lifecycle.drain_pressed_keys():
            if code == self._START_KEY:
                self.request_start()
            elif code == self._STOP_KEY:
                self.request_stop()
            elif code == self._RESET_KEY:
                self.reset(pause=True)

        result = self._session_lifecycle.last_step_result
        buttons = None if result is None else result.get("spacemouse_buttons")
        if buttons is None or buttons.is_none:
            return
        pressed = bool(buttons[0][self._SPACEMOUSE_RIGHT_BUTTON])
        if pressed and not self._prev_spacemouse_right_pressed:
            self.reset(pause=True)
        self._prev_spacemouse_right_pressed = pressed

    # ------------------------------------------------------------------
    # Controller button polling (glue between session and anchor manager)
    # ------------------------------------------------------------------

    def _poll_buttons(self) -> None:
        """Poll controller buttons and trigger actions on rising edges.

        Called once per :meth:`advance` frame, after ``session.step()`` has
        executed the pipeline so the controller ``TensorGroup`` is fresh.

        Currently handles:
            * Right controller primary button ("A") -- toggles anchor rotation.
        """
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

        right_data = self._session_lifecycle.last_right_controller
        if right_data is None or right_data.is_none:
            return

        current = float(right_data[ControllerInputIndex.PRIMARY_CLICK]) > 0.5
        if current and not self._prev_right_a_pressed:
            self._anchor_manager.toggle_anchor_rotation()
        self._prev_right_a_pressed = current


def _enable_teleop_bridge() -> None:
    """Enable the XR teleop bridge extension and configure carb settings.

    Must be called after the Kit launcher has started.
    """
    import carb.settings
    import omni.kit.app

    carb.settings.get_settings().set("/persistent/xr/openxr/disableInputBindings", True)
    carb.settings.get_settings().set('/xr/openxr/components/"isaacsim.kit.xr.teleop.bridge"/enabled', True)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.kit.xr.teleop.bridge", True)


def create_isaac_teleop_device(
    cfg: IsaacTeleopCfg,
    sim_device: str | None = None,
    callbacks: dict[str, Callable] | None = None,
    cloudxr_env_file: str | None = None,
    auto_launch_cloudxr: bool = True,
    use_kit_xr_bridge: bool = True,
    mcap_record_path: str | None = None,
    mcap_replay_path: str | None = None,
    enable_debug_visualization: bool = False,
    haptic_cfg: HapticFeedbackCfg | None = None,
    *,
    env_prim_path: str | None = None,
    visualizers: Sequence[BaseVisualizer] = (),
) -> IsaacTeleopDevice:
    """Create an :class:`IsaacTeleopDevice` with required Omniverse extension setup.

    This helper centralises the boilerplate that every script must execute
    before constructing an :class:`IsaacTeleopDevice`:

    1. Disable default OpenXR input bindings (prevents conflicts).
    2. Enable the ``isaacsim.kit.xr.teleop.bridge`` extension (live mode
       only -- replay mode skips this since it never touches the XR
       runtime).
    3. Optionally override :attr:`IsaacTeleopCfg.sim_device` so action tensors
       land on the same device the caller uses for the simulation.

    Note:
        Supplying *env_prim_path* copies the configuration before expanding paths.
        Otherwise, *sim_device* overrides ``cfg.sim_device`` in place.

    Args:
        cfg: Isaac Capture configuration.
        sim_device: If provided, overrides ``cfg.sim_device`` so action tensors
            are placed on the requested torch device (e.g. ``"cuda:0"``).
        callbacks: Optional mapping of command keys (e.g. ``"START"``, ``"STOP"``,
            ``"RESET"``) to callables registered on the device.
        cloudxr_env_file: Optional path to a CloudXR ``.env`` file.  When
            provided and *auto_launch_cloudxr* is ``True``, the CloudXR
            runtime and WSS proxy are launched automatically during session
            start.  When ``None``, no CloudXR runtime is launched.
        auto_launch_cloudxr: Whether to auto-launch the CloudXR runtime
            when *cloudxr_env_file* is set.  Set to ``False`` to skip the
            launch (e.g. when running the runtime externally).  Ignored
            when *cloudxr_env_file* is ``None``.
        use_kit_xr_bridge: Whether to drive the session from Kit's XR bridge
            (the full XR rendering / anchor path).  When ``True`` (default) the
            ``isaacsim.kit.xr.teleop.bridge`` extension is enabled and the
            session sources its OpenXR handles from Kit.  When ``False`` the
            session runs standalone -- the bridge is left untouched and
            ``isaacteleop`` creates its own OpenXR session through the CloudXR
            runtime, so teleop I/O works headless without Kit XR rendering.
            Typically wired to the ``--xr`` CLI flag.
        mcap_record_path: Optional MCAP file path to record the live teleop
            session into.  Debug-grade only.  Mutually exclusive with
            *mcap_replay_path*.
        mcap_replay_path: Optional MCAP file path to replay.  When set, the
            returned device runs in :class:`SessionMode.REPLAY` and the XR
            teleop bridge is left untouched.  Mutually exclusive with
            *mcap_record_path*.
        enable_debug_visualization: Whether tracking debug visualization is
            enabled at session start.  See
            :paramref:`IsaacTeleopDevice.enable_debug_visualization`.
        haptic_cfg: Optional haptic-feedback configuration.  When provided, the
            returned device implements
            :class:`~isaaclab_teleop.HapticFeedbackReceiver` and renders per-hand
            output vectors on the configured device (controller, glove, ...).

        env_prim_path: Concrete environment root used for ``{ENV_REGEX_NS}`` in target and XR anchor paths.

        visualizers: Live sequence of windows supplying keyboard input, usually ``sim.visualizers``.

    Returns:
        A fully configured :class:`IsaacTeleopDevice` ready for use in a
        ``with`` block.
    """
    if mcap_record_path is not None and mcap_replay_path is not None:
        raise ValueError(
            "mcap_record_path and mcap_replay_path are mutually exclusive; "
            "set at most one to switch between LIVE recording and REPLAY playback."
        )

    # Replay sessions never talk to Kit's XR bridge, and standalone sessions
    # (use_kit_xr_bridge=False) deliberately bypass it, so loading/enabling the
    # bridge extension would only add startup latency, noisy log lines, and --
    # for standalone -- pull in the Kit XR rendering stack we want to avoid.
    if mcap_replay_path is None and use_kit_xr_bridge:
        _enable_teleop_bridge()

    if env_prim_path is not None:
        cfg = replace(cfg)
        if cfg.target_frame_prim_path is not None:
            cfg.target_frame_prim_path = cfg.target_frame_prim_path.replace("{ENV_REGEX_NS}", env_prim_path)
        if cfg.xr_cfg.anchor_prim_path is not None:
            cfg.xr_cfg.anchor_prim_path = cfg.xr_cfg.anchor_prim_path.replace("{ENV_REGEX_NS}", env_prim_path)

    if sim_device is not None:
        cfg.sim_device = sim_device

    if mcap_replay_path is not None:
        logger.info("Using Isaac Capture stack for teleoperation (REPLAY mode)")
    else:
        logger.info("Using Isaac Capture stack for teleoperation")
    device = IsaacTeleopDevice(
        cfg,
        cloudxr_env_file=cloudxr_env_file,
        auto_launch_cloudxr=auto_launch_cloudxr,
        use_kit_xr_bridge=use_kit_xr_bridge,
        mcap_record_path=mcap_record_path,
        mcap_replay_path=mcap_replay_path,
        enable_debug_visualization=enable_debug_visualization,
        haptic_cfg=haptic_cfg,
        visualizers=visualizers,
    )

    if callbacks is not None:
        for key, func in callbacks.items():
            device.add_callback(key, func)

    return device
