# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop-based teleoperation device for Isaac Lab."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from .async_retarget_loop import AsyncRetargetLoop, EmaTimingEstimatorCfg, TimingEstimatorCfg
from .command_handler import CommandHandler
from .control_events import ControlEvents
from .isaac_teleop_cfg import IsaacTeleopCfg
from .session_lifecycle import TeleopSessionLifecycle
from .xr_anchor_manager import XrAnchorManager

if TYPE_CHECKING:
    from .session_lifecycle import SupportsDLPack

logger = logging.getLogger(__name__)


class IsaacTeleopDevice:
    """A IsaacTeleop-based teleoperation device for Isaac Lab.

    This device provides an interface between IsaacTeleop's retargeting pipeline
    and Isaac Lab environments.  It composes three focused collaborators:

    * :class:`XrAnchorManager` -- XR anchor prim setup, synchronization,
      and coordinate-frame transform computation.
    * :class:`TeleopSessionLifecycle` -- pipeline building, OpenXR handle
      acquisition, session creation/destruction, and action-tensor extraction.
    * :class:`CommandHandler` -- callback registration for START / STOP / RESET
      commands, bridged from the pipeline-based control events.

    Together they manage:

    1. XR anchor configuration and synchronization
    2. IsaacTeleop session lifecycle
    3. Action tensor generation from the retargeting pipeline

    The device uses IsaacTeleop's TensorReorderer to flatten pipeline outputs
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
        registered directly via :meth:`add_callback`.

    Example:
        .. code-block:: python

            cfg = IsaacTeleopCfg(
                pipeline_builder=my_pipeline_builder,
                sim_device="cuda:0",
            )

            # Poses in world frame (default)
            with IsaacTeleopDevice(cfg) as device:
                while running:
                    action = device.advance()
                    env.step(action.repeat(num_envs, 1))

            # Config-driven rebase into robot base frame
            cfg.target_frame_prim_path = "/World/Robot/base_link"
            with IsaacTeleopDevice(cfg) as device:
                while running:
                    action = device.advance()
                    env.step(action.repeat(num_envs, 1))

            # Explicit rebase into robot base frame
            with IsaacTeleopDevice(cfg) as device:
                while running:
                    robot_T_world = get_robot_base_transform()
                    action = device.advance(target_T_world=robot_T_world)
                    env.step(action.repeat(num_envs, 1))
    """

    def __init__(
        self,
        cfg: IsaacTeleopCfg,
        cloudxr_env_file: str | None = None,
        auto_launch_cloudxr: bool = True,
    ):
        """Initialize the IsaacTeleop device.

        Args:
            cfg: Configuration object for IsaacTeleop settings.
            cloudxr_env_file: Optional path to a CloudXR ``.env`` file.
                When provided and *auto_launch_cloudxr* is ``True``, the
                CloudXR runtime is launched automatically during session
                start.  When ``None``, no CloudXR runtime is launched.
            auto_launch_cloudxr: Whether to auto-launch the CloudXR runtime
                when *cloudxr_env_file* is set.  Ignored when
                *cloudxr_env_file* is ``None``.
        """
        self._cfg = cfg

        self._anchor_manager = XrAnchorManager(cfg.xr_cfg)
        self._command_handler = CommandHandler()
        self._session_lifecycle = TeleopSessionLifecycle(
            cfg,
            cloudxr_env_file=cloudxr_env_file,
            auto_launch_cloudxr=auto_launch_cloudxr,
        )

        self._prev_right_a_pressed = False
        self._prev_control_is_active: bool | None = None

        # Async advance state: background retarget loop with paced
        # scheduling that produces results consumed by advance().
        self._async_enabled = True
        self._async_blocking = True
        self._async_timing_cfg: TimingEstimatorCfg = EmaTimingEstimatorCfg()
        self._retarget_loop: AsyncRetargetLoop | None = None
        self._cached_action: torch.Tensor | None = None

        # Ensure the retarget loop is stopped before the lifecycle clears
        # session/pipeline state (covers _on_pre_shutdown and other external
        # shutdown paths that bypass __exit__).
        self._session_lifecycle.add_on_stop_callback(self._stop_retarget_loop)

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, "_command_handler"):
            self._command_handler.cleanup()
        if hasattr(self, "_anchor_manager"):
            self._anchor_manager.cleanup()

    def __str__(self) -> str:
        """Returns a string containing information about the IsaacTeleop device."""
        xr_cfg = self._cfg.xr_cfg
        msg = f"IsaacTeleop Device: {self.__class__.__name__}\n"
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
        """Enter the context manager and prepare the IsaacTeleop session.

        Builds the retargeting pipeline and attempts to acquire OpenXR handles
        from Kit's XR bridge extension.  If the handles are not yet available
        (e.g. the user has not clicked "Start AR"), session creation is deferred
        and will be retried automatically on each :meth:`advance` call.

        Returns:
            Self for context manager protocol.
        """
        self._saved_async_enabled = self._async_enabled
        self._session_lifecycle.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up the IsaacTeleop session."""
        self._stop_retarget_loop()
        self._async_enabled = self._saved_async_enabled
        self._anchor_manager.cleanup()
        self._session_lifecycle.stop(exc_type, exc_val, exc_tb)
        return False

    def reset(self) -> None:
        """Reset the device state.

        Resets the XR anchor synchronizer and schedules a
        ``reset`` :class:`~isaacteleop.retargeting_engine.interface.execution_events.ExecutionEvents`
        for the next pipeline step so that all retargeters reinitialize
        their cross-step state.
        """
        self._anchor_manager.reset()
        self._session_lifecycle.request_reset()

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

    def set_async(
        self,
        enabled: bool = True,
        blocking: bool = True,
        *,
        timing_cfg: TimingEstimatorCfg | None = None,
    ) -> None:
        """Enable or disable asynchronous advance mode.

        When enabled, retargeting runs in a background thread that
        combines a **consumption gate** with **paced scheduling**:

        1. After producing a result the thread waits for
           :meth:`advance` to consume it, guaranteeing a strict 1:1
           ratio between retarget invocations and ``advance()`` calls.
        2. Once the gate opens the thread sleeps until the predicted
           optimal moment, then reads inputs and retargets, delivering
           the freshest possible result just before the next
           :meth:`advance`.

        With ``blocking=True`` (default), :meth:`advance` waits for the
        background thread to produce a fresh result before returning.

        With ``blocking=False``, :meth:`advance` returns immediately
        with whatever result is available (may be stale if the background
        thread hasn't finished yet).

        The first call after enabling always runs synchronously to seed
        the pipeline.

        Args:
            enabled: Whether to enable async mode.
            blocking: When ``True`` (default), :meth:`advance` blocks
                until a fresh result is available.  When ``False``,
                returns immediately (possibly stale).  Ignored when
                *enabled* is ``False``.
            timing_cfg: Configuration for the timing estimator used by the
                background loop.  Defaults to :class:`EmaTimingEstimatorCfg`
                when ``None``.  Pass a subclass to use a different
                estimation strategy.
        """
        if timing_cfg is None:
            timing_cfg = self._async_timing_cfg

        needs_restart = (
            enabled != self._async_enabled
            or (enabled and blocking != self._async_blocking)
            or (enabled and timing_cfg != self._async_timing_cfg)
        )
        if not needs_restart:
            return
        self._stop_retarget_loop()
        self._async_enabled = enabled
        self._async_blocking = blocking
        self._async_timing_cfg = timing_cfg
        self._cached_action = None
        mode = "off"
        if enabled:
            mode = "blocking" if blocking else "non-blocking"
        logger.info("IsaacTeleop async advance: %s", mode)

    def advance(self, target_T_world: np.ndarray | torch.Tensor | SupportsDLPack | None = None) -> torch.Tensor | None:
        """Process current device state and return control commands.

        If the IsaacTeleop session has not been started yet (because the OpenXR
        handles were not available at ``__enter__`` time), this method will
        attempt to start it on each call.  Once the user clicks "Start AR" and
        the handles become available, the session is created transparently.

        When async mode is enabled (see :meth:`set_async`), the retargeting
        pipeline runs in a background thread with paced scheduling.  By
        default the call blocks until a fresh result is ready (the block
        is almost always instantaneous because the pacer finishes just in
        time).  This lets retargeting overlap with ``env.step()`` while
        guaranteeing exactly one retarget per ``advance()`` call.

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
        # Auto-compute target_T_world from config if not explicitly provided
        if target_T_world is None and self._cfg.target_frame_prim_path is not None:
            target_T_world = self._get_target_frame_T_world()

        if self._async_enabled:
            return self._advance_async(target_T_world)
        return self._advance_sync(target_T_world)

    def _advance_sync(self, target_T_world=None) -> torch.Tensor | None:
        """Synchronous (original) advance path."""
        action = self._session_lifecycle.step(
            anchor_world_matrix_fn=self._anchor_manager.get_world_matrix,
            target_T_world=target_T_world,
        )
        if action is not None:
            self._poll_buttons()

        self._dispatch_control_callbacks()

        return action

    def _advance_async(self, target_T_world=None) -> torch.Tensor | None:
        """Async advance: read from the paced, consumption-gated loop.

        All USD/Kit interactions (anchor matrix, target-frame transform,
        button polling) run on the calling (main) thread.  Only the
        retarget computation runs in the background.

        First call runs synchronously to seed the cache and starts the
        background loop only once the session is confirmed alive.
        Subsequent calls push fresh inputs from the main thread and
        consume the latest result, blocking for freshness when
        ``self._async_blocking`` is set.

        If the background thread died with an exception, it is re-raised
        here on the main thread after cleaning up the loop.
        """
        if self._retarget_loop is None:
            if not self._seed_async_loop(target_T_world):
                return None
        else:
            action, fresh = self._consume_async_loop(target_T_world)
            if action is not None:
                self._cached_action = action
            elif fresh:
                # Background thread explicitly produced None (session down).
                # Stop the loop so the next advance() re-enters the sync seed
                # path, keeping session management on the main thread.
                self._cached_action = None
                self._stop_retarget_loop()

        if self._cached_action is not None:
            self._poll_buttons()

        self._dispatch_control_callbacks()

        return self._cached_action

    def _seed_async_loop(self, target_T_world=None) -> bool:
        """Run sync once to seed cache, then create/start background loop.

        Returns:
            ``True`` when the loop is running and cache is seeded; ``False``
            when no synchronous action was produced yet.
        """
        action = self._advance_sync(target_T_world)
        if action is None:
            return False
        self._cached_action = action
        self._retarget_loop = AsyncRetargetLoop(
            step_fn=lambda anchor_matrix, target_transform: self._session_lifecycle.step(
                anchor_world_matrix_fn=lambda: anchor_matrix,
                target_T_world=target_transform,
            ),
            timing_cfg=self._async_timing_cfg,
        )
        self._retarget_loop.update_inputs(
            self._anchor_manager.get_world_matrix(),
            target_T_world,
        )
        self._retarget_loop.start()
        return True

    def _consume_async_loop(self, target_T_world=None) -> tuple[torch.Tensor | None, bool]:
        """Push main-thread inputs and consume latest result from loop."""
        loop = self._retarget_loop
        if loop is None:
            raise RuntimeError("Async retarget loop is not initialized")
        # Push current inputs from main thread so the background thread
        # never touches USD/Kit APIs.
        loop.update_inputs(
            self._anchor_manager.get_world_matrix(),
            target_T_world,
        )
        try:
            return loop.consume(block=self._async_blocking)
        except Exception:
            self._stop_retarget_loop()
            raise

    def _stop_retarget_loop(self) -> None:
        """Stop and discard the background retarget loop (idempotent)."""
        if self._retarget_loop is not None:
            self._retarget_loop.stop()
            self._retarget_loop = None

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
    # Target frame transform (config-driven rebase)
    # ------------------------------------------------------------------

    def _get_target_frame_T_world(self) -> np.ndarray | None:
        """Read the target-frame prim's world matrix from Fabric and return its inverse.

        Uses USDRT to read the prim's hierarchical world matrix, matching the
        pattern used by :class:`XrAnchorSynchronizer` for anchor prim reads.

        Returns:
            A (4, 4) float32 :class:`numpy.ndarray` representing the inverse
            of the prim's world transform (i.e. ``target_T_world``), or
            ``None`` if the prim cannot be read.
        """
        try:
            import omni.usd
            import usdrt
            from pxr import UsdUtils
            from usdrt import Rt

            stage = omni.usd.get_context().get_stage()
            stage_cache = UsdUtils.StageCache.Get()
            stage_id = stage_cache.GetId(stage).ToLongInt()
            if stage_id < 0:
                stage_id = stage_cache.Insert(stage).ToLongInt()
            rt_stage = usdrt.Usd.Stage.Attach(stage_id)
            if rt_stage is None:
                return None

            rt_prim = rt_stage.GetPrimAtPath(self._cfg.target_frame_prim_path)
            if not rt_prim.IsValid():
                return None

            rt_xformable = Rt.Xformable(rt_prim)
            if not rt_xformable.GetPrim().IsValid():
                return None

            world_matrix_attr = rt_xformable.GetFabricHierarchyWorldMatrixAttr()
            if world_matrix_attr is None:
                return None

            rt_matrix = world_matrix_attr.Get()
            if rt_matrix is None:
                return None

            pos = rt_matrix.ExtractTranslation()
            rt_quat = rt_matrix.ExtractRotationQuat()

            from scipy.spatial.transform import Rotation

            quat_xyzw = [
                float(rt_quat.GetImaginary()[0]),
                float(rt_quat.GetImaginary()[1]),
                float(rt_quat.GetImaginary()[2]),
                float(rt_quat.GetReal()),
            ]

            R = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
            t = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)

            inv_mat = np.eye(4, dtype=np.float32)
            inv_mat[:3, :3] = R.T
            inv_mat[:3, 3] = -(R.T @ t)
            return inv_mat
        except Exception as e:
            logger.warning(f"Failed to read target frame prim '{self._cfg.target_frame_prim_path}': {e}")
            return None

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

    Must be called after the Omniverse AppLauncher has started.
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
) -> IsaacTeleopDevice:
    """Create an :class:`IsaacTeleopDevice` with required Omniverse extension setup.

    This helper centralises the boilerplate that every script must execute
    before constructing an :class:`IsaacTeleopDevice`:

    1. Disable default OpenXR input bindings (prevents conflicts).
    2. Enable the ``isaacsim.kit.xr.teleop.bridge`` extension.
    3. Optionally override :attr:`IsaacTeleopCfg.sim_device` so action tensors
       land on the same device the caller uses for the simulation.

    Note:
        When *sim_device* is provided, ``cfg.sim_device`` is mutated in place
        before the device is constructed.

    Args:
        cfg: IsaacTeleop configuration.
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

    Returns:
        A fully configured :class:`IsaacTeleopDevice` ready for use in a
        ``with`` block.
    """
    _enable_teleop_bridge()

    if sim_device is not None:
        cfg.sim_device = sim_device

    logger.info("Using IsaacTeleop stack for teleoperation")
    device = IsaacTeleopDevice(
        cfg,
        cloudxr_env_file=cloudxr_env_file,
        auto_launch_cloudxr=auto_launch_cloudxr,
    )

    if callbacks is not None:
        for key, func in callbacks.items():
            device.add_callback(key, func)

    return device
