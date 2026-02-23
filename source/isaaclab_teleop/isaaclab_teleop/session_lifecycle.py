# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop session lifecycle management."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaacteleop.oxr import OpenXRSessionHandles
    from isaacteleop.retargeting_engine_ui import MultiRetargeterTuningUIImGui
    from isaacteleop.teleop_session_manager import TeleopSession

from .isaac_teleop_cfg import IsaacTeleopCfg

logger = logging.getLogger(__name__)


class TeleopSessionLifecycle:
    """Manages the IsaacTeleop session lifecycle.

    This class is responsible for:

    1. Building the retargeting pipeline from configuration
    2. Discovering ``ControllerTracker`` instances within the pipeline
    3. Acquiring OpenXR handles from Kit's XR bridge extension
    4. Creating, entering, and exiting the ``TeleopSession``
    5. Building external inputs for pipeline leaf nodes (e.g. world-to-anchor transform)
    6. Stepping the session and extracting the flattened action tensor
    7. Managing the optional retargeting tuning UI
    """

    WORLD_T_ANCHOR_INPUT_NAME = "world_T_anchor"
    """Well-known name for the ValueInput node that receives the
    world-to-XR-anchor 4x4 transform matrix."""

    def __init__(self, cfg: IsaacTeleopCfg):
        """Initialize the session lifecycle manager.

        Args:
            cfg: Configuration for IsaacTeleop settings.
        """
        self._cfg = cfg
        self._device = torch.device(cfg.sim_device)

        # Session state (populated during start)
        self._session: TeleopSession | None = None
        self._pipeline = None
        self._controller_tracker = None
        self._session_start_deferred_logged = False

        # Retargeting tuning UI (created in start, closed in stop)
        self._retargeting_ui_ctx: MultiRetargeterTuningUIImGui | None = None
        self._retargeting_ui = None

    @property
    def is_active(self) -> bool:
        """Whether the teleop session is currently running."""
        return self._session is not None

    @property
    def pipeline(self):
        """The retargeting pipeline, or ``None`` if not yet built."""
        return self._pipeline

    @property
    def controller_tracker(self):
        """The ``ControllerTracker`` discovered or created for the pipeline."""
        return self._controller_tracker

    @property
    def deviceio_session(self):
        """The underlying ``DeviceIOSession``, or ``None`` if the session hasn't started."""
        if self._session is not None:
            return self._session.deviceio_session
        return None

    # ------------------------------------------------------------------
    # Lifecycle: start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the pipeline and attempt to start the session.

        Builds the retargeting pipeline, discovers the ``ControllerTracker``,
        attempts to acquire OpenXR handles, and opens the retargeting tuning
        UI if retargeters are configured.

        If the OpenXR handles are not yet available (e.g. user hasn't clicked
        "Start AR"), session creation is deferred and will be retried on each
        :meth:`step` call.
        """
        # Build the pipeline from the config
        self._pipeline = self._cfg.pipeline_builder()
        self._session_start_deferred_logged = False

        # Discover the ControllerTracker from the pipeline's DeviceIO source
        # nodes instead of creating a new one.  Creating a second
        # ControllerTracker would cause an XR_ERROR_NAME_DUPLICATED because
        # the OpenXR action set name is fixed.
        self._controller_tracker = self._find_controller_tracker()

        # Try to start the session now; it may be deferred
        self._try_start_session()

        # Open the retargeting tuning UI and keep it alive until stop()
        retargeters = self._cfg.retargeters_to_tune() if self._cfg.retargeters_to_tune else []
        if retargeters:
            from isaacteleop.retargeting_engine_ui import MultiRetargeterTuningUIImGui

            print("Opening Retargeting UI...")
            self._retargeting_ui_ctx = MultiRetargeterTuningUIImGui(retargeters, title="Hand Retargeting Tuning")
            self._retargeting_ui = self._retargeting_ui_ctx.__enter__()

    def stop(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """Shut down the session and clean up resources.

        Closes the retargeting tuning UI and exits the ``TeleopSession``
        context manager.  If the underlying OpenXR session was already torn
        down externally (e.g. "Stop AR"), cleanup errors are suppressed.

        Args:
            exc_type: Exception type (for context manager protocol).
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        # Close the retargeting tuning UI first
        if self._retargeting_ui_ctx is not None:
            self._retargeting_ui_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._retargeting_ui_ctx = None
            self._retargeting_ui = None

        if self._session is not None:
            try:
                self._session.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                # The OpenXR session may have already been torn down externally
                # (e.g. user clicked "Stop AR"), so destroying spaces/action
                # sets will fail with XR_ERROR_HANDLE_INVALID.  This is
                # expected and safe to suppress.
                logger.debug(f"Suppressed error during IsaacTeleop session cleanup: {e}")
            self._session = None
            self._pipeline = None
        logger.info("IsaacTeleop session ended")

    # ------------------------------------------------------------------
    # Deferred session creation
    # ------------------------------------------------------------------

    def try_start_session(self) -> bool:
        """Public wrapper for deferred session creation.

        Returns:
            ``True`` if the session is running, ``False`` if still deferred.
        """
        return self._try_start_session()

    def _try_start_session(self) -> bool:
        """Attempt to create and start the IsaacTeleop session.

        Tries to acquire OpenXR handles from Kit's XR bridge.  If the handles
        are available, creates and enters the ``TeleopSession``.  If not (e.g.
        the user hasn't started AR yet), the attempt is silently deferred and
        will be retried on the next :meth:`step` call.

        Returns:
            ``True`` if the session was successfully started (or was already
            running), ``False`` if session creation was deferred.
        """
        if self._session is not None:
            return True

        from isaacteleop.oxr import OpenXRSessionHandles
        from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

        oxr_handles = self._acquire_kit_oxr_handles(OpenXRSessionHandles)

        if oxr_handles is None:
            if not self._session_start_deferred_logged:
                logger.info(
                    "OpenXR handles not yet available (waiting for AR to start); IsaacTeleop session creation deferred"
                )
                self._session_start_deferred_logged = True
            return False

        # Determine whether the controller tracker was auto-discovered from
        # the pipeline.  If so, TeleopSession will collect it automatically
        # and we must NOT list it again in ``trackers`` (OpenXR forbids
        # duplicate action set names).
        manual_trackers: list = []
        pipeline_tracker = self._find_controller_tracker_in(self._pipeline)
        if pipeline_tracker is None and self._controller_tracker is not None:
            # The tracker was created as a fallback; pass it manually.
            manual_trackers.append(self._controller_tracker)

        # Create TeleopSession config
        session_config = TeleopSessionConfig(
            app_name=self._cfg.app_name,
            trackers=manual_trackers,
            pipeline=self._pipeline,
            plugins=self._cfg.plugins,
            oxr_handles=oxr_handles,
        )

        # Create and enter the TeleopSession
        self._session = TeleopSession(session_config)
        self._session.__enter__()

        logger.info(f"IsaacTeleop session started: {self._cfg.app_name}")
        return True

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, anchor_world_matrix_fn: Callable[[], np.ndarray] | None = None) -> torch.Tensor | None:
        """Execute one step of the teleop session and return the action tensor.

        If the session has not been started yet (because OpenXR handles were
        not available), this method will attempt to start it.  Once the user
        clicks "Start AR" and the handles become available, the session is
        created transparently.

        If the underlying OpenXR session is torn down externally (e.g. the
        user clicks "Stop AR"), the error is caught, the session is cleaned
        up, and ``None`` is returned so the caller can continue rendering
        while waiting for a potential restart.

        Args:
            anchor_world_matrix_fn: Optional callable returning the (4, 4)
                world-to-anchor transform.  Used to build external inputs
                for ``ValueInput`` leaf nodes in the pipeline.

        Returns:
            A flattened action :class:`torch.Tensor` ready for the Isaac Lab
            environment, or ``None`` if the session has not started yet
            or the XR session was torn down externally.

        Raises:
            RuntimeError: If called before :meth:`start`.
        """
        if self._pipeline is None:
            raise RuntimeError("TeleopSessionLifecycle.start() must be called before step()")

        # Lazily start the session when OpenXR handles become available
        if self._session is None:
            if not self._try_start_session():
                return None

        # Build external inputs (e.g. world-to-anchor transform) if the
        # pipeline contains ValueInput leaf nodes.
        external_inputs = self._build_external_inputs(anchor_world_matrix_fn)

        # Execute one step of the teleop session.
        # If the underlying OpenXR session was destroyed externally (e.g.
        # user clicked "Stop AR"), the step call will fail.  We catch the
        # error, tear down the dead session, and return None so the caller
        # can continue rendering (or wait for the session to restart).
        assert self._session is not None  # guaranteed by _try_start_session above
        try:
            result = self._session.step(external_inputs=external_inputs)
        except Exception as e:
            logger.warning(f"IsaacTeleop session step failed (XR session likely torn down): {e}")
            self._teardown_dead_session()
            return None

        # Extract the flattened action array from TensorReorderer output
        action_np = result["action"][0]

        # Convert to torch tensor and move to device
        action = torch.from_numpy(np.asarray(action_np, dtype=np.float32)).to(self._device)

        return action

    # ------------------------------------------------------------------
    # Dead session teardown
    # ------------------------------------------------------------------

    def _teardown_dead_session(self) -> None:
        """Clean up a session whose underlying OpenXR handles are no longer valid.

        This is called when :meth:`step` detects that the XR session was
        destroyed externally (e.g. user clicked "Stop AR").  The
        ``TeleopSession`` is exited with error suppression (since its XR
        resources are already gone), and the internal state is reset so that
        the deferred-start logic in :meth:`step` can re-acquire handles if
        the user restarts AR.
        """
        if self._session is not None:
            try:
                self._session.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"Suppressed error tearing down dead session: {e}")
            self._session = None
        self._session_start_deferred_logged = False
        logger.info("IsaacTeleop session torn down after external XR shutdown")

    # ------------------------------------------------------------------
    # External input building
    # ------------------------------------------------------------------

    def _build_external_inputs(self, anchor_world_matrix_fn: Callable[[], np.ndarray] | None) -> dict | None:
        """Build external inputs for non-DeviceIO leaf nodes in the pipeline.

        Checks whether the active ``TeleopSession`` has external (non-DeviceIO)
        leaf nodes and, for each recognized leaf, constructs the corresponding
        ``TensorGroup`` data.

        Args:
            anchor_world_matrix_fn: Callable returning the (4, 4)
                world-to-anchor transform matrix.

        Returns:
            A dict suitable for ``TeleopSession.step(external_inputs=...)``,
            or ``None`` when no external inputs are required.
        """
        if self._session is None or not self._session.has_external_inputs():
            return None

        from isaacteleop.retargeting_engine.interface import TensorGroup, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        ext_specs = self._session.get_external_input_specs()
        external_inputs: dict = {}

        for leaf_name in ext_specs:
            if leaf_name == self.WORLD_T_ANCHOR_INPUT_NAME:
                if anchor_world_matrix_fn is not None:
                    anchor_matrix = anchor_world_matrix_fn()
                else:
                    anchor_matrix = np.eye(4, dtype=np.float32)
                xform_tg = TensorGroup(TransformMatrix())
                xform_tg[0] = anchor_matrix
                external_inputs[leaf_name] = {ValueInput.VALUE: xform_tg}
            else:
                logger.warning(
                    f"Unrecognized external leaf node '{leaf_name}' in pipeline; "
                    "IsaacTeleopDevice does not know how to provide its inputs"
                )

        return external_inputs if external_inputs else None

    # ------------------------------------------------------------------
    # Controller tracker discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_controller_tracker_in(pipeline):
        """Recursively search *pipeline* for an existing ControllerTracker.

        Checks leaf nodes for ``IDeviceIOSource`` instances whose tracker is a
        ``ControllerTracker``.

        Returns:
            The first ``ControllerTracker`` found, or ``None``.
        """
        from isaacteleop.deviceio import ControllerTracker
        from isaacteleop.retargeting_engine.deviceio_source_nodes import IDeviceIOSource

        try:
            leaf_nodes = pipeline.get_leaf_nodes()
        except AttributeError:
            return None

        for node in leaf_nodes:
            if isinstance(node, IDeviceIOSource):
                tracker = node.get_tracker()
                if isinstance(tracker, ControllerTracker):
                    return tracker
        return None

    def _find_controller_tracker(self):
        """Find a ControllerTracker from the pipeline, or create one as fallback.

        The pipeline typically contains a ``ControllersSource`` that owns a
        ``ControllerTracker``.  Reusing it avoids the
        ``XR_ERROR_NAME_DUPLICATED`` error that occurs when two trackers try to
        register the same OpenXR action set name.

        Returns:
            A ``ControllerTracker`` instance.
        """
        tracker = self._find_controller_tracker_in(self._pipeline)
        if tracker is not None:
            logger.debug("Reusing ControllerTracker from pipeline source node")
            return tracker

        # Fallback: pipeline has no ControllersSource (unlikely but possible)
        from isaacteleop.deviceio import ControllerTracker

        logger.debug("No ControllerTracker found in pipeline; creating a new one")
        return ControllerTracker()

    # ------------------------------------------------------------------
    # OpenXR handle acquisition
    # ------------------------------------------------------------------

    @staticmethod
    def _acquire_kit_oxr_handles(handles_cls: type[OpenXRSessionHandles]) -> OpenXRSessionHandles | None:
        """Acquire OpenXR session handles from Kit's XR bridge extension.

        Imports ``isaacsim.kit.xr.teleop.bridge`` and reads the four raw handle
        values (XrInstance, XrSession, XrSpace, xrGetInstanceProcAddr) that Kit's
        OpenXR system exposes.  The handles are returned as an
        ``OpenXRSessionHandles`` instance ready for ``DeviceIOSession.run()``.

        Args:
            handles_cls: The ``OpenXRSessionHandles`` class (passed in to avoid
                a module-level import of ``isaacteleop.oxr``).

        Returns:
            An ``OpenXRSessionHandles`` instance, or ``None`` if the bridge
            extension is not available (e.g. running outside Isaac Sim).
        """
        try:
            import isaacsim.kit.xr.teleop.bridge as xr_bridge
        except (ImportError, ModuleNotFoundError):
            logger.info("isaacsim.kit.xr.teleop.bridge not available; IsaacTeleop will create its own OpenXR session")
            return None

        instance = xr_bridge.get_instance_handle()
        session = xr_bridge.get_session_handle()
        space = xr_bridge.get_stage_space_handle()
        proc_addr = xr_bridge.get_instance_proc_addr()

        if not all((instance, session, space, proc_addr)):
            logger.debug(
                "XR bridge returned incomplete handles "
                f"(instance={instance}, session={session}, space={space}, proc_addr={proc_addr})"
            )
            return None

        logger.info("Acquired OpenXR handles from Kit XR bridge")
        return handles_cls(instance, session, space, proc_addr)
