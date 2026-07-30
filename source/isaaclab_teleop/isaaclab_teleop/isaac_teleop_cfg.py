# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration class for IsaacTeleop-based teleoperation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

from .control_events import TELEOP_CONTROL_CHANNEL_UUID
from .xr_cfg import XrCfg

_CLOUDXR_ENV_DIR = Path(__file__).resolve().parent

CLOUDXR_AVP_ENV: str = str(_CLOUDXR_ENV_DIR / "avp-cloudxr.env")
"""Absolute path to the Apple Vision Pro CloudXR ``.env`` profile (``auto-native``)."""

CLOUDXR_JS_ENV: str = str(_CLOUDXR_ENV_DIR / "cloudxrjs-cloudxr.env")
"""Absolute path to the CloudXR JS (Quest/Pico) ``.env`` profile (``auto-webrtc``)."""

CLOUDXR_STANDALONE_ENV: str = str(_CLOUDXR_ENV_DIR / "cloudxr-standalone.env")
"""Absolute path to the standalone (headless, no XR client) CloudXR ``.env`` profile.

Default profile for teleop scripts run without ``--xr``, where IsaacTeleop is a
pure input/output transport and creates its own OpenXR session. It forces a
``quest3`` device profile so the CloudXR runtime advertises an OpenXR system with
no client connected, working around ``XR_ERROR_FORM_FACTOR_UNAVAILABLE`` (``-35``).
"""

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner
    from isaacteleop.teleop_session_manager import PluginConfig, RetargetingExecutionConfig


@configclass
class XrCameraFeedCfg:
    """Configuration for one camera image panel shown in XR."""

    camera_name: str = MISSING
    """Name of the :class:`~isaaclab.sensors.Camera` in the interactive scene."""

    enabled: bool = True
    """Whether to create and update this feed."""

    panel_width_m: float = 0.48
    """Physical panel width [m]."""

    distance_m: float = 0.8
    """Distance in front of the viewer anchor [m].

    This value is unused when :attr:`XrCameraFeedLayoutCfg.placement` is
    ``"world"``.
    """

    offset_m: tuple[float, float] = (0.0, 0.0)
    """Horizontal and vertical panel offset in the selected placement frame [m]."""

    max_update_hz: float = 30.0
    """Maximum provider upload rate [Hz]. Set to zero to update after every rendered frame."""

    label: str | None = None
    """Optional short label shown above the image."""


@configclass
class XrCameraFeedLayoutCfg:
    """Declarative placement and packing for enabled XR camera feeds."""

    mode: Literal["manual", "horizontal", "vertical", "grid"] = "manual"
    """Layout mode. Manual preserves each feed's offset and distance."""

    placement: Literal["viewer_start", "head_locked", "world"] = "viewer_start"
    """Reference frame used to place the panels.

    ``"viewer_start"`` captures the first valid viewer eye position and yaw,
    then leaves the panels fixed in the world. ``"head_locked"`` follows the
    viewer with full pose. ``"world"`` uses :attr:`world_position_m` and
    :attr:`world_orientation_xyzw` as a fixed pose in the Isaac Lab USD stage
    world.
    """

    center_offset_m: tuple[float, float] = (0.0, 0.0)
    """Horizontal and vertical center of an automatic layout [m]."""

    distance_m: float = 0.8
    """Distance of every automatically placed panel from the viewer anchor [m].

    This value is unused when :attr:`placement` is ``"world"``.
    """

    panel_gap_m: float = 0.04
    """Edge-to-edge gap between automatically placed panels [m]."""

    max_columns: int = 2
    """Maximum number of columns in grid mode."""

    world_position_m: tuple[float, float, float] | None = None
    """Layout-plane center in the Isaac Lab USD stage world [m].

    Isaac Lab stages are Z-up. This value is required when :attr:`placement`
    is ``"world"``.
    """

    world_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Panel-local-to-world orientation as a quaternion in ``xyzw`` order.

    Panel local +X is image right, +Y is image up, and +Z points from the
    panel's readable side toward the viewer. Feed and layout offsets are
    applied in the resulting local XY plane.
    """


@configclass
class IsaacTeleopCfg:
    """Configuration for IsaacTeleop-based teleoperation.

    This configuration class defines the parameters needed to create a IsaacTeleop
    teleoperation session integrated with Isaac Lab environments.

    The pipeline_builder is a callable that constructs the IsaacTeleop retargeting
    pipeline. It should return an OutputCombiner with a single "action" output
    that contains the flattened action tensor (typically via TensorReorderer).

    If the pipeline builder also produces retargeters that should be exposed in
    the tuning UI, the env cfg should call the builder, unpack the results, and
    populate both ``pipeline_builder`` and ``retargeters_to_tune`` explicitly.
    Both fields must be callables (lambdas / functions) so they survive the
    ``deepcopy`` performed by ``@configclass`` on mutable attributes.

    Example:
        .. code-block:: python

            def build_pipeline():
                controllers = ControllersSource(name="controllers")
                se3 = Se3AbsRetargeter(cfg, name="ee_pose")
                # ... connect and flatten with TensorReorderer ...
                pipeline = OutputCombiner({"action": reorderer.output("output")})
                return pipeline, [se3]  # return retargeters separately


            pipeline, retargeters = build_pipeline()
            teleop_cfg = IsaacTeleopCfg(
                xr_cfg=XrCfg(anchor_pos=(0.5, 0.0, 0.5)),
                pipeline_builder=lambda: pipeline,
                retargeters_to_tune=lambda: retargeters,
            )
    """

    xr_cfg: XrCfg = field(default_factory=XrCfg)
    """XR anchor configuration for positioning the user in the simulation.

    This includes anchor position, rotation, and optional dynamic anchoring
    to follow a prim (e.g., robot base) during locomotion tasks.
    """

    xr_camera_feeds: list[XrCameraFeedCfg] = field(default_factory=list)
    """Existing task camera outputs to show as XR image panels.

    The default empty list disables PiP.
    """

    xr_camera_feed_layout: XrCameraFeedLayoutCfg = field(default_factory=XrCameraFeedLayoutCfg)
    """Placement and packing applied to the ordered enabled camera feeds."""

    pipeline_builder: Callable[[], OutputCombiner] = MISSING
    """Callable that builds the IsaacTeleop retargeting pipeline.

    The function should return an OutputCombiner with an "action" output
    containing the flattened action tensor matching the Isaac Lab action space.
    Use TensorReorderer to flatten multiple retargeter outputs into a single array.

    To expose retargeters for the tuning UI, populate :attr:`retargeters_to_tune`
    directly when constructing this config rather than encoding them into the
    builder's return value.
    """

    plugins: list[PluginConfig] = field(default_factory=list)
    """List of IsaacTeleop plugin configurations.

    Plugins can provide additional functionality like synthetic hand tracking
    from controller inputs.
    """

    sim_device: str = "cuda:0"
    """Torch device string for placing output action tensors."""

    retargeting_execution: RetargetingExecutionConfig | None = None
    """IsaacTeleop retargeting execution settings.

    Left as ``None`` by default so that importing and constructing this config
    never requires the optional ``isaacteleop`` package (e.g. on platforms where
    it is not installed). When ``None``, Isaac Lab resolves it at session start to
    IsaacTeleop's pipelined, deadline-paced default
    (``RetargetingExecutionConfig(mode="pipelined", pacing=DeadlinePacingConfig(safety_margin_s=0.025))``),
    where ``isaacteleop`` is guaranteed to be available. Set this explicitly to
    ``RetargetingExecutionConfig(mode="sync")`` for exact current-frame
    retargeting while debugging or comparing behavior.
    """

    teleoperation_active_default: bool = False
    """Whether teleoperation should be active by default when the session starts.

    When ``False`` (the default), the teleop session remains inactive until a
    ``"START"`` command is received from xr_core via the message bus.
    """

    retargeters_to_tune: Callable[[], list[BaseRetargeter]] | None = None
    """Optional callable returning retargeters to expose in the tuning UI.

    Must be a callable (e.g. ``lambda: [retargeter1, retargeter2]``) rather
    than a plain list because ``@configclass`` deep-copies mutable attributes
    and retargeter objects often contain non-picklable C++/SWIG handles.
    Wrapping in a callable makes the value opaque to ``deepcopy``.

    When set and the tuning UI is enabled, the returned retargeters will be
    displayed in the ``MultiRetargeterTuningUIImGui`` window, allowing
    real-time adjustment of their tunable parameters.  Only retargeters that
    have a ``ParameterState`` (i.e. tunable parameters) will appear.

    If ``None``, the tuning UI will not be opened.
    """

    control_channel_uuid: bytes | None = TELEOP_CONTROL_CHANNEL_UUID
    """16-byte UUID for the teleop control message channel.

    Defaults to :data:`~isaaclab_teleop.TELEOP_CONTROL_CHANNEL_UUID`
    (``uuid5(NAMESPACE_DNS, "teleop_command")``), which is the well-known
    channel both the Isaac Lab server and CloudXR JS client use to
    exchange start/stop/reset commands.

    When set, a ``teleop_control_pipeline`` is created automatically
    using :class:`~isaaclab_teleop.teleop_message_processor.TeleopMessageProcessor`
    and :class:`~isaacteleop.teleop_session_manager.DefaultTeleopStateManager`.
    The remote client sends UTF-8 control commands over the OpenXR opaque
    data channel identified by this UUID, and the results are exposed via
    :func:`~isaaclab_teleop.poll_control_events`.

    Set to ``None`` to disable the control channel entirely.
    """

    target_frame_prim_path: str | None = None
    """Optional USD prim path whose world frame becomes the target coordinate
    frame for all output poses.

    When set, the device automatically reads this prim's world transform each
    frame and uses its inverse as the ``target_T_world`` rebase matrix in
    :meth:`~isaaclab_teleop.IsaacTeleopDevice.advance`.  An explicit
    ``target_T_world`` argument to :meth:`~isaaclab_teleop.IsaacTeleopDevice.advance`
    takes precedence over this config.

    Typical usage: set to the robot base link prim path so that an IK
    controller receives end-effector poses in the robot's base frame.

    Example::

        IsaacTeleopCfg(
            target_frame_prim_path="/World/envs/env_0/Robot/base_link",
            ...
        )
    """

    app_name: str = "IsaacLabTeleop"
    """Application name for the IsaacTeleop session."""
