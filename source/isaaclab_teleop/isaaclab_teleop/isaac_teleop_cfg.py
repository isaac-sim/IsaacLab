# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration class for IsaacTeleop-based teleoperation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .xr_cfg import XrCfg

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner
    from isaacteleop.teleop_session_manager import PluginConfig


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

    app_name: str = "IsaacLabTeleop"
    """Application name for the IsaacTeleop session."""
