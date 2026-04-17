# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording pipeline hooks after visualizers step.

Keeps :class:`~isaaclab.sim.SimulationContext` free of imports from ``isaaclab_physx``,
``isaaclab_newton``, and other recording backends. Each integration is loaded lazily so
optional extensions are not required at import time.
"""

from __future__ import annotations

from typing import Any


def run_recording_hooks_after_visualizers(sim: Any) -> None:
    """Run recording-related work after :meth:`~isaaclab.sim.SimulationContext.render` steps visualizers.

    Dispatches to Isaac Sim (Kit/RTX) and Newton follow-ups. Optional extensions are
    imported inside each helper so minimal installs still work.

    Args:
        sim: Active :class:`~isaaclab.sim.SimulationContext` instance.
    """
    _recording_followup_isaac_sim(sim)
    _recording_followup_newton(sim)


def _recording_followup_isaac_sim(sim: Any) -> None:
    """Isaac Sim: keep RTX / Replicator outputs fresh when recording video without a Kit visualizer.

    When ``--video`` uses ``rgb_array`` / :class:`~gymnasium.wrappers.RecordVideo`, Replicator
    render products must see Kit's event loop pumped. :class:`~isaaclab_visualizers.kit.KitVisualizer`
    already calls ``omni.kit.app.get_app().update()`` in its ``step()``; if no such visualizer
    is active, we pump here (guarded by ``/isaaclab/video/enabled`` and ``is_rendering``).

    Implemented by ``pump_kit_app_for_headless_video_render_if_needed`` in
    :mod:`isaaclab_physx.renderers.isaac_rtx_renderer_utils`.
    """
    try:
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
            pump_kit_app_for_headless_video_render_if_needed,
        )
    except ImportError:
        return
    pump_kit_app_for_headless_video_render_if_needed(sim)


def _recording_followup_newton(sim: Any) -> None:
    """Newton: recording pipeline after visualizers (e.g. Newton GL video).

    Implementation lives under :mod:`isaaclab_newton.video_recording` so Newton-specific
    capture stays out of core :mod:`isaaclab.sim`.
    """
    try:
        from isaaclab_newton.video_recording.recording_hooks import (
            recording_followup_after_visualizers,
        )
    except ImportError:
        return
    recording_followup_after_visualizers(sim)
