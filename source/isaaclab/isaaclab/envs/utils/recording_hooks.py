# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hooks that run after visualizers during :meth:`~isaaclab.sim.SimulationContext.render`.

Lives alongside :mod:`video_recorder` / :mod:`video_recorder_cfg` because both tie into
``--video`` / ``rgb_array`` recording. Keeps :class:`~isaaclab.sim.SimulationContext` free
of imports from ``isaaclab_physx``, ``isaaclab_newton``, and other recording backends.
Each integration is loaded lazily so optional extensions are not required at import time.
"""

from __future__ import annotations

from typing import Any


def run_recording_hooks_after_visualizers(sim: Any) -> None:
    """Run recording-related work after :meth:`~isaaclab.sim.SimulationContext.render` steps visualizers.

    Isaac Sim / RTX follow-up is loaded lazily so minimal installs still work.
    Newton GL video is handled by :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`
    (e.g. :class:`~isaaclab_newton.video_recording.newton_gl_perspective_video.NewtonGlPerspectiveVideo`),
    not here.

    Args:
        sim: Active :class:`~isaaclab.sim.SimulationContext` instance.
    """
    _recording_followup_isaac_sim(sim)


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
