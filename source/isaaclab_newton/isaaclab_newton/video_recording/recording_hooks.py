# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hooks for Newton-based video recording after visualizers have stepped."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


def recording_followup_after_visualizers(sim: SimulationContext) -> None:
    """Newton extension hook: recording pipeline after visualizers have stepped.

    Called from :func:`isaaclab.envs.utils.recording_hooks.run_recording_hooks_after_visualizers`.
    Wire **Newton GL** / Newton-specific video capture here (e.g. perspective video,
    frame sync with ``NewtonVisualizer``). Stay lightweight and no-op when Newton
    recording is inactive.

    The Isaac Sim / RTX path (``omni.kit.app`` pump for Replicator ``rgb_array``) lives in
    :mod:`isaaclab_physx.renderers.isaac_rtx_renderer_utils` — not here.

    Args:
        sim: Active simulation context.
    """
    _ = sim  # Reserved until Newton GL video paths are hooked up.
