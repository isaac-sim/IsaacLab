# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.utils import has_kit
from isaaclab.utils.version import get_isaac_sim_version

logger = logging.getLogger(__name__)

SIMPLE_SHADING_MODES: dict[str, int] = {
    "simple_shading_constant_diffuse": 0,
    "simple_shading_diffuse_mdl": 1,
    "simple_shading_full_mdl": 2,
}
SIMPLE_SHADING_MODE_SETTING: str = "/rtx/sdg/simpleShading/mode"

# Module-level dedup stamp: tracks the last (sim instance, physics step) at
# which Kit's ``app.update()`` was pumped.  Keyed on ``id(sim)`` so that a
# new ``SimulationContext`` (e.g. in a new test) automatically invalidates
# any stale stamp from a previous instance.
_last_render_update_key: tuple[int, int] = (0, -1)


def ensure_isaac_rtx_render_update() -> None:
    """Ensure the Isaac RTX renderer has been pumped for the current physics step.

    This keeps the Kit-specific ``app.update()`` logic inside the renderers
    package rather than in the backend-agnostic ``SimulationContext``.

    Safe to call from multiple ``Camera`` / ``TiledCamera`` instances per step —
    only the first call triggers ``app.update()``.  Subsequent calls are no-ops
    because the module-level ``_last_render_update_key`` already matches the
    current ``(id(sim), step_count)`` pair.

    The key is a ``(sim_instance_id, step_count)`` tuple so that creating a new
    ``SimulationContext`` (e.g. in a subsequent test) automatically invalidates
    any stale stamp left over from a previous instance.

    No-op conditions:
        * Already called this step (dedup across camera instances).
        * A visualizer already pumps ``app.update()`` (e.g. KitVisualizer).
        * Rendering is not active.
    """
    global _last_render_update_key

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return

    key = (id(sim), sim._physics_step_count)
    if _last_render_update_key == key:
        return  # Already pumped this step (by another camera or a visualizer)

    # If a visualizer already pumps the Kit app loop, mark as done and skip.
    if any(viz.pumps_app_update() for viz in sim.visualizers):
        _last_render_update_key = key
        return

    if not sim.is_rendering:
        return

    # Sync physics results → Fabric so RTX sees updated positions.
    # physics_manager.step() only runs simulate()/fetch_results() and does NOT
    # call _update_fabric(), so without this the render would lag one frame behind.
    sim.physics_manager.forward()

    import omni.kit.app

    sim.set_setting("/app/player/playSimulations", False)
    omni.kit.app.get_app().update()
    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key


def configure_isaac_rtx_settings(data_types: Sequence[str]) -> None:
    """Configure global Isaac RTX settings based on requested data types.

    On Isaac Sim 6.0+ this enables the fast SDG path when no RGB/RGBA
    annotators are requested (unless a GUI viewport is active).

    On older Isaac Sim versions it logs warnings for data types that are
    not yet supported (``albedo``, simple-shading modes).

    Args:
        data_types: The sensor data types requested by the camera.
    """
    if not has_kit():
        return

    settings = get_settings_manager()
    isaac_sim_version = get_isaac_sim_version()

    if isaac_sim_version.major >= 6:
        needs_color_render = "rgb" in data_types or "rgba" in data_types
        if not needs_color_render:
            settings.set_bool("/rtx/sdg/force/disableColorRender", True)
        if settings.get("/isaaclab/has_gui"):
            settings.set_bool("/rtx/sdg/force/disableColorRender", False)
    else:
        if "albedo" in data_types:
            logger.warning(
                "Albedo annotator is only supported in Isaac Sim 6.0+. The albedo data type will be ignored."
            )
        if any(dt in SIMPLE_SHADING_MODES for dt in data_types):
            logger.warning(
                "Simple shading annotators are only supported in Isaac Sim 6.0+. The simple shading data types"
                " will be ignored."
            )


def resolve_simple_shading_mode(data_types: Sequence[str]) -> int | None:
    """Resolve the requested simple shading mode from data types.

    Args:
        data_types: The sensor data types requested by the camera.

    Returns:
        The integer mode value for the first matching simple-shading data type,
        or ``None`` if none were requested.
    """
    requested = [dt for dt in data_types if dt in SIMPLE_SHADING_MODES]
    if not requested:
        return None
    if len(requested) > 1:
        logger.warning(
            "Multiple simple shading modes requested (%s). Using '%s' only.",
            requested,
            requested[0],
        )
    return SIMPLE_SHADING_MODES[requested[0]]


DEPTH_DATA_TYPES: frozenset[str] = frozenset({"distance_to_camera", "distance_to_image_plane", "depth"})
"""Data types that represent depth measurements and are eligible for depth clipping."""

ANNOTATOR_CHANNEL_COUNTS: dict[str, int] = {
    "motion_vectors": 2,
    "normals": 3,
    "rgb": 3,
    **{mode: 3 for mode in SIMPLE_SHADING_MODES},
}
"""Number of output channels for annotator data types that need channel slicing.

Replicator annotators may return more channels than the data type requires.
For example, ``"motion_vectors"`` returns a 4-channel buffer but only the first
2 channels (x, y) are meaningful.  Data types not in this mapping do not need
channel slicing.

See also `GitHub #2003 <https://github.com/isaac-sim/IsaacLab/issues/2003>`_
(motion vectors) and `GitHub #4239 <https://github.com/isaac-sim/IsaacLab/issues/4239>`_
(normals) for context on why alignment-correct slicing is required.
"""

SEGMENTATION_COLORIZE_FIELDS: dict[str, str] = {
    "semantic_segmentation": "colorize_semantic_segmentation",
    "instance_segmentation_fast": "colorize_instance_segmentation",
    "instance_id_segmentation_fast": "colorize_instance_id_segmentation",
}
"""Mapping from segmentation data type to the camera-config attribute that controls
colorization.

When colorization is enabled the raw ``uint32`` segmentation buffer is
reinterpreted as a 4-channel ``uint8`` RGBA image.
"""


def slice_output_channels(data: torch.Tensor, data_type: str) -> torch.Tensor:
    """Slice annotator output to the expected number of channels.

    Replicator annotators often return 4-channel buffers where only a subset
    of channels carry meaningful data.  This function slices to the correct
    channel count based on :data:`ANNOTATOR_CHANNEL_COUNTS`.

    Args:
        data: Annotator output tensor with channels in the last dimension.
        data_type: Camera data type name (e.g. ``"motion_vectors"``).

    Returns:
        A view of *data* with the trailing dimension sliced, or *data*
        unchanged when no slicing rule exists for *data_type*.
    """
    n = ANNOTATOR_CHANNEL_COUNTS.get(data_type)
    if n is not None:
        return data[..., :n]
    return data


def apply_depth_clipping(
    output: torch.Tensor,
    data_type: str,
    clipping_range: tuple[float, float],
    behavior: Literal["max", "zero", "none"],
) -> None:
    """Apply depth clipping to a camera output tensor in-place.

    Two operations are performed sequentially:

    1. For ``"distance_to_camera"`` data, values exceeding the far clipping plane are
       set to infinity. The ``distance_to_camera`` annotator returns the radial distance
       to the camera optical center, but the renderer clips w.r.t. the image plane, so
       some values can exceed the far plane.
    2. For all depth data types (see :data:`DEPTH_DATA_TYPES`), infinite values are
       replaced according to *behavior*: ``"zero"`` maps them to ``0.0``, ``"max"``
       maps them to the far clipping distance, and ``"none"`` leaves them as infinity.

    Args:
        output: Depth tensor to clip, modified in-place.
        data_type: The camera data type name (e.g. ``"distance_to_camera"``).
        clipping_range: Near and far clipping distances [m].
        behavior: Clipping behavior for values beyond the far plane.
    """
    far = clipping_range[1]

    if data_type == "distance_to_camera":
        output[output > far] = torch.inf

    if data_type in DEPTH_DATA_TYPES and behavior != "none":
        output[torch.isinf(output)] = 0.0 if behavior == "zero" else far
