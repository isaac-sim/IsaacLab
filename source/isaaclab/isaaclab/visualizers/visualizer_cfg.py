# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from ..utils import configclass
from ..utils.string import string_to_callable

if TYPE_CHECKING:
    from .base_visualizer import BaseVisualizer


VISUALIZER_TYPES = {
    "kit": "kit:KitVisualizerCfg",
    "newton_gl": "newton:NewtonGLVisualizerCfg",
    "newton_rtx": "newton:NewtonRTXVisualizerCfg",
    "rerun": "rerun:RerunVisualizerCfg",
    "viser": "viser:ViserVisualizerCfg",
}
"""Canonical visualizer type names, for ``--visualizer`` and :attr:`VisualizerCfg.visualizer_type`, mapped to
the ``<isaaclab_visualizers subpackage>:<class>`` of their default config, imported only when needed."""

VISUALIZER_ALIASES = {"newton": "newton_gl"}
"""Deprecated ``--visualizer`` names and their replacements."""

USD_DEFAULT_VERTICAL_APERTURE_MM = 15.2908
"""Vertical aperture [mm] that visualizers use to turn a focal length into a vertical field of view."""

_VISUALIZER_EXTRAS = {
    "kit": "isaacsim",
    "rerun": "rerun",
    "viser": "viser",
}


def get_visualizer_install_hint(visualizer_type: str) -> str:
    """Return the uv command needed to run a visualizer backend."""
    extra = _VISUALIZER_EXTRAS.get(visualizer_type)
    if extra is None:
        return "Run your command with: uv run <command>."
    return f"Run your command with: uv run --extra {extra} <command>."


def parse_visualizer_csv(value: str | list[str]) -> list[str]:
    """Parse a ``--visualizer`` comma-separated list, or a list of names, into canonical names.

    Parsing canonical names again returns them unchanged.
    """
    names = value.split(",") if isinstance(value, str) else list(value)
    # an undocumented alias of omitting --visualizer, kept so older commands keep running
    if names == ["none"]:
        return []
    invalid = [name for name in names if name not in (*VISUALIZER_TYPES, *VISUALIZER_ALIASES)]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid --visualizer value {value!r}: use a comma-separated list, without spaces, of "
            f"{', '.join(VISUALIZER_TYPES)}."
        )
    for name in names:
        if name in VISUALIZER_ALIASES:
            warnings.warn(
                f"--viz '{name}' is deprecated. Use '--viz {VISUALIZER_ALIASES[name]}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
    return list(dict.fromkeys(VISUALIZER_ALIASES.get(name, name) for name in names))


def _make_visualizer_cfg(visualizer_type: str) -> VisualizerCfg:
    """Construct the default config of a visualizer type, importing only its backend package."""
    try:
        cfg_class = string_to_callable(f"isaaclab_visualizers.{VISUALIZER_TYPES[visualizer_type]}")
    except ValueError as exc:  # string_to_callable reports a missing module as ValueError
        raise RuntimeError(
            f"Visualizer '{visualizer_type}' is not available: {exc}. {get_visualizer_install_hint(visualizer_type)}"
        ) from exc
    return cfg_class()


def resolve_visualizer_cfgs(
    visualizer_cfgs: list[VisualizerCfg] | VisualizerCfg | None,
    visualizers: list[str] | None,
    max_visible_envs=None,
    headless_visualizers: tuple[str, ...] | list[str] = (),
) -> list[VisualizerCfg]:
    """Return the visualizers a run uses: exactly the selected types, configured by *visualizer_cfgs*.

    Each selected type reuses the configured visualizer of that type, with its settings, or else gets its default
    config; configured visualizers of unselected types do not run. Each type of *headless_visualizers* the
    selection lacks is added the same way but runs headless, e.g. a visualizer only a video recorder uses.

    Args:
        visualizer_cfgs: Configured visualizers, e.g. :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`.
        visualizers: Selection in canonical names (see :func:`parse_visualizer_csv`), empty for no visualizers.
            None applies no selection and keeps the configured visualizers, for a simulation built without a
            launch.
        max_visible_envs: ``--max_visible_envs`` applied to every resulting visualizer, or None.
        headless_visualizers: Capture-capable types (``kit``, ``newton_gl``, ``newton_rtx``) to add headless when
            not selected, in canonical names.
    """
    if visualizer_cfgs is None:
        visualizer_cfgs = []
    elif not isinstance(visualizer_cfgs, list):
        visualizer_cfgs = [visualizer_cfgs]
    if visualizers is not None:
        configured = {cfg.visualizer_type: cfg for cfg in reversed(visualizer_cfgs)}
        added = [name for name in headless_visualizers if name not in visualizers]
        visualizer_cfgs = [cfg for cfg in visualizer_cfgs if cfg.visualizer_type in visualizers]
        configured_types = {cfg.visualizer_type for cfg in visualizer_cfgs}
        visualizer_cfgs += [_make_visualizer_cfg(name) for name in visualizers if name not in configured_types]
        for name in added:
            # a headless copy of the configured visualizer of the type keeps its settings, e.g. the camera pose
            cfg = configured[name].copy() if name in configured else _make_visualizer_cfg(name)
            cfg.headless = True
            visualizer_cfgs.append(cfg)
    if max_visible_envs is not None:
        for cfg in visualizer_cfgs:
            cfg.max_visible_envs = int(max_visible_envs)
    return visualizer_cfgs


@configclass
class PerspectiveCameraCfg:
    """Initial pose and optics for a visualizer-owned interactive perspective camera."""

    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Eye position in world coordinates [m]."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look-at target in world coordinates [m]."""

    focal_length: float = 12.0
    """Perspective camera focal length [mm]."""


@configclass
class SceneCameraCfg:
    """Select an existing scene camera's output for display in the visualizer.

    This does not create a sensor. Declare its pose, optics, and renderer in the scene's CameraCfg.
    """

    prim_path: str = MISSING
    """Scene camera's configured prim path, including ``{ENV_REGEX_NS}`` when applicable."""


@configclass
class TrackingCameraCfg(SceneCameraCfg):
    """A scene camera that follows an asset, or stays fixed in each environment, created for the visualizer.

    Unlike :class:`SceneCameraCfg`, this declares the camera instead of referring to one: when a visualizer is
    selected or recorded, the launcher adds a :class:`~isaaclab.sensors.CameraCfg` with this pose, optics, and
    renderer to every environment, so runs without a visualizer pay nothing. The streaming view shows it like any
    other scene camera, e.g. recorded with ``--video viz:newton_gl:streaming_view``.
    """

    prim_path: str = "{ENV_REGEX_NS}/TrackingCamera"
    """Prim path of the created camera, including ``{ENV_REGEX_NS}``. Its last segment is the scene sensor name."""

    eye: tuple[float, float, float] | None = None
    """Camera eye offset [m] from the tracked asset, or from the environment origin when :attr:`track_path` is None.

    None uses :attr:`VisualizerCfg.eye` of the visualizer config that declares the camera, so a task adjusts both
    views through one value.
    """

    lookat: tuple[float, float, float] | None = None
    """Camera look-at offset [m], in the same frame as :attr:`eye`. None uses :attr:`VisualizerCfg.lookat`."""

    focal_length: float | None = None
    """Camera focal length [mm]. None uses :attr:`VisualizerCfg.focal_length`."""

    resolution: tuple[int, int] = (1920, 1080)
    """Camera image size as (width, height) [px]."""

    track_path: str | None = None
    """Scene asset to follow, or None to keep the camera fixed relative to each environment origin.

    Use ``"robot"`` for an asset root or ``"robot/base"`` for a named body, which must match exactly one body.
    """

    follow_heading: bool = False
    """Rotate :attr:`eye` and :attr:`lookat` with the tracked asset's yaw, keeping the horizon level.

    Without it the offsets stay aligned with the world axes. Only used with :attr:`track_path`.
    """

    heading_smoothing_time_constant: float = 0.0
    """Exponential heading-filter time constant [s]; zero follows the heading immediately.

    Larger values damp rapid turns more, with more lag. Only used with :attr:`follow_heading`.
    """

    renderer_cfg: Any = None
    """Renderer of the camera sensor, e.g. ``NewtonWarpRendererCfg(enable_shadows=True)`` to trade speed for
    quality. None uses the :class:`~isaaclab.sensors.CameraCfg` default."""

    data_types: tuple[str, ...] = ("rgb",)
    """Camera output channels; include every channel :attr:`VisualizerCfg.streaming_gt_types` displays."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.heading_smoothing_time_constant) or self.heading_smoothing_time_constant < 0.0:
            raise ValueError("heading_smoothing_time_constant must be finite and non-negative.")


@configclass
class VisualizerCfg:
    """Base configuration for all visualizer backends.

    Note:
        This configuration can be used directly as
        :attr:`~isaaclab.sim.SimulationCfg.default_visualizer_cfg` to provide shared defaults.
        To create a visualizer, use a concrete config from ``isaaclab_visualizers``, such as
        ``KitVisualizerCfg`` or ``NewtonGLVisualizerCfg``.
    """

    class_type: type[BaseVisualizer] | str | None = None
    """Visualizer implementation class. Concrete configs must set this field."""

    cloning_contexts: tuple[type | str, ...] = ()
    """Clone contexts that build this visualizer's scene representation from the asset plan."""

    # Camera sources
    cameras: list[PerspectiveCameraCfg | SceneCameraCfg] | None = None
    """Camera sources the visualizer displays, in display order. None uses :attr:`eye`, :attr:`lookat` and
    :attr:`focal_length` for the main viewport and shows no scene camera unless :attr:`streaming_view` is set.

    * :class:`PerspectiveCameraCfg`: the interactive main viewport, which you can move.
    * :class:`SceneCameraCfg`: the output of a camera sensor the scene already declares, shown in the streaming
      panel. A :class:`TrackingCameraCfg` is a scene camera that the launcher creates for the visualizer.

    Any scene source turns :attr:`streaming_view` on, and every one must provide all
    :attr:`streaming_gt_types` channels. Kit, Newton GL, Rerun, and Viser show the first scene source by default,
    and Newton GL also lets you switch between them. Newton RTX supports only perspective sources.
    """

    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Interactive visualizer camera eye position in world coordinates."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Interactive visualizer camera look-at target in world coordinates."""

    focal_length: float = 12.0
    """Camera focal length in millimeters for visualizer camera views."""

    background_color: tuple[float, float, float] | None = (0.30, 0.55, 0.82)
    """Solid background color as normalized RGB values in ``[0, 1]``.

    Kit, Newton GL, and Newton RTX honor this field. Set it to ``None`` to preserve the
    backend's native background. Scene lighting remains independent of the visible background.
    """

    # ── Streaming view ────────────────────────────────────────────────────────
    # Reads pixels from a scene-declared camera sensor, tiles them
    # across envs and GT types, and shows the result as an image panel in interactive
    # visualizers (Newton GL, Kit) or pushes it per-step to sink-based ones (Rerun, Viser).

    streaming_view: bool = False
    """Enable the streaming camera image view (opt-in, disabled by default)."""

    streaming_sensor_prim_path: str | None = None
    """Prim path of a scene-declared :class:`~isaaclab.sensors.Camera` to display.

    Use the camera's configured prim path, including ``{ENV_REGEX_NS}`` when applicable.
    None selects the first compatible camera in the scene. Without one the panel stays empty.
    The scene owns camera construction, updates, and lifetime; visualizers only read its output.
    """

    # Shared settings
    streaming_envs: int | list[int] = 32
    """Environments to display.

    * ``int`` — sample this many envs once at initialization (from all visible envs).
    * ``list[int]`` — display exactly these env indices.
    """

    streaming_gt_types: tuple[str, ...] = ("rgb",)
    """GT data types displayed left-to-right per environment row.

    Valid values: ``"rgb"``, ``"depth"``, ``"segmentation"``, ``"normals"``.
    Validated against :data:`~isaaclab.envs.utils.camera_colorizer.SUPPORTED_GT_TYPES`
    at initialization time (only when :attr:`streaming_view` is ``True``).
    """

    streaming_depth_min: float = 0.1
    """Near-clip for the turbo depth colormap [m].  Used when ``"depth"`` is in
    :attr:`streaming_gt_types`."""

    streaming_depth_max: float = 10.0
    """Far-clip for the turbo depth colormap [m].  Used when ``"depth"`` is in
    :attr:`streaming_gt_types`."""

    # Partial visualization settings
    max_visible_envs: int | None = None
    """Upper bound on how many envs are shown.

    * If visible_env_indices is not None, then this field will apply also
      to the explicit env indices set to the visible_env_indices.
    """

    visible_env_indices: list[int] | None = None
    """env indices to visualize in order (out-of-range indices are dropped)."""

    randomly_sample_visible_envs: bool = True
    """If ``max_visible_envs`` is provided, when enabled, selected visible envs are randomly sampled.
       If disabled, the first ``max_visible_envs`` envs are selected.

    * Note: ``visible_env_indices`` overrides this field.
    """

    # Visualization Markers
    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    # Live Plots
    enable_live_plots: bool = True
    """Stream per-step scalar data (manager terms, episode reward, episode length) into the visualizer.

    Plot windows start hidden or collapsed by default and can be toggled open at runtime.
    Set to ``False`` to disable live plots entirely and avoid any collection overhead.
    """

    live_plots_update_interval: int = 5
    """Collect and push live plot data every ``N`` simulation steps (default: every 5 steps)."""

    # Internal
    visualizer_type: str | None = None
    """Type identifier (e.g., 'newton', 'rerun', 'viser', 'kit'). Must be overridden by subclasses."""

    def __post_init__(self) -> None:
        if self.background_color is not None:
            if len(self.background_color) != 3 or any(not 0.0 <= value <= 1.0 for value in self.background_color):
                raise ValueError("background_color must contain three normalized RGB values in [0, 1].")
            self.background_color = tuple(float(value) for value in self.background_color)

        if self.cameras is not None:
            if not self.cameras:
                raise ValueError("cameras must contain at least one display source.")
            self.streaming_view = any(isinstance(camera, SceneCameraCfg) for camera in self.cameras)
            if isinstance(camera := self.cameras[0], PerspectiveCameraCfg):
                self.eye, self.lookat, self.focal_length = camera.eye, camera.lookat, camera.focal_length
