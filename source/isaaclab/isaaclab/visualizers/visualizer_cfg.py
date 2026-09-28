# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import configclass

if TYPE_CHECKING:
    from .base_visualizer import BaseVisualizer


_VISUALIZER_TYPES = ("kit", "newton_gl", "newton_rtx", "rerun", "viser")
"""Canonical visualizer type names, for ``--visualizer`` and :attr:`VisualizerCfg.visualizer_type`."""

_VISUALIZER_ALIASES = {"newton": "newton_gl"}
"""Deprecated ``--visualizer`` names and their replacements."""

_VISUALIZER_EXTRAS = {
    "kit": "isaacsim",
    "rerun": "rerun",
    "viser": "viser",
}


def _get_visualizer_install_hint(visualizer_type: str) -> str:
    """Return the uv command needed to run a visualizer backend."""
    extra = _VISUALIZER_EXTRAS.get(visualizer_type)
    if extra is None:
        return "Run your command with: uv run <command>."
    return f"Run your command with: uv run --extra {extra} <command>."


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

    # Primary interactive camera settings
    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Interactive visualizer camera eye position in world coordinates."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Interactive visualizer camera look-at target in world coordinates."""

    focal_length: float = 12.0
    """Camera focal length in millimeters for visualizer camera views."""

    background_color: tuple[float, float, float] | None = None
    """Solid background color as normalized RGB values in ``[0, 1]``.

    None preserves the scene HDR background in Kit and Newton RTX, or Newton GL's procedural sky.
    An explicit color changes only the visible background, not scene lighting or reflections.
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
    None selects the first camera in the scene. Without a scene camera the panel stays empty.
    The scene owns camera construction, updates, and lifetime; visualizers only read its output.
    """

    # Shared settings
    streaming_envs: int | list[int] = 32
    """Environments to capture.

    * ``int`` — sample this many envs once at initialization (from all visible envs).
    * ``list[int]`` — capture exactly these env indices.
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

    # Deprecated aliases kept for one-release compatibility. Remove in the next major release.
    tiled_cam_view: bool | None = None
    """Deprecated. Use :attr:`streaming_view` instead."""

    tiled_cam_num: int | None = None
    """Deprecated. Use :attr:`streaming_envs` (int) instead."""

    tiled_cam_env_indices: list[int] | None = None
    """Deprecated. Use :attr:`streaming_envs` (list[int]) instead."""

    tiled_cam_prim_path: str | None = None
    """Deprecated. Use :attr:`streaming_sensor_prim_path` instead."""

    def __post_init__(self) -> None:
        import warnings

        if self.background_color is not None:
            if len(self.background_color) != 3 or any(not 0.0 <= value <= 1.0 for value in self.background_color):
                raise ValueError("background_color must contain three normalized RGB values in [0, 1].")
            self.background_color = tuple(float(value) for value in self.background_color)

        _simple = [
            ("tiled_cam_view", "streaming_view"),
            ("tiled_cam_prim_path", "streaming_sensor_prim_path"),
        ]
        for old, new in _simple:
            val = getattr(self, old)
            if val is not None:
                warnings.warn(f"{old!r} is deprecated; use {new!r} instead.", DeprecationWarning, stacklevel=3)
                setattr(self, new, val)
                setattr(self, old, None)
        # tiled_cam_env_indices takes priority over tiled_cam_num
        env_indices = getattr(self, "tiled_cam_env_indices")
        if env_indices is not None:
            warnings.warn(
                "'tiled_cam_env_indices' is deprecated; use 'streaming_envs' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            self.streaming_envs = env_indices
            self.tiled_cam_env_indices = None
            self.tiled_cam_num = None
        else:
            num = getattr(self, "tiled_cam_num")
            if num is not None:
                warnings.warn(
                    "'tiled_cam_num' is deprecated; use 'streaming_envs' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                self.streaming_envs = num
                self.tiled_cam_num = None
