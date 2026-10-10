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
from typing import TYPE_CHECKING

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
    """Select an existing scene camera's output as the visualizer's main view.

    This does not create a sensor. Declare its pose, optics, and renderer in the scene's CameraCfg.
    """

    prim_path: str = MISSING
    """Scene camera's configured prim path, including ``{ENV_REGEX_NS}`` when applicable."""


@configclass
class WindowCfg:
    """Native window presentation, independent of the camera's image resolution."""

    size: tuple[int, int] = (1920, 1080)
    """Initial window width and height [px]."""

    fps: float = 30.0
    """Maximum Newton window update rate [Hz], measured in wall-clock time.

    Headless on-demand recording is independent of this limit. Kit owns its application update cadence.
    """

    def validate_config(self) -> None:
        """Require positive dimensions and a finite, positive presentation rate."""
        if len(self.size) != 2 or any(not isinstance(value, int) or value < 1 for value in self.size):
            raise ValueError("WindowCfg.size must contain positive integer width and height.")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("WindowCfg.fps must be finite and positive.")


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

    window: WindowCfg = WindowCfg()
    """Native window settings used by Newton and Kit; network visualizers manage their own presentation."""

    headless: bool = False
    """Render on demand without opening a native viewer window."""

    # Primary interactive camera settings
    cameras: list[PerspectiveCameraCfg | SceneCameraCfg] | None = None
    """Camera sources available to the visualizer.

    PerspectiveCameraCfg configures the interactive view; SceneCameraCfg refers to an existing scene
    sensor. Newton GL and RTX display the first source and offer a dropdown for switching sources.
    Kit, Rerun, and Viser display the first scene source in their camera panel.
    None uses eye/lookat/focal_length and the streaming settings below.
    Every explicit scene source must provide all requested streaming_gt_types channels.
    """

    eye: tuple[float, float, float] = (4.0, -4.0, 3.0)
    """Interactive visualizer camera eye position in world coordinates."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Interactive visualizer camera look-at target in world coordinates."""

    focal_length: float = 12.0
    """Camera focal length in millimeters for visualizer camera views."""

    background_color: tuple[float, float, float] | None = None
    """Viewer perspective-camera background as normalized RGB values in ``[0, 1]``.

    None preserves the scene HDR background in Kit and Newton RTX, or Newton GL's procedural sky.
    An explicit color changes only the visible background, not scene lighting or reflections.
    Scene-camera images retain their sensor's :attr:`~isaaclab.sensors.CameraCfg.background_color`.
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
