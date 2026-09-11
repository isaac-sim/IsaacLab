# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scene and measurement for the tiled-camera tile-identity tests.

A tiled camera returns one image per environment, sliced out of a single tiled framebuffer. Nothing in
the returned tensor's *shape* proves that slice ``i`` holds environment ``i``'s view: a renderer that
produces fewer tiles than were requested still yields a correctly shaped tensor, and the reshape then
hands every environment someone else's pixels. Counting black images does not catch that either, because
the mis-sliced tiles usually still contain plausible scene content.

This builds a scene where each environment is *visually identifiable*: every environment holds one cube,
raised to one of ``num_levels`` heights chosen from the environment index. Each camera sees only its own
cube, so the vertical position of the bright pixels in a tile encodes the level of the environment that
tile belongs to. Grouping the measured positions by assigned level must therefore produce cleanly
separated bands. Under a correct mapping the bands are a pixel or so wide and separated by ~10 px; if
tiles cross a level boundary -- permuted, duplicated, or truncated between environments assigned different
levels -- every band fills with a mixture of levels and they overlap.

Only ``num_levels`` heights are used, shared across every environment at that level, so this does not
prove tile ``i`` holds exactly environment ``i``'s pixels: a swap or duplication between two tiles that
happen to share a level leaves the bands clean. The guarantee is narrower -- it catches layout errors that
move a tile across a level boundary, which is what the 4096-tile regression this module targets does (see
:mod:`test_tiled_camera_tile_identity_ovrtx`).

The measurement is renderer-agnostic (it uses the per-row pixel spread rather than assuming a black
background), so the same helper serves the OVRTX and Isaac RTX camera backends.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

SIM_DT = 1.0 / 60.0
"""Simulation step [s]."""

CUBE_SIZE = 0.4
"""Edge length of the per-environment marker cube [m]."""

BASE_Z = 0.5
"""Height of the lowest marker level above the environment origin [m]."""

LEVEL_STEP = 0.6
"""Vertical spacing between adjacent marker levels [m]. Sized so neighbouring levels land in
clearly separated image rows at the tile sizes these tests use."""

ENV_SPACING = 12.0
"""Distance between environment origins [m]. Wide enough that no camera sees a neighbour's cube."""

WARMUP_STEPS = 6
"""Steps rendered before measuring, so the marker poses and the first frame have settled."""

MIN_BAND_GAP_PX = 2.0
"""Smallest acceptable gap between adjacent level bands [px]. The correct mapping leaves roughly
10 px, so this rejects overlap without being sensitive to render noise."""


@configclass
class TileIdentitySceneCfg(InteractiveSceneCfg):
    """One marker cube per environment, raised to a per-environment height after reset.

    Gravity is disabled so the cube stays exactly where the test writes it, making the rendered
    position a direct readout of the assigned level rather than of the physics state.
    """

    marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Marker",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, BASE_Z)),
    )


@dataclass
class TileIdentityResult:
    """Per-tile measurement of a tile-identity render.

    Args:
        level: Level assigned to each environment, shape ``[num_envs]``.
        centroid: Vertical position of the marker in each tile [px], shape ``[num_envs]``.
        empty: Whether a tile contains no marker at all, shape ``[num_envs]``.
        num_levels: Number of distinct marker heights used.
    """

    level: torch.Tensor
    centroid: torch.Tensor
    empty: torch.Tensor
    num_levels: int

    def level_bands(self) -> list[tuple[float, float]]:
        """Return the ``(min, max)`` measured position of each level, over the tiles that rendered."""
        bands = []
        for value in range(self.num_levels):
            selected = self.centroid[(self.level == value) & ~self.empty]
            bands.append((float(selected.min()), float(selected.max())) if selected.numel() else (float("nan"),) * 2)
        return bands

    def smallest_band_gap(self) -> float:
        """Return the smallest gap between adjacent level bands [px].

        Marker height increases with level while image rows are numbered downwards, so a correct
        mapping yields strictly *decreasing* bands. A negative result means two levels overlap, i.e.
        tiles assigned different levels rendered at the same position and cannot all be correct.
        """
        bands = self.level_bands()
        return min(bands[value][0] - bands[value + 1][1] for value in range(self.num_levels - 1))

    def mismatched_tiles(self) -> torch.Tensor:
        """Return the indices of tiles whose measured position does not match their assigned level.

        Each level's expected position is taken as the median over the tiles assigned to it, so this
        needs no calibration against hard-coded pixel values. It exists to make a failure legible --
        the assertions use :meth:`smallest_band_gap` -- so it reports *which* environments are wrong.
        """
        expected = torch.stack(
            [
                self.centroid[(self.level == value) & ~self.empty].median()
                if ((self.level == value) & ~self.empty).any()
                else torch.tensor(float("nan"), device=self.centroid.device)
                for value in range(self.num_levels)
            ]
        )
        decoded = (self.centroid[:, None] - expected[None, :]).abs().argmin(dim=1)
        return ((decoded != self.level) | self.empty).nonzero().flatten()


def assign_levels(num_envs: int, num_levels: int, device: str) -> torch.Tensor:
    """Assign each environment a marker level.

    Uses a multiplicative hash rather than ``index % num_levels`` so the levels do not line up with
    the rows or columns of the tiled framebuffer. A layout error that shifts tiles by whole rows would
    otherwise map one level onto another and stay invisible.

    Args:
        num_envs: Number of environments.
        num_levels: Number of distinct marker heights.
        device: Device to build the assignment on.

    Returns:
        Level index per environment, shape ``[num_envs]``.
    """
    index = torch.arange(num_envs, device=device, dtype=torch.int64)
    return ((index * 2654435761) >> 16) % num_levels


def render_tile_identity(
    num_envs: int,
    renderer_cfg,
    physics_cfg,
    device: str = "cuda:0",
    tile_size: int = 64,
    num_levels: int = 4,
) -> TileIdentityResult:
    """Render ``num_envs`` identifiable environments and measure each tile.

    Args:
        num_envs: Number of environments, and therefore of camera tiles.
        renderer_cfg: Renderer configuration for the camera under test.
        physics_cfg: Physics backend configuration.
        device: Device to simulate and render on.
        tile_size: Per-camera tile width and height [px].
        num_levels: Number of distinct marker heights.

    Returns:
        The per-tile measurement.
    """
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(SimulationCfg(dt=SIM_DT, physics=physics_cfg, device=device))
    scene = InteractiveScene(TileIdentitySceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING, replicate_physics=True))
    # Centre the camera on the middle level so the highest and lowest markers stay inside the frame.
    camera = Camera(
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=tile_size,
            width=tile_size,
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg(
                pos=(-4.0, 0.0, BASE_Z + LEVEL_STEP * (num_levels - 1) / 2.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="world",
            ),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.1, 40.0)),
            renderer_cfg=renderer_cfg,
        )
    )
    try:
        sim.reset()

        level = assign_levels(num_envs, num_levels, sim.device)
        marker: RigidObject = scene["marker"]
        pose = marker.data.root_pose_w.torch.clone()
        pose[:, 2] = scene.env_origins[:, 2] + BASE_Z + level.float() * LEVEL_STEP
        marker.write_root_pose_to_sim(pose)

        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(SIM_DT, force_recompute=True)

        images = camera.data.output["rgb"].torch.float()
        return _measure_tiles(images[..., :3], level, num_levels)
    finally:
        del camera
        del scene
        sim.stop()
        sim.clear_instance()


def _measure_tiles(images: torch.Tensor, level: torch.Tensor, num_levels: int) -> TileIdentityResult:
    """Locate the marker in every tile.

    The marker is the only structure in an otherwise flat tile, so a row containing it has a much
    larger pixel spread than an empty one. Weighting each row by that spread and taking the centroid
    gives the marker's vertical position without assuming any particular background colour, which
    keeps the measurement usable across renderers.

    Args:
        images: Per-environment RGB images, shape ``[num_envs, H, W, 3]``.
        level: Level assigned to each environment, shape ``[num_envs]``.
        num_levels: Number of distinct marker heights.

    Returns:
        The per-tile measurement.
    """
    luma = images.mean(dim=-1)
    row_spread = luma.std(dim=2)
    weight = (row_spread - row_spread.amin(dim=1, keepdim=True)).clamp(min=0.0)
    total = weight.sum(dim=1)
    rows = torch.arange(luma.shape[1], device=luma.device, dtype=torch.float32)
    empty = total <= 1e-6
    centroid = (weight * rows).sum(dim=1) / total.clamp(min=1e-6)
    return TileIdentityResult(level=level, centroid=centroid, empty=empty, num_levels=num_levels)
