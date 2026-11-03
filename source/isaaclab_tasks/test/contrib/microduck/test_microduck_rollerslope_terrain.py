# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the MicroDuck roller-slope sub-terrain.

The sub-terrain is pure geometry -- it takes a difficulty and returns meshes plus a spawn origin --
so the tests build it directly and then *measure the built mesh* rather than re-checking the
formulas that built it. Surface heights are read by casting a ray straight down onto the assembled
tile, which is the only way to catch a gap or a step at the two joins; the expected profile is the
independent piecewise-linear reference
``0 -> -(x - flat_length) * tan(angle) -> -drop`` worked out from
``artifacts/microduck/upstream_reference_tasks4.md`` section 2.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from isaaclab_tasks.contrib.microduck.rollerslope.terrain import (
    MICRODUCK_RAMP_DEG_MAX,
    MICRODUCK_RAMP_DEG_MIN,
    FlatRampTerrainCfg,
    flat_ramp_terrain,
)

pytestmark = pytest.mark.unit


TILE_SIZE = (15.0, 4.0)
"""Sub-terrain tile [m] the roller-slope task generates, sized for the longest ramp plus margin."""

DIFFICULTY_TO_DEGREES = [
    (-1.0, MICRODUCK_RAMP_DEG_MIN),
    (0.0, MICRODUCK_RAMP_DEG_MIN),
    (0.25, 6.5),
    (0.5, 11.0),
    (0.75, 15.5),
    (1.0, MICRODUCK_RAMP_DEG_MAX),
    (2.0, MICRODUCK_RAMP_DEG_MAX),
]
"""The difficulty-to-ramp-angle map, including the two clamped ends.

Every interior entry is ``2 + 18 * difficulty`` evaluated by hand, which is upstream's
``deg_min + d * (deg_max - deg_min)`` (``slope_terrain.py:26-31``).
"""


def _make_cfg(**kwargs) -> FlatRampTerrainCfg:
    """Build a sub-terrain configuration on the task's tile, with a pinned ramp length by default.

    A degenerate ``ramp_length_range`` removes the only random draw in the terrain, so a test that
    does not care about the draw gets a deterministic tile. This is the same pin the accuracy-gate
    harness uses.
    """
    kwargs.setdefault("ramp_length_range", (5.0, 5.0))
    cfg = FlatRampTerrainCfg(**kwargs)
    cfg.size = TILE_SIZE
    return cfg


def _surface_height(meshes: list[trimesh.Trimesh], x: float, y: float) -> float:
    """Height [m] of the highest terrain surface under a point, measured by a downward ray.

    Args:
        meshes: The sub-terrain meshes, in the sub-terrain's own frame.
        x: Sample position along the slope [m].
        y: Sample position across the slope [m].

    Returns:
        The surface height [m], or ``nan`` where the tile has no geometry.
    """
    tile = trimesh.util.concatenate(meshes)
    hits, _, _ = tile.ray.intersects_location(
        ray_origins=np.array([[x, y, 10.0]]), ray_directions=np.array([[0.0, 0.0, -1.0]])
    )
    if len(hits) == 0:
        return float("nan")
    return float(np.max(hits[:, 2]))


def _expected_height(cfg: FlatRampTerrainCfg, ramp_length: float, angle: float, x: float) -> float:
    """The piecewise-linear reference profile, derived independently of the implementation."""
    if x <= cfg.flat_length:
        return 0.0
    if x <= cfg.flat_length + ramp_length:
        return -(x - cfg.flat_length) * math.tan(angle)
    return -ramp_length * math.tan(angle)


##
# The difficulty-to-angle map
##


@pytest.mark.parametrize("difficulty, degrees", DIFFICULTY_TO_DEGREES)
def test_the_ramp_angle_interpolates_the_difficulty_and_clamps_outside_it(difficulty, degrees):
    cfg = _make_cfg()
    _, origin = flat_ramp_terrain(difficulty, cfg)

    # the surface at the spawn distance pins the angle: z = -spawn_on_ramp * tan(angle)
    angle = math.atan(-origin[2] / cfg.spawn_on_ramp)
    assert math.degrees(angle) == pytest.approx(degrees, abs=1e-9)


def test_the_difficulty_endpoints_reach_the_documented_two_and_twenty_degrees():
    # Restated as an explicit endpoint check because the terrain generator never draws exactly 0.0
    # or 1.0 -- it draws (row + eta) / num_rows -- so the endpoints are a property of this map only.
    cfg = _make_cfg()
    shallow = flat_ramp_terrain(0.0, cfg)[1]
    steep = flat_ramp_terrain(1.0, cfg)[1]

    assert math.degrees(math.atan(-shallow[2] / cfg.spawn_on_ramp)) == pytest.approx(2.0, abs=1e-9)
    assert math.degrees(math.atan(-steep[2] / cfg.spawn_on_ramp)) == pytest.approx(20.0, abs=1e-9)


##
# The built surface
##


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_the_built_surface_follows_the_flat_ramp_runout_profile_without_a_gap_or_a_step(difficulty):
    ramp_length = 5.0
    cfg = _make_cfg(ramp_length_range=(ramp_length, ramp_length))
    meshes, _ = flat_ramp_terrain(difficulty, cfg)
    angle = math.radians(MICRODUCK_RAMP_DEG_MIN + difficulty * (MICRODUCK_RAMP_DEG_MAX - MICRODUCK_RAMP_DEG_MIN))

    # dense either side of both joins, where a missing shim would open a gap, plus the interiors
    top_join = cfg.flat_length
    bottom_join = cfg.flat_length + ramp_length
    samples = np.concatenate(
        [
            np.linspace(top_join - 0.1, top_join + 0.1, 21),
            np.linspace(bottom_join - 0.1, bottom_join + 0.1, 21),
            np.linspace(0.05, bottom_join + cfg.runout_length - 0.05, 15),
        ]
    )
    for x in samples:
        measured = _surface_height(meshes, float(x), cfg.size[1] / 2.0)
        assert measured == pytest.approx(_expected_height(cfg, ramp_length, angle, float(x)), abs=1e-6), (
            f"surface breaks at x={x:.4f} on a {math.degrees(angle):.1f} degree ramp"
        )


def test_dropping_the_shim_would_have_opened_a_gap_the_test_above_can_see():
    # Guards the guard: the join test is only meaningful if the tolerance is far below the error a
    # naive implementation makes. Upstream's shim is -(thickness / 2) * sin(angle) in x
    # (``slope_terrain.py:78-84``), so that is exactly the gap it closes.
    cfg = _make_cfg()
    naive_gap = (cfg.thickness / 2.0) * math.sin(math.radians(MICRODUCK_RAMP_DEG_MAX))

    assert naive_gap == pytest.approx(0.0855, abs=1e-4)
    assert naive_gap > 1e-6 * 1000


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_the_spawn_origin_sits_on_the_incline_surface(difficulty):
    cfg = _make_cfg()
    meshes, origin = flat_ramp_terrain(difficulty, cfg)

    assert origin[0] == pytest.approx(cfg.flat_length + cfg.spawn_on_ramp)
    assert _surface_height(meshes, float(origin[0]), float(origin[1])) == pytest.approx(origin[2], abs=1e-6)
    # past the flat, so a robot placed here rolls under gravity rather than needing a push
    assert origin[2] < 0.0


def test_the_sub_terrain_is_centred_on_the_tile_rather_than_on_its_edge():
    # Deliberate deviation from upstream, which places the boxes and the origin at local y = 0 and so
    # straddles the tile boundary (addendum section 2.3 / 9.1). Isaac Lab sub-terrains span
    # (0, 0) to size, so the ramp is centred and the origin is on the centreline.
    cfg = _make_cfg()
    meshes, origin = flat_ramp_terrain(0.5, cfg)
    bounds = trimesh.util.concatenate(meshes).bounds

    assert origin[1] == pytest.approx(cfg.size[1] / 2.0)
    assert bounds[0][1] == pytest.approx(0.0)
    assert bounds[1][1] == pytest.approx(cfg.size[1])


def test_a_steeper_ramp_drops_the_runout_further_for_the_same_ramp_length():
    cfg = _make_cfg()
    # inside the run-out, which spans x in [7, 11] for the pinned 5 m ramp
    shallow = _surface_height(flat_ramp_terrain(0.0, cfg)[0], 10.0, cfg.size[1] / 2.0)
    steep = _surface_height(flat_ramp_terrain(1.0, cfg)[0], 10.0, cfg.size[1] / 2.0)

    assert steep < shallow < 0.0
    # 5 m of ramp at 20 degrees, which is what the void-guard constant is sized against
    assert steep == pytest.approx(-5.0 * math.tan(math.radians(20.0)), abs=1e-6)


##
# Configuration handling
##


def test_a_tile_too_short_for_the_longest_ramp_is_rejected_rather_than_silently_truncated():
    cfg = _make_cfg(ramp_length_range=(3.0, 8.0))
    cfg.size = (10.0, 4.0)  # 2 + 8 + 4 = 14 > 10

    with pytest.raises(ValueError, match="does not fit"):
        flat_ramp_terrain(0.5, cfg)


def test_the_ramp_length_is_drawn_once_per_tile_inside_its_range():
    cfg = _make_cfg(ramp_length_range=(3.0, 8.0))
    lengths = set()
    for _ in range(16):
        meshes, _ = flat_ramp_terrain(0.5, cfg)
        # the runout's top surface starts one ramp length past the flat
        bottom = trimesh.util.concatenate(meshes[2:]).bounds
        lengths.add(round(float(bottom[0][0]) - cfg.flat_length, 6))

    assert len(lengths) > 1, "the ramp length should vary between tiles"
    assert all(3.0 <= length <= 8.0 for length in lengths)


def test_pinning_the_ramp_length_range_makes_the_tile_deterministic():
    # What the accuracy-gate harness relies on: the sub-terrain draws from the global numpy state,
    # which the terrain generator does not seed, so the gate pins the range instead.
    cfg = _make_cfg(ramp_length_range=(5.0, 5.0))
    first = trimesh.util.concatenate(flat_ramp_terrain(0.5, cfg)[0]).bounds
    second = trimesh.util.concatenate(flat_ramp_terrain(0.5, cfg)[0]).bounds

    assert np.allclose(first, second)
