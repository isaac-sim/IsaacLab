# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for isaaclab_newton.cloner.contact_filter.

These tests exercise the pure-Python resolution helpers without requiring a
running simulator.  A real :class:`newton.ModelBuilder` is used for the
integration tests so :func:`build_contact_sensor` is tested end-to-end
(pattern → indices → SensorContact construction).
"""

from __future__ import annotations

import numpy as np
import pytest
from isaaclab_newton.cloner.contact_filter import (
    _match_labels_in_world,
    _normalize_for_labels,
    _resolve_pattern,
    _to_fnmatch,
    build_contact_sensor,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BODY_NAMES = ["base", "shoulder", "elbow", "hand"]
SHAPE_NAMES = ["base_col", "shoulder_col", "elbow_col", "hand_col", "finger_l", "finger_r"]
NUM_WORLDS = 3


def _make_body_labels(n_worlds: int = NUM_WORLDS) -> list[str]:
    return [f"/World/envs/env_{w}/Robot/{name}" for w in range(n_worlds) for name in BODY_NAMES]


def _make_shape_labels(n_worlds: int = NUM_WORLDS) -> list[str]:
    return [f"/World/envs/env_{w}/Robot/{name}" for w in range(n_worlds) for name in SHAPE_NAMES]


def _make_world_start(per_world: int, n_worlds: int = NUM_WORLDS) -> np.ndarray:
    return np.arange(n_worlds + 1) * per_world


def _make_model(n_worlds: int = 1):
    """Build a minimal Newton model with rigid bodies and shapes."""
    from newton import ModelBuilder

    builder = ModelBuilder()
    for w in range(n_worlds):
        if n_worlds > 1:
            builder.begin_world()
        for name in ["base", "link1", "link2", "tip"]:
            builder.add_body(label=name)
            builder.add_shape_box(
                body=builder.body_count - 1,
                hx=0.05,
                hy=0.05,
                hz=0.05,
                label=f"{name}_col",
            )
        if n_worlds > 1:
            builder.end_world()
    return builder.finalize(device="cpu")


# ===================================================================
# _to_fnmatch
# ===================================================================


def test_to_fnmatch_none():
    assert _to_fnmatch(None) is None


def test_to_fnmatch_string():
    assert _to_fnmatch(".*_link") == "*_link"


def test_to_fnmatch_list():
    assert _to_fnmatch(["palm_.*", ".*finger.*"]) == ["palm_*", "*finger*"]


def test_to_fnmatch_no_regex():
    assert _to_fnmatch("hand") == "hand"


# ===================================================================
# _normalize_for_labels
# ===================================================================


def test_normalize_none_expr():
    assert _normalize_for_labels(None, ["a", "b"]) is None


def test_normalize_empty_labels():
    assert _normalize_for_labels("foo", []) == "foo"


def test_normalize_labels_have_paths_expr_has_paths():
    assert _normalize_for_labels("/World/Robot/*", ["/World/Robot/base"]) == "/World/Robot/*"


def test_normalize_labels_bare_expr_has_paths():
    assert _normalize_for_labels("/World/Robot/*", ["base", "hand"]) == "*"


def test_normalize_labels_bare_expr_bare():
    assert _normalize_for_labels("hand", ["base", "hand"]) == "hand"


def test_normalize_list():
    result = _normalize_for_labels(["/a/b/base", "/c/d/hand"], ["base", "hand"])
    assert result == ["base", "hand"]


def test_normalize_preserves_str_type():
    result = _normalize_for_labels("/a/base", ["base"])
    assert isinstance(result, str)
    assert result == "base"


# ===================================================================
# _match_labels_in_world
# ===================================================================


def test_match_single_pattern_all_worlds():
    labels = _make_body_labels()
    ws = _make_world_start(len(BODY_NAMES))
    assert _resolve_pattern("*hand*", labels, ws, NUM_WORLDS) == [3, 7, 11]


def test_match_multi_pattern():
    labels = _make_body_labels()
    ws = _make_world_start(len(BODY_NAMES))
    assert _resolve_pattern(["*base*", "*hand*"], labels, ws, NUM_WORLDS) == [0, 3, 4, 7, 8, 11]


def test_match_no_match():
    labels = _make_body_labels()
    ws = _make_world_start(len(BODY_NAMES))
    assert _resolve_pattern("*nonexistent*", labels, ws, NUM_WORLDS) == []


def test_match_single_world():
    labels = _make_body_labels(1)
    ws = _make_world_start(len(BODY_NAMES), 1)
    assert _match_labels_in_world(labels, "*elbow*", 0, ws) == [2]


# ===================================================================
# _resolve_pattern
# ===================================================================


def test_resolve_none():
    assert _resolve_pattern(None, ["a", "b"], None, 1) is None


def test_resolve_single_world_no_world_start():
    assert _resolve_pattern("*and*", ["base", "hand", "elbow"], None, 1) == [1]


def test_resolve_multi_world_with_world_start():
    labels = _make_body_labels()
    ws = _make_world_start(len(BODY_NAMES))
    assert _resolve_pattern("*shoulder*", labels, ws, NUM_WORLDS) == [1, 5, 9]


def test_resolve_multi_world_no_world_start_falls_back():
    """world_start=None with world_count>1 falls back to flat matching."""
    assert _resolve_pattern("*", ["base", "hand", "elbow"], None, 3) == [0, 1, 2]


def test_resolve_world_start_too_small():
    """world_start.size < 2 falls back to flat matching."""
    assert _resolve_pattern("*", ["base", "hand"], np.array([0]), 2) == [0, 1]


# ===================================================================
# Heterogeneous worlds
# ===================================================================


def test_heterogeneous_different_entity_counts():
    labels = ["apple", "banana", "cherry", "dog", "elephant", "fig", "grape", "honey", "ice"]
    ws = np.array([0, 3, 5, 9])
    result = _resolve_pattern("*a*", labels, ws, 3)
    assert result == [0, 1, 4, 6]


def test_heterogeneous_empty_world():
    labels = ["a", "b"]
    ws = np.array([0, 2, 2, 2])
    assert _resolve_pattern("*", labels, ws, 3) == [0, 1]


# ===================================================================
# build_contact_sensor — integration (real Newton ModelBuilder)
# ===================================================================


def test_build_single_world_body_sensor():
    sensor = build_contact_sensor(_make_model(1), body_names_expr="*tip*", prune_noncolliding=False)
    assert sensor.shape[0] == 1


def test_build_single_world_shape_sensor():
    sensor = build_contact_sensor(_make_model(1), shape_names_expr="*_col", prune_noncolliding=False)
    assert sensor.shape[0] == 4


def test_build_multi_world_body_sensor():
    sensor = build_contact_sensor(_make_model(3), body_names_expr="*tip*", prune_noncolliding=False)
    assert sensor.shape[0] == 3  # 1 per world × 3


def test_build_multi_world_shape_with_counterpart():
    sensor = build_contact_sensor(
        _make_model(3),
        shape_names_expr="*tip_col*",
        contact_partners_shape_expr="*base_col*",
        prune_noncolliding=False,
    )
    assert sensor.shape[0] == 3  # 1 sensing per world × 3
    # Replicated sensor: template has total + 1 base_col from world 0 = 2
    assert sensor.shape[1] == 2


def test_build_multi_world_all_shapes():
    sensor = build_contact_sensor(_make_model(3), shape_names_expr="*_col", prune_noncolliding=False)
    assert sensor.shape[0] == 12  # 4 shapes × 3 worlds


def test_build_regex_to_fnmatch():
    """``.*`` expressions are converted to ``*`` before matching."""
    sensor = build_contact_sensor(_make_model(1), body_names_expr=".*tip.*", prune_noncolliding=False)
    assert sensor.shape[0] == 1


def test_build_net_force_shape():
    sensor = build_contact_sensor(_make_model(2), body_names_expr="*base*", prune_noncolliding=False)
    assert sensor.net_force.shape[0] == 2
    assert sensor.net_force.shape[1] == sensor.shape[1]


def test_build_sensing_objs_global_indices():
    """Multi-world sensing_objs should contain global body indices."""
    sensor = build_contact_sensor(_make_model(3), body_names_expr="*tip*", prune_noncolliding=False)
    indices = [idx for idx, _ in sensor.sensing_objs]
    # "tip" is body 3 in each world of 4 bodies → 3, 7, 11
    assert indices == [3, 7, 11]


def test_build_no_match_returns_empty():
    sensor = build_contact_sensor(_make_model(1), body_names_expr="*nonexistent*", prune_noncolliding=False)
    assert sensor.shape == (0, 0)
    assert sensor.sensing_objs == []


# ===================================================================
# Filter correctness: replicated vs plain produce same sensing_objs
# ===================================================================


def test_replicated_sensing_objs_match_plain():
    """Replicated sensor must produce the same sensing_objs indices as plain path.

    Builds both paths for a small world count (where plain is still fast) and
    asserts that sensing_objs indices are identical.
    """
    from isaaclab_newton.cloner.contact_filter import (
        _normalize_for_labels,
        _resolve_pattern,
        _to_fnmatch,
    )
    from newton.sensors import SensorContact as NewtonContactSensor

    n_worlds = 4
    model = _make_model(n_worlds)

    replicated = build_contact_sensor(model, body_names_expr="*tip*", prune_noncolliding=False)

    body_labels = list(model.body_label)
    body_start = model.body_world_start.numpy()
    pattern = _normalize_for_labels(_to_fnmatch("*tip*"), body_labels)
    all_indices = _resolve_pattern(pattern, body_labels, body_start, n_worlds)
    plain = NewtonContactSensor(
        model,
        sensing_obj_bodies=all_indices,
        include_total=True,
        prune_noncolliding=False,
    )

    rep_indices = [idx for idx, _ in replicated.sensing_objs]
    plain_indices = [idx for idx, _ in plain.sensing_objs]
    assert rep_indices == plain_indices, f"Replicated sensing_objs {rep_indices} != plain {plain_indices}"


def test_replicated_counterparts_match_plain():
    """Replicated sensor counterparts should match the world-0 template structure."""
    from isaaclab_newton.cloner.contact_filter import (
        _match_labels_in_world,
        _normalize_for_labels,
        _to_fnmatch,
    )
    from newton.sensors import SensorContact as NewtonContactSensor

    n_worlds = 4
    model = _make_model(n_worlds)

    replicated = build_contact_sensor(
        model,
        shape_names_expr="*tip_col*",
        contact_partners_shape_expr="*base_col*",
        prune_noncolliding=False,
    )

    shape_labels = list(model.shape_label)
    shape_start = model.shape_world_start.numpy()
    sensing_pat = _normalize_for_labels(_to_fnmatch("*tip_col*"), shape_labels)
    counter_pat = _normalize_for_labels(_to_fnmatch("*base_col*"), shape_labels)
    ss0 = _match_labels_in_world(shape_labels, sensing_pat, 0, shape_start)
    cs0 = _match_labels_in_world(shape_labels, counter_pat, 0, shape_start)

    world0_plain = NewtonContactSensor(
        model,
        sensing_obj_shapes=ss0,
        counterpart_shapes=cs0,
        include_total=True,
        prune_noncolliding=False,
    )

    # Replicated counterparts should equal world-0 template counterparts
    rep_counter = [(idx, getattr(k, "value", k)) for idx, k in replicated.counterparts]
    plain_counter = [(idx, getattr(k, "value", k)) for idx, k in world0_plain.counterparts]
    assert rep_counter == plain_counter


def test_filter_multi_pattern_exact_indices():
    """Verify exact resolved indices when using multiple patterns."""
    from isaaclab_newton.cloner.contact_filter import (
        _normalize_for_labels,
        _resolve_pattern,
        _to_fnmatch,
    )

    n_worlds = 3
    model = _make_model(n_worlds)
    body_labels = list(model.body_label)
    body_start = model.body_world_start.numpy()

    # Match "base" and "tip" across 3 worlds (4 bodies per world)
    pattern = _normalize_for_labels(_to_fnmatch([".*base.*", ".*tip.*"]), body_labels)
    indices = _resolve_pattern(pattern, body_labels, body_start, n_worlds)

    # base=index 0 in each world, tip=index 3 in each world
    # world order: [base0, tip0, base1, tip1, base2, tip2]
    assert indices == [0, 3, 4, 7, 8, 11]


def test_filter_shape_names_exact_indices():
    """Verify exact resolved shape indices for multi-world model."""
    from isaaclab_newton.cloner.contact_filter import (
        _normalize_for_labels,
        _resolve_pattern,
        _to_fnmatch,
    )

    n_worlds = 2
    model = _make_model(n_worlds)
    shape_labels = list(model.shape_label)
    shape_start = model.shape_world_start.numpy()

    # Match "link2_col" — index 2 within each world's 4 shapes
    pattern = _normalize_for_labels(_to_fnmatch("*link2_col*"), shape_labels)
    indices = _resolve_pattern(pattern, shape_labels, shape_start, n_worlds)

    assert indices == [2, 6]  # 4 shapes per world: world0=2, world1=6


# ===================================================================
# Performance benchmarks
# ===================================================================


def test_build_multi_world_uses_replicated():
    """Multi-world models should use _ReplicatedContactSensor for O(1) setup per world."""
    from isaaclab_newton.cloner.contact_filter import _ReplicatedContactSensor

    sensor = build_contact_sensor(_make_model(4), body_names_expr="*tip*", prune_noncolliding=False)
    assert isinstance(sensor, _ReplicatedContactSensor)
    assert sensor.shape[0] == 4


def test_build_single_world_uses_newton_sensor():
    """Single-world models should use plain NewtonContactSensor."""
    from newton.sensors import SensorContact

    sensor = build_contact_sensor(_make_model(1), body_names_expr="*tip*", prune_noncolliding=False)
    assert isinstance(sensor, SensorContact)


@pytest.mark.parametrize("n_worlds", [128, 512, 4096])
def test_build_sensor_perf(n_worlds):
    """build_contact_sensor should complete in < 2s even for large world counts."""
    import time

    model = _make_model(n_worlds)

    t0 = time.perf_counter()
    sensor = build_contact_sensor(
        model,
        body_names_expr="*tip*",
        contact_partners_body_expr="*base*",
        prune_noncolliding=False,
    )
    elapsed = time.perf_counter() - t0

    assert sensor.shape[0] == n_worlds
    print(f"\n  build_contact_sensor({n_worlds} worlds): {elapsed:.3f}s")
    assert elapsed < 2.0, f"Took {elapsed:.3f}s for {n_worlds} worlds — too slow"
