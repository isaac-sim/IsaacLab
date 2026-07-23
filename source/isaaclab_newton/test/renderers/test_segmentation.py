# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Newton Warp renderer's segmentation mapping (no GPU / sim required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("warp")
pytest.importorskip("pxr")

from isaaclab_newton.renderers.segmentation import NewtonSegmentationMapper

from pxr import Usd

# The color palette / reserved ids live in core and are unit-tested there
# (``isaaclab/test/renderers/test_segmentation_colors.py``); here they are only an oracle for the
# mapper's info-dict keys.
from isaaclab.renderers.segmentation_colors import BACKGROUND_ID, UNLABELLED_ID, pack_rgba, random_color_from_id
from isaaclab.sim.utils.semantics import add_labels


def _cfg(**overrides):
    """Minimal renderer-cfg stand-in exposing only the fields the mapper reads."""
    base = {
        "semantic_filter": "*:*",
        "semantic_segmentation_mapping": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _scene():
    """Two cartpole robots (each with labelled root) plus an unlabelled ground plane.

    Returns the in-memory stage and the per-shape prim-path list (``model.shape_label``).
    """
    stage = Usd.Stage.CreateInMemory()
    shape_paths = [
        "/World/envs/env_0/Robot/pole/geom",
        "/World/envs/env_0/Robot/cart/geom",
        "/World/envs/env_1/Robot/pole/geom",
        "/World/ground/geom",
    ]
    for path in shape_paths:
        stage.DefinePrim(path, "Mesh")
    # Semantic label authored on the robot roots; inherited by the descendant geom shapes.
    for robot in ("/World/envs/env_0/Robot", "/World/envs/env_1/Robot"):
        add_labels(stage.GetPrimAtPath(robot), labels=["cartpole"], instance_name="class")
    return stage, shape_paths


def _model(shape_paths):
    return SimpleNamespace(shape_label=list(shape_paths), device="cpu")


def test_semantic_segmentation_shares_class_id_across_envs():
    """All cartpole shapes across envs share one class id; the unlabelled ground is UNLABELLED."""
    stage, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), stage, _cfg())
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    # env_0 pole, env_0 cart, env_1 pole all share the cartpole class id; ground is UNLABELLED.
    assert ids[0] == ids[1] == ids[2]
    assert ids[0] >= 2
    assert ids[3] == UNLABELLED_ID
    labels = mapping.info["idToLabels"]
    assert labels[BACKGROUND_ID] == {"class": "BACKGROUND"}
    assert labels[UNLABELLED_ID] == {"class": "UNLABELLED"}
    assert labels[ids[0]] == {"class": "cartpole"}


def test_instance_segmentation_groups_by_labelled_ancestor():
    """Shapes group by their nearest labelled ancestor; idToSemantics carries the class label."""
    stage, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), stage, _cfg())
    mapper.build_mapping("instance_segmentation_fast", colorize=False)
    mapping = mapper.get_mapping("instance_segmentation_fast", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    # env_0 pole and cart share the env_0/Robot instance; env_1 is a separate instance; ground unlabelled.
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
    assert ids[3] == UNLABELLED_ID
    assert mapping.info["idToLabels"][ids[0]] == "/World/envs/env_0/Robot"
    assert mapping.info["idToSemantics"][ids[0]] == {"class": "cartpole"}


def test_colorize_info_keys_are_color_tuples():
    """With colorization, info keys are ``(r, g, b, a)`` color tuples and a color palette is built."""
    stage, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), stage, _cfg())
    mapper.build_mapping("semantic_segmentation", colorize=True)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=True)

    assert mapping.shape_to_color is not None
    assert random_color_from_id(BACKGROUND_ID) in mapping.info["idToLabels"]
    assert random_color_from_id(UNLABELLED_ID) in mapping.info["idToLabels"]


def test_semantic_filter_excludes_non_matching_types():
    """A filter restricted to an absent type marks every shape UNLABELLED."""
    stage, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), stage, _cfg(semantic_filter=["shape"]))
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    assert mapping.shape_to_id.numpy().tolist() == [UNLABELLED_ID] * len(shape_paths)


def test_semantic_filter_comma_separated_type_clauses():
    """Comma-separated ``type:label`` pairs within one semicolon group each match independently.

    Filter ``"class:cartpole, material:wood"`` must label shapes annotated with ``class:cartpole``
    *and* shapes annotated with ``material:wood``; before the fix only the first type clause was
    parsed, leaving ``material:wood`` shapes UNLABELLED.
    """
    stage = Usd.Stage.CreateInMemory()
    shape_paths = [
        "/World/robot/geom",
        "/World/shelf/geom",
        "/World/ground/geom",
    ]
    for path in shape_paths:
        stage.DefinePrim(path, "Mesh")
    # Different semantic types on distinct parent prims.
    add_labels(stage.GetPrimAtPath("/World/robot"), labels=["cartpole"], instance_name="class")
    add_labels(stage.GetPrimAtPath("/World/shelf"), labels=["wood"], instance_name="material")

    mapper = NewtonSegmentationMapper(_model(shape_paths), stage, _cfg(semantic_filter="class:cartpole, material:wood"))
    mapper.build_mapping("semantic_segmentation", colorize=False)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=False)

    ids = mapping.shape_to_id.numpy().tolist()
    assert ids[0] >= 2, "class:cartpole shape should be labelled"
    assert ids[1] >= 2, "material:wood shape should be labelled (was UNLABELLED before fix)"
    assert ids[0] != ids[1], "different label payloads must map to different ids"
    assert ids[2] == UNLABELLED_ID


def test_ancestor_cache_prevents_redundant_get_labels_calls():
    """Each unique ancestor prim is queried at most once even when many shapes share ancestry.

    Before the fix, a labelled robot root with N leaf shapes caused N redundant ``get_labels``
    calls on the same root prim because the cache was only consulted for the original leaf path,
    not for each intermediate ancestor during the walk.
    """
    from unittest.mock import patch

    import isaaclab.sim.utils.semantics as _semantics_mod

    stage, shape_paths = _scene()
    mapper = NewtonSegmentationMapper(_model(shape_paths), stage, _cfg())

    original_get_labels = _semantics_mod.get_labels
    queried_paths: list[str] = []

    def counting_get_labels(prim):
        queried_paths.append(prim.GetPath().pathString)
        return original_get_labels(prim)

    with patch.object(_semantics_mod, "get_labels", side_effect=counting_get_labels):
        mapper.build_mapping("semantic_segmentation", colorize=False)

    duplicates = [p for p in set(queried_paths) if queried_paths.count(p) > 1]
    assert not duplicates, f"Prims queried more than once (ancestor cache missed): {duplicates}"


def test_semantic_segmentation_mapping_overrides_color():
    """``semantic_segmentation_mapping`` forces the class color and its info key."""
    stage, shape_paths = _scene()
    override = (255, 36, 66, 255)
    mapper = NewtonSegmentationMapper(
        _model(shape_paths), stage, _cfg(semantic_segmentation_mapping={"class:cartpole": override})
    )
    mapper.build_mapping("semantic_segmentation", colorize=True)
    mapping = mapper.get_mapping("semantic_segmentation", colorize=True)

    # The cartpole class id must be colored with the override, and keyed by it in idToLabels.
    assert override in mapping.info["idToLabels"]
    assert mapping.info["idToLabels"][override] == {"class": "cartpole"}
    packed = pack_rgba(override)
    assert packed in mapping.shape_to_color.numpy().tolist()
