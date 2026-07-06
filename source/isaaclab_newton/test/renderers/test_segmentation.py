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

from isaaclab_newton.renderers.segmentation import SegmentationMapper

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
    mapper = SegmentationMapper(_model(shape_paths), stage, _cfg())
    plan = mapper.plan("semantic_segmentation", colorize=False)

    ids = plan.shape_to_id.numpy().tolist()
    # env_0 pole, env_0 cart, env_1 pole all share the cartpole class id; ground is UNLABELLED.
    assert ids[0] == ids[1] == ids[2]
    assert ids[0] >= 2
    assert ids[3] == UNLABELLED_ID
    labels = plan.info["idToLabels"]
    assert labels[str(BACKGROUND_ID)] == {"class": "BACKGROUND"}
    assert labels[str(UNLABELLED_ID)] == {"class": "UNLABELLED"}
    assert labels[str(ids[0])] == {"class": "cartpole"}


def test_instance_id_segmentation_is_per_prim():
    """Each shape maps to a unique id and its own leaf prim path."""
    stage, shape_paths = _scene()
    mapper = SegmentationMapper(_model(shape_paths), stage, _cfg())
    plan = mapper.plan("instance_id_segmentation_fast", colorize=False)

    ids = plan.shape_to_id.numpy().tolist()
    assert len(set(ids)) == len(ids)  # all distinct
    labels = plan.info["idToLabels"]
    for shape_index, path in enumerate(shape_paths):
        assert labels[str(ids[shape_index])] == path


def test_instance_segmentation_groups_by_labelled_ancestor():
    """Shapes group by their nearest labelled ancestor; idToSemantics carries the class label."""
    stage, shape_paths = _scene()
    mapper = SegmentationMapper(_model(shape_paths), stage, _cfg())
    plan = mapper.plan("instance_segmentation_fast", colorize=False)

    ids = plan.shape_to_id.numpy().tolist()
    # env_0 pole and cart share the env_0/Robot instance; env_1 is a separate instance; ground unlabelled.
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
    assert ids[3] == UNLABELLED_ID
    assert plan.info["idToLabels"][str(ids[0])] == "/World/envs/env_0/Robot"
    assert plan.info["idToSemantics"][str(ids[0])] == {"class": "cartpole"}


def test_colorize_info_keys_are_color_tuples():
    """With colorization, info keys are ``(r, g, b, a)`` color strings and a color palette is built."""
    stage, shape_paths = _scene()
    mapper = SegmentationMapper(_model(shape_paths), stage, _cfg())
    plan = mapper.plan("semantic_segmentation", colorize=True)

    assert plan.shape_to_color is not None
    assert str(random_color_from_id(BACKGROUND_ID)) in plan.info["idToLabels"]
    assert str(random_color_from_id(UNLABELLED_ID)) in plan.info["idToLabels"]


def test_semantic_filter_excludes_non_matching_types():
    """A filter restricted to an absent type marks every shape UNLABELLED."""
    stage, shape_paths = _scene()
    mapper = SegmentationMapper(_model(shape_paths), stage, _cfg(semantic_filter=["shape"]))
    plan = mapper.plan("semantic_segmentation", colorize=False)

    assert plan.shape_to_id.numpy().tolist() == [UNLABELLED_ID] * len(shape_paths)


def test_semantic_segmentation_mapping_overrides_color():
    """``semantic_segmentation_mapping`` forces the class color and its info key."""
    stage, shape_paths = _scene()
    override = (255, 36, 66, 255)
    mapper = SegmentationMapper(
        _model(shape_paths), stage, _cfg(semantic_segmentation_mapping={"class:cartpole": override})
    )
    plan = mapper.plan("semantic_segmentation", colorize=True)

    # The cartpole class id must be colored with the override, and keyed by it in idToLabels.
    assert str(override) in plan.info["idToLabels"]
    assert plan.info["idToLabels"][str(override)] == {"class": "cartpole"}
    packed = pack_rgba(override)
    assert packed in plan.shape_to_color.numpy().tolist()
