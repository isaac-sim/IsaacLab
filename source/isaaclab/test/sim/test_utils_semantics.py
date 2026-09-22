# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import isaaclab.sim as sim_utils

pytestmark = pytest.mark.unit


@pytest.fixture
def labeled_prims():
    """A prim carrying two label instances with a labeled nested child."""
    stage = sim_utils.create_new_stage()
    prim = stage.DefinePrim("/test", "Xform")
    nested = stage.DefinePrim("/test/nested", "Xform")
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")
    sim_utils.add_labels(nested, ["nested_label"], instance_name="class")
    sim_utils.add_labels(nested, ["nested_shape"], instance_name="shape")
    return prim, nested


def test_add_and_get_labels(labeled_prims):
    prim, nested = labeled_prims
    assert sim_utils.get_labels(prim) == {"class": ["label_a", "label_b"], "shape": ["shape_a"]}
    assert sim_utils.get_labels(nested) == {"class": ["nested_label"], "shape": ["nested_shape"]}

    # appending keeps existing labels and skips duplicates; overwriting replaces them
    sim_utils.add_labels(prim, ["label_b", "label_c"], instance_name="class", overwrite=False)
    assert sim_utils.get_labels(prim)["class"] == ["label_a", "label_b", "label_c"]
    sim_utils.add_labels(prim, ["replaced"], instance_name="class", overwrite=True)
    assert sim_utils.get_labels(prim) == {"class": ["replaced"], "shape": ["shape_a"]}


def test_remove_labels(labeled_prims):
    prim, nested = labeled_prims

    # a single instance on the prim only
    sim_utils.remove_labels(prim, instance_name="shape")
    assert sim_utils.get_labels(prim) == {"class": ["label_a", "label_b"]}
    assert sim_utils.get_labels(nested) == {"class": ["nested_label"], "shape": ["nested_shape"]}
    # a single instance including descendants
    sim_utils.remove_labels(prim, instance_name="class", include_descendants=True)
    assert sim_utils.get_labels(prim) == {}
    assert sim_utils.get_labels(nested) == {"shape": ["nested_shape"]}
    # everything including descendants
    sim_utils.add_labels(prim, ["label_a"], instance_name="class")
    sim_utils.remove_labels(prim, include_descendants=True)
    assert sim_utils.get_labels(prim) == {}
    assert sim_utils.get_labels(nested) == {}


def test_check_missing_and_count_labels():
    sim_utils.create_new_stage()
    for i in range(3):
        sim_utils.create_prim(f"/World/Test/Object{i}", "Cube", semantic_label="cube")
    sim_utils.create_prim("/World/Test/Object3", "Sphere")
    nested = sim_utils.create_prim("/World/Test/Object0/Nested", "Cube")
    sim_utils.add_labels(nested, ["nested"], instance_name="shape")

    # only geometry prims are inspected
    assert sim_utils.check_missing_labels() == ["/World/Test/Object3"]
    assert sim_utils.check_missing_labels(prim_path="/World/Test/Object0") == []
    assert sim_utils.check_missing_labels(prim_path="/World/Test/Invalid") == []

    assert sim_utils.count_total_labels() == {"missing_labels": 1, "cube": 3, "nested": 1}
    assert sim_utils.count_total_labels(prim_path="/World/Test/Object0") == {
        "missing_labels": 0,
        "cube": 1,
        "nested": 1,
    }
    assert sim_utils.count_total_labels(prim_path="/World/Test/Invalid") == {"missing_labels": 0}
