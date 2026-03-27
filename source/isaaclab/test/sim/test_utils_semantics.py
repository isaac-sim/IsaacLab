# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# note: need to enable cameras to be able to make replicator core available
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest

import isaaclab.sim as sim_utils


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""
    # Setup: Create a new stage
    sim_utils.create_new_stage()
    sim_utils.update_stage()

    # Yield for the test
    yield

    # Teardown: Clear stage after each test
    sim_utils.clear_stage()


def create_test_environment_with_labels():
    """Creates a test environment with objects with labels."""
    # create 3 cubes with label "cube"
    for i in range(3):
        sim_utils.create_prim(f"/World/Test/Object{i}", "Cube", semantic_label="cube")
    # create a sphere without any labels
    sim_utils.create_prim("/World/Test/Object3", "Sphere")
    # create a nested prim with label "nested"
    nested_prim = sim_utils.create_prim("/World/Test/Object0/Nested", "Cube")
    sim_utils.add_labels(nested_prim, ["nested"], instance_name="shape")

    return [f"/World/Test/Object{i}" for i in range(4)] + [str(nested_prim.GetPrimPath())]


"""
Tests.
"""


def test_add_and_get_labels():
    """Test add_labels() and get_labels() functions."""
    # get stage handle
    stage = sim_utils.get_current_stage()
    # create a test prim
    prim = stage.DefinePrim("/test", "Xform")
    nested_prim = stage.DefinePrim("/test/nested", "Xform")

    # Apply semantics
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")
    sim_utils.add_labels(nested_prim, ["nested_label"], instance_name="class")

    # Get labels
    labels_dict = sim_utils.get_labels(prim)
    # Check labels are added correctly
    assert "class" in labels_dict
    assert sorted(labels_dict["class"]) == sorted(["label_a", "label_b"])
    assert "shape" in labels_dict
    assert labels_dict["shape"] == ["shape_a"]
    nested_labels_dict = sim_utils.get_labels(nested_prim)
    assert "class" in nested_labels_dict
    assert nested_labels_dict["class"] == ["nested_label"]


def test_add_labels_with_overwrite():
    """Test add_labels() function with overwriting existing labels."""
    # get stage handle
    stage = sim_utils.get_current_stage()
    # create a test prim
    prim = stage.DefinePrim("/test", "Xform")

    # Add labels
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")

    # Overwrite existing labels for a specific instance
    sim_utils.add_labels(prim, ["replaced_label"], instance_name="class", overwrite=True)
    labels_dict = sim_utils.get_labels(prim)
    assert labels_dict["class"] == ["replaced_label"]
    assert "shape" in labels_dict
    assert labels_dict["shape"] == ["shape_a"]


def test_add_labels_without_overwrite():
    """Test add_labels() function without overwriting existing labels."""
    # get stage handle
    stage = sim_utils.get_current_stage()
    # create a test prim
    prim = stage.DefinePrim("/test", "Xform")

    # Add labels
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")

    # Re-add labels with overwrite=False (should append)
    sim_utils.add_labels(prim, ["label_c"], instance_name="class", overwrite=False)
    labels_dict = sim_utils.get_labels(prim)
    assert sorted(labels_dict["class"]) == sorted(["label_a", "label_b", "label_c"])


def test_remove_all_labels():
    """Test removing of all labels from a prim and its descendants."""
    # get stage handle
    stage = sim_utils.get_current_stage()
    # create a test prim
    prim = stage.DefinePrim("/test", "Xform")
    nested_prim = stage.DefinePrim("/test/nested", "Xform")

    # Add labels
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")
    sim_utils.add_labels(nested_prim, ["nested_label"], instance_name="class")

    # Remove all labels
    sim_utils.remove_labels(prim)
    # Check labels are removed correctly
    labels_dict = sim_utils.get_labels(prim)
    assert len(labels_dict) == 0
    # Check nested prim labels are not removed
    nested_labels_dict = sim_utils.get_labels(nested_prim)
    assert "class" in nested_labels_dict
    assert nested_labels_dict["class"] == ["nested_label"]

    # Re-add labels
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")
    sim_utils.add_labels(nested_prim, ["nested_label"], instance_name="class")
    # Remove all labels
    sim_utils.remove_labels(prim, include_descendants=True)
    # Check labels are removed correctly
    labels_dict = sim_utils.get_labels(prim)
    assert len(labels_dict) == 0
    # Check nested prim labels are removed
    nested_labels_dict = sim_utils.get_labels(nested_prim)
    assert len(nested_labels_dict) == 0


def test_remove_specific_labels():
    """Test removing of specific labels from a prim and its descendants."""
    # get stage handle
    stage = sim_utils.get_current_stage()
    # create a test prim
    prim = stage.DefinePrim("/test", "Xform")
    nested_prim = stage.DefinePrim("/test/nested", "Xform")

    # Add labels
    sim_utils.add_labels(prim, ["label_a", "label_b"], instance_name="class")
    sim_utils.add_labels(prim, ["shape_a"], instance_name="shape")
    sim_utils.add_labels(nested_prim, ["nested_label"], instance_name="class")
    sim_utils.add_labels(nested_prim, ["nested_shape"], instance_name="shape")

    # Remove specific labels
    sim_utils.remove_labels(prim, instance_name="shape")
    # Check labels are removed correctly
    labels_dict = sim_utils.get_labels(prim)
    assert "shape" not in labels_dict
    assert "class" in labels_dict
    assert sorted(labels_dict["class"]) == sorted(["label_a", "label_b"])
    # Check nested prim labels are not removed
    nested_labels_dict = sim_utils.get_labels(nested_prim)
    assert "class" in nested_labels_dict
    assert nested_labels_dict["class"] == ["nested_label"]

    # Remove specific labels
    sim_utils.remove_labels(prim, instance_name="class", include_descendants=True)
    # Check labels are removed correctly
    labels_dict = sim_utils.get_labels(prim)
    assert len(labels_dict) == 0
    # Check nested prim labels are removed
    nested_labels_dict = sim_utils.get_labels(nested_prim)
    assert "shape" in nested_labels_dict
    assert nested_labels_dict["shape"] == ["nested_shape"]


def test_check_missing_labels():
    """Test the check_missing_labels() function."""
    # create a test environment with labels
    object_paths = create_test_environment_with_labels()

    # Check from root
    missing_paths = sim_utils.check_missing_labels()

    # Only the sphere should be missing
    assert len(missing_paths) == 1
    assert object_paths[3] in missing_paths  # Object3 should be missing

    # Check from specific subtree
    missing_paths_subtree = sim_utils.check_missing_labels(prim_path="/World/Test/Object0")
    # Object0 and Nested both have labels
    assert len(missing_paths_subtree) == 0

    # Check from invalid path
    missing_paths_invalid = sim_utils.check_missing_labels(prim_path="/World/Test/Invalid")
    assert len(missing_paths_invalid) == 0


def test_count_labels_in_scene():
    """Test the count_labels_in_scene() function."""
    # create a test environment with labels
    create_test_environment_with_labels()

    # Count from root
    labels_dict = sim_utils.count_total_labels()
    # Object0 and Nested both have labels
    assert labels_dict.get("cube", 0) == 3
    assert labels_dict.get("nested", 0) == 1
    assert labels_dict.get("missing_labels", 0) == 1

    # Count from specific subtree
    labels_dict_subtree = sim_utils.count_total_labels(prim_path="/World/Test/Object0")
    assert labels_dict_subtree.get("cube", 0) == 1
    assert labels_dict_subtree.get("nested", 0) == 1
    assert labels_dict_subtree.get("missing_labels", 0) == 0

    # Count from invalid path
    labels_dict_invalid = sim_utils.count_total_labels(prim_path="/World/Test/Invalid")
    assert labels_dict_invalid.get("missing_labels", 0) == 0
