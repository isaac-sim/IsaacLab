# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: While we don't actually use the simulation app in this test, we still need to launch it
#       because warp is only available in the context of a running simulation
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import random

import pytest

import isaaclab.utils.string as string_utils


def test_case_conversion():
    """Test case conversion between camel case and snake case."""
    # test camel case to snake case
    assert string_utils.to_snake_case("CamelCase") == "camel_case"
    assert string_utils.to_snake_case("camelCase") == "camel_case"
    assert string_utils.to_snake_case("CamelCaseString") == "camel_case_string"
    # test snake case to camel case
    assert string_utils.to_camel_case("snake_case", to="CC") == "SnakeCase"
    assert string_utils.to_camel_case("snake_case_string", to="CC") == "SnakeCaseString"
    assert string_utils.to_camel_case("snake_case_string", to="cC") == "snakeCaseString"


def test_resolve_matching_names_with_basic_strings():
    """Test resolving matching names with a basic expression."""
    # list of strings
    target_names = ["a", "b", "c", "d", "e"]
    # test matching names
    query_names = ["a|c", "b"]
    index_list, names_list = string_utils.resolve_matching_names(query_names, target_names)
    assert index_list == [0, 1, 2]
    assert names_list == ["a", "b", "c"]
    # test matching names with regex
    query_names = ["a.*", "b"]
    index_list, names_list = string_utils.resolve_matching_names(query_names, target_names)
    assert index_list == [0, 1]
    assert names_list == ["a", "b"]
    # test duplicate names
    query_names = ["a|c", "b", "a|c"]
    with pytest.raises(ValueError):
        _ = string_utils.resolve_matching_names(query_names, target_names)
    # test no regex match
    query_names = ["a|c", "b", "f"]
    with pytest.raises(ValueError):
        _ = string_utils.resolve_matching_names(query_names, target_names)


def test_resolve_matching_names_with_joint_name_strings():
    """Test resolving matching names with joint names."""
    # list of strings
    robot_joint_names = []
    for i in ["hip", "thigh", "calf"]:
        for j in ["FL", "FR", "RL", "RR"]:
            robot_joint_names.append(f"{j}_{i}_joint")
    # test matching names
    index_list, names_list = string_utils.resolve_matching_names(".*", robot_joint_names)
    assert index_list == list(range(len(robot_joint_names)))
    assert names_list == robot_joint_names
    # test matching names with regex
    index_list, names_list = string_utils.resolve_matching_names(".*_joint", robot_joint_names)
    assert index_list == list(range(len(robot_joint_names)))
    assert names_list == robot_joint_names
    # test matching names with regex
    index_list, names_list = string_utils.resolve_matching_names(["FL.*", "FR.*"], robot_joint_names)
    ground_truth_index_list = [0, 1, 4, 5, 8, 9]
    assert index_list == ground_truth_index_list
    assert names_list == [robot_joint_names[i] for i in ground_truth_index_list]
    # test matching names with regex
    query_list = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
    ]
    index_list, names_list = string_utils.resolve_matching_names(query_list, robot_joint_names)
    ground_truth_index_list = [0, 1, 4, 5, 8, 9]
    assert names_list != query_list
    assert index_list == ground_truth_index_list
    assert names_list == [robot_joint_names[i] for i in ground_truth_index_list]
    # test matching names with regex but shuffled
    # randomize order of previous query list
    random.shuffle(query_list)
    index_list, names_list = string_utils.resolve_matching_names(query_list, robot_joint_names)
    ground_truth_index_list = [0, 1, 4, 5, 8, 9]
    assert names_list != query_list
    assert index_list == ground_truth_index_list
    assert names_list == [robot_joint_names[i] for i in ground_truth_index_list]


def test_resolve_matching_names_with_preserved_order():
    """Test resolving matching names with preserved order."""
    # list of strings and query list
    robot_joint_names = []
    for i in ["hip", "thigh", "calf"]:
        for j in ["FL", "FR", "RL", "RR"]:
            robot_joint_names.append(f"{j}_{i}_joint")
    query_list = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FL_calf_joint",
        "FR_calf_joint",
    ]
    # test return in target ordering with sublist
    query_list.reverse()
    index_list, names_list = string_utils.resolve_matching_names(query_list, robot_joint_names, preserve_order=True)
    ground_truth_index_list = [9, 8, 5, 1, 4, 0]
    assert names_list == query_list
    assert index_list == ground_truth_index_list
    # test return in target ordering with regex expression
    index_list, names_list = string_utils.resolve_matching_names(
        ["FR.*", "FL.*"], robot_joint_names, preserve_order=True
    )
    ground_truth_index_list = [1, 5, 9, 0, 4, 8]
    assert index_list == ground_truth_index_list
    assert names_list == [robot_joint_names[i] for i in ground_truth_index_list]
    # test return in target ordering with a mix of regex and non-regex expression
    index_list, names_list = string_utils.resolve_matching_names(
        ["FR.*", "FL_calf_joint", "FL_thigh_joint", "FL_hip_joint"], robot_joint_names, preserve_order=True
    )
    ground_truth_index_list = [1, 5, 9, 8, 4, 0]
    assert index_list == ground_truth_index_list
    assert names_list == [robot_joint_names[i] for i in ground_truth_index_list]


def test_resolve_matching_names_values_with_basic_strings():
    """Test resolving matching names with a basic expression."""
    # list of strings
    target_names = ["a", "b", "c", "d", "e"]
    # test matching names
    data = {"a|c": 1, "b": 2}
    index_list, names_list, values_list = string_utils.resolve_matching_names_values(data, target_names)
    assert index_list == [0, 1, 2]
    assert names_list == ["a", "b", "c"]
    assert values_list == [1, 2, 1]
    # test matching names with regex
    data = {"a|d|e": 1, "b|c": 2}
    index_list, names_list, values_list = string_utils.resolve_matching_names_values(data, target_names)
    assert index_list == [0, 1, 2, 3, 4]
    assert names_list == ["a", "b", "c", "d", "e"]
    assert values_list == [1, 2, 2, 1, 1]
    # test matching names with regex
    data = {"a|d|e|b": 1, "b|c": 2}
    with pytest.raises(ValueError):
        _ = string_utils.resolve_matching_names_values(data, target_names)
    # test no regex match
    query_names = {"a|c": 1, "b": 0, "f": 2}
    with pytest.raises(ValueError):
        _ = string_utils.resolve_matching_names_values(query_names, target_names)


def test_resolve_matching_names_values_with_basic_strings_and_preserved_order():
    """Test resolving matching names with a basic expression."""
    # list of strings
    target_names = ["a", "b", "c", "d", "e"]
    # test matching names
    data = {"a|c": 1, "b": 2}
    index_list, names_list, values_list = string_utils.resolve_matching_names_values(
        data, target_names, preserve_order=True
    )
    assert index_list == [0, 2, 1]
    assert names_list == ["a", "c", "b"]
    assert values_list == [1, 1, 2]
    # test matching names with regex
    data = {"a|d|e": 1, "b|c": 2}
    index_list, names_list, values_list = string_utils.resolve_matching_names_values(
        data, target_names, preserve_order=True
    )
    assert index_list == [0, 3, 4, 1, 2]
    assert names_list == ["a", "d", "e", "b", "c"]
    assert values_list == [1, 1, 1, 2, 2]
    # test matching names with regex
    data = {"a|d|e|b": 1, "b|c": 2}
    with pytest.raises(ValueError):
        _ = string_utils.resolve_matching_names_values(data, target_names, preserve_order=True)
    # test no regex match
    query_names = {"a|c": 1, "b": 0, "f": 2}
    with pytest.raises(ValueError):
        _ = string_utils.resolve_matching_names_values(query_names, target_names, preserve_order=True)
