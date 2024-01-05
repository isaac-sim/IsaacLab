# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
import unittest

import omni.isaac.orbit.utils.string as string_utils


class TestStringUtilities(unittest.TestCase):
    """Test fixture for checking string utilities."""

    def test_case_conversion(self):
        """Test case conversion between camel case and snake case."""
        # test camel case to snake case
        self.assertEqual(string_utils.to_snake_case("CamelCase"), "camel_case")
        self.assertEqual(string_utils.to_snake_case("camelCase"), "camel_case")
        self.assertEqual(string_utils.to_snake_case("CamelCaseString"), "camel_case_string")
        # test snake case to camel case
        self.assertEqual(string_utils.to_camel_case("snake_case", to="CC"), "SnakeCase")
        self.assertEqual(string_utils.to_camel_case("snake_case_string", to="CC"), "SnakeCaseString")
        self.assertEqual(string_utils.to_camel_case("snake_case_string", to="cC"), "snakeCaseString")

    def test_resolve_matching_names_with_basic_strings(self):
        """Test resolving matching names with a basic expression."""
        # list of strings
        target_names = ["a", "b", "c", "d", "e"]
        # test matching names
        query_names = ["a|c", "b"]
        index_list, names_list = string_utils.resolve_matching_names(query_names, target_names)
        self.assertEqual(index_list, [0, 1, 2])
        self.assertEqual(names_list, ["a", "b", "c"])
        # test matching names with regex
        query_names = ["a.*", "b"]
        index_list, names_list = string_utils.resolve_matching_names(query_names, target_names)
        self.assertEqual(index_list, [0, 1])
        self.assertEqual(names_list, ["a", "b"])
        # test duplicate names
        query_names = ["a|c", "b", "a|c"]
        with self.assertRaises(ValueError):
            _ = string_utils.resolve_matching_names(query_names, target_names)
        # test no regex match
        query_names = ["a|c", "b", "f"]
        with self.assertRaises(ValueError):
            _ = string_utils.resolve_matching_names(query_names, target_names)

    def test_resolve_matching_names_with_joint_name_strings(self):
        """Test resolving matching names with joint names."""
        # list of strings
        robot_joint_names = []
        for i in ["hip", "thigh", "calf"]:
            for j in ["FL", "FR", "RL", "RR"]:
                robot_joint_names.append(f"{j}_{i}_joint")
        # test matching names
        index_list, names_list = string_utils.resolve_matching_names(".*", robot_joint_names)
        self.assertEqual(index_list, list(range(len(robot_joint_names))))
        self.assertEqual(names_list, robot_joint_names)
        # test matching names with regex
        index_list, names_list = string_utils.resolve_matching_names(".*_joint", robot_joint_names)
        self.assertEqual(index_list, list(range(len(robot_joint_names))))
        self.assertEqual(names_list, robot_joint_names)
        # test matching names with regex
        index_list, names_list = string_utils.resolve_matching_names(["FL.*", "FR.*"], robot_joint_names)
        ground_truth_index_list = [0, 1, 4, 5, 8, 9]
        self.assertEqual(index_list, ground_truth_index_list)
        self.assertEqual(names_list, [robot_joint_names[i] for i in ground_truth_index_list])
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
        self.assertNotEqual(names_list, query_list)
        self.assertEqual(index_list, ground_truth_index_list)
        self.assertEqual(names_list, [robot_joint_names[i] for i in ground_truth_index_list])
        # test matching names with regex but shuffled
        # randomize order of previous query list
        random.shuffle(query_list)
        index_list, names_list = string_utils.resolve_matching_names(query_list, robot_joint_names)
        ground_truth_index_list = [0, 1, 4, 5, 8, 9]
        self.assertNotEqual(names_list, query_list)
        self.assertEqual(index_list, ground_truth_index_list)
        self.assertEqual(names_list, [robot_joint_names[i] for i in ground_truth_index_list])

    def test_resolve_matching_names_values_with_basic_strings(self):
        """Test resolving matching names with a basic expression."""
        # list of strings
        target_names = ["a", "b", "c", "d", "e"]
        # test matching names
        data = {"a|c": 1, "b": 2}
        index_list, names_list, values_list = string_utils.resolve_matching_names_values(data, target_names)
        self.assertEqual(index_list, [0, 1, 2])
        self.assertEqual(names_list, ["a", "b", "c"])
        self.assertEqual(values_list, [1, 2, 1])
        # test matching names with regex
        data = {"a|d|e": 1, "b|c": 2}
        index_list, names_list, values_list = string_utils.resolve_matching_names_values(data, target_names)
        self.assertEqual(index_list, [0, 1, 2, 3, 4])
        self.assertEqual(names_list, ["a", "b", "c", "d", "e"])
        self.assertEqual(values_list, [1, 2, 2, 1, 1])
        # test matching names with regex
        data = {"a|d|e|b": 1, "b|c": 2}
        with self.assertRaises(ValueError):
            _ = string_utils.resolve_matching_names_values(data, target_names)
        # test no regex match
        query_names = {"a|c": 1, "b": 0, "f": 2}
        with self.assertRaises(ValueError):
            _ = string_utils.resolve_matching_names(query_names, target_names)


if __name__ == "__main__":
    unittest.main()
