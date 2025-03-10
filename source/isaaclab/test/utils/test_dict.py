# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# NOTE: While we don't actually use the simulation app in this test, we still need to launch it
#       because warp is only available in the context of a running simulation
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import unittest

import isaaclab.utils.dict as dict_utils


def test_function(x):
    """Test function for string <-> callable conversion."""
    return x**2


def test_lambda_function(x):
    """Test function for string <-> callable conversion."""
    return x**2


class TestDictUtilities(unittest.TestCase):
    """Test fixture for checking dictionary utilities in Isaac Lab."""

    def test_print_dict(self):
        """Test printing of dictionary."""
        # create a complex nested dictionary
        test_dict = {
            "a": 1,
            "b": 2,
            "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
            "i": 7,
            "j": lambda x: x**2,  # noqa: E731
            "k": dict_utils.class_to_dict,
        }
        # print the dictionary
        dict_utils.print_dict(test_dict)

    def test_string_callable_function_conversion(self):
        """Test string <-> callable conversion for function."""

        # convert function to string
        test_string = dict_utils.callable_to_string(test_function)
        # convert string to function
        test_function_2 = dict_utils.string_to_callable(test_string)
        # check that functions are the same
        self.assertEqual(test_function(2), test_function_2(2))

    def test_string_callable_function_with_lambda_in_name_conversion(self):
        """Test string <-> callable conversion for function which has lambda in its name."""

        # convert function to string
        test_string = dict_utils.callable_to_string(test_lambda_function)
        # convert string to function
        test_function_2 = dict_utils.string_to_callable(test_string)
        # check that functions are the same
        self.assertEqual(test_function(2), test_function_2(2))

    def test_string_callable_lambda_conversion(self):
        """Test string <-> callable conversion for lambda expression."""

        # create lambda function
        func = lambda x: x**2  # noqa: E731
        # convert function to string
        test_string = dict_utils.callable_to_string(func)
        # convert string to function
        func_2 = dict_utils.string_to_callable(test_string)
        # check that functions are the same
        self.assertEqual(test_string, "lambda x: x**2")
        self.assertEqual(func(2), func_2(2))

    def test_dict_to_md5(self):
        """Test MD5 hash generation for dictionary."""
        # create a complex nested dictionary
        test_dict = {
            "a": 1,
            "b": 2,
            "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
            "i": random.random(),
            "k": dict_utils.callable_to_string(dict_utils.class_to_dict),
        }
        # generate the MD5 hash
        md5_hash_1 = dict_utils.dict_to_md5_hash(test_dict)

        # check that the hash is correct even after multiple calls
        for _ in range(200):
            md5_hash_2 = dict_utils.dict_to_md5_hash(test_dict)
            self.assertEqual(md5_hash_1, md5_hash_2)


if __name__ == "__main__":
    run_tests()
