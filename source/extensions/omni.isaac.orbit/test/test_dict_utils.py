# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import omni.isaac.orbit.utils.dict as dict_utils


class TestDictUtilities(unittest.TestCase):
    """Test fixture for checking Kit utilities in Orbit."""

    def test_print_dict(self):
        """Test printing of dictionary."""
        # create a complex nested dictionary
        test_dict = {
            "a": 1,
            "b": 2,
            "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
            "i": 7,
            "j": lambda x: x**2,
            "k": dict_utils.class_to_dict,
        }
        # print the dictionary
        dict_utils.print_dict(test_dict)


if __name__ == "__main__":
    unittest.main()
