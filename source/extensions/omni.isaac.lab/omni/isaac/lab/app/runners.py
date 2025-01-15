# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with runners to simplify running main via unittest."""

import unittest


def run_tests(verbosity: int = 2, **kwargs):
    """Wrapper for running tests via ``unittest``.

    Args:
        verbosity: Verbosity level for the test runner.
        **kwargs: Additional arguments to pass to the `unittest.main` function.
    """
    # run main
    unittest.main(verbosity=verbosity, exit=True, **kwargs)
