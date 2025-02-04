# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with runners to simplify running main via unittest."""

import inspect
import os
import unittest

import coverage


def run_tests(verbosity: int = 2, coverage_dir="logs/coverage", **kwargs):
    """Wrapper for running tests via ``unittest``.

    Args:
        verbosity: Verbosity level for the test runner.
        coverage_dir: Directory to store the coverage report. Defaults to None.
                      If None, coverage data collection isn't performed.
        **kwargs: Additional arguments to pass to the `unittest.main` function.

    """
    if coverage_dir is not None:
        # get the calling file's name
        calling_file = str(inspect.stack()[1].filename)
        # remove the path up until source and the .py extension
        calling_file = calling_file[calling_file.find("source") : -3]
        # replace any / or . with _
        calling_file = calling_file.replace("/", "_").replace(".", "_")
        coverage_path = os.path.join(coverage_dir, calling_file)
        # create the directory if it doesn't exist
        if not os.path.exists(coverage_dir):
            os.makedirs(coverage_dir)
        cov = coverage.Coverage(data_file=coverage_path)
        cov.start()

    # run main
    unittest.main(verbosity=verbosity, exit=True, **kwargs)

    if coverage_dir is not None:
        # stop coverage and save the data
        cov.stop()
        cov.save()
