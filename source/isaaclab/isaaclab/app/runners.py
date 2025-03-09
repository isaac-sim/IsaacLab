# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with runners to simplify running main via pytest."""

import pytest

def run_tests(file: str, verbose: bool = True, **kwargs):
    """Wrapper for running tests via ``pytest`` for a specific file.

    Args:
        file: The path to the test file to run.
        verbose: Whether to run tests with verbose output.
        **kwargs: Additional arguments to pass to the `pytest.main` function.
    """

    # Add verbosity flag if verbose is True
    verbosity_flag = ["-v"] if verbose else []

    # Run pytest with `--capture=no` to avoid getting stuck
    return pytest.main([file, "--capture=no"] + verbosity_flag + list(kwargs.get("extra_args", [])))

