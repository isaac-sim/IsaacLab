# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A dictionary of tests and their timeouts in seconds.

Any tests not listed here will use the default timeout.
"""
PER_TEST_TIMEOUTS = {
    "test_environments.py": 1200,  # This test runs through all the environments for 100 steps each
    "test_env_rendering_logic.py": 300,
    "test_rsl_rl_wrapper.py": 200,
    "test_sb3_wrapper.py": 200,
    "test_skrl_wrapper.py": 200,
}
