# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# The following tests are skipped by run_tests.py
TESTS_TO_SKIP = [
    # orbit
    "test_argparser_launch.py",  # app.close issue
    "test_env_var_launch.py",  # app.close issue
    "test_kwarg_launch.py",  # app.close issue
    "test_differential_ik.py",  # Failing
    # orbit_tasks
    "test_data_collector.py",  # Failing
    "test_record_video.py",  # Failing
]
