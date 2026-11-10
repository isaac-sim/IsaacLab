# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record of the test output pytest captures and never prints.

pytest captures each test's output at the file-descriptor level, native writes included, and prints it only in
the report of a failing test. The CI runner reads the test process's own stdout and stderr, so it never
receives what a passing test printed. ``tools/conftest.py`` fails a test file whose output exceeds its budget,
and a passing test printing one warning per environment is the case that budget exists for. Loaded as a pytest
plugin by the repo-root ``conftest.py``, this module appends that unprinted output to the file named by
:data:`PATH_ENV_VAR`, which the runner counts together with the streams it read itself.
"""

import os

PATH_ENV_VAR = "ISAACLAB_TEST_CAPTURED_OUTPUT"
"""Environment variable naming the file unprinted output is appended to. Unset (the default) disables it.

The runner sets it per test file, mirroring the hang dump's ``ISAACLAB_HANG_DUMP``.
"""


def pytest_runtest_logreport(report):
    """Append the output a passing or skipped test phase captured, which no report prints."""
    path = os.environ.get(PATH_ENV_VAR)
    # Under pytest-xdist the controller receives every worker's reports, so it alone records them.
    if not path or report.failed or os.environ.get("PYTEST_XDIST_WORKER"):
        return
    # A report also carries the sections of the phases before it; only this phase's are new.
    names = (f"Captured stdout {report.when}", f"Captured stderr {report.when}")
    if texts := [text for name, text in report.sections if name in names]:
        with open(path, "a", encoding="utf-8", errors="replace") as file:
            file.writelines(texts)
