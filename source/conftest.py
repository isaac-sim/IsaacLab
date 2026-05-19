# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dep-manifest diagnostic: prints numpy version + bundled OpenBLAS hash at pytest session start.

Loaded by pytest when collecting tests under ``source/``. Importing numpy here registers
its vendored OpenBLAS ``pthread_atfork`` handler in pytest's process, which is the same
process that later calls ``fork()`` via ``SimulationApp()``. The output identifies which
numpy + OpenBLAS bundle actually landed in each CI test container.
"""

import os

import numpy

print(f"\n[dep-manifest] numpy {numpy.__version__}", flush=True)
_libs_dir = os.path.join(os.path.dirname(numpy.__file__), os.pardir, "numpy.libs")
if os.path.isdir(_libs_dir):
    for _f in sorted(os.listdir(_libs_dir)):
        if "openblas" in _f.lower():
            print(f"[dep-manifest] bundled openblas: {_f}", flush=True)
