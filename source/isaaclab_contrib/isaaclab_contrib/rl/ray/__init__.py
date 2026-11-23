# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contributed Ray job-dispatch and hyperparameter-tuning workflows."""

import os

# These tools use preinstalled cluster environments and explicitly upload job files.
# Ray's automatic uv environment instead uploads the checkout and rejects per-job pip dependencies.
# Configure this before importing Ray, as in the RLinf entry point.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
