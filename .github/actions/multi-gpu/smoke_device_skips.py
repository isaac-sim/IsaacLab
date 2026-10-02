#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Count multi-GPU training smoke cases skipped because the runner has too few GPUs.

The smoke skips a case whose rank count exceeds the visible devices, which keeps it runnable
on a workstation but lets a smaller CI runner pass without running it. This reads one JUnit
report, warns once per such case, and writes ``device_skips=<n>`` to ``$GITHUB_OUTPUT`` so a
separate, non-required job can turn the lost coverage red.

Usage: ``smoke_device_skips.py <junit-report>``
"""

import os
import sys
import xml.etree.ElementTree as ET

# The reason ``_require_devices`` in test_multi_gpu_training_smoke.py skips with.
_DEVICE_SKIP = "visible CUDA devices"

report = sys.argv[1]
skipped = []
if os.path.exists(report):
    for case in ET.parse(report).getroot().iter("testcase"):
        skip = case.find("skipped")
        if skip is not None and _DEVICE_SKIP in skip.get("message", ""):
            skipped.append(f"{case.get('name')}: {skip.get('message')}")
else:
    print(f"::warning::{report} not found; smoke coverage unknown")

for case in skipped:
    print(f"::warning::multi-GPU smoke did not run {case}")
with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    fh.write(f"device_skips={len(skipped)}\n")
