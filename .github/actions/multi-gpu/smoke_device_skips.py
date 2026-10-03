#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Count multi-GPU training smoke cases skipped because the runner has too few GPUs.

The smoke skips every case on a host with fewer than two GPUs, which keeps it collectable on a
workstation but would let an undersized CI runner pass without running it. This reads one JUnit
report, warns once per such case, and writes ``device_skips=<n>`` to ``$GITHUB_OUTPUT`` so a
separate, non-required job can turn the lost coverage red. A missing or unreadable report writes
``device_skips=unknown``: no evidence that any case ran is not full coverage.

Usage: ``smoke_device_skips.py <junit-report>``
"""

import os
import sys
import xml.etree.ElementTree as ET

# The reason ``_visible_gpus`` in test_multi_gpu_training_smoke.py skips with.
_DEVICE_SKIP = "visible CUDA devices"

report = sys.argv[1]
try:
    cases = list(ET.parse(report).getroot().iter("testcase"))
except (OSError, ET.ParseError) as err:
    print(f"::warning::cannot read {report} ({err}); smoke coverage unknown")
    cases = None

if cases is None:
    device_skips = "unknown"
else:
    skipped = []
    for case in cases:
        skip = case.find("skipped")
        if skip is not None and _DEVICE_SKIP in skip.get("message", ""):
            skipped.append(f"{case.get('name')}: {skip.get('message')}")
    for case in skipped:
        print(f"::warning::multi-GPU smoke did not run {case}")
    device_skips = len(skipped)
with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
    fh.write(f"device_skips={device_skips}\n")
