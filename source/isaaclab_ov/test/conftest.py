# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared configuration for kitless OVPhysX tests."""

import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap when OpenUSD parses payloads concurrently in kitless processes.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"
