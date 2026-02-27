# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .record_cpu_info import CPUInfoRecorder
    from .record_gpu_info import GPUInfoRecorder
    from .record_memory_info import MemoryInfoRecorder
    from .record_version_info import VersionInfoRecorder

from isaaclab.utils.module import lazy_export

lazy_export(
    ("record_cpu_info", "CPUInfoRecorder"),
    ("record_gpu_info", "GPUInfoRecorder"),
    ("record_memory_info", "MemoryInfoRecorder"),
    ("record_version_info", "VersionInfoRecorder"),
)
