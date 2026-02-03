# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Package containing implementation of Isaac Lab Mimic data generation."""

__version__ = "1.0.0"

# Configure deprecation warnings to show only once per session (regardless of call site)
# This prevents repeated warnings when deprecated properties are accessed from multiple locations
import warnings

warnings.filterwarnings("once", category=DeprecationWarning, module=r"isaaclab_mimic.*")
warnings.filterwarnings("once", category=FutureWarning, module=r"isaaclab_mimic.*")
