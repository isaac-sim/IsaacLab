# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ...utils.backend_utils import FactoryBase
from .base_usd_exporter import BaseUsdExporter


class UsdExporter(FactoryBase, BaseUsdExporter):
    """Factory for the USD exporter of the active physics backend."""

    def __new__(cls, *args, **kwargs) -> BaseUsdExporter:
        """Create the exporter of the active physics backend."""
        return super().__new__(cls, *args, **kwargs)
