# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX PVA data container."""

from isaaclab.sensors.pva.pva_data import _PvaDataStorage


class PvaData(_PvaDataStorage):
    """PhysX PVA data backed by shared output storage.

    .. deprecated::
        Direct use of this backend class is deprecated. Read ``sensor.data`` or construct
        :class:`isaaclab.sensors.PvaData` instead. No removal release is scheduled.
    """
