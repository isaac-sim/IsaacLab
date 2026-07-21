# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class CableObjectData:
    """Unsupported OpenUSD PhysX cable object data implementation."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the cable object data."""

    def __init__(self, root_view: object, device: str):
        """Raise because OpenUSD PhysX does not support cable object data.

        Args:
            root_view: The backend root view.
            device: The device used for processing.

        Raises:
            NotImplementedError: Always raised.
        """
        del root_view, device
        raise NotImplementedError("CableObjectData is not supported by the OpenUSD PhysX backend.")
