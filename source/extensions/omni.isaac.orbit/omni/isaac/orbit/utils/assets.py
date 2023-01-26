# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Defines the host-server where assets and resources are stored.

By default, we use the Isaac Sim Nucleus Server for hosting assets and resources. This makes
distribution of the assets easier and makes the repository smaller in size code-wise.

For more information on Omniverse Nucleus:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus
"""

import io
import os

import carb
import omni.client
import omni.isaac.core.utils.nucleus as nucleus_utils

# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    carb.log_error(msg)
    raise RuntimeError(msg)


ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""

ISAAC_ORBIT_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac/Samples/Orbit"
"""Path to the `Isaac/Samples/Orbit` directory on the NVIDIA Nucleus Server."""


def check_file_path(path: str) -> int:
    """Checks if a file exists on the Nucleus Server or locally.

    Args:
        path (str): The path to the file.

    Returns:
        int: The status of the file. Possible values are: :obj:`0` if the file does not exist
            :obj:`1` if the file exists locally, or :obj:`2` if the file exists on the Nucleus Server.
    """
    if os.path.isfile(path):
        return 1
    elif omni.client.stat(path)[0] == omni.client.Result.OK:
        return 2
    else:
        return 0


def read_file(path: str) -> io.BytesIO:
    """Reads a file from the Nucleus Server or locally.

    Args:
        path (str): The path to the file.

    Raises:
        FileNotFoundError: When the file not found locally or on Nucleus Server.

    Returns:
        io.BytesIO: The content of the file.
    """
    # check file status
    file_status = check_file_path(path)
    if file_status == 1:
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    elif file_status == 2:
        file_content = omni.client.read_file(path)[2]
        return io.BytesIO(memoryview(file_content).tobytes())
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")
