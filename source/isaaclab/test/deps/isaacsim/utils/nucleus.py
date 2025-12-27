# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import carb
import omni.client
from omni.client import Result

logger = logging.getLogger(__name__)

DEFAULT_ASSET_ROOT_PATH_SETTING = "/persistent/isaac/asset_root/default"
"""The setting name for the default path to the Isaac Sim assets on a Nucleus server."""

DEFAULT_ASSET_ROOT_TIMEOUT_SETTING = "/persistent/isaac/asset_root/timeout"
"""The setting name for the default timeout for checking the Isaac Sim assets on a Nucleus server."""

def check_server(server: str, path: str, timeout: float = 10.0) -> bool:
    """Check a specific server for a path.

    Args:
        server: The name of the Nucleus server
        path: The path to search
        timeout: The timeout for the check operation. Defaults to 10 seconds.

    Returns:
        True if the path is found, False otherwise.
    """
    logger.info(f"Checking path: {server}{path}")
    # Increase hang detection timeout
    omni.client.set_hang_detection_time_ms(20000)
    result, _ = omni.client.stat(f"{server}{path}")
    if result == Result.OK:
        logger.info(f"Success: {server}{path}")
        return True
    else:
        logger.info(f"Failure: {server}{path} not accessible")
        return False


def get_assets_root_path(*, skip_check: bool = False) -> str:
    """Tries to find the root path to the Isaac Sim assets on a Nucleus server.

    Args:
        skip_check: If True, skip the checking step to verify that the resolved path exists.
            Defaults to False.

    Returns:
        The URL of the Nucleus server with the root path to the assets folder.

    Raises:
        RuntimeError: If the assets root path is not set or if the root path is not found.
    """
    # get the timeout for the check operation
    timeout = carb.settings.get_settings().get(DEFAULT_ASSET_ROOT_TIMEOUT_SETTING)
    if not isinstance(timeout, (int, float)):
        timeout = 10.0

    # resolve path
    logger.info(f"Checking '{DEFAULT_ASSET_ROOT_PATH_SETTING}' setting...")
    default_asset_root = carb.settings.get_settings().get(DEFAULT_ASSET_ROOT_PATH_SETTING)
    if not default_asset_root:
        raise RuntimeError(f"The setting '{DEFAULT_ASSET_ROOT_PATH_SETTING}' is not set")

    # if skip_check is True, return the default asset root path
    if skip_check:
        return default_asset_root

    # check path
    result = check_server(default_asset_root, "/Isaac", timeout)
    if result:
        result = check_server(default_asset_root, "/NVIDIA", timeout)
        if result:
            logger.info(f"Assets root found at {default_asset_root}")
            return default_asset_root

    raise RuntimeError(f"Could not find assets root folder: {default_asset_root}")
