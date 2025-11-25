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
DEFAULT_ASSET_ROOT_TIMEOUT_SETTING = "/persistent/isaac/asset_root/timeout"


def check_server(server: str, path: str, timeout: float = 10.0) -> bool:
    """Check a specific server for a path

    Args:
        server (str): Name of Nucleus server
        path (str): Path to search
        timeout (float): Default value: 10 seconds

    Returns:
        bool: True if folder is found
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
    """Tries to find the root path to the Isaac Sim assets on a Nucleus server

    Args:
        skip_check (bool): If True, skip the checking step to verify that the resolved path exists.

    Raises:
        RuntimeError: if the root path setting is not set.
        RuntimeError: if the root path is not found.

    Returns:
        url (str): URL of Nucleus server with root path to assets folder.
    """

    # get timeout
    timeout = carb.settings.get_settings().get(DEFAULT_ASSET_ROOT_TIMEOUT_SETTING)
    if not isinstance(timeout, (int, float)):
        timeout = 10.0

    # resolve path
    logger.info(f"Check {DEFAULT_ASSET_ROOT_PATH_SETTING} setting")
    default_asset_root = carb.settings.get_settings().get(DEFAULT_ASSET_ROOT_PATH_SETTING)
    if not default_asset_root:
        raise RuntimeError(f"The '{DEFAULT_ASSET_ROOT_PATH_SETTING}' setting is not set")
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
