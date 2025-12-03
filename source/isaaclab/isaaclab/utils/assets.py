# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module that defines the host-server where assets and resources are stored.

By default, we use S3 or other cloud storage for hosting assets and resources. This makes
distribution of the assets easier and makes the repository smaller in size code-wise.
"""

import asyncio
import io
import logging
import os
import time
import tempfile
from typing import Literal
from urllib.parse import urlparse
from . import client

logger = logging.getLogger(__name__)

NUCLEUS_ASSET_ROOT_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
"""Path to the root directory on the cloud storage."""

NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""

USD_EXTENSIONS = {".usd", ".usda", ".usdz"}


def _is_usd_path(path: str) -> bool:
    ext = os.path.splitext(urlparse(path).path)[1].lower()
    return ext in client.USD_EXTENSIONS


def check_file_path(path: str) -> Literal[0, 1, 2]:
    """Checks if a file exists on cloud storage or locally.

    Args:
        path: The path to the file.

    Returns:
        The status of the file. Possible values are listed below.

        * :obj:`0` if the file does not exist
        * :obj:`1` if the file exists locally
        * :obj:`2` if the file exists on cloud storage (S3 or HTTP/HTTPS)
    """
    if os.path.isfile(path):
        return 1
    # we need to convert backslash to forward slash on Windows for client API
    elif client.stat(path.replace(os.sep, "/"))[0] == client.Result.OK:
        return 2
    else:
        return 0


def retrieve_file_path(
    path: str,
    download_dir: str | None = None,
    force_download: bool = True,
) -> str:
    """Resolve a path to a local file, downloading from Nucleus/HTTP/S3 if needed.

    Behavior:
        * Local file returns its absolute path.
        * Remote USD pulls the USD and all referenced assets into ``download_dir`` and returns the
          absolute path to the local root USD.
        * Other remote files are copied once into ``download_dir`` and that local path is returned.

    Args:
        path: Local path or remote URL.
        download_dir: Directory to place downloads. Defaults to ``tempfile.gettempdir()``.
        force_download: If True, re-download even if the target already exists.

    Raises:
        FileNotFoundError: If the path is neither local nor reachable remotely.

    Returns:
        Absolute path to the resolved local file.
    """
    status = check_file_path(path)

    # Local file
    if status == 1:
        return os.path.abspath(path)

    # Remote file
    if status == 2:
        if download_dir is None:
            download_dir = tempfile.gettempdir()
        download_dir = os.path.abspath(download_dir)
        os.makedirs(download_dir, exist_ok=True)

        url = path.replace(os.sep, "/")

        # USD → USD + dependencies
        if _is_usd_path(url):
            mapping = client.download_usd_with_references_sync(
                root_url=url,
                download_root=download_dir,
                force_overwrite=force_download,
                progress_callback=lambda done, total, src: logger.debug(
                    "  [%s] %d / %s bytes", src, done, "?" if total is None else str(total)
                ),
            )
            local_root = mapping.get(client._normalize_url(url))
            if local_root is None:
                key = urlparse(url).path.lstrip("/")
                local_root = os.path.join(download_dir, key)
            return os.path.abspath(local_root)

        # Non-USD → single file download
        file_name = os.path.basename(client.break_url(url).path)
        target_path = os.path.join(download_dir, file_name)

        if not os.path.isfile(target_path) or force_download:
            result = client.copy(url, target_path, client.CopyBehavior.OVERWRITE)
            if result != client.Result.OK and force_download:
                raise RuntimeError(f"Unable to copy file: '{path}' from cloud storage.")
        return os.path.abspath(target_path)

    # Not found anywhere
    raise FileNotFoundError(f"Unable to find the file: {path}")


def read_file(path: str) -> io.BytesIO:
    """Reads a file from cloud storage or locally.

    Args:
        path: The path to the file.

    Raises:
        FileNotFoundError: When the file not found locally or on cloud storage.

    Returns:
        The content of the file.
    """
    # check file status
    file_status = check_file_path(path)
    if file_status == 1:
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    elif file_status == 2:
        file_content = client.read_file(path.replace(os.sep, "/"))[2]
        return io.BytesIO(memoryview(file_content).tobytes())
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


"""
Nucleus Connection.
"""


def check_usd_path_with_timeout(usd_path: str, timeout: float = 300, log_interval: float = 30) -> bool:
    """Checks whether the given USD file path is available on the NVIDIA Nucleus server.

    This function synchronously runs an asynchronous USD path availability check,
    logging progress periodically until it completes. The file is available on the server
    if the HTTP status code is 200. Otherwise, the file is not available on the server.

    This is useful for checking server responsiveness before attempting to load a remote
    asset. It will block execution until the check completes or times out.

    Args:
        usd_path: The remote USD file path to check.
        timeout: Maximum time (in seconds) to wait for the server check.
        log_interval: Interval (in seconds) at which progress is logged.

    Returns:
        Whether the given USD path is available on the server.
    """
    start_time = time.time()
    loop = asyncio.get_event_loop()

    coroutine = _is_usd_path_available(usd_path, timeout)
    task = asyncio.ensure_future(coroutine)

    next_log_time = start_time + log_interval

    first_log = True
    while not task.done():
        now = time.time()
        if now >= next_log_time:
            elapsed = int(now - start_time)
            if first_log:
                logger.warning(f"Checking server availability for USD path: {usd_path} (timeout: {timeout}s)")
                first_log = False
            logger.warning(f"Waiting for server response... ({elapsed}s elapsed)")
            next_log_time += log_interval
        loop.run_until_complete(asyncio.sleep(0.1))  # Yield to allow async work

    return task.result()


"""
Helper functions.
"""


async def _is_usd_path_available(usd_path: str, timeout: float) -> bool:
    """Checks whether the given USD path is available on the Omniverse Nucleus server.

    This function is a asynchronous routine to check the availability of the given USD path on the Omniverse Nucleus server.
    It will return True if the USD path is available on the server, False otherwise.

    Args:
        usd_path: The remote or local USD file path to check.
        timeout: Timeout in seconds for the async stat call.

    Returns:
        Whether the given USD path is available on the server.
    """
    try:
        result, _ = await asyncio.wait_for(client.stat_async(usd_path), timeout=timeout)
        return result == client.Result.OK
    except asyncio.TimeoutError:
        logger.warning(f"Timed out after {timeout}s while checking for USD: {usd_path}")
        return False
    except Exception as ex:
        logger.warning(f"Exception during USD file check: {type(ex).__name__}: {ex}")
        return False
