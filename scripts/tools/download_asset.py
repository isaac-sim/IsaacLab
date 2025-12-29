#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to download assets from Nucleus server.

This script downloads assets from Omniverse Nucleus server, supporting:

- Single files and recursive directory downloads
- Path placeholders for common asset locations

Usage:

.. code-block:: bash
    # Download single file
    ./isaaclab.sh -p scripts/tools/download_asset.py \
        --path "{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd" \
        --output ./assets/ur10e

    # Download directory recursively
    ./isaaclab.sh -p scripts/tools/download_asset.py \
        --path "{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned" \
        --recursive --output ./assets/ycb

Placeholders:
    {ISAACLAB_NUCLEUS_DIR} - Isaac Lab assets
    {ISAAC_NUCLEUS_DIR}    - Isaac Sim assets
    {NVIDIA_NUCLEUS_DIR}   - NVIDIA assets
    {NUCLEUS_ASSET_ROOT_DIR} - Root directory
"""

import argparse

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Download assets from Nucleus server.")
parser.add_argument("--path", type=str, required=True, help="Path to asset on Nucleus server (supports placeholders)")
parser.add_argument("--output", type=str, default=None, help="Local download directory (default: temp directory)")
parser.add_argument("--force", action="store_true", default=True, help="Overwrite existing files (default: True)")
parser.add_argument("--no-force", dest="force", action="store_false", help="Skip existing files")
parser.add_argument("--recursive", action="store_true", help="Download directories recursively")
parser.add_argument("--check-only", action="store_true", help="Check if asset exists without downloading")
parser.add_argument(
    "--log-level",
    type=str,
    default="info",
    choices=["info", "debug", "warning", "error", "critical"],
    help="Logging level (default: info)",
)

args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import logging
import os
import sys
import tempfile
from typing import Literal, Tuple

import omni.client

import isaaclab.utils.assets as assets_utils

# Configure logger
logger = logging.getLogger()
logger.setLevel(getattr(logging, args_cli.log_level.upper()))
# Remove any existing handlers and add our own
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Add console handler with our format
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console_handler)


NUCLEUS_DIR_PLACEHOLDERS = {
    "{ISAACLAB_NUCLEUS_DIR}": assets_utils.ISAACLAB_NUCLEUS_DIR,
    "{ISAAC_NUCLEUS_DIR}": assets_utils.ISAAC_NUCLEUS_DIR,
    "{NVIDIA_NUCLEUS_DIR}": assets_utils.NVIDIA_NUCLEUS_DIR,
    "{NUCLEUS_ASSET_ROOT_DIR}": assets_utils.NUCLEUS_ASSET_ROOT_DIR,
}
"""Dictionary of placeholder variables and their values."""


ERROR_MESSAGES = {
    omni.client.Result.ERROR_ACCESS_DENIED: "Permission denied",
    omni.client.Result.ERROR_ACCESS_LOST: "Connection lost",
    omni.client.Result.ERROR_NOT_FOUND: "File not found",
}
"""Dictionary of error messages for the omni.client.Result enum."""


def check_path_type(path: str) -> Literal["file", "directory", "not_found"]:
    """Check if the input path is a file, directory, or doesn't exist.

    Args:
        path: The path to check.

    Returns:
        Whether the path is a file, directory, or doesn't exist.
    """
    # Check if the path exists
    result, entry = omni.client.stat(path)
    # If the path doesn't exist, return "not_found"
    if result != omni.client.Result.OK:
        return "not_found"
    # If the path is a directory, return "directory"
    # Note: Directories in omni.client have the CAN_HAVE_CHILDREN flag set
    if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
        return "directory"
    else:
        return "file"


def download_single_file(src_path: str, dst_path: str, overwrite: bool = True) -> bool:
    """Download a single file from Nucleus to local filesystem.

    Args:
        src_path: The path to the file to download.
        dst_path: The path to the destination file.
        overwrite: Whether to overwrite existing files.

    Returns:
        Whether the file was downloaded successfully.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # If the file already exists and overwrite is not allowed, skip
    if os.path.exists(dst_path) and not overwrite:
        logging.debug(f"Skipping existing file: {dst_path}")
        return True

    # Download the file
    behavior = omni.client.CopyBehavior.OVERWRITE if overwrite else omni.client.CopyBehavior.ERROR_IF_EXISTS
    result = omni.client.copy(src_path.replace(os.sep, "/"), dst_path, behavior)

    # If the file was downloaded successfully, return True
    if result == omni.client.Result.OK:
        return True
    else:
        error_msg = ERROR_MESSAGES.get(result, "Unknown error")
        logging.error(f"Failed to download: {src_path} (error: {error_msg})")
        return False


async def list_directory(path: str) -> tuple[list[str], list[str]]:
    """List files and subdirectories in a directory.

    Args:
        path: The path to the directory to list.

    Returns:
        A tuple of lists of files and directories.
    """
    # If the path doesn't end with a slash, add one
    if not path.endswith("/"):
        path += "/"

    result, entries = await asyncio.wait_for(omni.client.list_async(path), timeout=30)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"Failed to list directory: {path}")

    files, dirs = [], []
    for entry in entries:
        full_path = omni.client.combine_urls(path, entry.relative_path)
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            dirs.append(full_path)
        else:
            files.append(full_path)

    return files, dirs


async def list_directory_recursive(path: str) -> list[str]:
    """Recursively list all files in a directory.

    Args:
        path: The path to the directory to list.

    Returns:
        A list of all files in the directory.
    """
    files, dirs = await list_directory(path)
    all_files = files.copy()

    # Recursively process subdirectories
    if dirs:
        tasks = [list_directory_recursive(d) for d in dirs]
        results = await asyncio.gather(*tasks)
        for result in results:
            all_files.extend(result)

    return all_files


async def download_directory(
    src_path: str, dst_dir: str, overwrite: bool = True, check_only: bool = False
) -> list[str]:
    """Download all files from a directory recursively.

    Args:
        src_path: The path to the directory to download.
        dst_dir: The path to the destination directory.
        overwrite: Whether to overwrite existing files.
        check_only: Whether to only check if the files exist without downloading them.
            If True, the files will not be downloaded, but the existence of the files will be checked.
            If False, the files will be downloaded.

    Returns:
        A list of the downloaded files.
    """
    # Collect files from the directory recursively
    logging.info(f"Collecting files from: {src_path}")

    # List all files in the directory recursively
    all_files = await list_directory_recursive(src_path)
    # If no files are found, exit
    if not all_files:
        logging.warning("No files to download. Exiting...")
        return []
    else:
        logging.info(f"Found {len(all_files)} files to download.")

    # If check_only is True, return early
    if check_only:
        return []

    # Log the files being downloaded
    logging.info("Starting download...")

    # Determine root for relative path calculation
    root = src_path if src_path.endswith("/") else src_path + "/"
    downloaded = []

    # Download each file
    for i, src_file in enumerate(all_files, 1):
        rel_path = os.path.relpath(src_file, root).replace("\\", "/")
        dst_file = os.path.join(dst_dir, rel_path)

        # Log the file being downloaded
        logging.info(f"[{i}/{len(all_files)}] Downloading: {rel_path}")
        # Download the file
        if download_single_file(src_file, dst_file, overwrite):
            downloaded.append(dst_file)

    return downloaded


def main() -> bool:
    """Main entry point."""

    # Validate Nucleus connection
    if assets_utils.NUCLEUS_ASSET_ROOT_DIR is None:
        logging.error("Cannot connect to Nucleus server. Please check your internet connection and try again.")
        return False

    # Log Nucleus root
    logging.debug(f"Found nucleus root: {assets_utils.NUCLEUS_ASSET_ROOT_DIR}")

    # Resolve nucleus path and output directory
    nucleus_path = args_cli.path
    for placeholder, value in NUCLEUS_DIR_PLACEHOLDERS.items():
        if placeholder in nucleus_path:
            nucleus_path = nucleus_path.replace(placeholder, value)
            logging.debug(f"Resolved {placeholder} -> {value}")

    # Resolve output directory
    if args_cli.output:
        output_dir = os.path.abspath(args_cli.output)
    else:
        output_dir = os.path.join(tempfile.gettempdir(), "isaaclab", "assets")
    os.makedirs(output_dir, exist_ok=True)

    # Log paths
    logging.info(f"Source: {nucleus_path}")
    logging.info(f"Destination: {output_dir}")

    # Check if path exists
    path_type = check_path_type(nucleus_path)
    if path_type == "not_found":
        logging.error(f"Path not found: {nucleus_path}")
        return False
    else:
        logging.info(f"Identified path type: {path_type}")

    try:
        loop = asyncio.get_event_loop()

        if path_type == "directory":
            if not args_cli.recursive:
                logging.error("Path is a directory. Please use '--recursive' flag to download directories recursively.")
                return False

            # Download directory recursively
            downloaded = loop.run_until_complete(
                download_directory(nucleus_path, output_dir, args_cli.force, args_cli.check_only)
            )

            # If check_only is True, return early
            if args_cli.check_only:
                logging.info("Check complete.")
                return True
            elif downloaded:
                logging.info(f"Downloaded {len(downloaded)} files successfully.")
                return True
            else:
                logging.error("Failed to download directory. Please try again.")
                return False

        else:
            # Check if file exists without downloading
            if args_cli.check_only:
                logging.info("Check complete.")
                return True

            # Download single file
            local_file = os.path.join(output_dir, os.path.basename(nucleus_path))
            logging.info(f"Downloading: {os.path.basename(nucleus_path)}")
            if download_single_file(nucleus_path, local_file, args_cli.force):
                logging.info(f"Downloaded to: {local_file}")
                return True
            else:
                logging.error("Failed to download file. Please try again.")
                return False

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return False


if __name__ == "__main__":
    # run the main function
    success = main()
    # close the app
    simulation_app.close()
    # exit with the success code
    exit(0 if success else 1)
