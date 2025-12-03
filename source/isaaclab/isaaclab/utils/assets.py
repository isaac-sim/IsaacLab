# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module that defines the host-server where assets and resources are stored.

By default, we use S3 or other cloud storage for hosting assets and resources. This makes
distribution of the assets easier and makes the repository smaller in size code-wise.
"""

import io
import os
import tempfile
from typing import Literal
from urllib.parse import urlparse

import boto3
import requests

NUCLEUS_ASSET_ROOT_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
"""Path to the root directory on the cloud storage."""

NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""


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

    # Check if it's a remote path (S3 or HTTP/HTTPS)
    parsed = urlparse(path)

    if parsed.scheme == "s3":
        # Check if file exists in S3
        try:
            s3_client = boto3.client("s3")
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            s3_client.head_object(Bucket=bucket, Key=key)
            return 2
        except Exception:
            return 0
    elif parsed.scheme in ["http", "https"]:
        # Check if file exists via HTTP/HTTPS
        try:
            response = requests.head(path, timeout=10)
            if response.status_code == 200:
                return 2
            else:
                return 0
        except Exception:
            return 0
    else:
        return 0


def retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = True) -> str:
    """Retrieves the path to a file from cloud storage or locally.

    If the file exists locally, then the absolute path to the file is returned.
    If the file exists on cloud storage (S3 or HTTP/HTTPS), then the file is downloaded to the local machine
    and the absolute path to the file is returned.

    Args:
        path: The path to the file.
        download_dir: The directory where the file should be downloaded. Defaults to None, in which
            case the file is downloaded to the system's temporary directory.
        force_download: Whether to force download the file from cloud storage. This will overwrite
            the local file if it exists. Defaults to True.

    Returns:
        The path to the file on the local machine.

    Raises:
        FileNotFoundError: When the file not found locally or on cloud storage.
        RuntimeError: When the file cannot be downloaded from cloud storage to the local machine.
    """
    # check file status
    file_status = check_file_path(path)
    if file_status == 1:
        return os.path.abspath(path)
    elif file_status == 2:
        # resolve download directory
        if download_dir is None:
            download_dir = tempfile.gettempdir()
        else:
            download_dir = os.path.abspath(download_dir)
        # create download directory if it does not exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # parse URL to get file name
        parsed = urlparse(path)
        file_name = os.path.basename(parsed.path)
        target_path = os.path.join(download_dir, file_name)

        # check if file already exists locally
        if not os.path.isfile(target_path) or force_download:
            # download file based on scheme
            try:
                if parsed.scheme == "s3":
                    # Download from S3
                    s3_client = boto3.client("s3")
                    bucket = parsed.netloc
                    key = parsed.path.lstrip("/")
                    s3_client.download_file(bucket, key, target_path)
                elif parsed.scheme in ["http", "https"]:
                    # Download via HTTP/HTTPS
                    response = requests.get(path, timeout=30)
                    response.raise_for_status()
                    with open(target_path, "wb") as f:
                        f.write(response.content)
                else:
                    raise RuntimeError(f"Unsupported URL scheme: {parsed.scheme}")
            except Exception as e:
                if force_download:
                    raise RuntimeError(f"Unable to download file: '{path}'. Error: {str(e)}")
        return os.path.abspath(target_path)
    else:
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
        # Read from remote storage
        parsed = urlparse(path)

        try:
            if parsed.scheme == "s3":
                # Read from S3
                s3_client = boto3.client("s3")
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                response = s3_client.get_object(Bucket=bucket, Key=key)
                file_content = response["Body"].read()
                return io.BytesIO(file_content)
            elif parsed.scheme in ["http", "https"]:
                # Read via HTTP/HTTPS
                response = requests.get(path, timeout=30)
                response.raise_for_status()
                return io.BytesIO(response.content)
            else:
                raise FileNotFoundError(f"Unsupported URL scheme: {parsed.scheme}")
        except Exception as e:
            raise FileNotFoundError(f"Unable to read file: '{path}'. Error: {str(e)}")
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")
