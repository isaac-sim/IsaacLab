# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module that defines the host-server where assets and resources are stored.

By default, we use the Isaac Sim Nucleus Server for hosting assets and resources. This makes
distribution of the assets easier and makes the repository smaller in size code-wise.

For more information, please check information on `Omniverse Nucleus`_.

.. _Omniverse Nucleus: https://docs.omniverse.nvidia.com/nucleus/latest/overview/overview.html
"""

import io
import logging
import os
import posixpath
import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import omni.client

logger = logging.getLogger(__name__)
from pxr import Sdf

NUCLEUS_ASSET_ROOT_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
"""Path to the root directory on the cloud storage."""

NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""

USD_EXTENSIONS = {".usd", ".usda", ".usdz"}


def check_file_path(path: str) -> Literal[0, 1, 2]:
    """Checks if a file exists on the Nucleus Server or locally.

    Args:
        path: The path to the file.

    Returns:
        The status of the file. Possible values are listed below.

        * :obj:`0` if the file does not exist
        * :obj:`1` if the file exists locally
        * :obj:`2` if the file exists on the Nucleus Server
    """
    if os.path.isfile(path):
        return 1
    # we need to convert backslash to forward slash on Windows for omni.client API
    elif omni.client.stat(path.replace(os.sep, "/"))[0] == omni.client.Result.OK:
        return 2
    else:
        return 0


def retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = True) -> str:
    """Retrieves the path to a file on the Nucleus Server or locally.

    If the file exists locally, then the absolute path to the file is returned.
    If the file exists on the Nucleus Server, then the file is downloaded to the local machine
    and the absolute path to the file is returned.

    Args:
        path: The path to the file.
        download_dir: The directory where the file should be downloaded. Defaults to None, in which
            case the file is downloaded to the system's temporary directory.
        force_download: Whether to force download the file from the Nucleus Server. This will overwrite
            the local file if it exists. Defaults to True.

    Returns:
        The path to the file on the local machine.

    Raises:
        FileNotFoundError: When the file not found locally or on Nucleus Server.
        RuntimeError: When the file cannot be copied from the Nucleus Server to the local machine. This
            can happen when the file already exists locally and :attr:`force_download` is set to False.
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
        # recursive download: mirror remote tree under download_dir
        remote_url = path.replace(os.sep, "/")
        to_visit = [remote_url]
        visited = set()
        local_root = None

        while to_visit:
            cur_url = to_visit.pop()
            if cur_url in visited:
                continue
            visited.add(cur_url)

            cur_rel = urlparse(cur_url).path.lstrip("/")
            target_path = os.path.join(download_dir, cur_rel)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if not os.path.isfile(target_path) or force_download:
                result = omni.client.copy(
                    cur_url.replace("collisions_convex_hull", "collisions"),
                    target_path,
                    omni.client.CopyBehavior.OVERWRITE,
                )  # HACK for kuka allegro....... definitely remove before merge
                if result != omni.client.Result.OK and force_download:
                    raise RuntimeError(f"Unable to copy file: '{cur_url}'. Is the Nucleus Server running?")

            if local_root is None:
                local_root = target_path

            # recurse into USD dependencies and referenced assets
            if Path(target_path).suffix.lower() in USD_EXTENSIONS:
                for ref in _find_usd_references(target_path):
                    ref_url = _resolve_reference_url(cur_url, ref)
                    if ref_url and ref_url not in visited:
                        to_visit.append(ref_url)

        return os.path.abspath(local_root)
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


def read_file(path: str) -> io.BytesIO:
    """Reads a file from the Nucleus Server or locally.

    Args:
        path: The path to the file.

    Raises:
        FileNotFoundError: When the file not found locally or on Nucleus Server.

    Returns:
        The content of the file.
    """
    # check file status
    file_status = check_file_path(path)
    if file_status == 1:
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    elif file_status == 2:
        file_content = omni.client.read_file(path.replace(os.sep, "/"))[2]
        return io.BytesIO(memoryview(file_content).tobytes())
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


def _is_downloadable_asset(path: str) -> bool:
    """Return True for USD or other asset types we mirror locally (textures, etc.)."""
    clean = path.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(clean).suffix.lower()

    if suffix == ".mdl":
        # MDL modules (OmniPBR.mdl, OmniSurface.mdl, ...) come from MDL search paths
        return False
    if not suffix:
        return False
    if suffix not in {".usd", ".usda", ".usdz", ".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff"}:
        return False
    return True


def _find_usd_references(local_usd_path: str) -> set[str]:
    """Use Sdf API to collect referenced assets from a USD layer."""
    try:
        layer = Sdf.Layer.FindOrOpen(local_usd_path)
    except Exception:
        logger.warning("Failed to open USD layer: %s", local_usd_path, exc_info=True)
        return set()

    if layer is None:
        return set()

    refs: set[str] = set()

    # Sublayers
    for sub_path in getattr(layer, "subLayerPaths", []) or []:
        if sub_path and _is_downloadable_asset(sub_path):
            refs.add(str(sub_path))

    def _walk_prim(prim_spec: Sdf.PrimSpec) -> None:
        # References
        ref_list = prim_spec.referenceList
        for field in ("addedItems", "prependedItems", "appendedItems", "explicitItems"):
            items = getattr(ref_list, field, None)
            if not items:
                continue
            for ref in items:
                asset_path = getattr(ref, "assetPath", None)
                if asset_path and _is_downloadable_asset(asset_path):
                    refs.add(str(asset_path))

        # Payloads
        payload_list = prim_spec.payloadList
        for field in ("addedItems", "prependedItems", "appendedItems", "explicitItems"):
            items = getattr(payload_list, field, None)
            if not items:
                continue
            for payload in items:
                asset_path = getattr(payload, "assetPath", None)
                if asset_path and _is_downloadable_asset(asset_path):
                    refs.add(str(asset_path))

        # AssetPath-valued attributes (this is where OmniPBR.mdl, textures, etc. show up)
        for attr_spec in prim_spec.attributes.values():
            default = attr_spec.default
            if isinstance(default, Sdf.AssetPath):
                if default.path and _is_downloadable_asset(default.path):
                    refs.add(default.path)
            elif isinstance(default, Sdf.AssetPathArray):
                for ap in default:
                    if ap.path and _is_downloadable_asset(ap.path):
                        refs.add(ap.path)

        # Variants - each variant set can have multiple variants with their own prim content
        for variant_set_spec in prim_spec.variantSets.values():
            for variant_spec in variant_set_spec.variants.values():
                variant_prim_spec = variant_spec.primSpec
                if variant_prim_spec is not None:
                    _walk_prim(variant_prim_spec)

        for child in prim_spec.nameChildren.values():
            _walk_prim(child)

    for root_prim in layer.rootPrims.values():
        _walk_prim(root_prim)

    return refs


def _resolve_reference_url(base_url: str, ref: str) -> str:
    """Resolve a USD asset reference against a base URL (http/local)."""
    ref = ref.strip()
    if not ref:
        return ref

    parsed_ref = urlparse(ref)
    if parsed_ref.scheme:
        return ref

    base = urlparse(base_url)
    if base.scheme == "":
        base_dir = os.path.dirname(base_url)
        return os.path.normpath(os.path.join(base_dir, ref))

    base_dir = posixpath.dirname(base.path)
    if ref.startswith("/"):
        new_path = posixpath.normpath(ref)
    else:
        new_path = posixpath.normpath(posixpath.join(base_dir, ref))
    return f"{base.scheme}://{base.netloc}{new_path}"
