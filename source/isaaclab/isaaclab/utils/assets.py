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
import re
import tempfile
from typing import Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_UDIM_RE = re.compile(r"<UDIM>", re.IGNORECASE)


def _parse_kit_asset_root() -> str:
    """Parse ``persistent.isaac.asset_root.cloud`` from ``apps/isaaclab.python.kit``."""
    _ISAACLAB_ROOT = os.path.join(os.path.dirname(__file__), *([".."] * 4))
    kit_path = os.path.normpath(os.path.join(_ISAACLAB_ROOT, "apps", "isaaclab.python.kit"))
    with open(kit_path) as f:
        for line in reversed(f.readlines()):  # read from the last line since it's the last setting defined
            m = re.match(r'\s*persistent\.isaac\.asset_root\.cloud\s*=\s*"([^"]*)"', line)
            if m:
                return m.group(1)
    return ""


NUCLEUS_ASSET_ROOT_DIR: str = _parse_kit_asset_root()
"""Path to the root directory on the Nucleus Server."""

NVIDIA_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR: str = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""


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

    import omni.client  # noqa: PLC0415

    if omni.client.stat(path.replace(os.sep, "/"))[0] == omni.client.Result.OK:
        return 2
    else:
        return 0


def retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = False) -> str:
    """Retrieves the path to a file on the Nucleus Server or locally.

    If the file exists locally, then the absolute path to the file is returned.
    If the file exists on the Nucleus Server, then the file is downloaded to the local machine
    and the absolute path to the file is returned.

    Args:
        path: The path to the file.
        download_dir: The directory where the file should be downloaded. Defaults to None, in which
            case the file is downloaded to the system's temporary directory.
        force_download: Whether to force download the file from the Nucleus Server. This will overwrite
            the local file if it exists. Defaults to False.

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
        import omni.client  # noqa: PLC0415

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

            # UDIM textures use a <UDIM> placeholder (e.g. texture.<UDIM>.png) that does not
            # correspond to a real file. Expand to individual tile URLs by probing tile numbers
            # starting at 1001; UDIM tiles are contiguous so stop at the first missing tile.
            if _UDIM_RE.search(cur_url):
                for tile in range(1001, 1101):
                    tile_url = _UDIM_RE.sub(str(tile), cur_url)
                    if omni.client.stat(tile_url.replace(os.sep, "/"))[0] == omni.client.Result.OK:
                        if tile_url not in visited:
                            to_visit.append(tile_url)
                    else:
                        break
                continue

            cur_rel = urlparse(cur_url).path.lstrip("/")
            target_path = os.path.join(download_dir, cur_rel)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if not os.path.isfile(target_path) or force_download:
                result = omni.client.copy(cur_url, target_path, omni.client.CopyBehavior.OVERWRITE)
                if result != omni.client.Result.OK and force_download:
                    raise RuntimeError(f"Unable to copy file: '{cur_url}'. Is the Nucleus Server running?")

            if local_root is None:
                local_root = target_path

            # recurse into USD dependencies (sublayers, references, payloads, textures, etc.)
            suffix = os.path.splitext(target_path)[1].lower()
            if suffix in {".usd", ".usda", ".usdc", ".usdz"}:
                for ref in _find_usd_dependencies(target_path):
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
        import omni.client  # noqa: PLC0415

        file_content = omni.client.read_file(path.replace(os.sep, "/"))[2]
        return io.BytesIO(memoryview(file_content).tobytes())
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


def _find_usd_dependencies(local_usd_path: str) -> set[str]:
    """Use UsdUtils to collect all asset dependencies from a USD file.

    This uses :func:`UsdUtils.ComputeAllDependencies` — the same approach as
    ``isaacsim.storage.native`` — to discover sublayers, references, payloads,
    and non-layer assets (textures, etc.) without maintaining a hardcoded list
    of file extensions.

    Args:
        local_usd_path: Path to a local USD file.

    Returns:
        Set of asset path strings as they appear in the USD layer (unresolved).
    """
    from pxr import Sdf, UsdUtils  # noqa: PLC0415

    try:
        layer = Sdf.Layer.FindOrOpen(local_usd_path)
    except Exception:
        logger.warning("Failed to open USD layer: %s", local_usd_path, exc_info=True)
        return set()

    if layer is None:
        return set()

    # Collect every asset path referenced from this layer.
    # UsdUtils.ModifyAssetPaths walks sublayers, references, payloads,
    # variant selections, and attribute values — exactly the set we need.
    refs: set[str] = set()

    def _collect(path: str) -> str:
        if path:
            refs.add(path)
        return path  # return unchanged — we are only reading, not modifying

    UsdUtils.ModifyAssetPaths(layer, _collect)

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
