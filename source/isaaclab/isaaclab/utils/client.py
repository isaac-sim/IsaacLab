# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Lightweight omni.client-like helpers for local/HTTP/S3 access.

This module implements a subset of omni.client behaviors used inside Isaac Lab:
path normalization, stat/read helpers, USD reference resolution, and simple
copy/download utilities. It supports local files, HTTP(S), and S3 paths and
provides a small Result enum for consistent status reporting.
"""

import asyncio
import logging
import os
import posixpath
from collections.abc import Callable
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError
from pxr import Sdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Basic types
# ---------------------------------------------------------------------------

OMNI_S3_HOST_PREFIX = "omniverse-content-production.s3-"


class Result(IntEnum):
    OK = 0
    ERROR_NOT_FOUND = 1
    ERROR_PERMISSION_DENIED = 2
    ERROR_NETWORK = 3
    ERROR_UNKNOWN = 4


class CopyBehavior(Enum):
    OVERWRITE = "overwrite"
    SKIP = "skip"


class UrlParts(NamedTuple):
    scheme: str
    authority: str
    path: str


def break_url(url: str) -> UrlParts:
    """Parse a URL into (scheme, authority, path) with empty parts when missing."""
    parsed = urlparse(url)
    return UrlParts(parsed.scheme or "", parsed.netloc or "", parsed.path or "")


_s3 = boto3.client("s3")


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _is_s3_url(path: str) -> bool:
    """Return True if the path uses the ``s3://`` scheme."""
    return path.startswith("s3://")


def _is_http_url(path: str) -> bool:
    """Return True if the path uses HTTP or HTTPS."""
    scheme = urlparse(path).scheme.lower()
    return scheme in ("http", "https")


def _is_local_path(path: str) -> bool:
    """Return True if the path has no URL scheme (treated as local)."""
    # Strong assumption: anything without a scheme is local
    return urlparse(path).scheme == ""


def _split_s3_url(path: str) -> tuple[str, str]:
    """Split an S3 URL into ``(bucket, key)`` or raise on invalid input."""
    parsed = urlparse(path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URL: {path}")
    return bucket, key


def _normalize_url(url: str) -> str:
    """Convert omniverse S3 URLs to HTTPS; leave others as-is."""
    if not _is_s3_url(url):
        return url
    parsed = urlparse(url)
    if parsed.netloc.startswith(OMNI_S3_HOST_PREFIX):
        return f"https://{parsed.netloc}{parsed.path}"
    return url


def _map_remote_to_local(download_root: str, url: str) -> str:
    """Mirror remote path structure under download_root."""
    parsed = urlparse(url)
    key = parsed.path.lstrip("/")
    local_path = os.path.join(download_root, key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path


def _resolve_reference_url(base_url: str, ref: str) -> str:
    """Resolve a USD asset reference against a base URL (http/s3/local)."""
    ref = ref.strip()
    if not ref:
        return ref

    parsed_ref = urlparse(ref)
    if parsed_ref.scheme:
        # Already absolute (http://, https://, s3://, etc.)
        return ref

    base = urlparse(base_url)
    if base.scheme == "":
        # Local base
        base_dir = os.path.dirname(base_url)
        return os.path.normpath(os.path.join(base_dir, ref))

    # Remote base
    base_dir = posixpath.dirname(base.path)
    if ref.startswith("/"):
        new_path = posixpath.normpath(ref)
    else:
        new_path = posixpath.normpath(posixpath.join(base_dir, ref))
    return f"{base.scheme}://{base.netloc}{new_path}"


# ---------------------------------------------------------------------------
# stat / read_file
# ---------------------------------------------------------------------------


def stat(path: str) -> tuple[Result, dict[str, Any] | None]:
    """Check whether a remote or local file exists and return basic metadata.

    Args:
        path: Local path or remote URL (HTTP/S3).

    Returns:
        Tuple of (:class:`Result`, info dict or None). On success, ``info`` may contain
        ``size``, ``etag``, ``last_modified``, and ``content_type`` when available.
        The :class:`Result` code is one of ``OK``, ``ERROR_NOT_FOUND``,
        ``ERROR_PERMISSION_DENIED``, ``ERROR_NETWORK``, or ``ERROR_UNKNOWN``.
    """
    url = _normalize_url(path)

    # HTTP(S)
    if _is_http_url(url):
        try:
            req = Request(url, method="HEAD")
            with urlopen(req) as resp:
                size_header = resp.headers.get("Content-Length")
                info = {
                    "size": int(size_header) if size_header is not None else None,
                    "etag": resp.headers.get("ETag"),
                    "last_modified": resp.headers.get("Last-Modified"),
                    "content_type": resp.headers.get("Content-Type"),
                }
                return Result.OK, info
        except HTTPError as exc:
            if exc.code == 404:
                return Result.ERROR_NOT_FOUND, None
            if exc.code == 403:
                return Result.ERROR_PERMISSION_DENIED, None
            logger.warning("HTTP error in stat(%s): %s", url, exc)
            return Result.ERROR_UNKNOWN, None
        except URLError as exc:
            logger.warning("Network error in stat(%s): %s", url, exc)
            return Result.ERROR_NETWORK, None
        except Exception as exc:
            logger.warning("Unexpected error in stat(%s): %s", url, exc)
            return Result.ERROR_UNKNOWN, None

    # S3 (non-omniverse)
    if _is_s3_url(url):
        bucket, key = _split_s3_url(url)
        try:
            response = _s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return Result.ERROR_NOT_FOUND, None
            if code in ("AccessDenied", "403"):
                return Result.ERROR_PERMISSION_DENIED, None
            logger.warning("Error in stat(%s): %s", url, exc)
            return Result.ERROR_UNKNOWN, None

        info = {
            "size": response.get("ContentLength"),
            "etag": response.get("ETag"),
            "last_modified": response.get("LastModified"),
            "content_type": response.get("ContentType"),
        }
        return Result.OK, info

    # Local
    if _is_local_path(url) and os.path.exists(url):
        try:
            size = os.path.getsize(url)
        except OSError:
            size = None
        info = {
            "size": size,
            "etag": None,
            "last_modified": None,
            "content_type": None,
        }
        return Result.OK, info

    return Result.ERROR_NOT_FOUND, None


async def stat_async(path: str) -> tuple[Result, dict[str, Any] | None]:
    """Async wrapper for :func:`stat`."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, stat, path)


def read_file(path: str) -> tuple[Result, dict[str, Any], memoryview]:
    """Read file content from HTTP(S), S3, or local."""
    url = _normalize_url(path)

    # HTTP(S)
    if _is_http_url(url):
        try:
            with urlopen(url) as resp:
                data_bytes = resp.read()
                meta = {
                    "size": len(data_bytes),
                    "content_type": resp.headers.get("Content-Type"),
                }
                return Result.OK, meta, memoryview(data_bytes)
        except HTTPError as exc:
            if exc.code == 404:
                return Result.ERROR_NOT_FOUND, {}, memoryview(b"")
            if exc.code == 403:
                return Result.ERROR_PERMISSION_DENIED, {}, memoryview(b"")
            logger.warning("HTTP error in read_file(%s): %s", url, exc)
            return Result.ERROR_UNKNOWN, {}, memoryview(b"")
        except URLError as exc:
            logger.warning("Network error in read_file(%s): %s", url, exc)
            return Result.ERROR_NETWORK, {}, memoryview(b"")
        except Exception as exc:
            logger.warning("Unexpected error in read_file(%s): %s", url, exc)
            return Result.ERROR_UNKNOWN, {}, memoryview(b"")

    # S3
    if _is_s3_url(url):
        bucket, key = _split_s3_url(url)
        try:
            obj = _s3.get_object(Bucket=bucket, Key=key)
            data_bytes = obj["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return Result.ERROR_NOT_FOUND, {}, memoryview(b"")
            if code in ("AccessDenied", "403"):
                return Result.ERROR_PERMISSION_DENIED, {}, memoryview(b"")
            logger.warning("Error in read_file(%s): %s", url, exc)
            return Result.ERROR_UNKNOWN, {}, memoryview(b"")

        meta = {
            "size": len(data_bytes),
            "content_type": obj.get("ContentType"),
        }
        return Result.OK, meta, memoryview(data_bytes)

    # Local
    if _is_local_path(url) and os.path.isfile(url):
        with open(url, "rb") as f:
            data_bytes = f.read()
        meta = {"size": len(data_bytes), "content_type": None}
        return Result.OK, meta, memoryview(data_bytes)

    return Result.ERROR_NOT_FOUND, {}, memoryview(b"")


async def read_file_async(path: str) -> tuple[Result, dict[str, Any], memoryview]:
    """Async wrapper for :func:`read_file`."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, read_file, path)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def copy(
    src: str,
    dst: str,
    behavior: CopyBehavior = CopyBehavior.OVERWRITE,
    progress_callback: Callable[[int, int | None, str], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Result:
    """Copy between local and remote (HTTP/S3) locations.

    Supported directions:
        * HTTP/HTTPS → local
        * S3 → local

    Args:
        src: Source path or URL.
        dst: Destination path or URL.
        behavior: Overwrite policy for local targets.
        progress_callback: Optional ``cb(done_bytes, total_bytes_or_None, src)``.
        chunk_size: Chunk size for streamed copies.

    Returns:
        Result enum indicating success or failure reason.
    """
    src = _normalize_url(src)

    if os.path.exists(dst) and behavior == CopyBehavior.SKIP:
        return Result.OK

    # HTTP(S) -> local
    if _is_http_url(src) and _is_local_path(dst):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        try:
            with urlopen(src) as resp:
                size_header = resp.headers.get("Content-Length")
                total_size = int(size_header) if size_header is not None else None

                transferred = 0
                with open(dst, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        transferred += len(chunk)
                        if progress_callback:
                            progress_callback(transferred, total_size, src)
            return Result.OK
        except HTTPError as exc:
            if exc.code == 404:
                return Result.ERROR_NOT_FOUND
            if exc.code == 403:
                return Result.ERROR_PERMISSION_DENIED
            logger.warning("HTTP error copying %s -> %s: %s", src, dst, exc)
            return Result.ERROR_UNKNOWN
        except URLError as exc:
            logger.warning("Network error copying %s -> %s: %s", src, dst, exc)
            return Result.ERROR_NETWORK
        except Exception as exc:
            logger.warning("Unexpected error copying %s -> %s: %s", src, dst, exc)
            return Result.ERROR_UNKNOWN

    # S3 -> local
    if _is_s3_url(src) and _is_local_path(dst):
        bucket, key = _split_s3_url(src)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        try:
            head = _s3.head_object(Bucket=bucket, Key=key)
            total_size = head.get("ContentLength")
            obj = _s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"]

            transferred = 0
            with open(dst, "wb") as f:
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    transferred += len(chunk)
                    if progress_callback:
                        progress_callback(transferred, total_size, src)
            return Result.OK
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return Result.ERROR_NOT_FOUND
            if code in ("AccessDenied", "403"):
                return Result.ERROR_PERMISSION_DENIED
            logger.warning("Error copying S3->local (%s -> %s): %s", src, dst, exc)
            return Result.ERROR_UNKNOWN

    logger.error("Copy combination not supported: %s -> %s", src, dst)
    return Result.ERROR_UNKNOWN


async def copy_async(
    src: str,
    dst: str,
    behavior: CopyBehavior = CopyBehavior.OVERWRITE,
    progress_callback: Callable[[int, int | None, str], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Result:
    """Async wrapper for :func:`copy` with the same arguments."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, copy, src, dst, behavior, progress_callback, chunk_size)


# ---------------------------------------------------------------------------
# USD dependency resolution
# ---------------------------------------------------------------------------

USD_EXTENSIONS = {".usd", ".usda", ".usdc"}


_DOWNLOADABLE_EXTS = {
    ".usd",
    ".usda",
    ".usdz",
    ".png",
    ".jpg",
    ".jpeg",
    ".exr",
    ".hdr",
    ".tif",
    ".tiff",
}


def _is_usd_file(path: str) -> bool:
    """Return True if path ends with a USD extension."""
    return Path(path).suffix.lower() in USD_EXTENSIONS


def _is_downloadable_asset(path: str) -> bool:
    """Return True for USD or other asset types we mirror locally (textures, etc.)."""
    clean = path.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(clean).suffix.lower()

    if suffix == ".mdl":
        # MDL modules (OmniPBR.mdl, OmniSurface.mdl, ...) come from MDL search paths
        return False
    if not suffix:
        return False
    if suffix not in _DOWNLOADABLE_EXTS:
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

        for child in prim_spec.nameChildren.values():
            _walk_prim(child)

    for root_prim in layer.rootPrims.values():
        _walk_prim(root_prim)

    return refs


async def download_usd_with_references(
    root_usd_s3_url: str,
    download_root: str,
    force_overwrite: bool = True,
    progress_callback: Callable[[int, int | None, str], None] | None = None,
) -> dict[str, str]:
    """Download a USD and all referenced assets to a local mirror.

    Traverses the USD dependency graph, downloading each referenced asset (USD, textures, etc.)
    into ``download_root`` while preserving the relative directory structure. Returns a mapping
    from normalized remote URLs to their local file paths.

    Args:
        root_usd_s3_url: Root USD URL (S3/HTTP).
        download_root: Local root directory to mirror into.
        force_overwrite: If True, overwrite existing files; otherwise skip.
        progress_callback: Optional ``cb(done_bytes, total_bytes_or_None, src)``.

    Returns:
        Dict mapping normalized remote URLs to local paths.
    """
    os.makedirs(download_root, exist_ok=True)

    root_url = _normalize_url(root_usd_s3_url)
    to_visit = [root_url]
    visited: set[str] = set()
    mapping: dict[str, str] = {}

    while to_visit:
        current_url = _normalize_url(to_visit.pop())
        if current_url in visited:
            continue
        visited.add(current_url)

        local_path = _map_remote_to_local(download_root, current_url)
        mapping[current_url] = local_path

        behavior = CopyBehavior.OVERWRITE if force_overwrite else CopyBehavior.SKIP
        logger.info("Downloading asset %s -> %s", current_url, local_path)
        res = await copy_async(current_url, local_path, behavior=behavior, progress_callback=progress_callback)
        if res != Result.OK:
            logger.warning("Failed to download %s (Result=%s)", current_url, res)
            continue

        if _is_usd_file(local_path):
            for ref in _find_usd_references(local_path):
                dep_url = _resolve_reference_url(current_url, ref)
                dep_url = _normalize_url(dep_url)
                if dep_url and dep_url not in visited:
                    to_visit.append(dep_url)

    return mapping


def download_usd_with_references_sync(
    root_url: str,
    download_root: str,
    force_overwrite: bool = True,
    progress_callback: Callable[[int, int | None, str], None] | None = None,
) -> dict[str, str]:
    """Synchronous wrapper for :func:`download_usd_with_references`. Safe for IsaacLab scripts

    NOT safe to call from inside a running event loop (e.g. Isaac Sim / Kit).
    In that case, call `await download_usd_with_references(...)` directly.
    """
    # If there's a running loop (Kit / Jupyter / etc.), don't try to block it.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop → safe to own one; asyncio.run handles creation/cleanup.
        return asyncio.run(
            download_usd_with_references(
                root_url,
                download_root,
                force_overwrite=force_overwrite,
                progress_callback=progress_callback,
            )
        )
    else:
        # Already inside an event loop: this wrapper must not be used.
        raise RuntimeError(
            "download_usd_with_references_sync() was called while an event loop is running.\n"
            "Use `await download_usd_with_references(...)` or schedule it with "
            "`asyncio.create_task` instead of calling the sync wrapper."
        )
