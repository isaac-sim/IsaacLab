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

import contextlib
import io
import json
import logging
import ntpath
import os
import posixpath
import re
import tempfile
import uuid
from collections.abc import Iterator
from types import ModuleType
from typing import Literal, NotRequired, TypedDict
from urllib.parse import urlparse

from filelock import FileLock

from ..paths import ISAACLAB_ROOT

logger = logging.getLogger(__name__)

# USDZ packages own their internal dependency layout and must not be rewritten.
_USD_EXTENSIONS = {".usd", ".usda", ".usdc"}


_KIT_EXPERIENCE_PATH = str(ISAACLAB_ROOT / "apps" / "isaaclab.python.kit")

# Isaac Sim resolves ``persistent.isaac.asset_root.default``, so it is read first. The
# legacy ``cloud`` setting is only consulted for experience files that predate it.
_KIT_ASSET_ROOT_SETTINGS = ("default", "cloud")


def _parse_kit_asset_root() -> str:
    """Parse the configured Isaac asset root.

    Returns:
        Value of ``persistent.isaac.asset_root.default``, or of the legacy
        ``persistent.isaac.asset_root.cloud``, from ``isaaclab.python.kit``.
    """
    with open(_KIT_EXPERIENCE_PATH) as f:
        lines = f.readlines()
    for setting in _KIT_ASSET_ROOT_SETTINGS:
        pattern = re.compile(rf'\s*persistent\.isaac\.asset_root\.{setting}\s*=\s*"([^"]*)"')
        for line in reversed(lines):  # read from the last line since it's the last setting defined
            m = pattern.match(line)
            if m:
                return m.group(1)
    return ""


_ASSET_REGION_PROFILE_ENV_VAR = "ISAACSIM_ASSET_REGION_PROFILE"
# Update this value when the China mirror moves to a new Isaac Sim asset release.
_ISAAC_SIM_ASSET_RELEASE = "6.1"
_CHINA_ASSET_ENDPOINT = "simready-cn.s3.oss-cn-shanghai.aliyuncs.com"
_US_ASSET_ROOT = _parse_kit_asset_root()


class _StorageProfile(TypedDict):
    """OmniClient routing and asset-root values for a named asset region profile."""

    asset_root: str
    endpoint: NotRequired[str]
    bucket: NotRequired[str]
    region: NotRequired[str]
    cdn_url: NotRequired[str]
    cdn_for_list: NotRequired[bool]


_STORAGE_PROFILES: dict[str, _StorageProfile] = {
    "us": {
        "asset_root": _US_ASSET_ROOT,
    },
    "china": {
        "endpoint": _CHINA_ASSET_ENDPOINT,
        "bucket": "simready-cn",
        "region": "oss-cn-shanghai",
        "cdn_url": "https://assets.simready.cn/",
        "cdn_for_list": False,
        "asset_root": f"https://{_CHINA_ASSET_ENDPOINT}/Assets/Isaac/{_ISAAC_SIM_ASSET_RELEASE}",
    },
}
_CONFIGURED_STORAGE_PROFILES: set[str] = set()


def _selected_storage_profile() -> tuple[str, _StorageProfile] | None:
    """Return the asset region profile selected by the environment, if it is known."""
    profile_name = os.getenv(_ASSET_REGION_PROFILE_ENV_VAR)
    if not profile_name:
        return None

    profile = _STORAGE_PROFILES.get(profile_name)
    if profile is None:
        logger.warning("Ignoring %s: no asset region profile named '%s'", _ASSET_REGION_PROFILE_ENV_VAR, profile_name)
        return None
    return profile_name, profile


def _configure_storage_profile(omni_client: ModuleType) -> None:
    """Configure the selected profile on an imported ``omni.client`` module."""
    selected_profile = _selected_storage_profile()
    if selected_profile is None:
        return

    profile_name, profile = selected_profile
    if profile_name in _CONFIGURED_STORAGE_PROFILES:
        return

    endpoint = profile.get("endpoint")
    if endpoint:
        result = omni_client.set_s3_configuration(
            url=endpoint,
            bucket=profile.get("bucket"),
            region=profile.get("region"),
            cloudfrontUrl=profile.get("cdn_url"),
            cloudfrontForList=profile.get("cdn_for_list", False),
            writeConfig=False,
        )
        if result != omni_client.Result.OK:
            raise RuntimeError(f"Asset region profile '{profile_name}' failed to configure {endpoint}: {result}")

    _CONFIGURED_STORAGE_PROFILES.add(profile_name)
    logger.info("Applied asset region profile '%s'", profile_name)


def configure_storage_profile() -> None:
    """Configure OmniClient routing for the selected asset region profile.

    The configuration is applied in memory and at most once per profile. Isaac Lab
    launchers and asset helpers call this automatically. Standalone kitless scripts
    should call it before using ``omni.client`` directly.

    Raises:
        RuntimeError: When OmniClient rejects the selected asset region profile.
    """
    selected_profile = _selected_storage_profile()
    if selected_profile is None or not selected_profile[1].get("endpoint"):
        return

    import omni.client  # noqa: PLC0415

    _configure_storage_profile(omni.client)


def configure_asset_region_profile() -> None:
    """Configure the selected asset region profile.

    This name matches the public Asset Region Profile terminology. The existing
    :func:`configure_storage_profile` initializer remains supported.
    """
    configure_storage_profile()


def _get_omni_client() -> ModuleType:
    """Import OmniClient lazily and apply the selected asset region profile."""
    import omni.client  # noqa: PLC0415

    _configure_storage_profile(omni.client)
    return omni.client


def _resolve_asset_root() -> str:
    """Resolve the configured Isaac asset root.

    The ``ISAACSIM_ASSET_ROOT`` environment variable follows the public Isaac Sim
    asset-root precedence. When it is unset, the asset root from the asset region profile
    named by ``ISAACSIM_ASSET_REGION_PROFILE`` is used. The kit file remains the fallback
    for kitless use.

    Returns:
        Value of ``ISAACSIM_ASSET_ROOT`` without its trailing separator, or the value
        selected by ``ISAACSIM_ASSET_REGION_PROFILE``, or the value configured in
        ``isaaclab.python.kit``.
    """
    # the value is used exactly as ``isaacsim.storage.native`` uses it, so both sides resolve
    # the same root; only the trailing separator is dropped, for ``/`` and for the documented
    # Windows ``\``
    asset_root = os.getenv("ISAACSIM_ASSET_ROOT")
    if asset_root:
        return asset_root.rstrip("/\\")

    selected_profile = _selected_storage_profile()
    if selected_profile is not None:
        return selected_profile[1]["asset_root"]

    return _parse_kit_asset_root()


NUCLEUS_ASSET_ROOT_DIR: str = _resolve_asset_root()
"""Path to the root directory on the Nucleus Server."""

NVIDIA_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR: str = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""

NEWTON_ASSET_DIR: str = os.environ.get(
    "NEWTON_ASSET_DIR", "https://raw.githubusercontent.com/newton-physics/newton-assets/main"
)
"""URL or local checkout directory of the Newton asset repository; resolve files with :func:`retrieve_file_path`."""

_MIRROR_FINGERPRINT_SUFFIX = ".isaaclab-cache.json"
"""Suffix of the sidecar file recording the remote revision a locally cached asset came from."""

_REMOTE_FINGERPRINTS: dict[str, dict] = {}
"""Successful remote metadata queries, refreshed by forced retrieval."""

_ANNOUNCED_MIRROR_DIRS: set[str] = set()
"""Cache directories already announced, so the banner is logged once per directory."""

_ANNOUNCED_MIRRORS: set[str] = set()
"""URLs already announced, so an asset consulted repeatedly is logged once."""

_ASSET_SOURCES: dict[str, tuple[str, str]] = {}
"""Original source and download directory of each managed copy."""


def _mirror_path(url: str, download_dir: str) -> str:
    """Local path a remote ``url`` mirrors to under ``download_dir``, or ``""`` if not a URL.

    The scheme and host are part of the mirror layout so that two servers exposing the same
    path (for instance a cloud and an on-prem Nucleus) do not share a cache entry.
    """
    parsed = urlparse(url.replace(os.sep, "/"))
    if not parsed.scheme or not parsed.path:
        return ""
    # ':' (port separator) is not a valid path character on Windows
    netloc = parsed.netloc.replace(":", "_")
    mirrored = os.path.join(download_dir, parsed.scheme, netloc, *parsed.path.lstrip("/").split("/"))
    # a host is what distinguishes a remote URL from a Windows drive letter, which ``urlparse``
    # also reports as a scheme
    if parsed.netloc:
        _ASSET_SOURCES[os.path.abspath(mirrored)] = (url, download_dir)
    return mirrored


def unmirror_file_path(path: str) -> str:
    """Maps a locally cached asset copy back to the URL it was downloaded from.

    :func:`retrieve_file_path` hands callers a local copy of a remote asset, so a stage built
    from one records a path that only resolves on the machine holding the cache. An export of
    that stage can use this to name the source asset instead.

    Only copies this process located are known, so a locally authored path is never mistaken
    for a cached copy. A copy mirrored by an earlier run is still recognised, because retrieval
    walks the whole dependency tree even when every file is already cached.

    Args:
        path: Local filesystem path, typically an asset path read from a USD layer.

    Returns:
        The URL the copy was cached from, or ``""`` when this process did not cache it.
    """
    source, _ = _ASSET_SOURCES.get(os.path.abspath(path), ("", ""))
    return source if _is_remote_path(source) else ""


def _remote_fingerprint(url: str) -> dict | None:
    """Provider metadata identifying the revision of ``url`` the server currently holds.

    Every reported field is kept, because which ones a provider fills in varies: a Nucleus
    server reporting a content hash and an HTTP host reporting only a size and a modification
    time both yield a usable revision marker.

    Successful queries are reused until forced retrieval. Failed queries are not cached,
    so a missing file or an unreachable server can be retried.

    Args:
        url: Remote asset URL.

    Returns:
        The reported metadata, or ``None`` when the server does not report the file. That
        covers both a missing file and an unreachable server, which ``omni.client`` does not
        distinguish here.
    """
    if url not in _REMOTE_FINGERPRINTS:
        omni_client = _get_omni_client()

        result, entry = omni_client.stat(url.replace(os.sep, "/"))
        if result != omni_client.Result.OK:
            return None
        _REMOTE_FINGERPRINTS[url] = {
            "hash": str(entry.hash or ""),
            "version": str(entry.version or ""),
            "size": int(entry.size or 0),
            "modified_time": str(entry.modified_time or ""),
        }
    return _REMOTE_FINGERPRINTS[url]


def _write_mirror_fingerprint(url: str, mirrored: str) -> None:
    """Record the remote revision a freshly cached copy was taken from."""
    fingerprint = _remote_fingerprint(url)
    if fingerprint is None:
        return
    try:
        with open(mirrored + _MIRROR_FINGERPRINT_SUFFIX, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f)
    except OSError as exc:
        # a copy we cannot annotate is simply re-fetched on the next run
        logger.debug("Unable to record the asset cache fingerprint for '%s': %s", url, exc)


def _mirror_is_current(url: str, mirrored: str) -> bool:
    """Whether the locally cached copy of ``url`` still matches what the server holds.

    A copy with no recorded fingerprint counts as outdated, so copies left by earlier Isaac Lab
    versions are re-fetched once and annotated. When the answer cannot be obtained at all --
    an unreachable server, or a provider that reports no metadata -- the copy is used anyway
    and the missing guarantee is logged, so offline runs keep working.
    """
    remote = _remote_fingerprint(url)
    if remote is None:
        logger.warning(
            "Asset server did not respond for '%s'. Using the local copy at '%s', which may be out of date.",
            url,
            mirrored,
        )
        return True
    if not any(remote.values()):
        logger.warning(
            "Asset server reports no revision metadata for '%s'. Using the local copy at '%s' without a"
            " freshness check.",
            url,
            mirrored,
        )
        return True
    try:
        with open(mirrored + _MIRROR_FINGERPRINT_SUFFIX, encoding="utf-8") as f:
            return json.load(f) == remote
    except (OSError, ValueError):
        return False


def _announce_local_asset(url: str, mirrored: str, download_dir: str) -> None:
    """Announce, once per cache directory and once per asset, that a local copy is being used."""
    if download_dir not in _ANNOUNCED_MIRROR_DIRS:
        _ANNOUNCED_MIRROR_DIRS.add(download_dir)
        logger.warning(
            "Serving remote assets from the local cache under '%s'. Each copy is checked against the server"
            " before use; delete the directory to force a full re-download.",
            download_dir,
        )
    if url not in _ANNOUNCED_MIRRORS:
        _ANNOUNCED_MIRRORS.add(url)
        logger.info("Loading local copy of remote asset '%s' from '%s'.", url, mirrored)


def _usable_mirror(url: str, download_dir: str | None = None) -> str:
    """Local copy to serve ``url`` from, or ``""`` when it has to come from the server."""
    download_dir = download_dir or tempfile.gettempdir()
    mirrored = _mirror_path(url, download_dir)
    if not mirrored or not os.path.isfile(mirrored) or not _mirror_is_current(url, mirrored):
        return ""
    _announce_local_asset(url, mirrored, download_dir)
    return mirrored


def _store_mirror(url: str, data: bytes) -> None:
    """Cache a payload that was just read from the server, so later runs can reuse it."""
    mirrored = _mirror_path(url, tempfile.gettempdir())
    if not mirrored:
        return
    try:
        os.makedirs(os.path.dirname(mirrored), exist_ok=True)
        with FileLock(mirrored + ".lock"):
            temporary_path = f"{mirrored}.{uuid.uuid4().hex}.partial"
            try:
                with open(temporary_path, "wb") as f:
                    f.write(data)
                os.replace(temporary_path, mirrored)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(temporary_path)
            _write_mirror_fingerprint(url, mirrored)
    except OSError as exc:
        logger.debug("Unable to cache the asset '%s' locally: %s", url, exc)


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

    # a locally cached copy that still matches the server answers this without a download
    if _usable_mirror(path):
        return 2

    return 2 if _remote_fingerprint(path) is not None else 0


def retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = False) -> str:
    """Retrieves the path to a file on the Nucleus Server or locally.

    USD layers are localized in the bound resolver context without modifying authored files or
    raw downloads. Each retrieval resolves dependencies afresh while reusing downloaded files.
    Unresolved references retain their anchors; USD decides which the selected composition requires.
    MDL modules, UDIM textures, and USDZ contents keep their native loader's dependency handling.
    Localization does not validate composition or rendering readiness.

    Args:
        path: The path to the file.
        download_dir: Download directory. Defaults to the system's temporary directory, or the original
            cache directory when reusing a managed copy.
        force_download: Whether to force download the file from the Nucleus Server. This will overwrite
            the local file if it exists. Defaults to False.

    Returns:
        The path to the file on the local machine.

    Raises:
        FileNotFoundError: When the requested file cannot be resolved or found.
        RuntimeError: When a download fails or a resolved USD copy cannot be saved.
    """
    root, cached_dir = _ASSET_SOURCES.get(os.path.abspath(path), (path, ""))
    download_dir = os.path.abspath(download_dir or cached_dir or tempfile.gettempdir())
    if os.path.splitext(root)[1].lower() in _USD_EXTENSIONS:
        from pxr import Ar, Sdf, UsdShade, UsdUtils  # noqa: PLC0415

        resolver = Ar.GetResolver()
        if not _is_remote_path(root):
            resolved = resolver.Resolve(root)
            if not resolved:
                raise FileNotFoundError(f"Unable to resolve the file: {root}")
            root = str(resolved)
    root = root.replace(os.sep, "/") if _is_remote_path(root) else os.path.abspath(root)
    if force_download:
        _REMOTE_FINGERPRINTS.pop(root, None)
    localized = {}
    copy_dir = os.path.join(download_dir, f"isaaclab_usd_{uuid.uuid4().hex}")

    def localize(source: str) -> str:
        if source in localized:
            return localized[source]
        remote = _is_remote_path(source)
        # Read detached contents while the download lock protects the raw mirror.
        with _download_file(source, download_dir, force_download) as local_path:
            suffix = os.path.splitext(local_path)[1].lower()
            if suffix in _USD_EXTENSIONS:
                layer = Sdf.Layer.OpenAsAnonymous(local_path)

        localized[source] = local_path
        if suffix not in _USD_EXTENSIONS:
            return local_path

        # Reserve before descending so shared children and cycles have one destination.
        output = os.path.join(copy_dir, str(len(localized)), os.path.basename(local_path))
        localized[source] = output
        anchor = Ar.ResolvedPath(local_path)
        changed = False

        def rewrite(ref: str) -> str:
            nonlocal changed
            if not ref:
                return ref
            dependency = _resolve_reference_url(source, ref)
            identifier = ref if _is_remote_path(ref) else resolver.CreateIdentifier(ref, anchor)
            if UsdShade.UdimUtils.IsUdimIdentifier(ref):
                # Keep the pattern on its source, not on a partially populated mirror.
                resolved = dependency
                if not _is_remote_path(dependency):
                    resolved = UsdShade.UdimUtils.ResolveUdimPath(ref, Sdf.Layer.FindOrOpen(local_path)) or dependency
            else:
                if not _is_remote_path(dependency):
                    dependency = str(resolver.Resolve(identifier)) or identifier
                if force_download and dependency not in localized:
                    _REMOTE_FINGERPRINTS.pop(dependency, None)
                exists = check_file_path(dependency)
                if ref.endswith(".mdl"):
                    # Keep native module names; anchor explicit paths to the source, not the download mirror.
                    resolved = dependency if exists or identifier != ref else identifier
                elif exists:
                    resolved = localize(dependency)
                else:
                    resolved = dependency
            changed |= resolved != (_resolve_reference_url(local_path, ref) if remote else dependency)
            return resolved

        UsdUtils.ModifyAssetPaths(layer, rewrite)
        if changed:
            os.makedirs(os.path.dirname(output))
            if not layer.Export(output):
                raise RuntimeError(f"Unable to save resolved USD layer: {output}")
            _ASSET_SOURCES[output] = (source, download_dir)
        else:
            localized[source] = local_path
        return localized[source]

    from ..app.loading_screen import report_activity

    report_activity("Loading assets")
    try:
        return localize(root)
    finally:
        report_activity(None)


@contextlib.contextmanager
def _download_file(source: str, download_dir: str, force_download: bool) -> Iterator[str]:
    """Yield a local file while holding its remote mirror's download lock."""
    if not _is_remote_path(source):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Unable to find the file: {source}")
        yield source
        return
    omni_client = _get_omni_client()
    target = _mirror_path(source, download_dir)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with FileLock(target + ".lock"):
        if force_download or not _usable_mirror(source, download_dir):
            temporary_path = f"{target}.{uuid.uuid4().hex}.partial"
            try:
                result = omni_client.copy(source, temporary_path, omni_client.CopyBehavior.OVERWRITE)
                if result != omni_client.Result.OK:
                    if check_file_path(source) == 0:
                        raise FileNotFoundError(f"Unable to find the file: {source}")
                    raise RuntimeError(f"Unable to copy file: '{source}' ({result})")
                os.replace(temporary_path, target)
                _write_mirror_fingerprint(source, target)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(temporary_path)
        yield target


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
        # Read the local copy when an earlier run already fetched this revision. Actuator
        # networks and similar payloads are read at every startup, so a remote read re-downloads
        # megabytes that :func:`retrieve_file_path` already cached.
        mirrored = _usable_mirror(path)
        if mirrored:
            with open(mirrored, "rb") as f:
                return io.BytesIO(f.read())

        omni_client = _get_omni_client()

        file_content = omni_client.read_file(path.replace(os.sep, "/"))[2]
        data = memoryview(file_content).tobytes()
        # cache what was just downloaded, so the next run reads it from disk
        _store_mirror(path, data)
        return io.BytesIO(data)
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


def _resolve_reference_url(base_url: str, ref: str) -> str:
    """Anchor a reference to its original URL or filesystem layer."""
    ref = ref.strip()
    if not ref:
        return ref

    parsed_ref = urlparse(ref)
    if parsed_ref.scheme:
        return ref

    base = urlparse(base_url)
    if not _is_remote_path(base_url):
        path_module = ntpath if ntpath.splitdrive(base_url)[0] else os.path
        base_dir = path_module.dirname(base_url)
        return path_module.normpath(path_module.join(base_dir, ref))

    base_dir = posixpath.dirname(base.path)
    if ref.startswith("/"):
        new_path = posixpath.normpath(ref)
    else:
        new_path = posixpath.normpath(posixpath.join(base_dir, ref))
    return f"{base.scheme}://{base.netloc}{new_path}"


def _is_remote_path(path: str) -> bool:
    """Return whether a path has a URL scheme rather than a Windows drive letter."""
    return len(urlparse(path).scheme) > 1 and not os.path.isabs(path)
