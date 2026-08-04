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
import json
import logging
import os
import posixpath
import re
import subprocess
import tempfile
from typing import Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_UDIM_RE = re.compile(r"<UDIM>", re.IGNORECASE)
_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
_MDL_RESOURCE_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"|/\*.*?\*/|//[^\r\n]*', re.DOTALL)
_MDL_TEXTURE_RE = re.compile(r"\.(?:bmp|dds|exr|hdr|ies|jpe?g|ktx2?|png|tga|tiff?|tx)(?:[?#].*)?$", re.IGNORECASE)
_MDL_IMPORT_RE = re.compile(r"\bimport\s+([^;]+);")
_MDL_USING_IMPORT_RE = re.compile(r"\busing\s+(.+?)\s+import\s+[^;]+;")
_MDL_RELATIVE_IMPORT_RE = re.compile(
    r"(?P<prefix>(?:\.\.::)+|\.::)(?P<module>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)(?P<wildcard>::\*)?"
)


_KIT_EXPERIENCE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), *([".."] * 4), "apps", "isaaclab.python.kit")
)

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


def _resolve_asset_root() -> str:
    """Resolve the configured Isaac asset root.

    The ``ISAACSIM_ASSET_ROOT`` environment variable follows the public Isaac Sim
    asset-root precedence. The kit file remains the fallback for kitless use.

    Returns:
        Value of ``ISAACSIM_ASSET_ROOT`` without its trailing separator, or the value
        configured in ``isaaclab.python.kit``.
    """
    # the value is used exactly as ``isaacsim.storage.native`` uses it, so both sides resolve
    # the same root; only the trailing separator is dropped, for ``/`` and for the documented
    # Windows ``\``
    asset_root = os.getenv("ISAACSIM_ASSET_ROOT")
    if asset_root:
        return asset_root.rstrip("/\\")
    return _parse_kit_asset_root()


NUCLEUS_ASSET_ROOT_DIR: str = _resolve_asset_root()
"""Path to the root directory on the Nucleus Server."""

NVIDIA_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR: str = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""

NEWTON_ASSET_REPO_URL: str = "https://github.com/newton-physics/newton-assets.git"
"""URL of the Newton asset repository."""

NEWTON_ASSET_DIR: str = os.environ.get("NEWTON_ASSET_DIR", NEWTON_ASSET_REPO_URL)
"""Git repository URL or local checkout directory used for Newton assets."""

GIT_ASSET_CACHE_DIR: str = os.path.join(tempfile.gettempdir(), "asset_cache")
"""Default local directory where git asset repositories are cached."""

_MIRROR_FINGERPRINT_SUFFIX = ".isaaclab-cache.json"
"""Suffix of the sidecar file recording the remote revision a locally cached asset came from."""

_REMOTE_FINGERPRINTS: dict[str, dict | None] = {}
"""Remote metadata per URL, resolved at most once per process."""

_ANNOUNCED_MIRROR_DIRS: set[str] = set()
"""Cache directories already announced, so the banner is logged once per directory."""

_ANNOUNCED_MIRRORS: set[str] = set()
"""URLs already announced, so an asset consulted repeatedly is logged once."""

_GIT_SSH_RE = re.compile(r"^[^@/:]+@[^:]+:.+")


def retrieve_git_asset_path(
    git_path: str, local_path: str, cache_dir: str | None = None, force_update: bool = False
) -> str:
    """Return a local path for an asset stored in a git repository.

    Remote repositories are cached under :data:`GIT_ASSET_CACHE_DIR`. If the requested
    asset is already cached, it is returned without running git.

    Args:
        git_path: Git repository URL, SSH path, or existing local checkout directory.
        local_path: Asset path relative to the git repository, or an absolute path inside it.
        cache_dir: Directory where remote repositories are cached. Defaults to
            :data:`GIT_ASSET_CACHE_DIR`.
        force_update: Whether to run ``git pull --ff-only`` for an existing checkout.

    Returns:
        Local path to the requested asset.

    Raises:
        FileNotFoundError: When :paramref:`git_path` points to a missing local directory, or the asset is missing.
        RuntimeError: When the git repository cannot be cloned or updated.
        ValueError: When :paramref:`local_path` is a URL, resolves outside the git repository, or a cache directory
            cannot be derived from :paramref:`git_path`.
    """
    if _is_git_remote_path(git_path):
        git_asset_dir = _get_git_asset_cache_dir(git_path, cache_dir)
        source_path = _resolve_git_asset_source_path(local_path, git_asset_dir)
        if not force_update and os.path.exists(source_path):
            return source_path

    git_asset_dir = _get_git_asset_dir(git_path, cache_dir, force_update)
    source_path = _resolve_git_asset_source_path(local_path, git_asset_dir)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Unable to find git asset: {source_path}")
    return source_path


def _get_git_asset_dir(git_path: str, cache_dir: str | None = None, force_update: bool = False) -> str:
    """Return a local checkout for a git asset repository.

    Args:
        git_path: Git repository URL, SSH path, or existing local checkout directory.
        cache_dir: Directory where remote repositories are cached.
        force_update: Whether to update an existing checkout.

    Returns:
        Path to a local repository checkout.

    Raises:
        FileNotFoundError: When a local :paramref:`git_path` does not exist.
        RuntimeError: When a remote checkout cannot be prepared.
    """
    if not _is_git_remote_path(git_path):
        git_asset_dir = os.path.abspath(os.path.expanduser(git_path))
        if not os.path.isdir(git_asset_dir):
            raise FileNotFoundError(f"Git asset path does not point to an existing directory: {git_asset_dir}")
        if force_update and os.path.isdir(os.path.join(git_asset_dir, ".git")):
            _run_git_command(["git", "-C", git_asset_dir, "pull", "--ff-only"])
        return git_asset_dir

    git_asset_dir = _get_git_asset_cache_dir(git_path, cache_dir)

    if os.path.isdir(os.path.join(git_asset_dir, ".git")):
        if force_update:
            _run_git_command(["git", "-C", git_asset_dir, "pull", "--ff-only"])
    elif os.path.exists(git_asset_dir):
        raise RuntimeError(f"Git asset cache exists but is not a git repository: {git_asset_dir}")
    else:
        os.makedirs(os.path.dirname(git_asset_dir), exist_ok=True)
        _run_git_command(["git", "clone", "--depth", "1", git_path, git_asset_dir])

    return git_asset_dir


def _get_git_asset_cache_dir(git_path: str, cache_dir: str | None = None) -> str:
    """Return the cache directory for a remote git repository.

    Args:
        git_path: Git repository URL or SSH path.
        cache_dir: Root cache directory. Defaults to :data:`GIT_ASSET_CACHE_DIR`.

    Returns:
        Cache checkout path for :paramref:`git_path`.
    """
    if cache_dir is None:
        cache_dir = GIT_ASSET_CACHE_DIR
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    return os.path.join(cache_dir, _get_git_asset_repo_name(git_path))


def _is_git_remote_path(git_path: str) -> bool:
    """Return whether a git path is remote.

    Args:
        git_path: Git repository path.

    Returns:
        True if :paramref:`git_path` is a URL or SSH git path.
    """
    return bool(urlparse(git_path).scheme) or _GIT_SSH_RE.match(git_path) is not None


def _get_git_asset_repo_name(git_path: str) -> str:
    """Return the cache directory name for a git repository.

    Args:
        git_path: Git repository URL or SSH path.

    Returns:
        Repository name without a trailing ``.git`` suffix.

    Raises:
        ValueError: When a repository name cannot be derived.
    """
    repo_path = urlparse(git_path).path
    if not repo_path and _GIT_SSH_RE.match(git_path):
        repo_path = git_path.rsplit(":", 1)[-1]
    repo_name = os.path.basename(repo_path.rstrip("/"))
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    if not repo_name:
        raise ValueError(f"Unable to determine git asset cache directory from git path: {git_path}")
    return repo_name


def _run_git_command(command: list[str]) -> None:
    """Run a git command.

    Args:
        command: Git command and arguments.

    Raises:
        RuntimeError: When git is missing or the command fails.
    """
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to clone git asset repositories.") from exc
    except subprocess.CalledProcessError as exc:
        command_str = " ".join(command)
        raise RuntimeError(f"Unable to run git asset repository command: {command_str}") from exc


def _resolve_git_asset_source_path(local_path: str, git_asset_dir: str) -> str:
    """Resolve an asset path inside a git checkout.

    Args:
        local_path: Asset path relative to :paramref:`git_asset_dir`, or an absolute path inside it.
        git_asset_dir: Local git repository checkout directory.

    Returns:
        Absolute asset path.

    Raises:
        ValueError: When :paramref:`local_path` is a URL or escapes :paramref:`git_asset_dir`.
    """
    if urlparse(local_path).scheme and not os.path.isabs(local_path):
        raise ValueError(f"Git asset paths must be local paths, got: {local_path}")

    if os.path.isabs(local_path):
        source_path = os.path.abspath(os.path.expanduser(local_path))
    else:
        source_path = os.path.abspath(os.path.join(git_asset_dir, os.path.expanduser(local_path)))

    try:
        if os.path.commonpath([git_asset_dir, source_path]) != git_asset_dir:
            raise ValueError(f"Git asset path resolves outside git repository: {local_path}")
    except ValueError as exc:
        raise ValueError(f"Git asset path resolves outside git repository: {local_path}") from exc
    return source_path


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
    return os.path.join(download_dir, parsed.scheme, netloc, *parsed.path.lstrip("/").split("/"))


def _remote_fingerprint(url: str) -> dict | None:
    """Provider metadata identifying the revision of ``url`` the server currently holds.

    Every reported field is kept, because which ones a provider fills in varies: a Nucleus
    server reporting a content hash and an HTTP host reporting only a size and a modification
    time both yield a usable revision marker.

    The answer is resolved once per URL per process, so the existence check, the freshness
    check and the download path share a single status probe.

    Args:
        url: Remote asset URL.

    Returns:
        The reported metadata, or ``None`` when the server does not report the file. That
        covers both a missing file and an unreachable server, which ``omni.client`` does not
        distinguish here.
    """
    if url not in _REMOTE_FINGERPRINTS:
        import omni.client  # noqa: PLC0415

        result, entry = omni.client.stat(url.replace(os.sep, "/"))
        _REMOTE_FINGERPRINTS[url] = (
            {
                "hash": str(entry.hash or ""),
                "version": str(entry.version or ""),
                "size": int(entry.size or 0),
                "modified_time": str(entry.modified_time or ""),
            }
            if result == omni.client.Result.OK
            else None
        )
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
        with open(mirrored, "wb") as f:
            f.write(data)
    except OSError as exc:
        logger.debug("Unable to cache the asset '%s' locally: %s", url, exc)
        return
    _write_mirror_fingerprint(url, mirrored)


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

            target_path = _mirror_path(cur_url, download_dir)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            is_root_asset = local_root is None
            # an outdated local copy is re-fetched, so a changed remote asset is picked up
            if force_download or not _usable_mirror(cur_url, download_dir):
                result = omni.client.copy(cur_url, target_path, omni.client.CopyBehavior.OVERWRITE)
                if result != omni.client.Result.OK:
                    if force_download or is_root_asset:
                        raise RuntimeError(f"Unable to copy file: '{cur_url}'. Is the Nucleus Server running?")
                    logger.debug("Skipping unavailable dependency: %s", cur_url)
                    continue
                _write_mirror_fingerprint(cur_url, target_path)

            if local_root is None:
                local_root = target_path

            # recurse into dependencies (USD references, payloads, MDL textures, etc.)
            for ref in _find_asset_dependencies(target_path):
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
        # Read the local copy when an earlier run already fetched this revision. Actuator
        # networks and similar payloads are read at every startup, so a remote read re-downloads
        # megabytes that :func:`retrieve_file_path` already cached.
        mirrored = _usable_mirror(path)
        if mirrored:
            with open(mirrored, "rb") as f:
                return io.BytesIO(f.read())

        import omni.client  # noqa: PLC0415

        file_content = omni.client.read_file(path.replace(os.sep, "/"))[2]
        data = memoryview(file_content).tobytes()
        # cache what was just downloaded, so the next run reads it from disk
        _store_mirror(path, data)
        return io.BytesIO(data)
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


def _find_asset_dependencies(local_asset_path: str) -> set[str]:
    """Collect external asset dependencies from a local asset file.

    USD layers are parsed with OpenUSD. MDL files are scanned for quoted texture
    resources and relative module imports because those references are resolved
    later by the MDL compiler and are not reported by USD dependency discovery.
    """
    suffix = os.path.splitext(local_asset_path)[1].lower()

    if suffix == ".mdl":
        try:
            with open(local_asset_path, encoding="utf-8") as f:
                source = f.read()
        except OSError as e:
            logger.warning("Failed to open MDL file: %s (%s)", local_asset_path, e)
            return set()

        return _find_mdl_dependencies(source)

    if suffix not in _USD_EXTENSIONS:
        return set()

    from pxr import Sdf, UsdUtils  # noqa: PLC0415

    try:
        layer = Sdf.Layer.FindOrOpen(local_asset_path)
    except Exception:
        logger.warning("Failed to open USD layer: %s", local_asset_path, exc_info=True)
        return set()

    if layer is None:
        return set()

    refs: set[str] = set()

    def _collect(path: str) -> str:
        """Record an asset path.

        Args:
            path: Asset path from the USD layer.

        Returns:
            The input path unchanged.
        """
        if path:
            refs.add(path)
        return path

    UsdUtils.ModifyAssetPaths(layer, _collect)

    return refs


def _find_mdl_dependencies(source: str) -> set[str]:
    """Collect local asset dependencies from MDL source text."""
    refs = set()

    for match in _MDL_RESOURCE_RE.finditer(source):
        ref = match.group(1)
        if ref and _MDL_TEXTURE_RE.search(ref.strip()):
            refs.add(ref.strip())

    source_code = _MDL_RESOURCE_RE.sub("", source)
    for match in _MDL_USING_IMPORT_RE.finditer(source_code):
        refs.update(_find_mdl_import_dependencies(match.group(1)))
    source_code = _MDL_USING_IMPORT_RE.sub("", source_code)
    for match in _MDL_IMPORT_RE.finditer(source_code):
        refs.update(_find_mdl_import_dependencies(match.group(1)))

    return refs


def _find_mdl_import_dependencies(import_clause: str) -> set[str]:
    """Collect local MDL modules referenced by an import clause."""
    refs = set()

    for match in _MDL_RELATIVE_IMPORT_RE.finditer(import_clause):
        prefix = match.group("prefix")
        package_prefix = [".."] * prefix.count("..::")
        components = [component for component in match.group("module").split("::") if component]
        if not components:
            continue

        if match.group("wildcard") is not None:
            candidate_lengths = (len(components),)
        else:
            # ``import .::A::B;`` can mean module ``A::B`` or symbol ``B`` from module ``A``.
            candidate_lengths = range(1, len(components) + 1)

        for length in candidate_lengths:
            refs.add(posixpath.join(*(package_prefix + components[:length])) + ".mdl")

    return refs


def _resolve_reference_url(base_url: str, ref: str) -> str:
    """Resolve a USD reference against a base URL.

    Args:
        base_url: URL or local path containing the reference.
        ref: Referenced asset path.

    Returns:
        Resolved URL or local path.
    """
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
