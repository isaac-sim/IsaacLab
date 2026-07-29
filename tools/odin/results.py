# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish and fetch benchmark bundles.

OSMO datasets were retired, so this module is the seam that isolates whatever
replaces them. Nothing else in OdinV2 imports a storage concept, so swapping the
implementation here is a one-file change.

The current implementation uses bucket URIs via ``osmo data upload`` and
``osmo data download``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

__all__ = ["fetch_results", "publish_command", "results_uri_for", "validate_bundle"]

# Upstream's SchemaBundleFile writes "<output_prefix>_<timestamp>.json" into
# --output_path, where output_prefix is "benchmark_<workflow>_<task>".
_BUNDLE_GLOB = "benchmark_*.json"


class _DownloaderProto(Protocol):
    def data_download(self, remote_uri: str, dest_dir: Path) -> None: ...


def results_uri_for(base_uri: str, dispatch_id: str, row_key: str) -> str:
    """Return the storage URI for one row's results.

    Args:
        base_uri: Bucket URI prefix from ``odin.yaml``.
        dispatch_id: Dispatch identifier.
        row_key: Row identity.

    Returns:
        ``<base_uri>/<dispatch_id>/<row_key>``, without a doubled separator.
    """
    return f"{base_uri.rstrip('/')}/{dispatch_id}/{row_key}"


def publish_command(*, base_uri: str, dispatch_id: str, row_key: str) -> str:
    """Return the shell snippet that uploads one row's output directory.

    The snippet tolerates its own failure: the benchmark's exit code is the
    task's verdict, and an upload problem must be visible in the logs without
    masking it.

    Args:
        base_uri: Bucket URI prefix from ``odin.yaml``.
        dispatch_id: Dispatch identifier.
        row_key: Row identity.

    Returns:
        A single-line shell command for splicing into the entry script.
    """
    uri = results_uri_for(base_uri, dispatch_id, row_key)
    return f'osmo data upload "{uri}" "$OUT" || echo "=== odin: results upload failed ==="'


def validate_bundle(bundle_dir: Path) -> bool:
    """Return ``True`` iff *bundle_dir* holds a readable schema bundle.

    A ``COMPLETED`` OSMO task that produced no parseable bundle is classified
    ``malformed_bundle`` by the caller.

    Args:
        bundle_dir: Directory the row's results were fetched into.

    Returns:
        Whether some bundle file exists, parses as JSON, and declares a
        ``schema_version``.
    """
    if not bundle_dir.is_dir():
        return False
    for candidate in sorted(bundle_dir.glob(_BUNDLE_GLOB)):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version"):
            return True
    return False


def fetch_results(
    *,
    client: _DownloaderProto,
    base_uri: str,
    dispatch_id: str,
    row_key: str,
    dest_dir: Path,
) -> Path:
    """Download one row's results into ``<dest_dir>/<row_key>``.

    Idempotent: a directory that already holds a valid bundle is not
    re-downloaded.

    Args:
        client: Provides ``data_download(remote_uri, dest_dir)``.
        base_uri: Bucket URI prefix from ``odin.yaml``.
        dispatch_id: Dispatch identifier.
        row_key: Row identity.
        dest_dir: Dispatch directory the row directory is created under.

    Returns:
        The row directory, whether or not the bundle within it validates.
    """
    row_dir = dest_dir / row_key
    if validate_bundle(row_dir):
        return row_dir
    client.data_download(results_uri_for(base_uri, dispatch_id, row_key), row_dir)
    return row_dir
