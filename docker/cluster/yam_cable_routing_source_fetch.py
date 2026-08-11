# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch one checksum-gated source package from a peer OSMO task."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

_FILES = ("source.tar.gz", "source.metadata", "git-status.txt", "source.sha256")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value(metadata: str, key: str) -> str:
    prefix = f"{key}="
    values = [line.removeprefix(prefix) for line in metadata.splitlines() if line.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise ValueError(f"source.metadata must contain exactly one non-empty {key} field")
    return values[0]


def _download(source_url: str, destination: Path, request_timeout_seconds: float) -> None:
    with urllib.request.urlopen(source_url, timeout=request_timeout_seconds) as response:
        if response.status != 200:
            raise OSError(f"HTTP {response.status} while fetching {source_url}")
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _fetch_once(
    source_url: str,
    destination: Path,
    expected_sha256: str,
    request_timeout_seconds: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-fetch-", dir=destination.parent))
    try:
        base_url = source_url.rstrip("/") + "/"
        for file_name in _FILES:
            file_url = urllib.parse.urljoin(base_url, urllib.parse.quote(file_name))
            _download(file_url, staging / file_name, request_timeout_seconds)

        published_sha256 = (staging / "source.sha256").read_text(encoding="utf-8").strip()
        if published_sha256 != expected_sha256:
            raise ValueError(f"source.sha256 contains {published_sha256!r}, expected {expected_sha256}")
        actual_sha256 = _sha256(staging / "source.tar.gz")
        if actual_sha256 != expected_sha256:
            raise ValueError(f"source.tar.gz digest {actual_sha256} != {expected_sha256}")

        metadata = (staging / "source.metadata").read_text(encoding="utf-8")
        if _metadata_value(metadata, "source_sha256") != expected_sha256:
            raise ValueError("source.metadata does not identify the expected source archive")
        expected_status_sha256 = _metadata_value(metadata, "git_status_sha256")
        if not _SHA256_PATTERN.fullmatch(expected_status_sha256):
            raise ValueError("source.metadata contains an invalid git_status_sha256")
        actual_status_sha256 = _sha256(staging / "git-status.txt")
        if actual_status_sha256 != expected_status_sha256:
            raise ValueError(f"git-status.txt digest {actual_status_sha256} != {expected_status_sha256}")

        destination.mkdir(parents=True, exist_ok=True)
        # source.sha256 is the readiness marker and must be published last.
        (destination / "source.sha256").unlink(missing_ok=True)
        for file_name in _FILES[:-1]:
            os.replace(staging / file_name, destination / file_name)
        os.replace(staging / "source.sha256", destination / "source.sha256")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def fetch_source_package(
    source_url: str,
    destination: Path,
    expected_sha256: str,
    wait_seconds: float,
    retry_seconds: float,
    request_timeout_seconds: float,
) -> None:
    """Fetch and atomically publish a validated package, retrying peer startup failures."""
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if wait_seconds <= 0 or retry_seconds <= 0 or request_timeout_seconds <= 0:
        raise ValueError("all timeout and retry values must be positive")

    deadline = time.monotonic() + wait_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            _fetch_once(source_url, destination, expected_sha256, request_timeout_seconds)
            print(
                f"source_fetch=complete attempts={attempt} sha256={expected_sha256} destination={destination}",
                flush=True,
            )
            return
        except OSError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Timed out fetching checksum-complete source {expected_sha256} "
                    f"from {source_url} after {attempt} attempts: {error}"
                ) from error
            print(
                f"source_fetch=retry attempt={attempt} remaining_seconds={remaining:.1f} error={error}",
                flush=True,
            )
            time.sleep(min(retry_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=1800.0)
    parser.add_argument("--retry-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    fetch_source_package(
        args.source_url,
        args.destination,
        args.expected_sha256,
        args.wait_seconds,
        args.retry_seconds,
        args.request_timeout_seconds,
    )


if __name__ == "__main__":
    main()
