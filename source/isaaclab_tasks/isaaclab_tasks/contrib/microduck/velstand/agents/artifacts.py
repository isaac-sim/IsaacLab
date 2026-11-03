# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anonymous, immutable teacher downloads; checkpoint weights are never packaged or redistributed."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "64d28ce092253c509b51031ec7ccb53790a00d0d"
ARTIFACTS = {
    "walk": (
        "bundles/velocity2/441tzs6d_3750/model_3750.pt",
        "702c01bd02aeda1d7869390342e00298193e7071d475f3776e0a6eedfc5c8209",
    ),
    "stand": (
        "bundles/microduck_stand/69u48n8l_9750/model_9750.pt",
        "12bfcfddcdb4270e4cbbfb43554d0b14fbdad01263699cd97424e48e76b302fe",
    ),
}


def verify_artifact(path: Path, expected_sha256: str) -> Path:
    """Refuse partial, stale or substituted checkpoint contents, including user-supplied paths."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return path


def resolve_artifact(name: str, cache_dir: str | Path | None = None, path: str | Path | None = None) -> Path:
    """Resolve a pinned public teacher without consulting credentials or logging into the Hub.

    The public training repository declares no weight license. Download into a private cache;
    do not include these files in source distributions or container images.
    """
    remote, digest = ARTIFACTS[name]
    if path:
        return verify_artifact(Path(path).expanduser(), digest)
    cache = Path(
        cache_dir or os.environ.get("MICRODUCK_VELSTAND_CACHE", "~/.cache/isaaclab/microduck/velstand")
    ).expanduser()
    target = cache / REVISION / f"{name}_{digest[:12]}.pt"
    if target.is_file():
        return verify_artifact(target, digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/pollen-robotics/microduck-training/resolve/{REVISION}/{remote}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                while data := response.read(1024 * 1024):
                    stream.write(data)
        verify_artifact(temporary, digest)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
