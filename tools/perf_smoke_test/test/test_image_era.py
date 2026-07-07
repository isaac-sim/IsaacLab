# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the source-derivable image-era key and manifest resolver."""

from __future__ import annotations

import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import image_era  # noqa: E402

_ENV_BASE = """
# General settings
ACCEPT_EULA=Y
ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
# version comment
ISAACSIM_VERSION=6.0.0-dev2
DOCKER_USER_HOME=/root
DOCKER_NAME_SUFFIX=""
"""


def test_parse_env_file_strips_comments_and_quotes() -> None:
    """Comments, blank lines, and surrounding quotes must not leak into values."""
    env = image_era.parse_env_file(_ENV_BASE)

    assert env["ISAACSIM_BASE_IMAGE"] == "nvcr.io/nvidia/isaac-sim"
    assert env["ISAACSIM_VERSION"] == "6.0.0-dev2"
    assert env["DOCKER_NAME_SUFFIX"] == ""
    assert "# General settings" not in env


def test_era_key_is_deterministic() -> None:
    """The same parsed env must always hash to the same era key."""
    env = image_era.parse_env_file(_ENV_BASE)

    assert image_era.compute_era_key(env) == image_era.compute_era_key(env)


def test_era_key_changes_when_isaacsim_version_changes() -> None:
    """A different Isaac Sim version is a different era (under-sensitivity guard)."""
    old = image_era.compute_era_key(image_era.parse_env_file(_ENV_BASE))
    new = image_era.compute_era_key(
        image_era.parse_env_file(_ENV_BASE.replace("6.0.0-dev2", "5.0.0"))
    )

    assert old != new


def test_era_key_ignores_cosmetic_and_non_era_fields() -> None:
    """Comment edits, reordering, and non-era fields must not split the era (over-sensitivity guard)."""
    base_key = image_era.compute_era_key(image_era.parse_env_file(_ENV_BASE))
    cosmetic = """
# a totally different comment
ISAACSIM_VERSION=6.0.0-dev2
ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
DOCKER_USER_HOME=/home/somethingelse
ACCEPT_EULA=N
"""

    assert image_era.compute_era_key(image_era.parse_env_file(cosmetic)) == base_key


def test_resolve_image_hit_returns_pinned_image() -> None:
    """A known era resolves to its immutable image and reports a match."""
    manifest = {
        "schema_version": 1,
        "fallback_image": "nvcr.io/nvidian/isaac-lab:latest-perf",
        "eras": {"era-abc": {"image": "nvcr.io/nvidian/isaac-lab:sha-deadbee"}},
    }

    image, matched = image_era.resolve_image("era-abc", manifest)

    assert matched is True
    assert image == "nvcr.io/nvidian/isaac-lab:sha-deadbee"


def test_resolve_image_miss_falls_back_to_latest_perf() -> None:
    """An unknown era degrades to the manifest fallback instead of failing."""
    manifest = {
        "schema_version": 1,
        "fallback_image": "nvcr.io/nvidian/isaac-lab:latest-perf",
        "eras": {},
    }

    image, matched = image_era.resolve_image("era-unknown", manifest)

    assert matched is False
    assert image == "nvcr.io/nvidian/isaac-lab:latest-perf"


def test_resolve_image_explicit_fallback_overrides_manifest() -> None:
    """An explicit fallback takes precedence over the manifest default."""
    image, matched = image_era.resolve_image(
        "era-unknown", {"eras": {}}, fallback_image="nvcr.io/nvidian/isaac-lab:sha-override"
    )

    assert matched is False
    assert image == "nvcr.io/nvidian/isaac-lab:sha-override"


def test_resolve_image_empty_manifest_uses_default_fallback() -> None:
    """A missing/empty manifest still yields the built-in default image."""
    image, matched = image_era.resolve_image("era-unknown", None)

    assert matched is False
    assert image == image_era.DEFAULT_FALLBACK_IMAGE


def test_era_key_from_commit_matches_tree_at_head() -> None:
    """The git-show path and the working-tree path must agree for the same source."""
    tree_key = image_era.era_key_from_tree(_REPO_ROOT)
    commit_key = image_era.era_key_from_commit("HEAD", _REPO_ROOT)

    assert tree_key == commit_key
