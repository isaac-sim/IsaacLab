# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test file selection shared by the CI test orchestrator and the CI marker job check.

Only the standard library and :mod:`test_settings` are imported, so the selection also runs on a
plain runner before any Isaac Lab image is pulled.
"""

from __future__ import annotations

import logging
import os

import test_settings  # isort: skip

logger = logging.getLogger(__name__)


def collect_test_files(
    source_dirs,
    filter_pattern,
    exclude_pattern,
    include_files,
    quarantined_only,
    curobo_only,
    shard_index="",
    shard_count="",
):
    """Collect test files from source directories, applying all active filters.

    Raises:
        FileNotFoundError: If a source directory does not exist.
    """
    test_files = []
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found at {source_dir}")

        for root, _, files in os.walk(source_dir):
            # source/isaaclab/test/install_ci/ has its own pytest config and conftest.
            # It is run via .github/actions/install-ci-run, never via this collector,
            # so skip the whole subtree to keep install_ci tests out of build.yaml jobs.
            if "install_ci" in root.replace("\\", "/").split("/"):
                continue

            for file in files:
                if not (file.startswith("test_") and file.endswith(".py")):
                    continue

                # Mode-exclusive filters (each bypasses TESTS_TO_SKIP)
                if quarantined_only:
                    if file not in test_settings.QUARANTINED_TESTS:
                        continue
                elif curobo_only:
                    if file not in test_settings.CUROBO_TESTS:
                        continue
                else:
                    # An explicit include_files entry overrides TESTS_TO_SKIP, allowing
                    # dedicated jobs (e.g. test-environments-training) to run tests that
                    # are otherwise excluded from general CI runs.
                    if file in test_settings.TESTS_TO_SKIP and file not in include_files:
                        logger.debug(f"Skipping {file} as it's in the skip list")
                        continue

                full_path = os.path.join(root, file)

                if filter_pattern and filter_pattern not in full_path:
                    logger.debug(f"Skipping {full_path} (does not match include pattern: {filter_pattern})")
                    continue
                if exclude_pattern and any(p.strip() in full_path for p in exclude_pattern.split(",")):
                    logger.debug(f"Skipping {full_path} (matches exclude pattern: {exclude_pattern})")
                    continue
                if include_files and file not in include_files:
                    logger.debug(f"Skipping {full_path} (not in include files list)")
                    continue

                test_files.append(full_path)

    # Sort test files deterministically to ensure consistent test ordering.
    test_files.sort()

    # Apply file-level sharding: select every Nth file from the deterministic order.
    # Skip when include_files is set — in that case the test's own conftest handles
    # sharding at the test-item level (e.g. parametrized test cases).
    if shard_index and shard_count and not include_files:
        shard_index = int(shard_index)
        shard_count = int(shard_count)
        test_files = [f for i, f in enumerate(test_files) if i % shard_count == shard_index]
        logger.info(f"Shard {shard_index}/{shard_count}: selected {len(test_files)} test files")

    return test_files


def files_with_marker(test_files, marker):
    """Keep the test files that carry ``marker``.

    Matches both ``@pytest.mark.<marker>`` (per-function) and ``pytestmark = pytest.mark.<marker>`` /
    ``pytestmark = [..., pytest.mark.<marker>, ...]`` (module-level) through their common
    ``pytest.mark.<marker>`` substring.

    Raises:
        RuntimeError: If a test file cannot be read, rather than silently dropping a marked file.
    """
    marker_token = f"pytest.mark.{marker}"
    marked_files = []
    for test_file in test_files:
        try:
            with open(test_file) as f:
                if marker_token in f.read():
                    marked_files.append(test_file)
        except OSError as exc:
            raise RuntimeError(
                f"ci_marker post-scan could not read {test_file}; refusing to silently drop a potentially"
                " marker-tagged file"
            ) from exc
    return marked_files
