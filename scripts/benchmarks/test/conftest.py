# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the benchmark smoke tests."""

import json
from pathlib import Path

import pytest

_TRAINING_BUNDLE_KEYS = {"run", "versions", "hardware", "runtime", "resources", "learning"}


@pytest.fixture
def load_training_bundle():
    """Provide a loader for the schema bundle in a formatter output directory."""

    def load(out_dir: Path) -> dict:
        candidates = sorted(out_dir.glob("*.json"))
        assert candidates, f"no *.json written to {out_dir}"
        for path in candidates:
            data = json.loads(path.read_text())
            if set(data) >= _TRAINING_BUNDLE_KEYS:
                return data
        pytest.fail(
            f"no bundle in {out_dir} contained keys {_TRAINING_BUNDLE_KEYS}; found {[path.name for path in candidates]}"
        )

    return load
