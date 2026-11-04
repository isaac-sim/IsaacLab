# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVStage compatibility selection without loading the native runtime."""

import importlib.metadata
import importlib.util
from unittest.mock import Mock

import pytest
from isaaclab_ov import ovstage_compat
from packaging.version import InvalidVersion

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize(
    "runtime_version, expected",
    [
        (None, False),
        ("0.3.0.1791986420", False),
        ("0.4.0.1793320890", True),
        ("0.4.0.dev1793793581+main.29a7740d", True),
        ("0.4.1", True),
        ("0.5.0.dev1", False),
        pytest.param("invalid-version", None, id="invalid-metadata"),
    ],
)
def test_cpu_hierarchy_preparation_selects_ovstage_release_line(monkeypatch, runtime_version, expected):
    """Select preparation from metadata once; missing packages disable it and invalid versions raise."""
    lookup = Mock(return_value=runtime_version)
    if runtime_version is None:
        lookup.side_effect = importlib.metadata.PackageNotFoundError("ovstage")
    monkeypatch.setattr(importlib.metadata, "version", lookup)
    spec = importlib.util.spec_from_file_location("isaaclab_ov._compat_probe", ovstage_compat.__file__)
    compat = importlib.util.module_from_spec(spec)
    if expected is None:
        with pytest.raises(InvalidVersion):
            spec.loader.exec_module(compat)
    else:
        spec.loader.exec_module(compat)
        assert compat.OVSTAGE_REQUIRES_CPU_HIERARCHY_BEFORE_CLONE is expected
    lookup.assert_called_once_with("ovstage")
