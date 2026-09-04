# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def test_lazy_export_preserves_absolute_import_alias(tmp_path, monkeypatch):
    """Absolute named imports in stubs should be exported under their alias."""
    package_name = "lazy_export_alias_test"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "from isaaclab.utils.module import lazy_export\n\nlazy_export()\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.pyi").write_text("from math import sqrt as square_root\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(package_name, None)
    try:
        module = importlib.import_module(package_name)
        assert module.square_root(9.0) == 3.0
        assert "square_root" in module.__all__
        assert "sqrt" not in module.__all__
    finally:
        sys.modules.pop(package_name, None)
