# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab.utils.io.yaml import dump_yaml, load_yaml

pytestmark = pytest.mark.unit


def test_dump_yaml_supports_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    dump_yaml("config.yaml", {"value": 42})

    assert (tmp_path / "config.yaml").is_file()
    assert load_yaml("config.yaml") == {"value": 42}


def test_dump_yaml_adds_extension_in_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    dump_yaml("config", {"value": 42})

    assert (tmp_path / "config.yaml").is_file()
