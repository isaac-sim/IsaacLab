# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionManager

pytestmark = pytest.mark.unit


def test_action_descriptor_export_does_not_write_to_stdout(capsys):
    """Exporting valid action descriptors should not emit debug output."""
    descriptor = GenericActionIODescriptor(
        name="test_action",
        full_path="tests.TestAction",
        description="Test action",
        shape=(1,),
        dtype="torch.float32",
        export=True,
    )
    manager = ActionManager.__new__(ActionManager)
    manager._terms = {"test": SimpleNamespace(IO_descriptor=descriptor)}

    exported = manager.get_IO_descriptors

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(exported) == 1
    assert exported[0]["name"] == "test_action"
