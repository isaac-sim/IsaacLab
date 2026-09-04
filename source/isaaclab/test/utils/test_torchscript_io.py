# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils.io.torchscript import load_torchscript_model

pytestmark = pytest.mark.unit


def test_load_torchscript_model_propagates_load_failure(monkeypatch, tmp_path):
    model_path = tmp_path / "invalid.pt"
    model_path.write_bytes(b"not a torchscript model")

    def fail_load(*args, **kwargs):
        raise ValueError("invalid archive")

    monkeypatch.setattr(torch.jit, "load", fail_load)

    with pytest.raises(RuntimeError, match="Failed to load TorchScript model") as exc_info:
        load_torchscript_model(str(model_path))

    assert isinstance(exc_info.value.__cause__, ValueError)
