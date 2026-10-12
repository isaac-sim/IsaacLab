# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the robomimic training entry point."""

import json
import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


class _Config:
    def __init__(self):
        self.train = SimpleNamespace(data=None, output_dir=None, cuda=False)

    def values_unlocked(self):
        return nullcontext()

    def update(self, values):
        pass

    def lock(self):
        pass


def test_training_failure_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A failed training run must fail the invoking process."""
    script = Path(__file__).resolve().parents[3] / "scripts/imitation_learning/robomimic/train.py"
    trainer = runpy.run_path(str(script), run_name="robomimic_train_test")
    module_globals = trainer["main"].__globals__
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"algo_name": "bc"}))
    config = _Config()
    task_spec = SimpleNamespace(kwargs={"robomimic_bc_cfg_entry_point": str(config_path)})
    args = SimpleNamespace(
        task="Isaac-Test",
        algo="bc",
        dataset=None,
        name=None,
        epochs=None,
        log_dir="robomimic",
        normalize_training_actions=False,
    )

    monkeypatch.setattr(module_globals["gym"], "spec", lambda _: task_spec)
    monkeypatch.setitem(module_globals, "config_factory", lambda _: config)
    monkeypatch.setattr(module_globals["TrainUtils"], "get_exp_dir", lambda _: ("logs", "models", "videos"))
    monkeypatch.setattr(module_globals["TorchUtils"], "get_torch_device", lambda **_: "cpu")

    def fail_training(*_):
        raise RuntimeError("training failed")

    monkeypatch.setitem(module_globals, "train", fail_training)

    with pytest.raises(RuntimeError, match="training failed"):
        trainer["main"](args)
