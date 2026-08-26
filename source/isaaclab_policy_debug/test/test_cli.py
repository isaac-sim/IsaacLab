# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
from isaaclab_policy_debug.cli import configure_policy_debug_args, find_newton_visualizer


def _args(path, **overrides):
    values = dict(
        policy_debug=str(path),
        policy_debug_max_policies=8,
        checkpoint=None,
        use_pretrained_checkpoint=False,
        num_envs=None,
        headless=False,
        visualizer=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_policy_debug_forces_newton_and_capacity(tmp_path):
    args = _args(tmp_path)
    configure_policy_debug_args(args)
    assert args.visualizer == ["newton_gl"]
    assert args.visualizer_explicit
    assert args.max_visible_envs == 8


def test_find_newton_visualizer_accepts_newton_gl():
    visualizer = SimpleNamespace(cfg=SimpleNamespace(visualizer_type="newton_gl"))
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace(visualizers=[visualizer])))
    assert find_newton_visualizer(env) is visualizer


@pytest.mark.parametrize(
    ("field", "value"),
    [("checkpoint", "model.pt"), ("use_pretrained_checkpoint", True), ("num_envs", 4), ("headless", True)],
)
def test_policy_debug_rejects_conflicts(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        configure_policy_debug_args(_args(tmp_path, **{field: value}))


def test_policy_debug_rejects_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        configure_policy_debug_args(_args(tmp_path / "missing"))


def test_policy_debug_rejects_non_positive_capacity(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        configure_policy_debug_args(_args(tmp_path, policy_debug_max_policies=0))
