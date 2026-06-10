# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab_tasks.utils import get_checkpoint_path


def test_get_checkpoint_path_prefers_requested_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "cartpole_camera_direct" / "2026-06-07_02-41-41" / "nn"
    checkpoint_dir.mkdir(parents=True)
    expected_checkpoint = checkpoint_dir / "cartpole_camera_direct.pth"
    expected_checkpoint.touch()
    (checkpoint_dir / "last_cartpole_camera_direct_ep_10_rew_1.0.pth").touch()

    checkpoint_path = get_checkpoint_path(
        str(tmp_path / "cartpole_camera_direct"), "2026-06-07_02-41-41", "cartpole_camera_direct.pth", other_dirs=["nn"]
    )

    assert checkpoint_path == str(expected_checkpoint)


def test_get_checkpoint_path_uses_latest_checkpoint_when_checkpoint_is_omitted(tmp_path):
    checkpoint_dir = tmp_path / "cartpole_camera_direct" / "2026-06-07_02-41-41" / "nn"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last_cartpole_camera_direct_ep_5_rew_0.5.pth").touch()
    expected_checkpoint = checkpoint_dir / "last_cartpole_camera_direct_ep_10_rew_1.0.pth"
    expected_checkpoint.touch()

    checkpoint_path = get_checkpoint_path(
        str(tmp_path / "cartpole_camera_direct"), "2026-06-07_02-41-41", other_dirs=["nn"]
    )

    assert checkpoint_path == str(expected_checkpoint)


def test_get_checkpoint_path_requested_checkpoint_stays_strict(tmp_path):
    checkpoint_dir = tmp_path / "cartpole_camera_direct" / "2026-06-07_02-41-41" / "nn"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last_cartpole_camera_direct_ep_10_rew_1.0.pth").touch()

    with pytest.raises(ValueError, match="cartpole_camera_direct.pth"):
        get_checkpoint_path(
            str(tmp_path / "cartpole_camera_direct"),
            "2026-06-07_02-41-41",
            "cartpole_camera_direct.pth",
            other_dirs=["nn"],
        )


def test_get_checkpoint_path_prefers_preferred_over_checkpoint(tmp_path):
    # rl_games: the best checkpoint must win over numbered ones when given as the preferred pattern
    checkpoint_dir = tmp_path / "cartpole_camera_direct" / "2026-06-07_02-41-41" / "nn"
    checkpoint_dir.mkdir(parents=True)
    expected_checkpoint = checkpoint_dir / "cartpole_camera_direct.pth"
    expected_checkpoint.touch()
    (checkpoint_dir / "last_cartpole_camera_direct_ep_10_rew_1.0.pth").touch()

    checkpoint_path = get_checkpoint_path(
        str(tmp_path / "cartpole_camera_direct"),
        "2026-06-07_02-41-41",
        ".*",
        other_dirs=["nn"],
        preferred_checkpoint="cartpole_camera_direct.pth",
    )

    assert checkpoint_path == str(expected_checkpoint)


def test_get_checkpoint_path_falls_back_when_preferred_missing(tmp_path):
    # rl_games short run: the best checkpoint is never written, so resolution falls back to the latest numbered one
    checkpoint_dir = tmp_path / "cartpole_camera_direct" / "2026-06-07_02-41-41" / "nn"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last_cartpole_camera_direct_ep_9_rew_0.9.pth").touch()
    expected_checkpoint = checkpoint_dir / "last_cartpole_camera_direct_ep_10_rew_1.0.pth"
    expected_checkpoint.touch()

    checkpoint_path = get_checkpoint_path(
        str(tmp_path / "cartpole_camera_direct"),
        "2026-06-07_02-41-41",
        ".*",
        other_dirs=["nn"],
        preferred_checkpoint="cartpole_camera_direct.pth",
    )

    assert checkpoint_path == str(expected_checkpoint)


def test_get_checkpoint_path_prefers_final_model_over_steps(tmp_path):
    # sb3: the final ``model.zip`` must win over numbered ``model_<n>_steps.zip`` snapshots
    run_dir = tmp_path / "2026-06-07_02-41-41"
    run_dir.mkdir(parents=True)
    expected_checkpoint = run_dir / "model.zip"
    expected_checkpoint.touch()
    (run_dir / "model_2000_steps.zip").touch()

    checkpoint_path = get_checkpoint_path(
        str(tmp_path), ".*", r"model_.*\.zip", sort_alpha=False, preferred_checkpoint=r"model\.zip"
    )

    assert checkpoint_path == str(expected_checkpoint)
