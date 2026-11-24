# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check replay validation against the public EpisodeData sample shape."""

import pytest
import torch
from isaaclab_teleop.entrypoints.replay_demos import compare_states

from isaaclab.utils.datasets import EpisodeData

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _recorded_and_runtime_states(
    runtime_env_index: int,
) -> tuple[dict[str, dict[str, dict[str, torch.Tensor]]], dict[str, dict[str, dict[str, torch.Tensor]]]]:
    recorded = {
        "articulation": {
            "robot": {
                "root_pose": torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]),
                "joint_position": torch.arange(9, dtype=torch.float32),
            }
        },
        "rigid_object": {"cube": {"root_pose": torch.tensor([0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0])}},
    }
    episode = EpisodeData()
    episode.data = {
        "states": {
            asset_type: {
                asset_name: {field_name: value.unsqueeze(0) for field_name, value in fields.items()}
                for asset_name, fields in assets.items()
            }
            for asset_type, assets in recorded.items()
        }
    }
    dataset_state = episode.get_state(0)
    assert dataset_state["articulation"]["robot"]["root_pose"].shape == (1, 7)
    runtime = {}
    for asset_type, assets in recorded.items():
        runtime[asset_type] = {}
        for asset_name, fields in assets.items():
            runtime[asset_type][asset_name] = {}
            for field_name, value in fields.items():
                batch = value.repeat(runtime_env_index + 1, 1) + 1.0
                batch[runtime_env_index] = value
                runtime[asset_type][asset_name][field_name] = batch
    return dataset_state, runtime


@pytest.mark.parametrize("flat_dataset", [False, True])
@pytest.mark.parametrize("different_value", [False, True])
def test_replay_compares_single_sample_to_selected_runtime_env(different_value: bool, flat_dataset: bool) -> None:
    runtime_env_index = 1
    dataset_state, runtime = _recorded_and_runtime_states(runtime_env_index)
    if flat_dataset:
        dataset_state = {
            asset_type: {
                asset_name: {field_name: value[0] for field_name, value in fields.items()}
                for asset_name, fields in assets.items()
            }
            for asset_type, assets in dataset_state.items()
        }
    if different_value:
        runtime["rigid_object"]["cube"]["root_pose"][runtime_env_index, 0] += 0.02
    matched, log = compare_states(dataset_state, runtime, runtime_env_index)
    assert matched is not different_value
    if different_value:
        assert 'State ["rigid_object"]["cube"]["root_pose"][0] don\'t match' in log
    else:
        assert log == ""


def test_replay_still_rejects_different_feature_counts() -> None:
    dataset_state, runtime = _recorded_and_runtime_states(0)
    runtime["articulation"]["robot"]["joint_position"] = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="State shape of joint_position for asset robot"):
        compare_states(dataset_state, runtime, 0)
