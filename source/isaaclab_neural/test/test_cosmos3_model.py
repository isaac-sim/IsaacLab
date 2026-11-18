# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The parts of Cosmos3 that do not load the Cosmos Framework."""

import numpy as np

from isaaclab.utils import configclass

from isaaclab_neural.models.cosmos3 import Cosmos3Cfg, Cosmos3Model, JointMappingCfg


class Mapping:
    def __init__(self, cfg):
        pass

    def encode(self, joint_targets, start_joint_state):
        controls = np.zeros((len(joint_targets), 59), np.float32)
        controls[:, 0] = joint_targets[:, 0] + start_joint_state[0]
        return controls


@configclass
class MappingCfg(JointMappingCfg):
    class_type: type = Mapping


def cosmos3_cfg(**kwargs):
    return Cosmos3Cfg(checkpoint_path="checkpoint", output_dir="out", **kwargs)


def test_a_session_serves_one_environment_and_a_window_any_number():
    assert Cosmos3Model(cosmos3_cfg(), 4).spec().max_envs is None
    assert Cosmos3Model(cosmos3_cfg(mode="session"), 1).spec().max_envs == 1


def test_joint_targets_are_mapped_to_controls_per_environment():
    model = Cosmos3Model(cosmos3_cfg(joint_mapping=MappingCfg()), 2)
    seen = []
    model.step_controls = lambda controls: seen.append(controls)
    targets = np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3)
    model.step(targets, np.array([[1, 0, 0], [5, 0, 0]], np.float32))
    assert seen[0].shape == (2, 16, 59)
    assert seen[0][1, 0, 0] == targets[1, 0, 0] + 5
