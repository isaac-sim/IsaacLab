# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ANYmal velocity-task symmetry augmentation."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.core.velocity.mdp.symmetry import anymal

# Native joint orders of ANYmal-D on the two physics backends.
PHYSX_ORDER = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]  # fmt: skip
NEWTON_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]  # fmt: skip

# Mirrored counterpart and sign of each joint.
LEFT_RIGHT = {
    "LF_HAA": ("RF_HAA", -1), "LF_HFE": ("RF_HFE", 1), "LF_KFE": ("RF_KFE", 1),
    "LH_HAA": ("RH_HAA", -1), "LH_HFE": ("RH_HFE", 1), "LH_KFE": ("RH_KFE", 1),
    "RF_HAA": ("LF_HAA", -1), "RF_HFE": ("LF_HFE", 1), "RF_KFE": ("LF_KFE", 1),
    "RH_HAA": ("LH_HAA", -1), "RH_HFE": ("LH_HFE", 1), "RH_KFE": ("LH_KFE", 1),
}  # fmt: skip
FRONT_BACK = {
    "LF_HAA": ("LH_HAA", 1), "LF_HFE": ("LH_HFE", -1), "LF_KFE": ("LH_KFE", -1),
    "LH_HAA": ("LF_HAA", 1), "LH_HFE": ("LF_HFE", -1), "LH_KFE": ("LF_KFE", -1),
    "RF_HAA": ("RH_HAA", 1), "RF_HFE": ("RH_HFE", -1), "RF_KFE": ("RH_KFE", -1),
    "RH_HAA": ("RF_HAA", 1), "RH_HFE": ("RF_HFE", -1), "RH_KFE": ("RF_KFE", -1),
}  # fmt: skip
DIAGONAL = {
    "LF_HAA": ("RH_HAA", -1), "LF_HFE": ("RH_HFE", -1), "LF_KFE": ("RH_KFE", -1),
    "LH_HAA": ("RF_HAA", -1), "LH_HFE": ("RF_HFE", -1), "LH_KFE": ("RF_KFE", -1),
    "RF_HAA": ("LH_HAA", -1), "RF_HFE": ("LH_HFE", -1), "RF_KFE": ("LH_KFE", -1),
    "RH_HAA": ("LF_HAA", -1), "RH_HFE": ("LF_HFE", -1), "RH_KFE": ("LF_KFE", -1),
}  # fmt: skip


@pytest.mark.parametrize("joint_names", [PHYSX_ORDER, NEWTON_ORDER], ids=["physx", "newton"])
def test_symmetry_mirrors_joints_by_name(joint_names):
    """Mirrored joint observations and actions come from the counterpart joint with the right sign."""
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(joint_names=joint_names)},
        observation_manager=SimpleNamespace(active_terms={"policy": []}),
    )
    env.unwrapped = env
    # tag every joint entry with a distinct value per joint and per observation block
    joint_values = torch.arange(1.0, 13.0)
    obs = torch.cat([torch.zeros(12), joint_values, joint_values + 100.0, joint_values + 200.0]).unsqueeze(0)
    obs_aug, actions_aug = anymal.compute_symmetric_states(
        env, TensorDict({"policy": obs}, batch_size=[1]), joint_values.unsqueeze(0)
    )

    value = {name: joint_values[i] for i, name in enumerate(joint_names)}
    for row, mirror in ((1, LEFT_RIGHT), (2, FRONT_BACK), (3, DIAGONAL)):
        expected = torch.stack([mirror[name][1] * value[mirror[name][0]] for name in joint_names])
        torch.testing.assert_close(actions_aug[row], expected)
        for offset, block in enumerate(range(12, 48, 12)):
            obs_block = obs_aug["policy"][row, block : block + 12]
            torch.testing.assert_close(obs_block, expected + torch.sign(expected) * 100.0 * offset)
