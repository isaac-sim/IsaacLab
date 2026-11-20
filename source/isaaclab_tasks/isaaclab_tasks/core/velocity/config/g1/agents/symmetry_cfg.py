# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/right symmetry augmentation shared by the G1 PPO and SAC agents."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


@configclass
class G1SymmetryCfg(RslRlSymmetryCfg):
    """Append sagittally mirrored G1 samples to every training batch."""

    use_data_augmentation: bool = True
    use_mirror_loss: bool = False
    data_augmentation_func: str = "isaaclab_tasks.core.velocity.mdp.symmetry.g1:compute_symmetric_states"
    mirror_loss_coeff: float = 0.0
