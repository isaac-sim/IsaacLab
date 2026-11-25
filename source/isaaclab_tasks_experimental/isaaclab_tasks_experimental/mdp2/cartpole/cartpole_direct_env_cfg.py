# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_experimental import mdp2

from isaaclab.utils import configclass, replace

from .cartpole_env_cfg import TASK, CartpoleEnvCfg, cart, pole


@configclass
class CartpoleDirectEnvCfg(CartpoleEnvCfg):
    """Cartpole MDP2 with the Direct environment's observation order: the cart and pole positions, then the cart and
    pole velocities."""

    task: mdp2.TaskCfg = replace(
        TASK,
        observations={
            "policy": {
                f"{joint}_{name}": replace(term, inputs={**term.inputs, "joint_ids": subset})
                for name, term in TASK.observations["policy"].items()
                for joint, subset in (("cart", cart), ("pole", pole))
            },
        },
    )
