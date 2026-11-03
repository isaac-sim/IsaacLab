# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck velocity tasks.

Every hyper-parameter is upstream's; see ``historical original source extraction`` section 6,
"RL agent cfg".

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ..velocity_env_cfg import MICRODUCK_STEPS_PER_ITERATION


@configclass
class MicroDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the MicroDuck velocity-tracking tasks."""

    # tied to the curriculum stage tables' step math, not restated as its own literal
    num_steps_per_env = MICRODUCK_STEPS_PER_ITERATION
    max_iterations = 50000
    save_interval = 250
    experiment_name = "microduck_velocity"
    # Asymmetric actor-critic, as upstream trains it: the actor reads the corrupted 61-wide deploy
    # vector and the critic reads the privileged group, which adds the base linear velocity and the
    # foot terms the robot has no sensor for.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckRollersPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck roller-skating task.

    Upstream's roller runner differs from its velocity one in exactly two fields, so everything else
    -- the network shapes, the optimizer, the asymmetric actor-critic wiring, the 24-step rollout and
    the 50 000-iteration budget -- is inherited rather than restated. See
    ``historical tasks2 source extraction`` section 5.9.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_velocity_rollers"
    # Three times the family's 0.01, and the only task in the family that raises it. Upstream's
    # comment calls it "roller-specific: higher exploration than the walk envs": the single positive
    # task reward pays nothing at all until the wheels turn, so the policy has to find a push before
    # it gets any gradient toward one.
    algorithm = MicroDuckPPORunnerCfg().algorithm.replace(entropy_coef=0.03)


@configclass
class MicroDuckSwizzlePPORunnerCfg(MicroDuckRollersPPORunnerCfg):
    """PPO runner for the MicroDuck swizzle task.

    Upstream builds this one by replacing exactly two string fields on the roller runner, so
    everything else -- including the raised exploration noise, which the swizzle needs for the same
    reason the stride does -- is inherited. See
    ``historical tasks3 source extraction`` section 7.6.

    Symmetry augmentation stays off, inherited silently through upstream's ``dataclasses.replace``.
    That is worth naming rather than fixing: this is the one task in the family whose defining
    property *is* left-right symmetry, and upstream trains it without the augmentation that would
    exploit that (section 11.4). Turning it on would be a retune with its own training run.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_velocity_swizzle"
