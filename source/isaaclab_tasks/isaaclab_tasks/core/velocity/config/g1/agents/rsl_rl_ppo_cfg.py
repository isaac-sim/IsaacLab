# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from isaaclab_tasks.utils import preset


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Newton needs ~1.7x the PPO iterations to match PhysX on G1. PhysX saturates near iter 3000
    # (reward ≈ +18, ep_len ≈ 980) and does not meaningfully improve on either metric past that —
    # reward oscillates +16 to +19 through iter 7500, ep_len stays flat. Newton reaches the same
    # (reward, ep_len) quality at iter 5000 (+16 / 984). Comparing reward alone is misleading:
    # ep_len confirms the robot is stable in both cases. The gap is sample-efficiency, not a
    # ceiling — no physics or reward tuning closes it.
    max_iterations = preset(default=3000, newton_mjwarp=5000)
    save_interval = 50
    experiment_name = "g1_rough"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
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
class G1RoughSymmetryPPORunnerCfg(G1RoughPPORunnerCfg):
    """The rough runner with every transition presented alongside its left-right mirror.

    On flat ground under a straight command this line walks with one knee bent thirty degrees more
    than the other and the pelvis rolled to one side, with the side chosen per seed -- see
    :mod:`~isaaclab_tasks.core.velocity.mdp.symmetry.g1_29dof` for the measurements. Nothing in the
    task states that the legs are interchangeable, so this states it in the only place that
    constrains the policy rather than a statistic about it.

    Data augmentation rather than the mirror loss: the loss adds a term whose weight is one more
    thing to tune, while augmentation is exact and free of a coefficient. It doubles the batch, so
    an iteration costs roughly twice the PPO update time; the rollout is unchanged.
    """

    def __post_init__(self):
        super().__post_init__()

        from isaaclab_tasks.core.velocity.mdp.symmetry import g1_29dof  # noqa: PLC0415

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1_29dof.compute_symmetric_states,
        )


@configclass
class G1RoughMjlabPPORunnerCfg(G1RoughPPORunnerCfg):
    """mjlab's PPO settings and its asymmetric observation split.

    Three things differ from ours and they are all here: observation normalisation on rather than
    off, an entropy coefficient of 0.01 rather than 0.008, and **30000 iterations rather than
    6000**. The last is not a detail -- every arm on this line needs 3000 to 4000 iterations just to
    leave ``success_rate`` 0.000, so our budget is barely two thousand iterations of real learning
    against their thirty thousand.

    ``obs_groups`` is where the split lands: the actor reads the corrupted, biased group and the
    critic the clean one with the foot state.
    """

    max_iterations = 30000
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()

        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
        self.algorithm.entropy_coef = 0.01


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]


@configclass
class G129DofRoughAirTime100DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distil the sighted w100 teacher into a proprioception-only student.

    The student keeps the actor architecture the teacher was trained with, so the only thing that
    changes across the pipeline is what goes in. ``init_std`` is 0.0 on the teacher because at
    distillation time it is an oracle being copied, not a policy being explored.

    ``obs_groups`` is where the two contracts meet: ``student`` reads the deployable group, and
    ``teacher`` the privileged one it was trained on. ``obs_normalization`` is False on both because
    :class:`G1RoughPPORunnerCfg` trains with it off -- a teacher checkpoint carries a normalizer's
    running statistics in its state dict only if it had one, and loading it into a model built the
    other way fails on unexpected or missing keys rather than training something subtly wrong.
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "g1_rough_29dof_distill"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class G129DofRoughAirTime100DepthDistillationRunnerCfg(G129DofRoughAirTime100DistillationRunnerCfg):
    """Distil the w100-line teacher into a student that reads a chest depth camera.

    Only the student changes against the blind stage. ``obs_groups`` gains the ``depth`` group and
    the student becomes a CNN model: ``rsl_rl`` splits observation sets by rank, sends each 4D group
    through a convolutional encoder, and concatenates the latent with the 1D groups before the MLP.
    The teacher is untouched, so the same checkpoint loads in both stages.

    ``max_grad_norm`` is set because the DR29 depth run without it swung between 0.63 and 0.82
    success on adjacent checkpoints while the loss sat flat -- single batches were free to move the
    policy a long way. 4000 iterations rather than the DR29 line's 3000 because the blind student on
    this line was still climbing at 1784.
    """

    max_iterations = 4000
    experiment_name = "g1_rough_29dof_depth_distill"
    obs_groups = {"student": ["policy", "depth"], "teacher": ["teacher"]}
    student = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            padding="zeros",
            activation="relu",
            # The MLP consumes a flat latent; without this, "The output of the CNN must be
            # flattened before passing it to the MLP".
            flatten=True,
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        # Normalizes the 1D groups only; the depth group is already scaled into [0, 1]. The teacher
        # keeps obs_normalization off to match this line's PPO config, which trains without it.
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )
