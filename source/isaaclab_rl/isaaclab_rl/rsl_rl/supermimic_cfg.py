# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed configuration for model-based SuperMimic learning."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg, RslRlMLPModelCfg


@configclass
class RslRlSuperMimicTaskCfg:
    """Task-owned differentiable state, reward and deployment features."""

    class_name: str = MISSING
    """The task class resolved by RSL-RL."""

    def configure_env(self, env_cfg, runner_cfg, *, inference: bool = False):
        """Resolve task observation settings before constructing the environment."""


@configclass
class RslRlSuperMimicActorCfg:
    """History-conditioned actor with a goal encoder."""

    class_name: str = "SuperMimicActor"
    """The actor class resolved by RSL-RL."""

    latent_dim: int = 32
    """Width of the encoded goal."""

    encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    """Hidden widths of the goal encoder."""

    decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    """Hidden widths of the history-conditioned action decoder."""

    activation: str = "elu"
    """Activation of the encoder and decoder layers."""

    norm: str = "none"
    """Normalization of the decoder layers, ``"none"`` or ``"layer"``."""

    encoder_norm: str = "layer"
    """Normalization of the goal encoder layers, ``"none"`` or ``"layer"``.

    LayerNorm keeps the hidden layers from growing a component shared by all goals, which would saturate the tanh
    latent so the policy stops seeing its goal.
    """

    output_gain: float = 0.01
    """Scale of the initial decoder output weights; values other than 1 also zero the output bias."""


@configclass
class RslRlWorldModelCfg:
    """Learned state increments and optional event-label predictions."""

    class_name: str = "WorldModel"
    """The world-model class resolved by RSL-RL."""

    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    """Hidden widths of the dynamics network."""

    label_hidden_dims: tuple[int, ...] = (512, 512)
    """Hidden widths of the optional event-label head."""

    activation: str = "swish"
    """Activation of the dynamics network."""

    normalize_output: bool = True
    """Whether to predict normalized increments and rescale them with running output statistics."""

    normalize_context: bool = True
    """Whether to normalize the dynamics context with running statistics; disable for context in a fixed range."""

    zero_initialize_output: bool = True
    """Whether to start the dynamics output layer at zero."""


@configclass
class RslRlLearnedDynamicsCfg:
    """Replay and training settings for SHAC's learned dynamics."""

    model: RslRlWorldModelCfg = RslRlWorldModelCfg()
    """Predictor trained on physical transitions."""

    replay_buffer_capacity: int = 1_000_000
    """Replay capacity in transitions across all environments."""

    max_replay_memory_fraction: float | None = None
    """Optional fraction of free CUDA memory available to replay at first insertion."""

    num_warmup_steps: int = 64
    """Stored steps per environment required before learning."""

    num_warmup_iterations: int = 500
    """Learner iterations fitting only dynamics."""

    num_updates: int = 10
    """Dynamics optimizer steps per learner iteration."""

    window_length: int = 8
    """Maximum autoregressive dynamics fitting horizon."""

    batch_size: int = 2048
    """Replay anchors per dynamics optimizer step."""

    mirror_fraction: float = 0.0
    """Fraction of each dynamics batch replaced by its left-right mirror image from the task's mirror hooks."""

    loss_reduction: str = "sum"
    """Sum the dynamics loss over each replay window (``"sum"``) or average it over all valid positions (``"mean"``)."""

    full_windows: bool = False
    """Whether to sample only complete windows inside an episode for dynamics fitting.

    Needs :attr:`num_warmup_steps` greater than :attr:`window_length`.
    """


@configclass
class RslRlSuperMimicAlgorithmCfg:
    """SHAC policy and value updates with optional learned dynamics."""

    class_name: str = "SHAC"
    """The algorithm class resolved by RSL-RL."""

    task_cfg: RslRlSuperMimicTaskCfg = MISSING
    """Differentiable task configured by the environment owner."""

    world_model_cfg: RslRlLearnedDynamicsCfg | None = RslRlLearnedDynamicsCfg()
    """Learned dynamics configuration for the SuperMimic recipe."""

    history_length: int = 5
    """Number of actor frames in the deployment history."""

    use_reference_action: bool = False
    """Whether the network output corrects the task's reference action, which exports then take as an input."""

    normalize_before_action: bool = True
    """Whether to update the observation and goal normalizers before choosing each collection action, not after."""

    updates_per_step: float = 0.4
    """Learner iterations accrued per collected environment step."""

    update_schedule: str = "after_warmup"
    """Accrue learner-update credit after warmup (``"after_warmup"``) or from the first collected step
    (``"from_start"``)."""

    num_policy_updates: int = 1
    """Actor updates and associated critic fits per learner iteration."""

    horizon: int = 32
    """Number of imagined policy steps."""

    policy_batch_size: int = 4096
    """Starts per actor update."""

    learning_rate: float = 2e-4
    """Initial learning rate of the world model and critic, and of the actor unless overridden."""

    actor_learning_rate: float | None = None
    """Initial actor learning rate. None uses :attr:`learning_rate`."""

    actor_learning_rate_multiplier: float = 1.0
    """Factor on the actor learning rate early in training, decaying linearly to 1 once a decay trigger is reached."""

    actor_learning_rate_decay_iterations: int = 300
    """Learner iterations over which the actor learning-rate multiplier decays to 1."""

    actor_learning_rate_decay_return: float | None = None
    """Mean return per maximum-length step of the episodes completed since the previous update that starts the decay.

    Each episode return is divided by the environment's maximum episode length, so short failed episodes count low.
    """

    actor_learning_rate_decay_start: int | None = None
    """Learner iteration at which the decay starts at the latest."""

    muon_models: tuple[str, ...] = ()
    """Models among ``"actor"`` and ``"world_model"`` whose hidden weight matrices are trained with Muon."""

    num_schedule_updates: int = 250_000
    """Learner iterations in the shared cosine learning-rate schedule."""

    final_learning_rate_ratio: float = 0.03
    """Final learning rate relative to its initial value."""

    learning_rate_schedule_unit: str = "learner_updates"
    """Measure cosine-schedule progress in learner updates (``"learner_updates"``) or in collected environment steps
    (``"env_steps"``)."""

    gamma: float = 0.99
    """Reward discount."""

    lam: float = 0.95
    """TD(lambda) mixing coefficient for value targets."""

    num_critic_epochs: int = 5
    """Epochs per critic fit."""

    num_critic_mini_batches: int = 4
    """Minibatches per critic epoch."""

    critic_survival_weighting: bool = True
    """Whether to weight value regression by imagined survival probabilities."""

    critic_bootstrap_clipping: bool = True
    """Whether to cap values with :attr:`max_step_reward` before computing critic return targets."""

    critic_shuffle_each_epoch: bool = True
    """Whether to draw new critic minibatches every epoch instead of reusing one permutation."""

    action_noise: float = 0.1
    """Standard deviation of additive collection noise."""

    actor_ema_halflife: int | None = None
    """Deployment-only actor exponential moving average (EMA) half-life in actor optimizer steps, e.g. 500.

    None disables it. Use the same setting for training and play/export via ``agent.algorithm.actor_ema_halflife=500``.
    """

    max_step_reward: float | None = None
    """Optional per-step reward bound used to cap values at the maximum discounted return."""

    use_mixed_precision: bool = False
    """Whether to use BF16 matrix operations during learning."""

    max_grad_norm: float = 1.0
    """Global gradient-norm limit for each model."""

    use_cuda_graph: bool = False
    """Whether to capture learner iterations and the collection calls ``act`` and ``process_env_step`` with CUDA graphs.

    Results match eager execution. Python values read inside the captured calls are fixed at capture time, for example
    the reward weights that the differentiable velocity task reads from the reward manager.
    """


@configclass
class RslRlSuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    """SHAC runner with physical successors on every step."""

    class_name: str = "OnPolicyRunner"
    """The runner class resolved by RSL-RL."""

    final_obs_every_step: bool = True
    """Capture physical observations before resets and interval events on every step."""

    separate_goal_terms: tuple[str, ...] = ()
    """Policy terms supplied to the goal encoder separately from the frame history."""

    actor: RslRlSuperMimicActorCfg = MISSING
    """History-conditioned deployment actor."""

    critic: RslRlMLPModelCfg = MISSING
    """Scalar value model consuming task-owned privileged features."""

    algorithm: RslRlSuperMimicAlgorithmCfg = MISSING
    """SuperMimic learning, task and replay configuration."""
