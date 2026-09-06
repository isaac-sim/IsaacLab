# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env  # for JAX

    env = wrap_env(env, wrapper="isaaclab")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Literal

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectRLEnv,
        ManagerBasedRLEnv,
    )


@configclass
class SkrlNetworkCfg:
    """Configuration for one network container in a skrl model."""

    name: str = "net"
    """Name of the network container. Defaults to ``"net"``."""

    input: str = "OBSERVATIONS"
    """Input expression consumed by the network container."""

    layers: list[int | str | dict[str, Any]] = MISSING
    """Layer definitions processed by the skrl model instantiator."""

    activations: str | list[str] | dict[str, Any] = "elu"
    """Activation function or functions. Defaults to ``"elu"``."""


@configclass
class SkrlModelCfg:
    """Base configuration for a model instantiated by the skrl runner."""

    class_name: str = MISSING
    """Name of the skrl model mixin used to instantiate the model."""

    network: list[SkrlNetworkCfg] = MISSING
    """Network containers used to build the model."""

    output: str | list[str] = MISSING
    """Output expression produced by the model."""

    clip_actions: bool = False
    """Whether to clip actions to the action space. Defaults to False."""


@configclass
class SkrlGaussianModelCfg(SkrlModelCfg):
    """Configuration for a Gaussian skrl model."""

    class_name: str = "GaussianMixin"
    """Name of the skrl model mixin. Defaults to ``"GaussianMixin"``."""

    output: str = "ACTIONS"
    """Output expression produced by the model. Defaults to ``"ACTIONS"``."""

    clip_log_std: bool = True
    """Whether to clip the log standard deviation. Defaults to True."""

    min_log_std: float = -20.0
    """Minimum log standard deviation. Defaults to -20.0."""

    max_log_std: float = 2.0
    """Maximum log standard deviation. Defaults to 2.0."""

    initial_log_std: float = 0.0
    """Initial log standard deviation. Defaults to 0.0."""

    clip_mean_actions: bool = False
    """Whether to clip mean actions to the action space. Defaults to False."""

    reduction: Literal["mean", "sum", "prod", "none"] = "sum"
    """Action log-probability reduction. Defaults to ``"sum"``."""

    fixed_log_std: bool = False
    """Whether the log standard deviation is non-trainable. Defaults to False."""


@configclass
class SkrlDeterministicModelCfg(SkrlModelCfg):
    """Configuration for a deterministic skrl model."""

    class_name: str = "DeterministicMixin"
    """Name of the skrl model mixin. Defaults to ``"DeterministicMixin"``."""

    output: str = "ONE"
    """Output expression produced by the model. Defaults to ``"ONE"``."""


@configclass
class SkrlModelsCfg:
    """Configuration for the models instantiated by the skrl runner."""

    policy: SkrlModelCfg = MISSING
    """Policy model configuration."""

    value: SkrlModelCfg = MISSING
    """Value model configuration."""

    separate: bool = False
    """Whether to instantiate separate policy and value models. Defaults to False."""


@configclass
class SkrlMemoryCfg:
    """Configuration for the skrl rollout memory."""

    class_name: str = "RandomMemory"
    """Name of the skrl memory class. Defaults to ``"RandomMemory"``."""

    memory_size: int = -1
    """Memory size. Defaults to -1, which uses the agent rollout length."""

    export: bool = False
    """Whether to export memory data. Defaults to False."""

    export_format: Literal["pt", "npz", "csv"] = "pt"
    """Memory export format. Defaults to ``"pt"``."""

    export_directory: str = ""
    """Memory export directory. Defaults to an empty string."""

    replacement: bool = False
    """Whether random sampling uses replacement. Defaults to False."""


@configclass
class SkrlExperimentCfg:
    """Configuration for skrl logging and checkpoints."""

    directory: str = MISSING
    """Directory below ``logs/skrl`` used for the experiment."""

    experiment_name: str = ""
    """Experiment name appended to the run directory. Defaults to an empty string."""

    write_interval: int | Literal["auto"] = "auto"
    """Metric write interval. Defaults to ``"auto"``."""

    checkpoint_interval: int | Literal["auto"] = "auto"
    """Checkpoint interval. Defaults to ``"auto"``."""

    store_separately: bool = False
    """Whether to store module checkpoints separately. Defaults to False."""

    wandb: bool = False
    """Whether to log the experiment with Weights & Biases. Defaults to False."""

    wandb_kwargs: dict[str, Any] = {}
    """Arguments passed to Weights & Biases. Defaults to an empty dictionary."""


@configclass
class SkrlPpoAgentCfg:
    """Configuration shared by skrl PPO-family agents."""

    class_name: Literal["PPO", "IPPO", "MAPPO"] = "PPO"
    """Name of the skrl agent class. Defaults to ``"PPO"``."""

    rollouts: int = MISSING
    """Number of environment steps collected per rollout."""

    learning_epochs: int = MISSING
    """Number of learning epochs per update."""

    mini_batches: int = MISSING
    """Number of mini-batches per learning epoch."""

    discount_factor: float = 0.99
    """Reward discount factor. Defaults to 0.99."""

    gae_lambda: float = 0.95
    """Generalized advantage estimation lambda. Defaults to 0.95."""

    learning_rate: float | tuple[float, float] = MISSING
    """Optimizer learning rate."""

    learning_rate_scheduler: str | None = "KLAdaptiveLR"
    """Learning-rate scheduler name. Defaults to ``"KLAdaptiveLR"``."""

    learning_rate_scheduler_kwargs: dict[str, Any] | None = {"kl_threshold": 0.008}
    """Arguments passed to the learning-rate scheduler."""

    observation_preprocessor: str | None = None
    """Observation preprocessor name. Defaults to None."""

    observation_preprocessor_kwargs: dict[str, Any] | None = None
    """Arguments passed to the observation preprocessor. Defaults to None."""

    state_preprocessor: str | None = None
    """State preprocessor name. Defaults to None."""

    state_preprocessor_kwargs: dict[str, Any] | None = None
    """Arguments passed to the state preprocessor. Defaults to None."""

    value_preprocessor: str | None = None
    """Value preprocessor name. Defaults to None."""

    value_preprocessor_kwargs: dict[str, Any] | None = None
    """Arguments passed to the value preprocessor. Defaults to None."""

    random_timesteps: int = 0
    """Number of random interaction steps. Defaults to 0."""

    learning_starts: int = 0
    """Timestep at which learning starts. Defaults to 0."""

    grad_norm_clip: float = 1.0
    """Maximum gradient norm. Defaults to 1.0."""

    ratio_clip: float = 0.2
    """PPO probability-ratio clipping value. Defaults to 0.2."""

    value_clip: float = 0.2
    """PPO value clipping value. Defaults to 0.2."""

    entropy_loss_scale: float = 0.0
    """Entropy-loss scale. Defaults to 0.0."""

    value_loss_scale: float = 2.0
    """Value-loss scale. Defaults to 2.0."""

    kl_threshold: float = 0.0
    """KL-divergence early-stop threshold. Defaults to 0.0."""

    rewards_shaper_scale: float | None = 1.0
    """Scale applied to rewards by the runner. Defaults to 1.0."""

    time_limit_bootstrap: bool = False
    """Whether to bootstrap truncated episodes. Defaults to False."""

    mixed_precision: bool = False
    """Whether to use automatic mixed precision. Defaults to False."""

    experiment: SkrlExperimentCfg = MISSING
    """Logging and checkpoint configuration."""


@configclass
class SkrlIppoAgentCfg(SkrlPpoAgentCfg):
    """Configuration for a skrl IPPO agent."""

    class_name: Literal["IPPO"] = "IPPO"
    """Name of the skrl agent class. Defaults to ``"IPPO"``."""


@configclass
class SkrlMappoAgentCfg(SkrlPpoAgentCfg):
    """Configuration for a skrl MAPPO agent."""

    class_name: Literal["MAPPO"] = "MAPPO"
    """Name of the skrl agent class. Defaults to ``"MAPPO"``."""


@configclass
class SkrlTrainerCfg:
    """Configuration for the skrl trainer."""

    class_name: str = "SequentialTrainer"
    """Name of the skrl trainer class. Defaults to ``"SequentialTrainer"``."""

    timesteps: int = MISSING
    """Total number of environment interaction steps."""

    headless: bool = False
    """Whether to disable environment rendering. Defaults to False."""

    render_interval: int = 1
    """Rendering interval in environment steps. Defaults to 1."""

    disable_progressbar: bool | None = False
    """Whether to disable the training progress bar. Defaults to False."""

    environment_info: str = "log"
    """Environment information handling mode. Defaults to ``"log"``."""

    close_environment_at_exit: bool = True
    """Whether the trainer closes the environment at exit. Defaults to True."""

    stochastic_evaluation: bool = False
    """Whether evaluation samples stochastic policy actions. Defaults to False."""


@configclass
class SkrlRunnerCfg:
    """Top-level configuration consumed by the skrl runner."""

    models: SkrlModelsCfg = MISSING
    """Model configurations."""

    agent: SkrlPpoAgentCfg = MISSING
    """Agent configuration."""

    trainer: SkrlTrainerCfg = MISSING
    """Trainer configuration."""

    seed: int = 42
    """Random seed for the experiment. Defaults to 42."""

    memory: SkrlMemoryCfg = SkrlMemoryCfg()
    """Rollout-memory configuration."""

    def to_runner_dict(self) -> dict[str, Any]:
        """Return the nested dictionary expected by :class:`skrl.utils.runner.Runner`."""
        return _replace_skrl_class_names(self.to_dict())


def _replace_skrl_class_names(value: Any) -> Any:
    """Replace Python-safe ``class_name`` fields with skrl's ``class`` keys."""
    if isinstance(value, dict):
        return {
            ("class" if key == "class_name" else key): _replace_skrl_class_names(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_skrl_class_names(item) for item in value]
    return value


def skrl_cfg_to_dict(cfg: SkrlRunnerCfg | dict[str, Any]) -> dict[str, Any]:
    """Convert an Isaac Lab skrl config class to the dictionary expected by skrl.

    Plain dictionaries are returned unchanged to preserve compatibility with YAML
    configuration entry points.
    """
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, SkrlRunnerCfg):
        return cfg.to_runner_dict()
    raise TypeError(f"Expected a SkrlRunnerCfg or dict, received: {type(cfg).__name__}")


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(
    env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
    ml_framework: Literal["torch", "jax", "warp"] = "torch",
    wrapper: Literal["auto", "isaaclab", "isaaclab-single-agent", "isaaclab-multi-agent"] = "isaaclab",
):
    """Wraps around Isaac Lab environment for skrl.

    This function wraps around the Isaac Lab environment. Since the wrapping
    functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.
        ml_framework: The ML framework to use for the wrapper. Defaults to "torch".
        wrapper: The wrapper to use. Defaults to "isaaclab": leave it to skrl to determine if the environment
            will be wrapped as single-agent or multi-agent.

    Raises:
        ValueError: When the environment is not an instance of any Isaac Lab environment interface.
        ValueError: If the specified ML framework is not valid.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """
    # check that input is valid
    # NOTE: import here (not at module level) to avoid loading heavy env classes before Isaac Sim is initialized.
    from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

    try:
        from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedRLEnvWarp
    except ImportError:
        DirectRLEnvWarp = None
        ManagerBasedRLEnvWarp = None

    allowed_types = (ManagerBasedRLEnv, DirectRLEnv, DirectMARLEnv)
    if DirectRLEnvWarp is not None:
        allowed_types += (DirectRLEnvWarp,)
    if ManagerBasedRLEnvWarp is not None:
        allowed_types += (ManagerBasedRLEnvWarp,)

    if not isinstance(env.unwrapped, allowed_types):
        raise ValueError(
            "The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv, DirectMARLEnv,"
            f" DirectRLEnvWarp or ManagerBasedRLEnvWarp. Environment type: {type(env)}"
        )

    # import statements according to the ML framework
    if ml_framework.startswith("torch"):
        from skrl.envs.wrappers.torch import wrap_env
    elif ml_framework.startswith("jax"):
        # preload submodule that skrl's distributed models use without importing (broken on recent JAX)
        import jax.experimental.multihost_utils  # noqa: F401
        from skrl.envs.wrappers.jax import wrap_env
    elif ml_framework.startswith("warp"):
        from skrl.envs.wrappers.warp import wrap_env
    else:
        raise ValueError(
            f"Invalid ML framework for skrl: {ml_framework}. Available options are: 'torch', 'jax', 'warp'"
        )

    # wrap and return the environment
    return wrap_env(env, wrapper)
