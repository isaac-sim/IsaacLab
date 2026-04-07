# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run hardcoded / deterministic inference with an RSL-RL agent.

This is a variant of ``play.py`` designed for the hardcoded-inference
environment where all randomization is disabled.  The IK grasp-pose solver
is still active so the robot properly grasps the gear; with every random
input zeroed out the result is identical on every reset.

Features on top of the standard play loop:

1. **Observation overrides** – hardcode the shaft-position and/or shaft-
   quaternion portions of the observation tensor so the policy always sees
   fixed values regardless of the sim state.  Edit the constants in the
   *OBSERVATION OVERRIDES* section below.

2. **Debug prints** – the initial observation is printed broken down by
   component so you can verify the values the policy will see.

3. **Old checkpoint support** – automatically converts pre-v5 rsl-rl
   checkpoints (single ``model_state_dict``) to the v5 split format.

The gear type, gear base pose, and gear height are configured in the
companion env-config file::

    .../gear_assembly/config/rizon_4s/hardcoded_inference_env_cfg.py

Usage::

    python scripts/reinforcement_learning/rsl_rl/play_hardcoded.py \\
        --num_envs 1 \\
        --checkpoint logs/rsl_rl/gear_assembly/2026-03-13_16-28-11/model_500.pt \\
        --visualizer kit
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVATION OVERRIDES                                                      ║
# ║  Set a value to inject a fixed observation every step, or leave as None     ║
# ║  to use the value computed from the simulation.                             ║
# ║                                                                             ║
# ║  Obs layout: [joint_pos(7) | joint_vel(7) | shaft_pos(3) | shaft_quat(4)]  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

OVERRIDE_SHAFT_POS = None   # e.g. [0.481, -0.073, -0.005]
OVERRIDE_SHAFT_QUAT = None  # e.g. [0.0, 0.0, 0.70711, -0.70711]

_SHAFT_POS_SLICE = slice(14, 17)
_SHAFT_QUAT_SLICE = slice(17, 21)


# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run hardcoded deterministic inference with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Deploy-GearAssembly-Rizon4s-Grav-Hardcoded-Inference-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


# -- old checkpoint conversion -----------------------------------------------
_OLD_TO_NEW_KEY_MAP = {
    # Actor MLP
    "actor.": "mlp.",
    # Critic MLP
    "critic.": "mlp.",
    # Actor RNN
    "memory_a.rnn.": "rnn.rnn.",
    # Critic RNN
    "memory_c.rnn.": "rnn.rnn.",
    # Actor obs normalizer
    "actor_obs_normalizer.": "obs_normalizer.",
    # Critic obs normalizer
    "critic_obs_normalizer.": "obs_normalizer.",
}

_ACTOR_PREFIXES = ("actor.", "memory_a.", "actor_obs_normalizer.")
_CRITIC_PREFIXES = ("critic.", "memory_c.", "critic_obs_normalizer.")


def _convert_old_checkpoint(loaded_dict: dict) -> dict:
    """Convert a pre-v5 rsl-rl checkpoint to the v5 format.

    Old format:  single ``model_state_dict`` with prefixes like ``actor.*``,
    ``memory_a.*``, ``actor_obs_normalizer.*``, etc.

    New format:  separate ``actor_state_dict`` and ``critic_state_dict`` with
    prefixes ``mlp.*``, ``rnn.rnn.*``, ``obs_normalizer.*``.
    """
    old_sd = loaded_dict["model_state_dict"]
    actor_sd: dict[str, torch.Tensor] = {}
    critic_sd: dict[str, torch.Tensor] = {}

    for old_key, tensor in old_sd.items():
        is_actor = old_key.startswith(_ACTOR_PREFIXES)
        is_critic = old_key.startswith(_CRITIC_PREFIXES)

        new_key = old_key
        for old_prefix, new_prefix in _OLD_TO_NEW_KEY_MAP.items():
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix):]
                break

        if is_actor:
            actor_sd[new_key] = tensor
        elif is_critic:
            critic_sd[new_key] = tensor

    print(f"[INFO] Converted old checkpoint: {len(actor_sd)} actor keys, {len(critic_sd)} critic keys")

    return {
        "actor_state_dict": actor_sd,
        "critic_state_dict": critic_sd,
        "optimizer_state_dict": loaded_dict.get("optimizer_state_dict", {}),
        "iter": loaded_dict.get("iter", 0),
        "infos": loaded_dict.get("infos"),
    }


def _load_checkpoint(runner, resume_path: str):
    """Load a checkpoint, automatically converting old (pre-v5) formats."""
    loaded_dict = torch.load(resume_path, weights_only=False)
    if "model_state_dict" in loaded_dict and "actor_state_dict" not in loaded_dict:
        print("[INFO] Detected old (pre-v5) rsl-rl checkpoint format — converting …")
        loaded_dict = _convert_old_checkpoint(loaded_dict)
    load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True}
    runner.alg.load(loaded_dict, load_cfg, strict=True)
    if load_cfg.get("iteration") and "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]


def _get_policy_tensor(obs):
    """Extract the flat policy tensor from obs (TensorDict or plain Tensor)."""
    if isinstance(obs, torch.Tensor):
        return obs
    # TensorDict — the wrapper stores the concatenated obs under the "policy" key
    return obs["policy"]


def _apply_obs_overrides(obs):
    """Optionally replace shaft_pos / shaft_quat in the observation."""
    t = _get_policy_tensor(obs)
    if OVERRIDE_SHAFT_POS is not None:
        t[:, _SHAFT_POS_SLICE] = torch.tensor(OVERRIDE_SHAFT_POS, device=t.device, dtype=t.dtype)
    if OVERRIDE_SHAFT_QUAT is not None:
        t[:, _SHAFT_QUAT_SLICE] = torch.tensor(OVERRIDE_SHAFT_QUAT, device=t.device, dtype=t.dtype)
    return obs


def _print_obs(obs, label: str = "Observation"):
    """Print observation components for the first environment."""
    t = _get_policy_tensor(obs)
    o = t[0]
    print(f"\n[{label}]")
    print(f"  joint_pos  : {o[0:7].tolist()}")
    print(f"  joint_vel  : {o[7:14].tolist()}")
    print(f"  shaft_pos  : {o[14:17].tolist()}")
    print(f"  shaft_quat : {o[17:21].tolist()}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run hardcoded inference with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # -- configure -------------------------------------------------------
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # -- resolve checkpoint ----------------------------------------------
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # -- create environment ----------------------------------------------
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # -- load policy -----------------------------------------------------
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        _load_checkpoint(runner, resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if version.parse(installed_version) >= version.parse("5.0.0"):
            policy_nn = runner.alg.actor
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        dt = env.unwrapped.step_dt

        # -- initial observation ---------------------------------------------
        obs = env.get_observations()
        obs = _apply_obs_overrides(obs)
        _print_obs(obs, label="Initial observation (after overrides)")

        # -- inference loop --------------------------------------------------
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    obs = _apply_obs_overrides(obs)

                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
