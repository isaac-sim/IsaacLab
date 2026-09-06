# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how evenly a biped policy uses its two feet.

``feet_air_time_positive_biped`` rewards the duration of the current single-stance phase, taking
the minimum over the two feet and clamping at its threshold. Nothing in it distinguishes left from
right or asks the feet to alternate, so a policy that keeps one foot planted and swings the other
scores at the clamp -- higher than a brisk symmetric gait, whose timers reset at every touchdown.
Raising the weight therefore rewards long single-stance phases, and a limp is the limit of that.

This rolls a checkpoint out under the evaluation conditions and reports, per foot, the mean air and
contact time and the share of steps spent airborne, so the impression "it limps" becomes a number.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True, help="Task the checkpoint was trained on.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to measure.")
parser.add_argument("--steps", type=int, default=1500, help="Control steps to roll out.")
parser.add_argument("--num_envs", type=int, default=64, help="Environments run in parallel.")
parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed; must differ from training's.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402


def _t(x):
    """Return a torch tensor for either a tensor or a backend ProxyArray."""
    return x.torch if hasattr(x, "torch") else x


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "curriculum", None) is not None:
        for term in [t for t in vars(env_cfg.curriculum) if not t.startswith("_")]:
            setattr(env_cfg.curriculum, term, None)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    import importlib.metadata as metadata  # noqa: PLC0415

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    sensor = env.unwrapped.scene.sensors["contact_forces"]
    feet = [i for i, n in enumerate(sensor.body_names) if n.endswith("ankle_roll_link")]
    names = [sensor.body_names[i] for i in feet]
    if len(feet) != 2:
        raise RuntimeError(f"expected two feet, matched {names}")

    device = env.unwrapped.device
    air_sum = torch.zeros(2, device=device)
    contact_sum = torch.zeros(2, device=device)
    airborne = torch.zeros(2, device=device)
    single = torch.zeros((), device=device)
    n = 0

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            stepped = env.step(policy(obs))
            obs = stepped[0]
            air = _t(sensor.data.current_air_time)[:, feet]
            contact = _t(sensor.data.current_contact_time)[:, feet]
            in_contact = contact > 0.0
            air_sum += air.mean(dim=0)
            contact_sum += contact.mean(dim=0)
            airborne += (~in_contact).float().mean(dim=0)
            single += (in_contact.sum(dim=1) == 1).float().mean()
            n += 1

    print(f"[symmetry] {args_cli.checkpoint}")
    print(f"[symmetry] steps {n}, single-stance share {single.item() / n:.3f}")
    for k, name in enumerate(names):
        print(
            f"[symmetry]   {name:24s} air {air_sum[k].item() / n:.4f}s  "
            f"contact {contact_sum[k].item() / n:.4f}s  airborne {airborne[k].item() / n:.3f}"
        )
    ratio = airborne[0].item() / max(airborne[1].item(), 1e-6)
    print(f"[symmetry] airborne-share ratio {names[0]}/{names[1]} = {ratio:.3f}  (1.0 is symmetric)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
