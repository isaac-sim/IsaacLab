# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a PPO actor with the contract a deployment needs, including what it cannot be given.

The counterpart of :mod:`export_depth_student` for a policy trained directly rather than distilled.
The reason it exists is the second half of the contract: a rough-terrain PPO actor on this task reads
328 values, of which ``base_lin_vel`` (3) and ``height_scan`` (187) are **privileged** -- the first
needs a base-velocity estimator the robot does not have, the second a terrain map it cannot see.
Exporting such a policy without saying so produces a file that loads, runs, and is unusable.

So the contract carries a ``deployable`` block naming every observation term the robot cannot
produce, and ``deployable: false`` when that list is non-empty. ``g1_deploy``'s ``core.py`` builds
exactly ``ang_vel + gravity + command + joint_pos + joint_vel + actions``; anything else has to be
either distilled away or synthesized, and which of the two is a decision, not a detail.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True, help="Task the policy was trained on.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to export.")
parser.add_argument("--out_dir", type=str, required=True, help="Directory for policy.pt and the contract.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

ROBOT_OBSERVABLE = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
"""Observation terms ``g1_deploy``'s ``core.observe`` can build from ``rt/lowstate`` and the command.

``base_ang_vel`` and ``projected_gravity`` come from the IMU, the joint terms from the motor states,
``actions`` from the previous step. Everything else -- base linear velocity, height scans, contact
states, anything privileged -- has no source on the robot.
"""


def _per_joint(value, name: str) -> float:
    """Resolve an actuator gain that may be a scalar or a regex-keyed dict.

    Args:
        value: Gain as configured -- a number, or a mapping from joint-name pattern to number.
        name: Joint the gain is wanted for.

    Returns:
        The gain for ``name``.
    """
    if not isinstance(value, dict):
        return value
    for pattern, gain in value.items():
        if re.fullmatch(pattern, name):
            return gain
    raise KeyError(f"no gain matches {name!r} in {sorted(value)}")


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 2
    env_raw = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env_raw, clip_actions=getattr(agent_cfg, "clip_actions", None))

    import importlib.metadata as metadata  # noqa: PLC0415

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)

    actor = runner.alg.actor
    actor.eval()
    exported = actor.as_jit().eval()

    u = env_raw.unwrapped
    groups = agent_cfg.obs_groups["actor"] if "actor" in agent_cfg.obs_groups else agent_cfg.obs_groups["policy"]
    dim = sum(int(u.observation_manager.group_obs_dim[g][0]) for g in groups)

    obs = torch.randn(4, dim, device=agent_cfg.device)
    with torch.inference_mode():
        got = exported(obs)
        from tensordict import TensorDict  # noqa: PLC0415

        want = actor.forward(TensorDict({g: obs for g in groups}, batch_size=[4]), stochastic_output=False)
    err = (got - want).abs().max().item()
    print(f"[export] max |exported - model| = {err:.3e} over {tuple(got.shape)} outputs")
    if err > 1e-5:
        raise RuntimeError(f"exported policy disagrees with the model by {err:.3e}; refusing to write it")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    policy_path = os.path.join(args_cli.out_dir, "policy.pt")
    torch.jit.save(torch.jit.script(exported.to("cpu")), policy_path)
    print(f"[export] wrote {policy_path}")

    robot = u.scene["robot"]
    n = len(robot.joint_names)
    stiffness, damping = [0.0] * n, [0.0] * n
    for group in env_cfg.scene.robot.actuators.values():
        ids, names = robot.find_joints(group.joint_names_expr)
        for i, name in zip(ids, names):
            stiffness[i] = float(_per_joint(group.stiffness, name))
            damping[i] = float(_per_joint(group.damping, name))

    terms = [
        {"name": name, "dim": int(shape[0]), "on_robot": name in ROBOT_OBSERVABLE}
        for name, shape in zip(
            u.observation_manager.active_terms["policy"], u.observation_manager.group_obs_term_dim["policy"]
        )
    ]
    missing = [t for t in terms if not t["on_robot"]]
    contract = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "stiffness": stiffness,
        "damping": damping,
        "action_dim": int(u.action_manager.total_action_dim),
        "action_scale": env_cfg.actions.joint_pos.scale,
        "use_default_offset": bool(env_cfg.actions.joint_pos.use_default_offset),
        "control_dt": float(env_cfg.sim.dt * env_cfg.decimation),
        "history_length": int(getattr(env_cfg.observations.policy.joint_pos, "history_length", None) or 1),
        "obs_dim": dim,
        "obs_terms": terms,
        "deployable": not missing,
        "unobservable_terms": [{"name": t["name"], "dim": t["dim"]} for t in missing],
        "unobservable_dim": sum(t["dim"] for t in missing),
    }
    contract_path = os.path.join(args_cli.out_dir, "contract.json")
    with open(contract_path, "w") as handle:
        json.dump(contract, handle, indent=2)
    print(f"[export] wrote {contract_path}")
    if missing:
        names = ", ".join(f"{t['name']} ({t['dim']})" for t in missing)
        print(f"[export] NOT DEPLOYABLE: {contract['unobservable_dim']} of {dim} inputs have no source on the robot: {names}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
