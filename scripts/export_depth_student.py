# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a depth-camera distillation student for deployment, and check the export reproduces it.

``export_policy_as_jit`` cannot be used here. It wraps the actor in a module whose ``forward`` takes
one flat tensor, which is true of an MLP policy and false of this one: the student reads a 690-value
proprioception group *and* a ``(3, 38, 64)`` depth group, and ``rsl_rl`` routes the second through a
convolutional encoder. The model's own :meth:`as_jit` produces the right shape --
``forward(obs_1d, obs_2d: list[Tensor])`` -- so that is what is exported.

The export is then run against the live model on the same random inputs and the outputs compared. A
silently wrong export is the expensive failure here: it loads, it runs, it produces plausible joint
targets, and the robot walks differently from the policy that was trained.

Alongside the policy it writes a contract JSON -- joint order, default pose, gains, observation
layout, camera intrinsics and extrinsics -- because a deployment that guesses any of those produces
a correctly shaped vector with the wrong meaning.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True, help="Task the student was trained on.")
parser.add_argument("--checkpoint", type=str, required=True, help="Student checkpoint to export.")
parser.add_argument("--out_dir", type=str, required=True, help="Directory for policy.pt and the contract.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 2
    env_raw = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env_raw, clip_actions=getattr(agent_cfg, "clip_actions", None))

    import importlib.metadata as metadata  # noqa: PLC0415

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)

    student = runner.alg.student
    student.eval()
    exported = student.as_jit().eval()

    u = env_raw.unwrapped
    groups = agent_cfg.obs_groups["student"]
    dim_1d = sum(
        int(u.observation_manager.group_obs_dim[g][0])
        for g in groups
        if len(u.observation_manager.group_obs_dim[g]) == 1
    )
    shapes_2d = [
        tuple(u.observation_manager.group_obs_dim[g])
        for g in groups
        if len(u.observation_manager.group_obs_dim[g]) == 3
    ]

    device = agent_cfg.device
    obs_1d = torch.randn(4, dim_1d, device=device)
    obs_2d = [torch.rand(4, *s, device=device) for s in shapes_2d]

    with torch.inference_mode():
        got = exported(obs_1d, obs_2d)
        # The live model consumes a TensorDict of named groups; build one from the same tensors so
        # the comparison is of the export, not of two different inputs.
        from tensordict import TensorDict  # noqa: PLC0415

        entries, it_2d = {}, iter(obs_2d)
        for g in groups:
            shape = u.observation_manager.group_obs_dim[g]
            entries[g] = next(it_2d) if len(shape) == 3 else obs_1d
        # forward with stochastic_output=False is the deterministic path the export reproduces.
        want = student.forward(TensorDict(entries, batch_size=[4]), stochastic_output=False)

    err = (got - want).abs().max().item()
    print(f"[export] max |exported - model| = {err:.3e} over {got.shape} outputs")
    if err > 1e-5:
        raise RuntimeError(f"exported policy disagrees with the model by {err:.3e}; refusing to write it")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    policy_path = os.path.join(args_cli.out_dir, "policy.pt")
    torch.jit.save(torch.jit.script(exported.to("cpu")), policy_path)
    print(f"[export] wrote {policy_path}")

    robot = u.scene["robot"]
    actuators = env_cfg.scene.robot.actuators
    n = len(robot.joint_names)
    stiffness = [0.0] * n
    damping = [0.0] * n
    for group in actuators.values():
        ids, names = robot.find_joints(group.joint_names_expr)
        for i, name in zip(ids, names):
            stiffness[i] = float(_per_joint(group.stiffness, name))
            damping[i] = float(_per_joint(group.damping, name))

    cam = env_cfg.scene.depth_camera
    contract = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "stiffness": stiffness,
        "damping": damping,
        "action_dim": int(u.action_manager.total_action_dim),
        "action_scale": float(env_cfg.actions.joint_pos.scale),
        "use_default_offset": bool(env_cfg.actions.joint_pos.use_default_offset),
        "control_dt": float(env_cfg.sim.dt * env_cfg.decimation),
        "history_length": int(env_cfg.observations.policy.joint_pos.history_length or 1),
        "obs_1d_dim": dim_1d,
        "obs_1d_terms": [
            {"name": name, "dim": int(dim[0])}
            for name, dim in zip(
                u.observation_manager.active_terms["policy"], u.observation_manager.group_obs_term_dim["policy"]
            )
        ],
        "depth_shape": list(shapes_2d[0]) if shapes_2d else None,
        "depth_max_range_m": float(env_cfg.observations.depth.image.clip[1]),
        "depth_frame_order": "oldest first along the channel axis",
        "camera": {
            "prim_path": cam.prim_path,
            "offset_pos_m": list(cam.offset.pos),
            "offset_rot_wxyz": list(cam.offset.rot),
            "convention": cam.offset.convention,
            "width": cam.width,
            "height": cam.height,
            "focal_length_mm": cam.spawn.focal_length,
            "horizontal_aperture_mm": cam.spawn.horizontal_aperture,
            "clipping_range_m": list(cam.spawn.clipping_range),
        },
    }
    contract_path = os.path.join(args_cli.out_dir, "contract.json")
    with open(contract_path, "w") as handle:
        json.dump(contract, handle, indent=2)
    print(f"[export] wrote {contract_path}")
    env.close()


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
    import re  # noqa: PLC0415

    for pattern, gain in value.items():
        if re.fullmatch(pattern, name):
            return gain
    raise KeyError(f"no gain matches {name!r} in {sorted(value)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
