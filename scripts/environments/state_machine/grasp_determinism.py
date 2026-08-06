# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Is the Newton soft-body grasp deterministic? Squeeze at a fixed force, measure object width.

No state machine and no arm trajectory: the end-effector is commanded to a single
constant pose (the object's centre of mass, read once right after reset) and held
there for the whole trial. Only the gripper is actuated. The finger joint effort
limit is the applied grasp force -- with the default finger stiffness the position
error saturates the actuator, so the commanded effort *is* the squeeze force.

Each trial measures the deformable's width along the finger-closing axis (world y,
nominal 0.05 m) once the squeeze has settled, then resets and repeats. Identical
initial conditions across trials means any spread in the measured width is
non-determinism in the solver, not in the setup.

By default the environment's reset randomisation (robot joint scale 0.9-1.1,
deformable position +-0.05 m) is switched off so the trials are genuinely
identical. Pass ``--randomize`` to leave it on, which instead measures how much
the grasp varies across the task's natural spread of initial conditions.

.. code-block:: bash

    # solver determinism: identical initial conditions, 5 trials at 5 N
    isaaclab.bat -p scripts/environments/state_machine/grasp_determinism.py --trials 5 --grip_effort 5.0

    # spread under the task's own reset randomisation
    isaaclab.bat -p scripts/environments/state_machine/grasp_determinism.py --trials 5 --randomize
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Determinism of the Newton soft-body grasp at fixed force.")
parser.add_argument("--trials", type=int, default=5, help="Number of squeeze trials.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--grip_effort", type=float, default=5.0, help="Finger joint effort limit [N] = grasp force.")
parser.add_argument("--grip_stiffness", type=float, default=0.0, help="Finger stiffness override (0 = keep 1000).")
parser.add_argument("--pregrasp_steps", type=int, default=130, help="Steps spent above the object before descending.")
parser.add_argument("--pregrasp_z", type=float, default=0.1, help="Height above the object COM for the pre-grasp [m].")
parser.add_argument("--settle_steps", type=int, default=140, help="Steps holding the pose with the gripper open.")
parser.add_argument("--squeeze_steps", type=int, default=200, help="Steps with the gripper closing.")
parser.add_argument("--measure_steps", type=int, default=40, help="Trailing squeeze steps averaged for the width.")
parser.add_argument("--randomize", action="store_true", default=False, help="Keep the env's reset randomisation.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

import statistics

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Isaac-Lift-Soft-Franka-v0"

# gripper command in the last action slot (see lift_franka_soft.py GripperState)
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# spawned deformable is a 0.3 x 0.05 x 0.05 cuboid; fingers close along world y
NOMINAL_WIDTH_Y = 0.05


def object_width_y(nodal_pos: torch.Tensor) -> torch.Tensor:
    """Width of each deformable along the finger-closing axis [m], shape [num_envs].

    Whole-body extent. The bar is 0.3 m long in x, so this inflates sharply if the
    object yaws; :func:`grasp_thickness` is the metric that survives that.
    """
    y = nodal_pos[..., 1]
    return y.max(dim=-1).values - y.min(dim=-1).values


def grasp_thickness(nodal_pos: torch.Tensor, x_center: torch.Tensor, half_slab: float = 0.04) -> torch.Tensor:
    """Thickness in y of the material actually between the fingers [m], shape [num_envs].

    Restricts to nodes within ``half_slab`` of the grasp point in x, so a yaw of the
    bar does not masquerade as a change in squeezed width.
    """
    out = []
    for e in range(nodal_pos.shape[0]):
        sel = (nodal_pos[e, :, 0] - x_center[e]).abs() <= half_slab
        y = nodal_pos[e, sel, 1] if bool(sel.any()) else nodal_pos[e, :, 1]
        out.append(y.max() - y.min())
    return torch.stack(out)


def main():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=[])  # default preset = Newton (MJWarp rigid + VBD soft)
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    # keep a trial from being cut short by the 5 s episode timeout
    trial_steps = args_cli.pregrasp_steps + args_cli.settle_steps + args_cli.squeeze_steps
    cfg.episode_length_s = max(30.0, (trial_steps + 60) * cfg.sim.dt * cfg.decimation)

    # the grasp force: cap finger effort so the closing torque saturates at a known value
    hand = cfg.scene.robot.actuators["panda_hand"]
    if args_cli.grip_stiffness > 0.0:
        hand.stiffness = args_cli.grip_stiffness
    hand.effort_limit_sim = args_cli.grip_effort

    # freeze the reset randomisation so every trial starts from an identical state
    if not args_cli.randomize:
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_deformable.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

    print(
        f"[DET] backend=newton trials={args_cli.trials} num_envs={args_cli.num_envs} "
        f"grip_effort={args_cli.grip_effort} N stiffness={hand.stiffness} "
        f"randomize={args_cli.randomize} seed={args_cli.seed}",
        flush=True,
    )

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger.*")
    deformable = env.unwrapped.scene["deformable"]

    per_trial = []
    for trial in range(args_cli.trials):
        env.reset(seed=args_cli.seed)

        # single constant EE target: the object's centre of mass, sampled once.
        # nodal/root positions are world-frame but the IK action is env-local, so shift by the env origin.
        obj_com = deformable.data.root_pos_w.torch.clone()
        obj_com_local = obj_com - env.unwrapped.scene.env_origins
        actions = torch.zeros((env.unwrapped.num_envs, env.unwrapped.action_space.shape[-1]), device=device)
        actions[:, 0:3] = obj_com_local
        actions[:, 3] = 1.0  # identity quaternion (w, x, y, z)
        actions[:, -1] = GRIPPER_OPEN

        grasp_x = obj_com[:, 0].clone()  # world frame, matches nodal_pos_w
        start_width = object_width_y(deformable.data.nodal_pos_w.torch)[0].item()
        if trial == 0:
            print(
                f"[DET] nodal_pos shape={tuple(deformable.data.nodal_pos_w.torch.shape)} "
                f"ee_target_local=({obj_com_local[0, 0]:.6f}, {obj_com_local[0, 1]:.6f}, {obj_com_local[0, 2]:.6f}) "
                f"rest_width_y={start_width:.6f}",
                flush=True,
            )

        # phase 0 - approach above the object first, mirroring the state machine's +0.1 m offset,
        # so the arm does not sweep sideways through the bar on its way in
        actions[:, 2] = obj_com_local[:, 2] + args_cli.pregrasp_z
        for _ in range(args_cli.pregrasp_steps):
            env.step(actions)

        # phase 1 - descend onto the grasp pose, gripper still open
        actions[:, 2] = obj_com_local[:, 2]
        for _ in range(args_cli.settle_steps):
            env.step(actions)

        pre_squeeze_thick = float(grasp_thickness(deformable.data.nodal_pos_w.torch, grasp_x)[0].item())

        # phase 2 - same pose, close the gripper at the capped force
        actions[:, -1] = GRIPPER_CLOSE
        thicks, widths, gaps, efforts = [], [], [], []
        measure_from = args_cli.squeeze_steps - args_cli.measure_steps
        for k in range(args_cli.squeeze_steps):
            env.step(actions)
            if k >= measure_from:
                nodal = deformable.data.nodal_pos_w.torch
                thicks.append(grasp_thickness(nodal, grasp_x).clone())
                widths.append(object_width_y(nodal).clone())
                gaps.append(float(robot.data.joint_pos[0, finger_ids].sum().item()))
                try:
                    efforts.append(float(robot.data.applied_torque[0, finger_ids].abs().max().item()))
                except Exception:
                    pass

        t = torch.stack(thicks)  # [measure_steps, num_envs]
        w = torch.stack(widths)
        final_t = float(t[-1, 0].item())
        mean_t = float(t[:, 0].mean().item())
        drift = float((t[-1, 0] - t[0, 0]).abs().item())  # still settling?
        spread_envs = float((t[-1].max() - t[-1].min()).item()) if args_cli.num_envs > 1 else 0.0
        eff = max(efforts) if efforts else float("nan")

        per_trial.append((final_t, mean_t, gaps[-1], eff))
        print(
            f"[DET] trial={trial} thickness_final={final_t:.9f} thickness_mean={mean_t:.9f} "
            f"pre_squeeze={pre_squeeze_thick:.6f} body_width_y={float(w[-1, 0].item()):.6f} "
            f"settling_drift={drift:.2e} finger_gap={gaps[-1]:.6f} max_finger_effort={eff:.3f} "
            f"across_env_spread={spread_envs:.2e}",
            flush=True,
        )

    finals = [t[0] for t in per_trial]
    spread = max(finals) - min(finals)
    mean = statistics.fmean(finals)
    stdev = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    identical = all(f == finals[0] for f in finals)
    compression = NOMINAL_WIDTH_Y - mean

    print(
        f"[DET] SUMMARY trials={len(finals)} grip_effort={args_cli.grip_effort} randomize={args_cli.randomize} "
        f"width_mean={mean:.9f} width_std={stdev:.3e} width_spread={spread:.3e} "
        f"spread_um={spread * 1e6:.3f} compression_mm={compression * 1e3:.4f} "
        f"bitwise_identical={identical}",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
