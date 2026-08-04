"""
Replay a pre-exported joint-space trajectory (see export_lerobot_episode.py in the
lerobot_openarm repo) inside Isaac Sim, by driving the robot's joints directly -- bypassing the
env's IK action term entirely, exactly the same mechanism eval_smolvla_jointspace.py's
_step_direct() uses for a live policy, just driven here by a fixed pre-recorded array instead.

WHY THIS EXISTS: use this to sanity-check the TRAINING DATA itself, independent of any trained
policy or real-hardware calibration. The dataset's recorded trajectory came FROM this same
simulator in the first place (via IsaacLab's convert_hdf5_to_lerobot.py, derived from
states/articulation/robot/joint_position) -- so replaying it back through the identical
physics/robot/PD-gain setup should reproduce what actually happened when it was captured. If a
recorded episode visibly collides with the pad, never reaches the cube, or the success check
never fires even on a recorded (supposedly successful) demo, that's evidence of a data-quality
issue in the recorded/generated dataset itself -- separate from anything about model training or
real-hardware calibration, and worth knowing before spending more time debugging either of those.

Isaac Sim's own Python environment can't directly import the `lerobot` package (see
eval_smolvla.py/smolvla_server.py's docstrings -- numpy 1.x vs 2.x incompatibility between these
two environments), so this script never touches a LeRobotDataset object at all -- it just loads
a plain .npy array that export_lerobot_episode.py already extracted in an environment that does
have `lerobot` installed.

Usage:
  Step 1 (lerobot_openarm venv, has `lerobot` installed):
    cd ~/Stanley_ws/lerobot_openarm && source .venv/bin/activate
    python export_lerobot_episode.py \\
        --repo-id ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints \\
        --episode 0 --output /tmp/episode0_actions.npy

  Step 2 (Isaac Sim):
    cd ~/Stanley_ws/IsaacLab
    ./isaaclab.sh -p scripts/imitation_learning/lerobot/replay_lerobot_dataset_sim.py \\
        --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
        --trajectory /tmp/episode0_actions.npy \\
        --enable_cameras

No policy server, no TCP connection, no --port -- this is a single-process, purely local replay.
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="Gym env id")
parser.add_argument("--trajectory", type=str, required=True, help="Path to a .npy file from export_lerobot_episode.py")
parser.add_argument("--playback-hz", type=float, default=30.0, help="Rate to step through the trajectory (dataset fps)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else after Isaac Sim is up."""

import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.stack  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Matches convert_hdf5_to_lerobot.py's LJ_IDX / eval_smolvla_jointspace.py's LJ_IDX exactly.
LJ_IDX = list(range(0, 14, 2)) + [14]


def _step_direct(env, target_left_q: np.ndarray) -> None:
    """Advance the sim by one env "step" worth of physics (env.cfg.decimation substeps), driving
    the 8 LJ joints directly instead of going through the env's IK action term -- identical to
    eval_smolvla_jointspace.py's _step_direct(), see that file's docstring for the exact
    ManagerBasedRLEnv.step() sequence this replicates (confirmed against its source, not guessed).
    """
    robot = env.scene["robot"]
    target_t = torch.as_tensor(target_left_q, dtype=torch.float32, device=env.device).unsqueeze(0)  # (1, 8)

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        robot.set_joint_position_target(target_t, joint_ids=LJ_IDX)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if (env._sim_step_counter % env.cfg.sim.render_interval == 0) and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)


def main():
    trajectory = np.load(args_cli.trajectory).astype(np.float32)
    print(f"Loaded trajectory: {trajectory.shape[0]} steps, {trajectory.shape[1]}D per step from {args_cli.trajectory}")
    if trajectory.shape[1] != 8:
        raise ValueError(f"Expected an 8D (LJ1.pos..LJ8.pos) trajectory, got {trajectory.shape[1]}D.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    dt = 1.0 / args_cli.playback_hz
    success_seen = False

    print("Replaying. Ctrl+C to stop early.")
    try:
        for step_idx, target_q in enumerate(trajectory):
            loop_start = time.time()

            _step_direct(env, target_q)

            if not success_seen and bool(success_term.func(env, **success_term.params)[0]):
                success_seen = True
                print(f"  [step {step_idx+1:3d}/{len(trajectory)}] SUCCESS condition reached")

            if step_idx % 10 == 0:
                print(f"  step {step_idx+1}/{len(trajectory)}")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nReplay finished. Success condition {'was' if success_seen else 'was NOT'} reached during playback.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
