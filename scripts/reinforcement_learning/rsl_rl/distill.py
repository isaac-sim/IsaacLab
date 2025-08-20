from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path

# --- Parse args FIRST (and include AppLauncher args) ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distill a privileged teacher policy into a vision-only student.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel envs")
parser.add_argument("--student_group", type=str, required=True, help="Obs group for student (e.g. camera_ext2)")
parser.add_argument("--teacher_group", type=str, required=True, help="Obs group for teacher (e.g. policy)")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to RL checkpoint (e.g. model_2950.pt)")
parser.add_argument("--total_steps", type=int, default=200_000, help="Total env steps to collect")
parser.add_argument("--update_every", type=int, default=32, help="Env steps per gradient step")
parser.add_argument("--lr", type=float, default=1e-3, help="Student optimizer LR")
parser.add_argument("--out_dir", type=str, default="outputs/distill_student", help="Output dir for ckpts/logs")
parser.add_argument("--log_interval", type=int, default=1000, help="Steps between logs")
parser.add_argument("--save_interval", type=int, default=10_000, help="Steps between checkpoints")

# Allow all AppLauncher args like --headless / --enable_cameras etc.
AppLauncher.add_app_launcher_args(parser)

# Split hydra args away from argparse
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# If you want cameras (student uses vision), ensure they’re on
if getattr(args_cli, "enable_cameras", None) is None:
    # default to enabling cameras if not explicitly set
    args_cli.enable_cameras = True

# --- Start Omniverse app BEFORE any isaaclab/isaacsim-dependent import ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # noqa: F401  (kept alive for lifetime of the script)

# --- Now safe to import isaaclab / rsl_rl modules ---
import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks import task_registry
make_task = task_registry.make

# Your models/algorithms
from rsl_rl.modules import StudentTeacher  # your class signature posted earlier
from rsl_rl.algorithms.distillation import Distillation  # your Distillation class (from your snippet)


def _ensure_out_dir(p: str | Path) -> Path:
    p = Path(p).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_dims(env: ManagerBasedRLEnv, student_group: str, teacher_group: str) -> tuple[int, int, int]:
    """
    Try a few common ways to get flattened obs dims per group and number of actions.
    Adjust these lines if your API differs slightly.
    """
    # Observation dims (common accessors in Isaac Lab)
    try:
        student_dim = env.observation_manager.get_group_obs_dim(student_group)  # preferred
        teacher_dim = env.observation_manager.get_group_obs_dim(teacher_group)
    except Exception:
        # Fallback: many envs expose group specs / shapes
        student_dim = env.observation_manager.group_specs[student_group].obs_dim
        teacher_dim = env.observation_manager.group_specs[teacher_group].obs_dim

    # Actions dim
    try:
        act_dim = env.action_manager.num_actions
    except Exception:
        act_dim = env.action_manager.action_dim  # fallback name in some versions

    return int(student_dim), int(teacher_dim), int(act_dim)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = _ensure_out_dir(args_cli.out_dir)

    # --- Build the environment ---
    # `make_task` usually accepts at least name + num_envs; hydra_args are already in sys.argv
    env: ManagerBasedRLEnv = make_task(name=args_cli.task, num_envs=args_cli.num_envs)

    # Get observation/action sizes
    student_dim, teacher_dim, act_dim = _get_dims(env, args_cli.student_group, args_cli.teacher_group)

    # --- Build student-teacher policy and distillation algo ---
    policy = StudentTeacher(
        num_student_obs=student_dim,
        num_teacher_obs=teacher_dim,
        num_actions=act_dim,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
    ).to(device)

    # Load teacher weights from an RL checkpoint
    teacher_ckpt = torch.load(args_cli.teacher_ckpt, map_location=device)
    _ = policy.load_state_dict(teacher_ckpt, strict=False)  # returns False for first-time load (OK)

    algo = Distillation(
        policy=policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=args_cli.lr,
        max_grad_norm=1.0,
        device=device,
        multi_gpu_cfg=None,
    )

    # Initialize rollout storage
    # We’ll pick a standard rollout size (update_every) and feed student+teacher obs + teacher actions
    algo.init_storage(
        training_type="feed_forward",
        num_envs=args_cli.num_envs,
        num_transitions_per_env=args_cli.update_every,
        student_obs_shape=(student_dim,),
        teacher_obs_shape=(teacher_dim,),
        actions_shape=(act_dim,),
    )

    # --- Rollout loop ---
    obs_dict = env.reset()
    steps = 0
    last_log = 0
    last_save = 0
    t0 = time.time()

    while steps < args_cli.total_steps:
        # Get current student/teacher obs
        # In ManagerBasedRLEnv, group obs are usually flattened tensors in obs_dict[group]
        student_obs = obs_dict[args_cli.student_group].to(device)
        teacher_obs = obs_dict[args_cli.teacher_group].to(device)

        # Student acts; algorithm also records teacher's privileged actions internally
        actions = algo.act(student_obs, teacher_obs)

        # Step environment
        obs_dict, rewards, dones, infos = env.step(actions)
        algo.process_env_step(rewards.to(device), dones.to(device), infos)

        steps += args_cli.num_envs

        # Update when rollout is full
        if (steps % (args_cli.update_every * args_cli.num_envs)) == 0:
            loss_dict = algo.update()

        # Logging
        if steps - last_log >= args_cli.log_interval:
            dt = time.time() - t0
            sps = int(steps / max(dt, 1e-6))
            msg = f"[distill] steps={steps:,}  SPS={sps}  "
            if 'loss_dict' in locals():
                msg += " | ".join(f"{k}={v:.5f}" for k, v in loss_dict.items())
            print(msg, flush=True)
            last_log = steps

        # Save checkpoints
        if steps - last_save >= args_cli.save_interval:
            ckpt_path = out_dir / f"student_{steps:09d}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[distill] Saved {ckpt_path}", flush=True)
            last_save = steps

    # Final save
    final_ckpt = out_dir / "student_final.pt"
    torch.save(policy.state_dict(), final_ckpt)
    print(f"[distill] Done. Saved final checkpoint to: {final_ckpt}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean shutdown of the app
        import omni.kit.app  # type: ignore
        omni.kit.app.get_app().post_quit()
