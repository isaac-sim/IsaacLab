# scripts/reinforcement_learning/rsl_rl/distill.py

from __future__ import annotations
import os, sys, argparse, torch
from isaaclab.app import AppLauncher

# ---- CLI
parser = argparse.ArgumentParser("Distill a camera student from a privileged PPO teacher.")
parser.add_argument("--task", required=True, help="Hydra task id, e.g. Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--student_group", required=True, help="Obs group name for student (e.g. camera_ext2)")
parser.add_argument("--teacher_group", required=True, help="Obs group name for teacher (e.g. policy)")
parser.add_argument("--teacher_ckpt", required=True, help="Path to PPO teacher checkpoint .pt")
parser.add_argument("--total_steps", type=int, default=200_000)
parser.add_argument("--update_every", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out_dir", type=str, default="outputs/distill_student")
# passthrough kit/hydra args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# ensure cameras are available only if you need them
# clear argv for hydra
sys.argv = [sys.argv[0]] + hydra_args

# ---- Launch kit
app = AppLauncher(args_cli).app

# ---- IsaacLab & task
import isaaclab
from isaaclab_tasks import make as make_task

# Build env with custom observation groups for distillation
env, _ = make_task(
    args_cli.task,
    env_index=0,
    num_envs=args_cli.num_envs,
    # Select obs groups for student/teacher
    obs_groups={"policy": args_cli.student_group, "teacher": args_cli.teacher_group},
    # Keep sim fabric, rendering config same as train.py defaults
)

# ---- RSL-RL wiring
import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

device = "cuda:0" if torch.cuda.is_available() else "cpu"
venv = VecEnv(env, rl_device=device, sim_device=device, graphics_device_id=0)

# training configuration (minimal, override PPO->Distillation)
train_cfg = {
    "runner_class_name": "OnPolicyRunner",
    "algorithm": {
        "class_name": "Distillation",
        "num_learning_epochs": 1,
        "gradient_length": 15,
        "learning_rate": args_cli.lr,
        "max_grad_norm": 1.0,
        "loss_type": "mse",
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
    "policy": {
        "class_name": "StudentTeacher",
        "student_hidden_dims": [256, 256, 256],
        "teacher_hidden_dims": [256, 256, 256],
        "activation": "elu",
        "init_noise_std": 0.1,
    },
    "empirical_normalization": False,
    "num_steps_per_env": args_cli.update_every,
    "save_interval": 50,  # save every 50 iters
    "logger": "tensorboard",
}

# create output dir
log_dir = os.path.join(args_cli.out_dir, "distill_" + os.path.basename(args_cli.teacher_ckpt).split(".")[0])
os.makedirs(log_dir, exist_ok=True)

runner = OnPolicyRunner(venv, train_cfg, log_dir=log_dir, device=device)

# ---- Load teacher (PPO ckpt). StudentTeacher.load_state_dict handles 'actor.' renaming.
runner.alg.policy.load_state_dict(torch.load(args_cli.teacher_ckpt, map_location=device)["model_state_dict"], strict=False)

# ---- Compute #iterations from total_steps
steps_per_iter = args_cli.update_every * args_cli.num_envs
num_learning_iterations = max(1, args_cli.total_steps // steps_per_iter)

# ---- GO
runner.learn(num_learning_iterations)
print(f"Done. Logs & student checkpoints in: {log_dir}")
