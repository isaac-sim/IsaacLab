# scripts/reinforcement_learning/rsl_rl/distill.py
import argparse, sys, os
import torch
from pathlib import Path

from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv

# === RSL-RL bits
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.utils.utils import set_seed
from rsl_rl.env import VecEnvWrapper  # if you already wrap env like PPO runner does

# Your StudentTeacher class (from your fork)
from rsl_rl.modules import StudentTeacher

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--student_group", type=str, default="camera_ext2",
                    help="Observation group for student (e.g., camera_ext2)")
parser.add_argument("--teacher_group", type=str, default="policy",
                    help="Observation group for teacher (privileged)")
parser.add_argument("--teacher_ckpt", type=str, required=True,
                    help="Path to PPO checkpoint (contains actor.* keys)")
parser.add_argument("--update_every", type=int, default=32,
                    help="How many env steps between updates")
parser.add_argument("--num_learning_epochs", type=int, default=1)
parser.add_argument("--grad_len", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_grad_norm", type=float, default=None)
parser.add_argument("--total_steps", type=int, default=20000)
parser.add_argument("--out_dir", type=str, default="outputs/distill_student")
# App/Kit args passthrough (Hydra)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args  # let Hydra parse its own args

# Always enable cameras if using a camera group
if args_cli.enable_cameras is None and args_cli.student_group.startswith("camera"):
    args_cli.enable_cameras = True

# --- Launch Isaac Sim ---
app = AppLauncher(args_cli).app

# === Build env ===
# old:
# from isaaclab_tasks import make as make_task

try:
    from isaaclab_tasks import make as make_task
except Exception:
    # Other builds expose it as task_registry.make
    from isaaclab_tasks import task_registry
    make_task = task_registry.make
 # same factory your PPO runner uses
env: ManagerBasedRLEnv = make_task(
    task_name=args_cli.task,
    num_envs=args_cli.num_envs,
    device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
    seed=args_cli.seed,
)

device = env.device
set_seed(args_cli.seed)

# Shapes
student_dim = env.obs_manager.group_obs_dim[args_cli.student_group]
teacher_dim = env.obs_manager.group_obs_dim[args_cli.teacher_group]
num_actions = env.action_manager.action_dim

# === Build StudentTeacher model ===
policy = StudentTeacher(
    num_student_obs=student_dim,
    num_teacher_obs=teacher_dim,
    num_actions=num_actions,
    student_hidden_dims=[256, 256, 256],
    teacher_hidden_dims=[256, 256, 256],
    activation="elu",
    init_noise_std=0.1,
).to(device)

# === Load the teacher (PPO) from actor.* weights ===
teacher_state = torch.load(args_cli.teacher_ckpt, map_location=device)
# Your StudentTeacher.load_state_dict() already knows how to strip "actor." and ignore critic.
policy.load_state_dict(teacher_state, strict=False)

# === Algo ===
algo = Distillation(
    policy=policy,
    num_learning_epochs=args_cli.num_learning_epochs,
    gradient_length=args_cli.grad_len,
    learning_rate=args_cli.lr,
    max_grad_norm=args_cli.max_grad_norm,
    device=device,
)

# Rollout storage sizing
horizon = args_cli.update_every
algo.init_storage(
    training_type="feedforward",
    num_envs=env.num_envs,
    num_transitions_per_env=horizon,
    student_obs_shape=(student_dim,),
    teacher_obs_shape=(teacher_dim,),
    actions_shape=(num_actions,),
)

# === Loop ===
out_dir = Path(args_cli.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

obs = env.reset()
steps = 0
while steps < args_cli.total_steps:
    for t in range(horizon):
        student_obs = obs[args_cli.student_group]["obs"].to(device)
        teacher_obs = obs[args_cli.teacher_group]["obs"].to(device)

        actions = algo.act(student_obs, teacher_obs)
        obs, rew, done, info = env.step(actions)
        algo.process_env_step(rew, done, info)
        steps += env.num_envs
        if steps >= args_cli.total_steps:
            break

    losses = algo.update()
    if env.rank == 0:
        print({k: float(v) for k, v in losses.items()})

    # save student snapshot
    if env.rank == 0:
        torch.save(policy.state_dict(), out_dir / "student.pt")

# tidy
app.close()
