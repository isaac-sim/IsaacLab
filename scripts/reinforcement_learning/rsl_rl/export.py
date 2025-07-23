# scripts/reinforcement_learning/rsl_rl/export.py
#
# Export a trained rsl_rl policy as TorchScript and K-Infer binary.
# The exported model exposes the following signature:
#     init() -> carry
#     step(joint_angles, joint_velocities, command, carry) -> (actions, carry)
#
# Example usage: python scripts/reinforcement_learning/rsl_rl/export.py --task=Isaac-Velocity-Rough-Kbot-v0 --checkpoint ~/Github/IsaacLab/logs/rsl_rl/kbot_rough/[path_to_checkpoint].pt

import argparse
import copy
import math
from pathlib import Path

import torch
from isaaclab.app import AppLauncher
from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    required=True,
    help="Gym id, e.g. Isaac-Humanoid-AMP-Dance-Direct-v0",
)
parser.add_argument(
    "--checkpoint",
    required=True,
    help=".pt file produced by the training / play script",
)

parser.add_argument("--num_envs", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Omniverse app (unclear if this is needed)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app  # noqa: F841 – kept for parity with other scripts


import gymnasium as gym  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401, E402 – ensure gym envs are registered
from isaaclab_tasks.utils import (  # noqa: E402
    load_cfg_from_registry,
    parse_env_cfg,
)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg, render_mode=None)

# Flatten multi-agent envs (rsl_rl works with single-agent vec envs)
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

env = RslRlVecEnvWrapper(env, clip_actions=None)

action_dim = env.action_space.shape[0]

obs_dim = env.observation_space["policy"].shape[-1]

task_name = args.task.split(":")[-1]
agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
agent_cfg.device = args.device

# Build runner & load checkpoint
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
print(f"[INFO] Loading checkpoint from: {args.checkpoint}")
runner.load(args.checkpoint)

try:
    policy_nn = runner.alg.policy
except AttributeError:
    policy_nn = runner.alg.actor_critic

if hasattr(policy_nn, "actor"):
    actor_net = copy.deepcopy(policy_nn.actor).cpu().eval()
elif hasattr(policy_nn, "student"):
    actor_net = copy.deepcopy(policy_nn.student).cpu().eval()
else:
    raise RuntimeError("Unsupported policy object – cannot locate actor network")

# Optionally include normalizer if present
normalizer = getattr(runner, "obs_normalizer", None)
if normalizer is None:
    normalizer = torch.nn.Identity()
else:
    normalizer = copy.deepcopy(normalizer).cpu().eval()

class ActorWrapper(torch.nn.Module):
    """Wraps (normalizer → actor) into a single forward pass."""

    def __init__(self, actor, norm):
        super().__init__()
        self.actor = actor
        self.norm = norm

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs))


wrapper = ActorWrapper(actor_net, normalizer).eval()
for p in wrapper.parameters():
    p.requires_grad = False

num_joints = len(env.unwrapped.scene["robot"].data.joint_names)
CARRY_SHAPE: tuple[int, ...] = (num_joints,)

NUM_COMMANDS = 3

_INIT_JOINT_POS = torch.tensor(
    [
        math.radians(20.0),  # dof_right_shoulder_pitch_03
        0.0,  # dof_right_shoulder_roll_03
        math.radians(-20.0),  # dof_right_shoulder_yaw_02
        0.0,  # dof_right_elbow_02
        0.0,  # dof_right_wrist_00
        math.radians(10.0),  # dof_left_shoulder_pitch_03
        0.0,  # dof_left_shoulder_roll_03
        math.radians(-10.0),  # dof_left_shoulder_yaw_02
        0.0,  # dof_left_elbow_02
        0.0,  # dof_left_wrist_00
        0.0,  # dof_right_hip_pitch_04
        0.0,  # dof_right_hip_roll_03
        math.radians(50.0),  # dof_right_hip_yaw_03
        math.radians(-90.0),  # dof_right_knee_04
        math.radians(-50.0),  # dof_right_ankle_02
        math.radians(90.0),  # dof_left_hip_pitch_04
        math.radians(-30.0),  # dof_left_hip_roll_03
        0.0,  # dof_left_hip_yaw_03
        math.radians(30.0),  # dof_left_knee_04
        0.0,  # dof_left_ankle_02
    ]
)


def _init_fn() -> torch.Tensor:
    """Returns the initial carry tensor (all zeros)."""
    return torch.zeros(CARRY_SHAPE)


def _step_fn(
    projected_gravity: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_angular_velocities: torch.Tensor,
    command: torch.Tensor,
    carry: torch.Tensor,
    gyroscope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy step."""
    offset_joint_angles = joint_angles - _INIT_JOINT_POS
    scaled_projected_gravity = projected_gravity / 9.81
    obs = torch.cat(
        (
            scaled_projected_gravity,
            command,
            offset_joint_angles,
            joint_angular_velocities,
            carry,
            gyroscope,
        ),
        dim=-1,
    )
    actions = wrapper(obs)
    return (actions * 0.5) + _INIT_JOINT_POS, actions


step_fn = torch.jit.trace(
    _step_fn,
    (
        torch.zeros(3),
        torch.zeros(num_joints),
        torch.zeros(num_joints),
        torch.zeros(NUM_COMMANDS),
        torch.zeros(*CARRY_SHAPE),
        torch.zeros(3),
    ),
)

init_fn = torch.jit.trace(_init_fn, ())

joint_names = list(env.unwrapped.scene["robot"].data.joint_names)
metadata = PyModelMetadata(
    joint_names=joint_names,
    num_commands=NUM_COMMANDS,
    carry_size=list(CARRY_SHAPE),
)

init_onnx = export_fn(init_fn, metadata)
step_onnx = export_fn(step_fn, metadata)

kinfer_blob = pack(init_fn=init_onnx, step_fn=step_onnx, metadata=metadata)

output_path = Path(args.checkpoint).with_suffix(".kinfer")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    f.write(kinfer_blob)

print(f"[OK] Export completed → {output_path}")
