# scripts/reinforcement_learning/skrl/export.py
#
# Export a trained skrl policy as TorchScript, with a custom
#   forward(joint_angles, joint_vel, quat, cmd, gyro, last_action)
# that internally builds the observation vector expected by the net.


import argparse

from isaaclab.app import AppLauncher
from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

parser = argparse.ArgumentParser()
parser.add_argument("--task",       required=True,
                    help="Gym id, e.g. Isaac-Humanoid-AMP-Dance-Direct-v0")
parser.add_argument("--checkpoint", required=True,
                    help=".pt file produced by training / play script")
parser.add_argument("--algorithm", required=True,
                    help="Algorithm used for training")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode=None))

# pick the right YAML for this task (AMP, PPO, …)
agent_yaml = load_cfg_from_registry(args.task, f"skrl_{args.algorithm.lower()}_cfg_entry_point")
runner = Runner(env, agent_yaml)
runner.agent.load(os.path.abspath(args.checkpoint))
policy = runner.agent.policy.eval()    # GaussianModel

policy_cpu = policy.to("cpu")

# turn the GaussianModel into TorchScript once
CARRY_SHAPE: tuple[int, ...] = (1,)

# Determine observation dimension for tracing the policy

obs_dim = env.observation_space.shape[0]

class ActorWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.base({"states": obs}, role="policy")[2]['mean_actions'].squeeze(0)
        return mu

wrapper = ActorWrapper(policy_cpu).eval()

for p in wrapper.parameters():
    p.requires_grad = False

policy_ts = torch.jit.trace(wrapper, torch.zeros(1, obs_dim, device="cpu"))


NUM_COMMANDS = 1 + 6 + 3 + 3 + 4 * 3


@torch.jit.script
def init_fn() -> torch.Tensor:
    """Returns the initial carry tensor."""
    return torch.zeros((1,))

def _step_fn(
    joint_angles: torch.Tensor,
    joint_angular_velocities: torch.Tensor,
    command: torch.Tensor,
    carry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step function for the policy."""
    obs = torch.cat(
        [joint_angles,
         joint_angular_velocities,
         command,
         ],
        dim=-1,
    )
    actions = wrapper(obs)
    return actions, carry


robot_data = env.unwrapped.scene["robot"].data
joint_names = list(robot_data.joint_names)

step_fn = torch.jit.trace(_step_fn, (
    torch.zeros(len(joint_names), device="cpu"),
    torch.zeros(len(joint_names), device="cpu"),
    torch.zeros(NUM_COMMANDS, device="cpu"),
    torch.zeros(*CARRY_SHAPE, device="cpu"),
))

metadata = PyModelMetadata(
    joint_names=joint_names,
    num_commands=NUM_COMMANDS,
    carry_size=CARRY_SHAPE,
)

init_onnx = export_fn(init_fn, metadata)
step_onnx = export_fn(step_fn, metadata)

from pathlib import Path

OUTPUT_PATH = Path(args.checkpoint).with_suffix(".kinfer")

kinfer_model = pack(init_fn=init_onnx, step_fn=step_onnx, metadata=metadata)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(kinfer_model)

print(f"[OK] Export completed → {OUTPUT_PATH}")
