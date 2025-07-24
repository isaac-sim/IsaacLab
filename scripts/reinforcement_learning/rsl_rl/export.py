# scripts/reinforcement_learning/rsl_rl/export.py
#
# Export a trained rsl_rl policy as Kinfer binary.
# Example usage: python scripts/reinforcement_learning/rsl_rl/export.py --task=Isaac-Velocity-Rough-Kbot-v0 --checkpoint ~/Github/IsaacLab/logs/rsl_rl/kbot_rough/[path_to_checkpoint].pt
# Or you can omit the checkpoint arg and it will use the latest checkpoint in the logs/rsl_rl/agent_name/ directory
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    required=True,
    help="Gym id, e.g. Isaac-Humanoid-AMP-Dance-Direct-v0",
)
parser.add_argument(
    "--checkpoint",
    help=".pt file to export. If not provided, the latest checkpoint in the logs/rsl_rl/agent_name/ directory will be used.",
)
parser.add_argument("--num_envs", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Omniverse app -- this is needed to be able to access isaacsim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app  # noqa: F841

"""Everything else???"""
import copy
import math
from pathlib import Path

import torch
from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata
import gymnasium as gym  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
)
from isaaclab_rl.rsl_rl.exporter import export_policy_as_jit

import isaaclab_tasks  # noqa: F401, E402 – ensure gym envs are registered
from isaaclab_tasks.utils import (  # noqa: E402
    load_cfg_from_registry,
    parse_env_cfg,
    get_checkpoint_path,
)

class TorchPolicyExporter(torch.nn.Module):
    """TorchScript-friendly stateless wrapper that works for FF, GRU, and LSTM nets."""

    def __init__(self, policy, normalizer=None):
        super().__init__()

        self.is_recurrent = policy.is_recurrent
        
        # Extract policy components
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy has neither actor nor student module.")

        # Set up RNN configuration
        if self.is_recurrent:
            self.rnn.cpu()
            self.num_layers = self.rnn.num_layers
            self.hidden_size = self.rnn.hidden_size
            rnn_name = type(self.rnn).__name__.lower()
            
            if "gru" in rnn_name:
                self.rnn_type = "gru"
                self.forward = self._forward_gru
            elif "lstm" in rnn_name:
                self.rnn_type = "lstm"
                self.forward = self._forward_lstm
            else:
                raise NotImplementedError(f"Unsupported RNN type: {rnn_name}")
                
            print(
                f"[DEBUG] RNN type: {self.rnn_type}, layers: {self.num_layers}, "
                f"hidden_size: {self.hidden_size}"
            )
        else:
            self.rnn_type = "ff"
            self.num_layers = 0
            self.hidden_size = 0
            self.forward = self._forward_ff

        # Set up normalizer
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def _forward_ff(self, obs: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feedforward forward pass (stateless for consistency)."""
        obs = self.normalizer(obs)
        return self.actor(obs), self.actor(obs)

    def _forward_gru(self, obs: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GRU forward pass with stateless carry management."""
        obs = self.normalizer(obs)
        
        # carry shape: (1, num_layers, hidden_size)
        if carry.dim() != 3 or carry.size(0) != 1:
            raise RuntimeError(f"Expected GRU carry shape (1, {self.num_layers}, {self.hidden_size}), got {carry.shape}")
        
        # Extract hidden state: (num_layers, 1, hidden_size)
        hidden = carry[0].unsqueeze(1)
        # Input needs to be 3D for RNN: (seq_len=1, batch_size=1, input_size)
        x = obs.unsqueeze(0).unsqueeze(0)
        out, new_hidden = self.rnn(x, hidden)
        actions = self.actor(out.squeeze(0).squeeze(0))
        # Reshape hidden back to carry format: (1, num_layers, hidden_size)
        new_carry = new_hidden.squeeze(1).unsqueeze(0)
        return actions, new_carry

    def _forward_lstm(self, obs: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LSTM forward pass with stateless carry management."""
        obs = self.normalizer(obs)
        
        # carry shape: (2, num_layers, hidden_size)
        if carry.dim() != 3 or carry.size(0) != 2:
            raise RuntimeError(f"Expected LSTM carry shape (2, {self.num_layers}, {self.hidden_size}), got {carry.shape}")
        
        # Extract hidden and cell states: (num_layers, 1, hidden_size)
        h = carry[0].unsqueeze(1)
        c = carry[1].unsqueeze(1)
        # Input needs to be 3D for RNN: (seq_len=1, batch_size=1, input_size)
        x = obs.unsqueeze(0).unsqueeze(0)
        out, (new_h, new_c) = self.rnn(x, (h, c))
        actions = self.actor(out.squeeze(0).squeeze(0))
        # Reshape states back to carry format: (2, num_layers, hidden_size)
        new_carry = torch.stack(
            (new_h.squeeze(1), new_c.squeeze(1)), dim=0
        )
        return actions, new_carry

    def get_carry_shape(self, num_joints: int) -> tuple[int, ...]:
        """Get the shape of the carry tensor for this policy."""
        if self.rnn_type == "ff":
            return (num_joints,)
        elif self.rnn_type == "gru":
            return (1, self.num_layers, self.hidden_size)
        elif self.rnn_type == "lstm":
            return (2, self.num_layers, self.hidden_size)
        else:
            raise RuntimeError(f"Unknown RNN type: {self.rnn_type}")

    def get_initial_carry(self, num_joints: int) -> torch.Tensor:
        """Get the initial (zero) carry tensor for this policy."""
        return torch.zeros(self.get_carry_shape(num_joints))

    def get_traced_module(self):
        """Return a scripted (CPU, eval) version of this module."""
        self.to("cpu").eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return torch.jit.script(self)


def main():
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

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # Get checkpoint path
    checkpoint_path = args.checkpoint or get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    runner.load(checkpoint_path)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Optionally include normalizer if present
    normalizer = getattr(runner, "obs_normalizer", None)
    if normalizer is None:
        normalizer = torch.nn.Identity()
    else:
        normalizer = copy.deepcopy(normalizer).cpu().eval()

    exporter = TorchPolicyExporter(policy_nn, normalizer)

    exporter.to("cpu").eval()
    for p in exporter.parameters():
        p.requires_grad_(False)

    ts_policy = exporter

    NUM_JOINTS = len(env.unwrapped.scene["robot"].data.joint_names)
    
    # Get carry shape from the exporter
    CARRY_SHAPE = exporter.get_carry_shape(NUM_JOINTS)

    # action_term = env.unwrapped.action_manager.active_terms[0]
    action_term_name = "joint_pos"
    action_term = env.unwrapped.action_manager.get_term(action_term_name)

    _INIT_JOINT_POS = action_term._offset
    _INIT_JOINT_POS = _INIT_JOINT_POS.to("cpu").squeeze(0)
    ACTION_SCALE = action_term._scale

    command_manager = env.unwrapped.command_manager
    command_term_names = command_manager.active_terms

    command_tensor = torch.cat([command_manager.get_command(name) for name in command_term_names], dim=-1)
    command_tensor = command_tensor.to("cpu").flatten()

    NUM_COMMANDS = command_tensor.shape[0]
    
    def construct_obs_rnn(
        projected_gravity: torch.Tensor,
        joint_angles: torch.Tensor,
        joint_angular_velocities: torch.Tensor,
        command: torch.Tensor,
        gyroscope: torch.Tensor,
        carry: torch.Tensor,
    ) -> torch.Tensor:
        offset_joint_angles = joint_angles - _INIT_JOINT_POS
        scaled_projected_gravity = projected_gravity / 9.81
        obs = torch.cat(
            (
                scaled_projected_gravity,
                command,
                offset_joint_angles,
                joint_angular_velocities,
                gyroscope,
            ),
            dim=-1,
        )
        return obs

    def construct_obs_ff(
        projected_gravity: torch.Tensor,
        joint_angles: torch.Tensor,
        joint_angular_velocities: torch.Tensor,
        command: torch.Tensor,
        gyroscope: torch.Tensor,
        carry: torch.Tensor,
    ) -> torch.Tensor:
        obs = construct_obs_rnn(projected_gravity, joint_angles, joint_angular_velocities, command, gyroscope, carry)
        obs = torch.cat((obs, carry), dim=-1)
        return obs

    # Recurrent or feedforward logic split out here so it doesn't get traced and err out on carry and matmul shape mismatches
    # Difference is that obs_rnn does not add the carry to the obs (holds previous action for purely feedforward policies)
    if exporter.is_recurrent:
        construct_obs = construct_obs_rnn
    else:
        construct_obs = construct_obs_ff

    def _step_fn(
        projected_gravity: torch.Tensor,
        joint_angles: torch.Tensor,
        joint_angular_velocities: torch.Tensor,
        command: torch.Tensor,
        gyroscope: torch.Tensor,
        carry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = construct_obs(projected_gravity, joint_angles, joint_angular_velocities, command, gyroscope, carry)

        actions, new_carry = ts_policy(obs, carry)
        
        return (actions * ACTION_SCALE) + _INIT_JOINT_POS, new_carry
    
    def _init_fn() -> torch.Tensor:
        return exporter.get_initial_carry(NUM_JOINTS)

    step_args = (
        torch.zeros(3),
        torch.zeros(NUM_JOINTS),
        torch.zeros(NUM_JOINTS),
        torch.zeros(NUM_COMMANDS),
        torch.zeros(3),
        torch.zeros(*CARRY_SHAPE),
    )

    step_fn = torch.jit.trace(_step_fn, step_args)
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

    output_path = Path(checkpoint_path).with_suffix(".kinfer")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_blob)

    print(f"[OK] Export completed → {output_path}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
