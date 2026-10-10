# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble original-head policy, native skrl PPO/RandomMemory/SequentialTrainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from scripts.reinforcement_learning.gr00t_skrl.agent import Gr00tPPO, RunState
from scripts.reinforcement_learning.gr00t_skrl.environment import RemoteEnvironment
from scripts.reinforcement_learning.gr00t_skrl.policy import ChainPolicy, FrozenEncoder, StateCritic, module_digest
from scripts.reinforcement_learning.gr00t_skrl.protocol import RunConfig


def checkpoint_metadata(cfg: RunConfig, encoder: FrozenEncoder) -> dict:
    """Describe the versioned probability contract and immutable model/processor inputs."""
    versions = {package: version(package) for package in ("skrl", "torch", "numpy", "transformers", "tensorboard")}
    if versions["skrl"] != "2.1.0":
        raise ValueError("This runner was validated against skrl 2.1.0")
    processor_files = {}
    for filename in (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
    ):
        processor_files[filename] = hashlib.sha256((Path(cfg.model_path) / filename).read_bytes()).hexdigest()
    return {
        "policy_version": "gr00t-n17-full-chain-v1",
        "model_path": cfg.model_path,
        "backbone_path": cfg.backbone_path,
        "processor_files": processor_files,
        "generation_steps": cfg.generation_steps,
        "sigma": cfg.sigma,
        "horizon": encoder.head.config.action_horizon,
        "action_dim": encoder.head.config.max_action_dim,
        "feature_layout": vars(encoder.layout),
        "dependencies": versions,
        "instruction": cfg.instruction,
        "task": cfg.task,
        "backend": "isaacsim_physx",
        "learning_rate": cfg.learning_rate,
        "rollouts": cfg.rollouts,
        "learning_epochs": cfg.learning_epochs,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ratio_clip": 0.2,
        "probability": "sum over K*H*D Gaussian transition log densities; x_0 constant omitted",
    }


def run(cfg: RunConfig) -> None:
    """Run a complete native rollout, checkpoint or inference, with reproducible diagnostics."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    encoder = FrozenEncoder(cfg)
    policy = ChainPolicy(encoder.head, encoder.layout, cfg.generation_steps, cfg.sigma, "cuda:0")
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("The complete original action head must remain trainable")
    critic = StateCritic(policy.observation_space, policy.action_space, "cuda:0")
    metadata = checkpoint_metadata(cfg, encoder)
    run_state = RunState(metadata)
    backbone_before = module_digest(encoder.backbone)
    agent_cfg = PPO_CFG(
        rollouts=cfg.rollouts,
        learning_epochs=cfg.learning_epochs,
        mini_batches=cfg.rollouts,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=cfg.learning_rate,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=0.0,
        time_limit_bootstrap=True,
        mixed_precision=False,
        experiment=ExperimentCfg(
            directory=cfg.run_dir, experiment_name="native", write_interval=cfg.rollouts, checkpoint_interval=0
        ),
    )
    memory = RandomMemory(memory_size=cfg.rollouts, num_envs=1, device="cuda:0")
    agent = Gr00tPPO(
        encoder=encoder,
        run_state=run_state,
        models={"policy": policy, "value": critic},
        memory=memory,
        observation_space=policy.observation_space,
        state_space=critic.state_space,
        action_space=policy.action_space,
        device="cuda:0",
        cfg=agent_cfg,
    )
    env = RemoteEnvironment(cfg, encoder, policy)
    try:
        trainer = SequentialTrainer(
            env=env,
            agents=agent,
            cfg={
                "timesteps": cfg.timesteps,
                "headless": True,
                "disable_progressbar": True,
                "close_environment_at_exit": False,
                "stochastic_evaluation": True,
            },
        )
        resume_summary = agent.resume(cfg.resume) if cfg.resume else None
        # Verify the real model and native memory interface before any parameter updates.
        observations, _ = env.reset()
        with torch.no_grad():
            chain, output = policy.act({"observations": observations})
            _, recomputed = policy.act({"observations": observations, "taken_actions": chain})
        consistency = float((output["log_prob"] - recomputed["log_prob"]).abs().max())
        if consistency > 0.01:
            raise RuntimeError(f"Sampling/recompute likelihood mismatch: {consistency}")
        sampling_peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            json.dumps(
                {
                    "metadata": metadata,
                    "trainable_head_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                    "backbone_trainable_parameters": sum(
                        p.numel() for p in encoder.backbone.parameters() if p.requires_grad
                    ),
                    "chain_size": policy.num_actions,
                    "log_prob_recompute_error": consistency,
                    "resume": resume_summary,
                }
            ),
            flush=True,
        )
        if resume_summary is not None:
            # The diagnostic above consumed RNG; restore again immediately before native trainer interaction.
            run_state.restore_rng()
        if cfg.mode == "train":
            trainer.train()
        else:
            trainer.eval()
        backbone_unchanged = module_digest(encoder.backbone) == backbone_before
        if not backbone_unchanged or any(p.grad is not None for p in encoder.backbone.parameters()):
            raise RuntimeError("Frozen backbone changed or received gradients")
        metrics = {
            "metadata": metadata,
            "seed": cfg.seed,
            "log_prob_recompute_error": consistency,
            "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "sampling_peak_allocated_gib": sampling_peak,
            "final_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "backbone_unchanged": backbone_unchanged,
            "backbone_sha256": backbone_before,
            "resume": resume_summary,
            "updates": agent.diagnostics,
            "transitions": env.metrics,
            "environment_steps": run_state.environment_steps,
            "optimizer_steps": run_state.optimizer_steps,
        }
        if not any(item["motion_m"] > 1e-6 for item in env.metrics):
            raise RuntimeError("No measured robot motion during real simulation")
        if not all(item["table_rgb_std"] > 1 and item["wrist_rgb_std"] > 1 for item in env.metrics):
            raise RuntimeError("Invalid or constant RGB camera observations")
        (Path(cfg.run_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(
            json.dumps(
                {
                    "environment_steps": run_state.environment_steps,
                    "optimizer_steps": run_state.optimizer_steps,
                    "backbone_unchanged": backbone_unchanged,
                }
            ),
            flush=True,
        )
    finally:
        env.close()
        if hasattr(agent, "writer"):
            agent.writer.close()


def main() -> None:
    """Read the launcher-owned configuration in the GR00T uv environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = RunConfig(**json.loads(Path(args.config).read_text()))
    cfg.validate()
    run(cfg)


if __name__ == "__main__":
    main()
