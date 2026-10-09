# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sample and update the original N1.7 action head without a distributed RL framework."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from torch import nn
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint
from transformers.feature_extraction_utils import BatchFeature

_STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
_NOISE_STD = 0.05


def _fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy())
    return digest.hexdigest()


def _state_dict(state: np.ndarray) -> dict[str, np.ndarray]:
    return {**{key: state[:, i : i + 1] for i, key in enumerate(_STATE_KEYS[:6])}, "gripper": state[:, 6:8]}


def _prepare(model, processor, observation: dict) -> tuple[dict, dict]:
    step = VLAStepData(
        images={"image": observation["table_cam"][0:1], "wrist_image": observation["wrist_cam"][0:1]},
        states=_state_dict(observation["state"]),
        actions={},
        text="Stack the red cube on the blue cube, then the green cube on the red cube.",
        embodiment=EmbodimentTag.LIBERO_PANDA,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    batch = processor.collator([processed])["inputs"]
    backbone_input, action_input = model.prepare_input(batch)
    with torch.no_grad():
        features = model.backbone(backbone_input)
    # Cache only frozen backbone outputs; projectors remain inside the trainable action head.
    return ({k: v.detach().cpu() for k, v in features.items()}, {k: v.detach().cpu() for k, v in action_input.items()})


def _velocity(model, features: dict, action_input: dict, latent: torch.Tensor, index: int, steps: int) -> torch.Tensor:
    head = model.action_head
    output = BatchFeature({k: v.to("cuda") for k, v in features.items()})
    inputs = BatchFeature({k: v.to("cuda") for k, v in action_input.items()})
    encoded = head._encode_features(output, inputs)
    times = torch.full((1,), int(index / steps * head.num_timestep_buckets), device="cuda", dtype=torch.long)
    action_features = head.action_encoder(latent.to(torch.bfloat16), times, inputs.embodiment_id)
    if head.config.add_pos_embed:
        positions = torch.arange(latent.shape[1], device="cuda")
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)
    hidden = torch.cat((encoded.state_features, action_features), dim=1)

    def run_dit(hidden_states):
        return head.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoded.backbone_features,
            timestep=times,
            image_mask=output.image_mask,
            backbone_attention_mask=output.backbone_attention_mask,
        )

    hidden = checkpoint(run_dit, hidden, use_reentrant=False) if torch.is_grad_enabled() else run_dit(hidden)
    return head.action_decoder(hidden, inputs.embodiment_id)[:, -head.action_horizon :].float()


def _sample(model, processor, observation: dict, steps: int) -> tuple[np.ndarray, dict]:
    features, inputs = _prepare(model, processor, observation)
    chain = [torch.randn((1, model.config.action_horizon, model.config.max_action_dim), device="cuda")]
    logprob = torch.zeros((), device="cuda")
    with torch.no_grad():
        for index in range(steps):
            mean = chain[-1] + _velocity(model, features, inputs, chain[-1], index, steps) / steps
            distribution = Normal(mean, _NOISE_STD)
            chain.append(distribution.sample())
            logprob += distribution.log_prob(chain[-1]).sum()
    decoded = processor.decode_action(
        chain[-1].cpu().numpy(),
        EmbodimentTag.LIBERO_PANDA,
        {k: v[None] for k, v in _state_dict(observation["state"]).items()},
    )
    action = np.concatenate([decoded[k][:, 0] for k in _STATE_KEYS], axis=-1).astype(np.float32)
    # Deterministic actuator limits are part of the latent-policy-to-environment mapping.
    action[:, :6] = np.clip(action[:, :6], -0.1, 0.1)
    return action, {
        "features": features,
        "inputs": inputs,
        "chain": [x.cpu() for x in chain],
        "old_logprob": logprob.cpu(),
        "state": torch.from_numpy(observation["state"]),
    }


def _logprob(model, sample: dict, steps: int) -> torch.Tensor:
    total = torch.zeros((), device="cuda")
    for index in range(steps):
        latent = sample["chain"][index].to("cuda")
        target = sample["chain"][index + 1].to("cuda")
        mean = latent + _velocity(model, sample["features"], sample["inputs"], latent, index, steps) / steps
        total = total + Normal(mean, _NOISE_STD).log_prob(target).sum()
    return total


def _advantages(samples: list[dict], critic: nn.Module, final_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        values = torch.cat([critic(s["state"].to("cuda")).flatten() for s in samples])
        final_value = critic(final_state.to("cuda")).squeeze()
        advantages = torch.zeros_like(values)
        gae = torch.zeros((), device="cuda")
        for i in reversed(range(len(samples))):
            sample = samples[i]
            next_value = values[i + 1] if i + 1 < len(samples) else final_value
            if sample["truncated"]:
                next_value = critic(torch.from_numpy(sample["final_state"]).to("cuda")).squeeze()
            bootstrap = 0.0 if sample["terminated"] else 1.0
            continuation = 0.0 if sample["terminated"] or sample["truncated"] else 1.0
            delta = sample["reward"] + 0.99 * bootstrap * next_value - values[i]
            gae = delta + 0.99 * 0.95 * continuation * gae
            advantages[i] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        return advantages, returns


def main() -> None:
    """Run a small on-policy rollout and optional PPO updates on one GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--backbone_path", type=Path, required=True)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--denoising_steps", type=int, default=2)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    cfg = Gr00tN1d7Config.from_pretrained(args.model_path)
    cfg.model_name = str(args.backbone_path)
    cfg.tune_visual = cfg.tune_llm = False
    cfg.tune_projector = cfg.tune_diffusion_model = cfg.tune_vlln = True
    cfg.load_bf16 = True
    model = Gr00tN1d7.from_pretrained(args.model_path, config=cfg, dtype=torch.bfloat16).eval().to("cuda")
    model.backbone.requires_grad_(False)
    processor = Gr00tN1d7Processor.from_pretrained(args.model_path, model_name=str(args.backbone_path))
    processor.eval()
    if "libero_sim" not in processor.modality_configs:
        raise ValueError("Provide an N1.7 stack-cube checkpoint with an 8-D state and 7-D action processor.")
    critic = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 1)).to("cuda")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.action_head.parameters(), "lr": 1e-4},
            {"params": critic.parameters(), "lr": 1e-4},
        ],
        weight_decay=0,
        foreach=False,
    )
    completed = 0
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["model_path"] != str(args.model_path) or saved["denoising_steps"] != args.denoising_steps:
            raise ValueError("Resume requires the same base model path and denoising configuration.")
        model.action_head.load_state_dict(saved["action_head"])
        critic.load_state_dict(saved["critic"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        completed = saved["updates"]
        del saved
    print("Loaded original N1.7 action head; frozen backbone; no RLinf.", flush=True)
    deadline = time.monotonic() + 180
    while True:
        try:
            connection = Client(
                args.socket_path, family="AF_UNIX", authkey=bytes.fromhex(os.environ["GR00T_LOCAL_AUTHKEY"])
            )
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                raise TimeoutError("Simulation server unavailable; inspect sim.log.") from None
            time.sleep(0.5)
    metrics = []
    with connection:
        observation = connection.recv()
        for _ in range(max(args.updates, 1)):
            model.backbone.to("cuda")
            samples = []
            for index in range(args.rollout_steps):
                action, sample = _sample(model, processor, observation, args.denoising_steps)
                connection.send(action)
                observation = connection.recv()
                sample.update({k: observation[k] for k in ("reward", "terminated", "truncated")})
                if observation["truncated"]:
                    sample["final_state"] = observation["final_state"]
                sample["action"] = action.tolist()
                samples.append(sample)
                print(f"Rollout {index + 1}/{args.rollout_steps}: reward={sample['reward']:.6f}", flush=True)
            if not args.updates:
                break
            model.backbone.to("cpu")
            torch.cuda.empty_cache()
            backbone_hash = _fingerprint(model.backbone)
            advantages, returns = _advantages(samples, critic, torch.from_numpy(observation["state"]))
            with torch.no_grad():
                errors = [
                    abs(float(_logprob(model, s, args.denoising_steps).cpu() - s["old_logprob"])) for s in samples
                ]
            if max(errors) > 0.01:
                raise RuntimeError(f"Old-policy probability mismatch before update: {errors}")
            optimizer.zero_grad(set_to_none=True)
            losses = []
            before = model.action_head.action_decoder.state_dict()
            before = {k: v.detach().cpu().clone() for k, v in before.items()}
            for index, sample in enumerate(samples):
                logprob = _logprob(model, sample, args.denoising_steps)
                ratio = torch.exp(logprob - sample["old_logprob"].to("cuda"))
                policy_loss = -torch.minimum(ratio * advantages[index], ratio.clamp(0.8, 1.2) * advantages[index])
                value_loss = 0.5 * (critic(sample["state"].to("cuda")).squeeze() - returns[index]).square()
                loss = (policy_loss + value_loss) / len(samples)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss")
                loss.backward()
                losses.append(float(loss.detach()))
            gradient_norm = nn.utils.clip_grad_norm_(
                list(model.action_head.parameters()) + list(critic.parameters()), 1.0
            )
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError(f"Invalid gradient norm: {gradient_norm}")
            optimizer.step()
            changed = sum(
                int(torch.count_nonzero(v.detach().cpu() != before[k]))
                for k, v in model.action_head.action_decoder.state_dict().items()
            )
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise RuntimeError("Frozen backbone received gradients")
            if _fingerprint(model.backbone) != backbone_hash:
                raise RuntimeError("Frozen backbone weights changed")
            if changed == 0:
                raise RuntimeError("PPO did not change action decoder weights")
            completed += 1
            metrics.append(
                {
                    "update": completed,
                    "loss": sum(losses),
                    "gradient_norm": float(gradient_norm),
                    "changed_decoder_elements": changed,
                    "optimizer_step": int(optimizer.state[next(model.action_head.parameters())]["step"]),
                    "frozen_backbone_sha256": backbone_hash,
                    "old_logprob_error": max(errors),
                    "model_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "rewards": [s["reward"] for s in samples],
                    "actions": [s["action"] for s in samples],
                }
            )
            optimizer.zero_grad(set_to_none=True)
            torch.save(
                {
                    "action_head": model.action_head.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "updates": completed,
                    "model_path": str(args.model_path),
                    "denoising_steps": args.denoising_steps,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                args.output_dir / "checkpoint.pt",
            )
            print(json.dumps(metrics[-1]), flush=True)
        connection.send(None)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
