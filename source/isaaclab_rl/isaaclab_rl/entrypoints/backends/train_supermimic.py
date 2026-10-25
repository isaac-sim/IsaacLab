# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import string_to_callable, to_dict
from isaaclab.utils.noise import NoiseCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.g1_tracking.g1_tracking_env_cfg import (
    G1TrackingEnvCfg,
    ObservationsCfg,
    TerminationsCfg,
    TransitionRecorderCfg,
)
from isaaclab_tasks.contrib.g1_tracking.tracking import MAX_REWARD_RATE, OBS_DIM
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...supermimic.learner import Learner
from ...supermimic.networks import WM_IN, WM_OUT, WorldModel
from ..common import (
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_seed,
    scoped_torch_backend_flags,
    wrap_sensor_capture,
    write_run_manifest,
)


def run(argv: list[str]) -> None:
    """Fit dynamics from real transitions and improve a policy through imagined tracking rollouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(
        parser,
        agent_default="supermimic_cfg_entry_point",
        agent_help="SuperMimic agent configuration.",
        include_distributed=False,
    )
    parser.add_argument("--checkpoint", help="Full native checkpoint to resume; physical state and replay restart.")
    parser.add_argument("--log_dir", help="Output directory; defaults to logs/supermimic/<experiment>/<timestamp>.")
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if args.frontend != "torch":
        parser.error("SuperMimic currently supports the native Torch environment frontend.")
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, overrides=overrides)
    _configure_training(args, env_cfg, agent_cfg)

    output = Path(
        args.log_dir
        or Path("logs/supermimic") / agent_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    resumed = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
    # Capture semantic configuration before adding run-specific log paths and video outputs.
    saved_env_cfg, saved_agent_cfg = to_dict(env_cfg), to_dict(agent_cfg)
    if resumed:
        if resumed["task"] != args.task or resumed["env_cfg"] != saved_env_cfg:
            raise ValueError("Resume requires the checkpoint's task and environment configuration.")
        mutable = {
            "max_iterations",
            "max_hours",
            "save_interval",
            "save_interval_seconds",
            "log_interval",
            "wandb_project",
            "wandb_entity",
            "wandb_run_id",
            "wandb_name",
        }
        changed = {
            key for key, value in saved_agent_cfg.items() if key not in mutable and resumed["agent_cfg"][key] != value
        }
        if changed or resumed["iteration"] >= agent_cfg.max_iterations:
            raise ValueError(f"Resume requires unchanged learning settings and a later iteration limit: {changed}.")
    identity = resumed.get("wandb") if resumed else None
    if identity and agent_cfg.wandb_run_id is not None and agent_cfg.wandb_run_id != identity["run_id"]:
        identity = None
    if identity:
        for key in ("project", "entity", "run_id"):
            value = getattr(agent_cfg, f"wandb_{key}")
            if value is not None and value != identity[key]:
                raise ValueError(f"Resuming a W&B run must preserve its {key}.")
            setattr(agent_cfg, f"wandb_{key}", identity[key])
    logging = nullcontext(None)
    if agent_cfg.wandb_project:
        import wandb

        logging = wandb.init(
            project=agent_cfg.wandb_project,
            entity=agent_cfg.wandb_entity,
            id=agent_cfg.wandb_run_id,
            name=agent_cfg.wandb_name,
            resume="allow",
            dir=str(output),
            config={"task": args.task, "agent": saved_agent_cfg, "env": saved_env_cfg},
            allow_val_change=True,
        )
    env_cfg.log_dir = str(output)
    enable_cameras_for_video(args)
    pre_launch_video_config(env_cfg, args)
    apply_video_recording(env_cfg, str(output), args)
    write_run_manifest(str(output), library="supermimic", task=args.task, metadata={"agent": args.agent})
    dump_train_configs(str(output), env_cfg, agent_cfg)
    torch.manual_seed(agent_cfg.seed)
    with (
        logging as run,
        scoped_torch_backend_flags(
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True, cudnn_deterministic=False, cudnn_benchmark=False
        ),
        launch_simulation(env_cfg, args),
    ):
        if run:
            run.define_metric("iteration")
            run.define_metric("*", step_metric="iteration")
            identity = {"entity": run.entity, "project": run.project, "run_id": run.id, "url": run.url}
            (output / "wandb-run.json").write_text(json.dumps(identity, indent=2) + "\n")
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        env = wrap_sensor_capture(env, str(output), args)
        metadata = {
            "task": args.task,
            "agent_cfg": saved_agent_cfg,
            "env_cfg": saved_env_cfg,
            "wandb": identity,
            "state_layout": "root-link-xyzw-v-w-q-qd",
        }
        try:
            _train(env, agent_cfg, output, resumed, run, metadata)
        finally:
            env.close()


def _configure_training(args, env_cfg, agent_cfg):
    """Apply native overrides and validate that real and imagined task semantics agree."""
    if not isinstance(env_cfg, G1TrackingEnvCfg):
        raise ValueError("SuperMimic currently requires the G1 motion-tracking task's analytical objective.")
    apply_env_overrides(args, env_cfg)
    if args.seed is not None:
        agent_cfg.seed = resolve_seed(args.seed)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = agent_cfg.seed
    if env_cfg.recorders.transition is None:
        env_cfg.recorders.transition = TransitionRecorderCfg()
    if min(agent_cfg.buffer_steps, agent_cfg.warmup_steps) < agent_cfg.world_model_window:
        raise ValueError("Replay capacity and warmup must each cover a full world-model window.")
    if (
        min(
            agent_cfg.max_iterations,
            agent_cfg.horizon,
            agent_cfg.world_model_window,
            agent_cfg.world_model_batches,
            agent_cfg.world_model_batch_size,
            agent_cfg.policy_batch_size,
            agent_cfg.critic_epochs,
            agent_cfg.critic_minibatches,
            agent_cfg.learning_rate_decay_iterations,
            agent_cfg.save_interval,
            agent_cfg.log_interval,
            env_cfg.observations.policy.history_length,
            env_cfg.commands.motion.goal_frames,
        )
        < 1
    ):
        raise ValueError("Learning counts, observation history and goal lengths must be positive.")
    if agent_cfg.critic_minibatches > agent_cfg.horizon * agent_cfg.policy_batch_size:
        raise ValueError("Critic minibatches must contain samples.")
    if not 0 <= env_cfg.commands.motion.gamma < 1 or not 0 <= agent_cfg.gae_lambda <= 1:
        raise ValueError("Discount must be in [0, 1) and GAE lambda in [0, 1].")
    if not math.isfinite(agent_cfg.action_noise) or agent_cfg.action_noise < 0:
        raise ValueError("Action noise must be finite and nonnegative.")
    if agent_cfg.world_model_only_iterations < 0:
        raise ValueError("World-model-only iterations must be nonnegative.")
    for value in (agent_cfg.update_to_data_ratio, agent_cfg.learning_rate, agent_cfg.save_interval_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rates, update/data ratio and checkpoint intervals must be finite and positive.")
    if agent_cfg.max_hours is not None and (not math.isfinite(agent_cfg.max_hours) or agent_cfg.max_hours <= 0):
        raise ValueError("The training-hour budget must be finite and positive.")
    if agent_cfg.amp not in ("none", "bf16") or agent_cfg.world_model not in ("mlp", "nerd-one-frame"):
        raise ValueError("Select amp=none/bf16 and world_model=mlp/nerd-one-frame.")
    if env_cfg.sim.dt * env_cfg.decimation != 0.02:
        raise ValueError("The reference motions and world model require a 20 ms control period.")
    # The same analytical observations are used inside imagined rollouts. Reject changes that
    # preserve tensor width but alter their meaning; native history and stateless noise remain configurable.
    template = ObservationsCfg()
    for name in ("policy", "goal"):
        expected, actual = to_dict(getattr(template, name)), to_dict(getattr(env_cfg.observations, name))
        if name == "policy":
            for group in (expected, actual):
                group.pop("history_length")
                group.pop("enable_corruption")
                for term in group.values():
                    if isinstance(term, dict):
                        term.pop("noise", None)
        if actual != expected or tuple(actual) != tuple(expected):
            raise ValueError(f"Analytical {name} observations require the task's declared functions and layout.")
    if to_dict(env_cfg.terminations) != to_dict(TerminationsCfg()):
        raise ValueError("Analytical termination requires the task's declared failure, clip-end and timeout terms.")


def _train(env, agent_cfg, output, resumed, run, metadata):
    """Own the learning schedule, checkpoint publication and signal handling for one environment."""
    if run:
        import wandb

    previous_handlers = {}
    try:
        native_env = env.unwrapped
        command = native_env.command_manager.get_term("motion")
        capture = native_env.recorder_manager.get_term("transition")
        observations, _ = env.reset()
        from isaaclab_tasks.contrib.g1_tracking.mdp import tracking_reward

        reward_cfg = native_env.reward_manager.get_term_cfg("tracking")
        reward_func = string_to_callable(reward_cfg.func) if isinstance(reward_cfg.func, str) else reward_cfg.func
        if (
            native_env.reward_manager.active_terms != ["tracking"]
            or reward_func is not tracking_reward
            or reward_cfg.weight <= 0
        ):
            raise ValueError("The imagined objective requires the task's single tracking reward with positive weight.")
        noise_terms, offset = [], 0
        group = native_env.cfg.observations.policy
        for name, shape in zip(
            native_env.observation_manager.active_terms["policy"],
            native_env.observation_manager.group_obs_term_dim["policy"],
            strict=True,
        ):
            term = getattr(group, name)
            noise = term.noise if group.enable_corruption else None
            if noise is not None and not isinstance(noise, NoiseCfg):
                raise ValueError("Analytical observations require stateless policy noise.")
            fn = (
                None if noise is None else string_to_callable(noise.func) if isinstance(noise.func, str) else noise.func
            )
            noise_terms.append((slice(offset, offset + shape[-1]), fn, noise))
            offset += shape[-1]
        if offset != OBS_DIM or int(command.motion.clip_len.max()) <= agent_cfg.horizon:
            raise ValueError("Observation width or motion lengths are incompatible with the learner.")
        network = None
        if agent_cfg.world_model == "nerd-one-frame":
            from physicsnemo_robotics.models.parallel_nerd import ParallelNeRD

            network = ParallelNeRD(WM_IN, WM_OUT, temporal_attention_spans=(1, 1, 1, 1))
            torch.nn.init.zeros_(network.head_output.weight)
            torch.nn.init.zeros_(network.head_output.bias)
        dtype = torch.bfloat16 if agent_cfg.amp == "bf16" else None
        world_model = WorldModel(network=network, amp_dtype=dtype).to(native_env.device)
        learner = Learner(
            agent_cfg,
            command.mdp,
            capture.latest,
            tuple(noise_terms),
            native_env.step_dt * reward_cfg.weight,
            world_model,
            goal_frames=command.cfg.goal_frames,
            history=group.history_length,
            gamma=command.cfg.gamma,
        )
        first_iteration, prior_wall_time, sim_steps = 0, 0.0, 0
        if resumed:
            for name in ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v"):
                getattr(learner, name).load_state_dict(resumed[name])
            command.bin_return.copy_(resumed["bin_return"])
            command.bin_weight.copy_(resumed["bin_weight"])
            first_iteration, prior_wall_time, sim_steps = (
                resumed["iteration"],
                resumed["wall_time"],
                resumed["sim_steps"],
            )
            del resumed
        stop_signal = 0

        def request_stop(signum, frame):
            nonlocal stop_signal
            stop_signal = signum

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        stats = {name: [] for name in ("reward", "length", "return", "failed", "done")}

        @torch.no_grad()
        def collect(count):
            nonlocal observations, sim_steps
            for _ in range(count):
                observations, _, _, _, extras = env.step(learner.act(observations))
                transition = capture.latest
                learner.observe(transition)
                done = transition["done"]
                done_count = done.sum()
                stats["reward"].append(transition["r"].mean() / (MAX_REWARD_RATE * learner.reward_scale))
                stats["length"].append(((transition["ep_step"] + 1) * done).sum())
                stats["return"].append(extras["log"].get("Metrics/motion/return", 0.0) * done_count)
                stats["failed"].append(transition["term"].sum())
                stats["done"].append(done_count)
                sim_steps += native_env.num_envs

        collect(agent_cfg.warmup_steps)
        training_start = log_start = checkpoint_time = time.time()
        step_accumulator = 0.0
        aggregate = {name: [] for name in ("wm", "wm_last", "pi", "v", "imagined_reward", "gradient", "survival")}
        columns = (
            "iteration",
            "wall_h",
            "sim_steps",
            "wm_loss",
            "wm_last_error",
            "pi_loss",
            "v_loss",
            "imagined_reward",
            "actor_gradient",
            "survival",
            "sim_reward",
            "episode_length",
            "episode_return",
            "failure_rate",
            "iterations_per_second",
            "learning_rate",
        )
        log_path = output / "train_log.csv"
        if first_iteration and log_path.exists():
            lines = log_path.read_text().splitlines()
            lines = lines[:1] + [line for line in lines[1:] if int(float(line.split(",")[0])) <= first_iteration]
            log_path.write_text("\n".join(lines) + "\n")
        else:
            log_path.write_text(",".join(columns) + "\n")
        with log_path.open("a") as log:
            for iteration in range(first_iteration, agent_cfg.max_iterations):
                fraction = min(iteration / agent_cfg.learning_rate_decay_iterations, 1.0)
                lr = agent_cfg.learning_rate * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * fraction)))
                for optimizer in (learner.opt_wm, learner.opt_pi, learner.opt_v):
                    for group_cfg in optimizer.param_groups:
                        group_cfg["lr"] = lr
                step_accumulator += 1.0 / agent_cfg.update_to_data_ratio
                count = int(step_accumulator)
                step_accumulator -= count
                collect(count)
                for _ in range(agent_cfg.world_model_batches):
                    loss, error = learner.wm_update()
                aggregate["wm"].append(loss)
                aggregate["wm_last"].append(error)
                if iteration >= agent_cfg.world_model_only_iterations:
                    results = learner.policy_update()
                    for name, value in zip(
                        ("pi", "v", "imagined_reward", "gradient", "survival"), results, strict=True
                    ):
                        aggregate[name].append(value)
                if (iteration + 1) % agent_cfg.log_interval == 0:
                    means = [float(torch.stack(values).mean()) if values else 0.0 for values in aggregate.values()]
                    totals = {
                        name: float(torch.stack(values).sum()) if values else 0.0 for name, values in stats.items()
                    }
                    done_count = max(totals["done"], 1.0)
                    wall_h = (prior_wall_time + time.time() - training_start) / 3600
                    row = [
                        iteration + 1,
                        wall_h,
                        sim_steps,
                        *means,
                        totals["reward"] / max(len(stats["reward"]), 1),
                        totals["length"] / done_count,
                        totals["return"] / done_count,
                        totals["failed"] / done_count,
                        agent_cfg.log_interval / (time.time() - log_start),
                        lr,
                    ]
                    if not all(math.isfinite(value) for value in row):
                        raise FloatingPointError(f"Nonfinite training statistics at update {iteration + 1}: {row}")
                    log.write(",".join(f"{value:.6g}" for value in row) + "\n")
                    log.flush()
                    if run:
                        run.log(dict(zip(columns, row, strict=True)))
                    print(
                        f"Iteration {iteration + 1}: dynamics={means[0]:.4f}, policy={means[2]:.4f}, "
                        f"value={means[3]:.4f}, hours={wall_h:.3f}",
                        flush=True,
                    )
                    log_start = time.time()
                    for values in (*stats.values(), *aggregate.values()):
                        values.clear()
                elapsed = prior_wall_time + time.time() - training_start
                stop = agent_cfg.max_hours is not None and elapsed >= agent_cfg.max_hours * 3600
                milestone = (
                    (iteration + 1) % agent_cfg.save_interval == 0 or iteration + 1 == agent_cfg.max_iterations or stop
                )
                if (
                    milestone
                    or stop_signal
                    or iteration == first_iteration
                    or time.time() - checkpoint_time >= agent_cfg.save_interval_seconds
                ):
                    checkpoint = {
                        **metadata,
                        **{
                            name: getattr(learner, name).state_dict()
                            for name in ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v")
                        },
                        "iteration": iteration + 1,
                        "wall_time": elapsed,
                        "sim_steps": sim_steps,
                        "bin_return": command.bin_return,
                        "bin_weight": command.bin_weight,
                        "saved_at": datetime.now().isoformat(),
                    }
                    torch.save(checkpoint, output / "last.pt.tmp")
                    os.replace(output / "last.pt.tmp", output / "last.pt")
                    if milestone:
                        policy_checkpoint = {
                            key: value
                            for key, value in checkpoint.items()
                            if key not in ("wm", "critic", "opt_wm", "opt_pi", "opt_v")
                        }
                        path = output / f"policy_{iteration + 1:06d}.pt"
                        torch.save(policy_checkpoint, path.with_suffix(".tmp"))
                        os.replace(path.with_suffix(".tmp"), path)
                    if run:
                        artifact = wandb.Artifact(
                            f"run-{run.id}-checkpoint", type="model", metadata={"iteration": iteration + 1}
                        )
                        artifact.add_file(str(output / "last.pt"), name="last.pt")
                        run.log_artifact(artifact, aliases=["latest"]).wait(timeout=180)
                        run.summary["durable_checkpoint_iteration"] = iteration + 1
                    checkpoint_time = time.time()
                if stop_signal:
                    raise SystemExit(128 + stop_signal)
                if stop:
                    break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
