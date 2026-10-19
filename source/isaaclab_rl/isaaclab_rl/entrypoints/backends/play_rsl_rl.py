# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.policy_ghost_sync import PolicyGhostSync
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection
from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from ..policy_ghosts import MAX_POLICY_GHOST_SLOTS, PolicyGhostRestart, PolicyGhostSession, relaunch
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--policy_ghosts",
        action="store_true",
        default=False,
        help=(
            "Overlay several checkpoints of the task in the Newton GL viewer. The --checkpoint drives the reference;"
            " ghosts are added from the viewer HUD. Simulates one slice of --num_envs per slot."
        ),
    )
    parser.add_argument(
        "--policy_ghost_slots",
        type=int,
        default=MAX_POLICY_GHOST_SLOTS,
        help=f"Number of slots, including the reference, with --policy_ghosts (2 to {MAX_POLICY_GHOST_SLOTS}).",
    )
    parser.add_argument(
        "--policy_ghost_root",
        default=None,
        help="Directory searched for ghost checkpoints. Defaults to the experiment's log directory.",
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    if remaining_args_env_registration is not None:
        _, remaining_args_env_registration = setup_preset_cli(parser, remaining_args_env_registration)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def build_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg) -> OnPolicyRunner | DistillationRunner:
    """Construct the RSL-RL runner of an agent config."""
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def run_policy_ghosts(
    env: RslRlVecEnvWrapper,
    args_cli: argparse.Namespace,
    agent_cfg: RslRlBaseRunnerCfg,
    reference_path: str,
    log_root_path: str,
    slice_size: int,
    restart: PolicyGhostRestart,
) -> None:
    """Play the reference checkpoint with ghost slots that the Newton GL viewer HUD fills."""

    def policy_factory(path: str):
        runner = build_runner(env, agent_cfg)
        runner.load(path)
        return runner.get_inference_policy(device=env.unwrapped.device)

    reference_file = Path(reference_path)
    session = PolicyGhostSession(
        env,
        slice_size=slice_size,
        num_slots=args_cli.policy_ghost_slots,
        checkpoint_root=os.path.abspath(args_cli.policy_ghost_root or log_root_path),
        task=args_cli.task,
        library="rsl_rl",
        checkpoint_pattern=r"model_.*\.pt",
        policy_factory=policy_factory,
        reference=PolicyGhostCheckpoint(
            reference_path, reference_file.parent.name, reference_file.name, reference_file.stat().st_mtime
        ),
        metadata={"agent": args_cli.agent},
    )
    attached = False
    for visualizer in env.unwrapped.sim.visualizers:
        if hasattr(visualizer, "attach_policy_ghosts"):
            visualizer.attach_policy_ghosts(session, restart.request)
            attached = True
    if not attached:
        raise RuntimeError("--policy_ghosts needs an initialized Newton GL visualizer.")
    try:
        run_playback(
            _stop_on_restart(session.step, restart),
            dt=env.unwrapped.step_dt,
            args_cli=args_cli,
            env_cfg=env.unwrapped.cfg,
        )
    finally:
        session.close()


def _stop_on_restart(step: Callable[[], None], restart: PolicyGhostRestart) -> Callable[[], None]:
    """Wrap a playback step so that a restart request ends the playback loop.

    The loop in :func:`~isaaclab_rl.entrypoints.common.run_playback` only ends on an interrupt, so the wrapper raises
    :class:`KeyboardInterrupt`, and the caller relaunches once the simulation has shut down.
    """

    def wrapped() -> None:
        if restart.requested:
            raise KeyboardInterrupt
        step()

    return wrapped


def _offer_policy_ghosts(env: RslRlVecEnvWrapper, env_cfg: object, restart: PolicyGhostRestart) -> None:
    """Add an Enable Policy Ghosts button to the Newton GL HUD of a run that started without the feature."""
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        return
    for visualizer in env.unwrapped.sim.visualizers:
        if hasattr(visualizer, "attach_policy_ghost_restart"):
            visualizer.attach_policy_ghost_restart(restart.request)


def _validate_policy_ghosts(args_cli: argparse.Namespace, env_cfg: object) -> None:
    """Reject configurations Policy Ghosts cannot run, before any simulation starts."""
    if not 2 <= args_cli.policy_ghost_slots <= MAX_POLICY_GHOST_SLOTS:
        raise SystemExit(f"--policy_ghost_slots must be between 2 and {MAX_POLICY_GHOST_SLOTS}.")
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise SystemExit("--policy_ghosts only supports manager-based environments.")
    visualizers = env_cfg.sim.visualizer_cfgs or []
    if not any(getattr(cfg, "visualizer_type", None) == "newton_gl" for cfg in visualizers):
        raise SystemExit("--policy_ghosts requires the Newton GL visualizer; pass --visualizer newton_gl.")


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent, restarting if the viewer HUD asks to toggle Policy Ghosts."""
    restart = PolicyGhostRestart()
    _run(argv, restart)
    if restart.requested:
        relaunch(restart.enable)


def _run(argv: list[str], restart: PolicyGhostRestart) -> None:
    """Play a checkpoint of an RSL-RL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        if args_cli.policy_ghosts:
            _validate_policy_ghosts(args_cli, env_cfg)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            # certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            logger.info(f"Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir
            if args_cli.policy_ghosts:
                slice_size = env_cfg.scene.num_envs
                env_cfg.scene.num_envs = slice_size * args_cli.policy_ghost_slots
                PolicyGhostSync.configure_env_cfg(env_cfg)
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            logger.info(f"Loading model checkpoint from: {resume_path}")
            if args_cli.policy_ghosts:
                screen.close()
                run_policy_ghosts(env, args_cli, agent_cfg, resume_path, log_root_path, slice_size, restart)
                return
            runner = build_runner(env, agent_cfg)
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

            obs = env.get_observations()

            def step() -> None:
                nonlocal obs
                obs, _, dones, _ = env.step(policy(obs))
                # reset recurrent states for episodes that have terminated
                policy.reset(dones)

            screen.close()
            _offer_policy_ghosts(env, env_cfg, restart)
            run_playback(_stop_on_restart(step, restart), dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
