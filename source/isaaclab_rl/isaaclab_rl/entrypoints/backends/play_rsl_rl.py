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
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.checkpoint_ghost_sync import CheckpointGhostSync
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection
from isaaclab.visualizers.checkpoint_ghost import CheckpointGhostCheckpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlSacRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..checkpoint_ghosts import (
    MAX_CHECKPOINT_GHOST_SLOTS,
    CheckpointGhostSession,
)
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
from . import cli_args_rsl_rl as cli_args
from .runner_rsl_rl import create_rsl_rl_runner

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
        "--checkpoint_ghosts",
        action="store_true",
        default=False,
        help=(
            "Enable Checkpoint Ghosts, the checkpoint visualization in the Newton GL viewer. The --checkpoint drives"
            " the reference; ghosts are added from the viewer HUD. --num_envs is the total simulated and is split"
            " evenly among the checkpoints shown."
        ),
    )
    parser.add_argument(
        "--checkpoint_ghost_slots",
        type=int,
        default=None,
        help=(
            f"Most checkpoints shown at once, including the reference (2 to {MAX_CHECKPOINT_GHOST_SLOTS}). Defaults to"
            f" {MAX_CHECKPOINT_GHOST_SLOTS}, capped at --num_envs."
        ),
    )
    parser.add_argument(
        "--checkpoint_ghost_root",
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
    if (
        args_cli.checkpoint_ghost_slots is not None
        and not 2 <= args_cli.checkpoint_ghost_slots <= MAX_CHECKPOINT_GHOST_SLOTS
    ):
        raise SystemExit(f"--checkpoint_ghost_slots must be between 2 and {MAX_CHECKPOINT_GHOST_SLOTS}.")
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        if isinstance(agent_cfg, RslRlSacRunnerCfg):
            raise ValueError("Published checkpoints are PPO policies and cannot be loaded by OffPolicyRunner agents.")
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


def create_checkpoint_ghost_session(
    env: RslRlVecEnvWrapper,
    args_cli: argparse.Namespace,
    agent_cfg: RslRlBaseRunnerCfg,
    reference_path: str,
    log_root_path: str,
    num_slots: int,
) -> CheckpointGhostSession:
    """Build a Checkpoint Ghost session whose reference slot plays ``reference_path``."""

    def policy_factory(path: str):
        runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
        runner.load(path)
        return runner.get_inference_policy(device=env.unwrapped.device)

    reference_file = Path(reference_path)
    return CheckpointGhostSession(
        env,
        num_slots=num_slots,
        task=args_cli.task,
        checkpoint_root=os.path.abspath(args_cli.checkpoint_ghost_root or log_root_path),
        checkpoint_pattern=r"model_.*\.pt",
        policy_factory=policy_factory,
        reference=CheckpointGhostCheckpoint(
            reference_path, reference_file.parent.name, reference_file.name, reference_file.stat().st_mtime
        ),
    )


def run_checkpoint_ghosts(
    env: RslRlVecEnvWrapper,
    args_cli: argparse.Namespace,
    agent_cfg: RslRlBaseRunnerCfg,
    reference_path: str,
    log_root_path: str,
) -> None:
    """Play the reference checkpoint with ghost slots that the Newton GL viewer HUD fills."""
    session = create_checkpoint_ghost_session(
        env, args_cli, agent_cfg, reference_path, log_root_path, args_cli.checkpoint_ghost_slots
    )
    attached = False
    for visualizer in env.unwrapped.sim.visualizers:
        if hasattr(visualizer, "attach_checkpoint_ghosts"):
            visualizer.attach_checkpoint_ghosts(session)
            attached = True
    if not attached:
        raise RuntimeError("--checkpoint_ghosts needs an initialized Newton GL visualizer.")
    try:
        run_playback(
            session.step,
            dt=env.unwrapped.step_dt,
            args_cli=args_cli,
            env_cfg=env.unwrapped.cfg,
        )
    finally:
        session.close()


def _offer_checkpoint_ghosts(env: RslRlVecEnvWrapper, env_cfg: object) -> None:
    """Show the greyed-out Checkpoint Ghosts section in the Newton GL HUD of a run that started without the feature."""
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        return
    if _scene_cameras_use_ovrtx(env_cfg):  # not supported with OVRTX cameras yet
        return
    for visualizer in env.unwrapped.sim.visualizers:
        if hasattr(visualizer, "offer_checkpoint_ghosts"):
            visualizer.offer_checkpoint_ghosts()


def _resolve_ghost_slots(args_cli: argparse.Namespace, env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Set ``args_cli.checkpoint_ghost_slots``, the most checkpoints that can be shown at once.

    ``--num_envs`` is the total simulated; the environments are split among the loaded checkpoints, so every slot
    needs at least one.
    """
    total = env_cfg.scene.num_envs
    if total < 2:
        raise SystemExit("--checkpoint_ghosts needs --num_envs of at least 2: one environment per checkpoint.")
    slots = args_cli.checkpoint_ghost_slots
    if slots is None:
        slots = MAX_CHECKPOINT_GHOST_SLOTS
    if slots > total:
        if args_cli.checkpoint_ghost_slots is not None:
            logger.warning(f"--checkpoint_ghost_slots {slots} exceeds --num_envs {total}; using {total} slots.")
        slots = total
    args_cli.checkpoint_ghost_slots = min(slots, MAX_CHECKPOINT_GHOST_SLOTS)


def _validate_checkpoint_ghosts(args_cli: argparse.Namespace, env_cfg: object) -> None:
    """Reject configurations Checkpoint Ghosts cannot run, before any simulation starts."""
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise SystemExit("--checkpoint_ghosts only supports manager-based environments.")
    visualizers = env_cfg.sim.visualizer_cfgs or []
    if not isinstance(visualizers, (list, tuple)):
        visualizers = [visualizers]
    # A shared visualizer configuration names its type on its backend; a deprecated combined one names it directly.
    types = []
    for cfg in visualizers:
        backend = getattr(cfg, "backend", None)
        for chosen in backend if isinstance(backend, (list, tuple)) else [backend]:  # --viz newton,viser selects both
            types.append(getattr(chosen, "visualizer_type", None) or getattr(cfg, "visualizer_type", None))
    if "newton" not in types:
        raise SystemExit("--checkpoint_ghosts requires the Newton GL visualizer; pass --visualizer newton.")
    curriculum = getattr(env_cfg, "curriculum", None)
    for name, term in vars(curriculum).items() if curriculum is not None else ():
        # Checkpoint Ghosts disable curriculum terms, so a task whose reset events read a term's state cannot run.
        if getattr(getattr(term, "func", None), "supports_checkpoint_ghosts", True) is False:
            raise SystemExit(
                f"--checkpoint_ghosts does not support this task: its curriculum term '{name}' sets the reset state."
            )
    if _scene_cameras_use_ovrtx(env_cfg):
        raise SystemExit(
            "--checkpoint_ghosts does not support tasks whose cameras use the OVRTX renderer yet."
            " Use another renderer, such as renderer=newton_warp."
        )


def _scene_cameras_use_ovrtx(env_cfg: object) -> bool:
    """Whether a scene sensor renders with OVRTX, which Checkpoint Ghosts do not support yet."""
    scene = getattr(env_cfg, "scene", None)
    for value in vars(scene).values() if scene is not None else ():
        if getattr(value, "render_on_demand", False) is True:
            continue  # the Newton viewer's own RTX camera, which launch adds and which no policy observes
        renderer_cfg = getattr(value, "renderer_cfg", None)
        if getattr(renderer_cfg, "renderer_type", None) == "ovrtx":
            return True
    return False


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        if args_cli.checkpoint_ghosts:
            _validate_checkpoint_ghosts(args_cli, env_cfg)
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
            if args_cli.checkpoint_ghosts:
                _resolve_ghost_slots(args_cli, env_cfg)
                CheckpointGhostSync.configure_env_cfg(env_cfg)
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
            if args_cli.checkpoint_ghosts:
                if args_cli.deterministic:
                    configure_seed(env_cfg.seed, torch_deterministic=True)
                screen.close()
                run_checkpoint_ghosts(env, args_cli, agent_cfg, resume_path, log_root_path)
                return
            runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path, map_location=agent_cfg.device)
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
            _offer_checkpoint_ghosts(env, env_cfg)
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
