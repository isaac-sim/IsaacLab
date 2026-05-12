# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI/automation entry point for replaying captured Isaac Teleop sessions.

This is the non-interactive counterpart to ``teleop_se3_agent.py``. It builds
a teleop environment, attaches an :class:`~isaaclab_teleop.IsaacTeleopDevice`
configured in :class:`isacteleop.teleop_session_manager.SessionMode.REPLAY`,
and pumps the simulation loop until ``--max_replay_duration_s`` elapses
(post-quit) or Kit is closed. The user-journey teleop script remains
``teleop_se3_agent.py``.

Inputs:
    ``--replay_file`` is an MCAP capture produced by Isaac Teleop's
    ``McapRecordingConfig`` path. The recorder lays down per-tracker
    flatbuffer messages (head / hands / controllers) plus a
    ``_teleop_control`` ``MessageChannelTracker`` for control events.
    TeleopCore's :class:`~isacteleop.deviceio_session.ReplaySession`
    only supports tracker-shaped data (no ``MessageChannelTracker``
    replay), so the control-events channel is recorded but cannot be
    re-emitted today; the replay agent therefore steps the env on
    every non-None action and bounds the run with
    ``--max_replay_duration_s``.

Warmup:
    Before stepping the env, the agent waits deterministically for Kit
    to finish loading the USD stage by polling
    ``omni.usd.UsdContext.get_stage_loading_status()`` until no assets
    are pending (bounded by ``--max_stage_load_wait_s`` as a safety net).
    It then pumps a fixed number of additional renderer-settle frames so
    shaders / articulation views finish warming up before any action
    lands. ``--replay_start_delay_s`` is available as an optional
    wall-clock buffer on top of the deterministic wait for hardware
    that needs more grace. During warmup the agent does not call
    :meth:`IsaacTeleopDevice.advance`, so ``ReplaySession.update()``
    does not advance through the MCAP.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Replay a captured Isaac Teleop MCAP session against an Isaac Lab environment. "
        "CI/automation entry point; for interactive teleoperation see teleop_se3_agent.py."
    )
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--replay_file",
    type=str,
    required=True,
    help="Absolute path to the Isaac Teleop MCAP capture to replay.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=1,
    help=(
        "Number of consecutive steps the task success term must hold before declaring success and"
        " resetting the env. Mirrors the equivalent flag in record_demos.py."
    ),
)
parser.add_argument(
    "--max_replay_duration_s",
    type=float,
    default=600.0,
    help=(
        "Maximum wall-clock seconds to keep the replay loop running before asking Kit to quit,"
        " measured from the end of the warmup window (see --replay_start_delay_s)."
        " TeleopCore's ReplaySession does not expose a playback-finished signal, so a hard cap"
        " is the deterministic way to terminate CI replays. Default is 600s (10 min)."
    ),
)
parser.add_argument(
    "--replay_start_delay_s",
    type=float,
    default=0.0,
    help=(
        "Optional wall-clock buffer added on top of the deterministic stage-load wait."
        " The agent always blocks until omni.usd reports no assets pending and then renders a"
        " fixed number of settle frames before consuming MCAP frames; this flag inserts an"
        " additional render-only window after that if the deterministic check is not enough"
        " for a given hardware/asset combination. Default is 0s -- bump it if you still see"
        " a race after the deterministic wait."
    ),
)
parser.add_argument(
    "--max_stage_load_wait_s",
    type=float,
    default=300.0,
    help=(
        "Safety cap on how long to wait for omni.usd to finish loading the stage before"
        " proceeding anyway. Hit only when something is misconfigured (missing asset, slow"
        " Nucleus, etc.); a warning is logged and replay continues. Default is 300s."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging
import time

import gymnasium as gym
import torch
from isaaclab_teleop import IsaacTeleopDevice, create_isaac_teleop_device

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)


def _prepare_env_cfg(task: str, num_envs: int, device: str) -> tuple[ManagerBasedRLEnvCfg, object | None]:
    """Build and tweak an env config suitable for non-interactive replay.

    Mirrors the env-config mutations performed by ``record_demos.py``'s
    :func:`create_environment_config`:

    * The ``success`` term is extracted and cleared from the env config so the
      script can drive success detection (and the matching reset cycle)
      explicitly via :func:`_process_success_condition`, gated by
      ``--num_success_steps``. This matches record_demos.py's pattern of
      manually counting consecutive success steps before resetting.
    * Every other termination term -- including ``time_out`` and any
      task-specific failure terms (e.g. ``object_dropping``,
      ``object_too_far``) -- is left active. ``env.step`` then auto-invokes
      ``_reset_idx`` for any env whose termination fires; the main loop
      detects this via the returned ``terminated``/``truncated`` tensors
      and completes the reset cycle (sim reinit + teleop device reset)
      so Pink IK starts the next attempt with fresh articulation views.

    Returns:
        Tuple ``(env_cfg, success_term)``. ``success_term`` is ``None`` when
        the env doesn't define a ``success`` termination term.
    """
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.env_name = task.split(":")[-1]
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "teleop_replay_agent only supports ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    success_term: object | None = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment;"
            " success-driven resets will not fire during replay."
        )
    env_cfg = remove_camera_configs(env_cfg)
    env_cfg.sim.render.antialiasing_mode = "DLSS"
    return env_cfg, success_term


def _handle_reset(env: gym.Env, teleop_interface: IsaacTeleopDevice) -> None:
    """Run the full env+teleop reset cycle used by ``record_demos.py``.

    Mirrors :func:`scripts.tools.record_demos.handle_reset` (sans the
    instruction-display update, which the headless replay agent doesn't
    own). ``env.sim.reset()`` does the hard physics reinit that keeps Pink
    IK seeded against fresh articulation views; see the initial-reset note
    in :func:`main`. ``env.recorder_manager.reset()`` is a no-op when no
    recorders are configured (the default for this script), but kept for
    parity with record_demos.py so future recorder additions don't have to
    re-derive the call sequence.
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    teleop_interface.reset()


def _process_success_condition(
    env: gym.Env,
    success_term: object | None,
    success_step_count: int,
    num_success_steps: int,
) -> tuple[int, bool]:
    """Track consecutive success steps and decide whether to reset.

    Mirrors :func:`scripts.tools.record_demos.process_success_condition`
    minus the recorder-export side effects, which this script does not own.

    Returns:
        Tuple ``(updated_success_step_count, reset_due_to_success)``.
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= num_success_steps:
            print(f"Success condition met after {success_step_count} consecutive steps; resetting env.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


_RENDERER_SETTLE_FRAMES: int = 30
"""Number of additional render frames pumped after the USD stage finishes loading.

Kit's stage-load status flips to ``count_loading == 0`` as soon as every referenced asset
has been resolved, but the renderer pipeline (shader compilation, articulation-view
binding, material warm-up) typically needs a few more event-loop ticks to converge. Thirty
frames at the default Kit render cadence is ~0.5 s on most machines and is deterministic
per-machine -- unlike a wall-clock delay it does not have to be tuned for hardware.
"""


def _wait_for_stage_load(simulation_app, max_wait_s: float) -> None:
    """Block until the USD stage finishes resolving every referenced asset.

    Polls :meth:`omni.usd.UsdContext.get_stage_loading_status`. The third element of
    the returned tuple is the count of assets Kit still has pending; when it reaches
    zero the stage is fully streamed in and the renderer pipeline is ready to draw
    against it. After the count reaches zero this function pumps an additional
    :data:`_RENDERER_SETTLE_FRAMES` ``simulation_app.update()`` calls so shaders,
    materials, and articulation views finish warming up before the caller begins
    consuming MCAP frames or stepping the env.

    Args:
        simulation_app: The :class:`isaaclab.app.SimulationApp` instance whose
            event loop to pump while waiting.
        max_wait_s: Upper bound on how long to spin on a non-zero loading count
            before warning and returning. Acts as a safety net for misconfigured
            scenes (missing assets, slow Nucleus); a successful run typically
            completes well within this bound.

    The function is best-effort: when ``omni.usd`` is unavailable (e.g. when
    running outside a Kit context) it returns immediately so callers do not
    need a separate code path.
    """
    try:
        import omni.usd
    except (ImportError, ModuleNotFoundError):
        logger.warning("omni.usd not available; skipping deterministic stage-load wait")
        return

    print("Waiting for USD stage to finish loading...")
    start_s = time.monotonic()
    last_progress_log_s = start_s
    while simulation_app.is_running():
        context = omni.usd.get_context()
        if context is None:
            break
        # get_stage_loading_status -> (message, count_loaded, count_loading)
        _, _, count_loading = context.get_stage_loading_status()
        if count_loading == 0:
            break
        elapsed_s = time.monotonic() - start_s
        if elapsed_s >= max_wait_s:
            logger.warning(
                "Stage still reports %d assets pending after %.1fs; proceeding anyway. Replay may race the renderer.",
                count_loading,
                max_wait_s,
            )
            break
        if time.monotonic() - last_progress_log_s >= 5.0:
            print(f"  stage loading: {count_loading} assets pending (elapsed {elapsed_s:.1f}s)")
            last_progress_log_s = time.monotonic()
        simulation_app.update()

    elapsed_s = time.monotonic() - start_s
    print(f"Stage load complete after {elapsed_s:.1f}s; settling renderer for {_RENDERER_SETTLE_FRAMES} frames...")
    for _ in range(_RENDERER_SETTLE_FRAMES):
        if not simulation_app.is_running():
            return
        simulation_app.update()


def _request_kit_quit() -> None:
    """Ask Kit to drain its event loop and exit.

    Used after ``--max_replay_duration_s`` elapses so CI processes
    terminate deterministically. The host process needs Kit to flip
    ``simulation_app.is_running()`` to ``False``; ``post_quit`` is the
    canonical Kit-side path. ``ReplaySession`` does not emit a
    playback-finished signal, and the recorded ``_teleop_control``
    message channel cannot be re-emitted by TeleopCore (no
    ``ReplayMessageChannelTrackerImpl``), so a wall-clock cap is what
    we have today.
    """
    try:
        import omni.kit.app

        omni.kit.app.get_app().post_quit()
    except Exception:
        logger.exception("Failed to post_quit at end of replay; the loop will keep running")


def main() -> None:
    """Replay a captured Isaac Teleop session against an Isaac Lab environment.

    Builds the env, attaches a replay-mode :class:`IsaacTeleopDevice`, and
    pumps the simulation loop until ``--max_replay_duration_s`` elapses or
    Kit is closed.

    Control flow:
        * Pre-loop warmup: ``_wait_for_stage_load`` polls
          ``omni.usd.UsdContext.get_stage_loading_status`` until Kit
          reports zero pending assets, then renders a fixed number of
          settle frames. An optional ``--replay_start_delay_s`` buffer
          can be appended for hardware that needs more grace.
          ``advance()`` is not called during warmup so
          ``ReplaySession.update`` does not consume MCAP frames yet.
        * Main loop: :meth:`IsaacTeleopDevice.advance` returns an action
          tensor derived from the MCAP-replayed tracker stream. The env
          steps on every non-None action; the agent does **not** gate on
          recorded START/STOP events because TeleopCore's
          ``ReplaySession`` rejects the ``MessageChannelTracker`` schema
          and therefore cannot reproduce the control channel during
          replay.
        * Failure terminations (``time_out`` / task failure) and success
          (gated by ``--num_success_steps``) still drive the full reset
          cycle so multi-episode recordings are replayed end-to-end as
          long as the trajectory remains within env tolerances.
        * After ``--max_replay_duration_s`` wall-clock seconds (measured
          from end-of-warmup) the agent asks Kit to ``post_quit`` so the
          host process exits cleanly.

    Resource cleanup is wrapped in a ``try/finally`` so that ``env.close()``
    always runs, even when device construction or any subsequent setup
    raises -- otherwise the USD stage would leak across CI runs.
    """
    env: gym.Env | None = None
    try:
        env_cfg, success_term = _prepare_env_cfg(args_cli.task, args_cli.num_envs, args_cli.device)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        if not hasattr(env_cfg, "isaac_teleop") or env_cfg.isaac_teleop is None:
            raise ValueError(
                f"Task '{args_cli.task}' does not configure an IsaacTeleop pipeline. "
                "MCAP replay requires env_cfg.isaac_teleop to be set."
            )

        teleop_interface = create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=args_cli.device,
            callbacks={},
            cloudxr_env_file=None,
            auto_launch_cloudxr=False,
            mcap_replay_path=args_cli.replay_file,
        )
        print(f"Using teleop device: {teleop_interface}")

        with teleop_interface:
            # Mirror the reset sequence used by ``record_demos.py``: ``sim.reset()``
            # does a hard physics reinit (re-binds articulation views, plays the
            # timeline) that ``env.reset()`` alone does not perform. Pink IK reads
            # ``data.joint_pos.torch`` every step to seed Pinocchio's configuration
            # and to compute ``target = curr + delta``; if the articulation view is
            # stale, every IK call produces zero-delta arm targets while the
            # hand-finger path (which bypasses IK) keeps tracking. See PR #5507.
            env.sim.reset()
            env.reset()
            teleop_interface.reset()

            # Deterministic warmup: block until omni.usd reports zero pending
            # assets, then pump a fixed number of renderer-settle frames.
            # ``TeleopSession.__enter__`` already opened the MCAP, but
            # ``ReplaySession.update`` only advances when ``advance()`` is
            # called -- so no MCAP frames are consumed during this window.
            _wait_for_stage_load(simulation_app, args_cli.max_stage_load_wait_s)

            # Optional extra wall-clock buffer on top of the deterministic
            # wait. Useful as an escape hatch when the deterministic check
            # is not enough (e.g. very slow shader compilation paths).
            if args_cli.replay_start_delay_s > 0:
                print(
                    f"Additional warmup buffer: rendering for {args_cli.replay_start_delay_s:.1f}s"
                    " before consuming MCAP frames."
                )
                buffer_start_s = time.monotonic()
                while simulation_app.is_running() and time.monotonic() - buffer_start_s < args_cli.replay_start_delay_s:
                    env.sim.render()

            print(
                f"Replay agent started; replaying MCAP from {args_cli.replay_file}"
                f" (max_replay_duration_s={args_cli.max_replay_duration_s:.1f})."
            )
            success_step_count = 0
            replay_start_s = time.monotonic()
            quit_requested = False

            while simulation_app.is_running():
                try:
                    with torch.inference_mode():
                        # Bound the run so CI does not hang if the MCAP has no
                        # natural end-of-stream signal.
                        elapsed_s = time.monotonic() - replay_start_s
                        if not quit_requested and elapsed_s >= args_cli.max_replay_duration_s:
                            print(
                                f"Replay reached max_replay_duration_s={args_cli.max_replay_duration_s:.1f};"
                                " asking Kit to quit."
                            )
                            _request_kit_quit()
                            quit_requested = True

                        action = teleop_interface.advance()

                        if action is None:
                            env.sim.render()
                            continue

                        actions = action.repeat(env.num_envs, 1)
                        _, _, terminated, truncated, _ = env.step(actions)

                        # Failure path: ``env.step`` already invoked ``_reset_idx``
                        # for any env whose ``time_out`` or task-specific failure
                        # term fired (success was extracted up front so it does
                        # not show up here). We still need to refresh sim physics
                        # state and the teleop device so Pink IK starts the next
                        # attempt with fresh articulation views.
                        if bool(terminated.any().item()) or bool(truncated.any().item()):
                            print("Failure condition met (terminated/timed-out); resetting env.")
                            _handle_reset(env, teleop_interface)
                            success_step_count = 0
                            continue

                        # Success path: ``success_term`` was cleared from the
                        # env cfg so ``env.step`` does not auto-reset on it.
                        # Mirror record_demos.py and trigger a reset only after
                        # the success condition has held for
                        # ``--num_success_steps`` consecutive steps.
                        success_step_count, reset_on_success = _process_success_condition(
                            env, success_term, success_step_count, args_cli.num_success_steps
                        )
                        if reset_on_success:
                            _handle_reset(env, teleop_interface)
                            success_step_count = 0
                except Exception:
                    # ``logger.exception`` preserves the full traceback; bare
                    # ``logger.error`` would only log the message.
                    logger.exception("Error during simulation step")
                    break
    finally:
        if env is not None:
            env.close()
            print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.update()
    simulation_app.close()
