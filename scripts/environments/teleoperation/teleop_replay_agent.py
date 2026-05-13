# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI/automation entry point for replaying captured Isaac Teleop sessions.

This is the non-interactive counterpart to ``teleop_se3_agent.py``. It builds
a teleop environment, attaches an :class:`~isaaclab_teleop.IsaacTeleopDevice`
configured in :class:`isacteleop.teleop_session_manager.SessionMode.REPLAY`,
and pumps the simulation loop until the recorded operator presses STOP (or
``--max_replay_duration_s`` elapses, or Kit is closed). The user-journey
teleop script remains ``teleop_se3_agent.py``.

Inputs:
    ``--replay_file`` is an MCAP capture produced by Isaac Teleop's
    ``McapRecordingConfig`` path (typically written by ``record_demos.py
    --mcap_record_path``). The recorder lays down per-tracker flatbuffer
    messages (head / hands / controllers) plus the ``_teleop_control``
    ``MessageChannelTracker`` that captured the operator's START / STOP /
    RESET gestures. TeleopCore's
    :class:`~isacteleop.deviceio_session.ReplaySession` re-emits all of
    them on the same monotonic-time cadence they were recorded on, so
    :func:`~isaaclab_teleop.poll_control_events` returns the same edges
    here that ``record_demos.py``'s loop saw at recording time.

Gating:
    The env-step loop mirrors ``teleop_se3_agent.py``: each iteration
    calls :meth:`IsaacTeleopDevice.advance` and
    :func:`~isaaclab_teleop.poll_control_events`, gates ``env.step()`` on
    ``ctrl.is_active`` (so pre-START operator setup frames are rendered
    without stepping), and handles mid-demo ``ctrl.should_reset`` events
    by running the full sim/env/teleop reset cycle.

End-of-replay termination:
    Four distinct signals ask Kit to ``post_quit`` and short-circuit
    the loop body to ``env.sim.render()`` until Kit drains:

    1. **The recorded operator STOP**, replayed via the
       ``_teleop_control`` ``MessageChannelTracker`` at the same
       recording-frame index it was captured at -- ``ctrl.is_active``
       transitions True->False on that frame.
    2. **The env's success condition firing for
       ``--num_success_steps`` consecutive steps**, the natural end of
       a ``record_demos.py``-style capture. Post-success MCAP frames
       are operator wind-down (releases, idle drift) that we have no
       use for in replay, so we skip the ``_handle_reset`` cycle the
       live agent would do.
    3. **A task-specific failure term** (``terminated`` or ``truncated``
       from ``env.step``) -- the recorded trajectory did not reproduce;
       the operator has no agency to recover during replay.
    4. **Wall-clock ``--max_replay_duration_s`` safety cap**, for
       recordings that produce neither a STOP, a success, nor a failure
       within the configured window.

Exit codes:
    The process exits with a status code that CI can branch on:

    * ``0`` -- the recorded ``success_term`` fired.
    * ``1`` -- the env terminated/truncated mid-trajectory, or the
      loop exited without any explicit terminator firing.
    * ``2`` -- ``--max_replay_duration_s`` was hit.

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

XR-active replay:
    Pass ``--cloudxr_env <shorthand-or-path>`` (and optionally
    ``--no-auto_launch_cloudxr``) to auto-spawn the CloudXR runtime and
    engage Kit's XR pipeline during replay. ``--cloudxr_env`` mirrors
    the flag on ``record_demos.py`` and accepts the same ``cloudxrjs``
    / ``avp`` shorthands. This is required (not optional) for two
    distinct reasons:

    A. **Performance parity with live teleop.** A pure-replay run (no
       XR, no CloudXR) skips the entire Kit XR rendering pipeline,
       so frame timings, render load, GPU/CPU contention, and any
       XR-side bottlenecks do not appear -- a captured trajectory
       that replayed at 90Hz under those conditions could easily run
       at 30Hz once XR is actually active. For perf regression or
       benchmarking the replay loop must reproduce the same Kit
       configuration the original recording ran under.

    B. **Correct ``world_T_anchor`` for playback.** The recorded
       tracker stream (head / hands / controllers) lives in
       OpenXR-local space; the world-frame poses the env consumes
       come from ``world_T_anchor @ oxr_pose``. With XR active,
       :class:`~isaaclab_teleop.XrAnchorManager` resolves
       ``world_T_anchor`` through ``XrAnchorSynchronizer`` (the same
       path used at record time), so the live anchor semantics --
       including any dynamic-anchor following of a prim and runtime
       recentering -- are reproduced. Without XR active, the manager
       falls back to the static :class:`~isaaclab_teleop.XrCfg`
       values, which only happen to match record-time semantics when
       the anchor never moved.

    The full incantation also needs ``AppLauncher``'s ``--xr`` flag
    plus a few Kit-side carb settings to flip the AR profile and load
    the teleop XR bridge (the replay path skips both for the
    headless-CI default; we have not yet promoted them to a single
    ``--xr_active`` knob)::

        ./isaaclab.sh -p teleop_replay_agent.py \\
            --task <task> --replay_file <X.mcap> \\
            --xr --device cuda:0 \\
            --cloudxr_env cloudxrjs \\
            --kit_args="--/xr/profile/ar/enabled=true \\
                        --enable isaacsim.kit.xr.teleop.bridge \\
                        --/persistent/xr/openxr/disableInputBindings=true"

    The headset is purely a viewer / anchor source -- the recorded MCAP
    remains the sole source of action; live controller input from the
    spectator's headset does not displace the replayed trajectory.
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
        " measured from the end of the warmup window. Safety net for malformed MCAPs that omit"
        " the operator's STOP gesture -- with a clean recording the agent exits on the replayed"
        " STOP edge well before this cap. Default is 600s (10 min)."
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
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default=None,
    help=(
        "Path to a CloudXR ``.env`` file, or a shorthand: 'cloudxrjs' (Quest/Pico) or 'avp'"
        " (Apple Vision Pro). Default is None -- CloudXR is not launched. Pair with"
        " AppLauncher's ``--xr`` and Kit-side AR-profile settings for spectate-on-headset"
        " replay; see the script docstring for the full command."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Auto-launch the CloudXR runtime when ``--cloudxr_env`` is set. Use"
        " ``--no-auto_launch_cloudxr`` to skip the launch (e.g. when running the"
        " runtime externally). Ignored when ``--cloudxr_env`` is omitted."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging
import sys
import time

import gymnasium as gym
import torch
from isaaclab_teleop import IsaacTeleopDevice, create_isaac_teleop_device, poll_control_events

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


def _resolve_cloudxr_env(value: str | None) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths.

    Mirrors :func:`scripts.tools.record_demos._resolve_cloudxr_env` so the same
    short names (``"cloudxrjs"``, ``"avp"``) behave identically on the
    recording and replay sides. Accepts ``"none"`` / empty / ``None`` to mean
    "no CloudXR" and otherwise returns the value unchanged.
    """
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


def _prepare_env_cfg(task: str, num_envs: int, device: str) -> tuple[ManagerBasedRLEnvCfg, object | None]:
    """Build and tweak an env config suitable for non-interactive replay.

    Mirrors the env-config mutations performed by ``record_demos.py``'s
    :func:`create_environment_config`:

    * The ``success`` term is extracted and cleared from the env config so the
      script can drive success detection (and the matching reset cycle)
      explicitly via :func:`_process_success_condition`, gated by
      ``--num_success_steps``. This matches record_demos.py's pattern of
      manually counting consecutive success steps before resetting.
    * The ``time_out`` term is cleared for the same reason it is cleared in
      :file:`scripts/tools/record_demos.py` and
      :file:`scripts/imitation_learning/robomimic/play.py`: a recorded
      trajectory often exceeds ``episode_length_s`` (pick-place is 20s by
      default; a successful operator demo can easily run 25-30s). With the
      term active, the env auto-truncates partway through the MCAP, resets
      to the default pose, and the remainder of the recorded actions get
      retargeted against the freshly-reset robot -- which manifests as
      "robot moves correctly for a bit, then snaps back / acts wrong."
      The recorder itself did not run with ``time_out`` enabled, so
      reproducing record-time semantics requires clearing it here too.
    * Other failure terms (e.g. ``object_dropping``, ``object_too_far``)
      are left active. ``env.step`` then auto-invokes ``_reset_idx`` for any
      env whose termination fires; the main loop detects this via the
      returned ``terminated``/``truncated`` tensors and completes the reset
      cycle (sim reinit + teleop device reset) so Pink IK starts the next
      attempt with fresh articulation views.

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
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
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

    Called from the main loop on any of:
      * the replayed STOP edge from the ``_teleop_control`` message
        channel (operator pressed Stop during recording),
      * the env's success condition firing for the required number of
        consecutive steps (natural end of a ``record_demos.py``
        single-episode capture),
      * the env terminating/truncating mid-trajectory (failed replay --
        recorded actions did not reproduce the recorded outcome),
      * the wall-clock ``--max_replay_duration_s`` safety cap.

    The caller is expected to update ``replay_outcome`` before calling
    this function so ``main()`` can map it to the correct exit code.

    ``post_quit`` runs on Kit's event loop asynchronously, so the caller
    must keep ``simulation_app.update()`` / ``env.sim.render()`` ticking
    until ``simulation_app.is_running()`` flips False -- which is what
    the main loop's ``if quit_requested: env.sim.render(); continue``
    short-circuit does.
    """
    try:
        import omni.kit.app

        omni.kit.app.get_app().post_quit()
    except Exception:
        logger.exception("Failed to post_quit at end of replay; the loop will keep running")


def main() -> int:
    """Replay a captured Isaac Teleop session against an Isaac Lab environment.

    Builds the env, attaches a replay-mode :class:`IsaacTeleopDevice`, and
    pumps the simulation loop until the recorded STOP edge fires, until
    ``--max_replay_duration_s`` elapses as a safety cap, or until Kit is
    closed.

    Control flow:
        * Pre-loop warmup: ``_wait_for_stage_load`` polls
          ``omni.usd.UsdContext.get_stage_loading_status`` until Kit
          reports zero pending assets, then renders a fixed number of
          settle frames. An optional ``--replay_start_delay_s`` buffer
          can be appended for hardware that needs more grace.
          ``advance()`` is not called during warmup so
          ``ReplaySession.update`` does not consume MCAP frames yet.
        * Main loop: :meth:`IsaacTeleopDevice.advance` returns an action
          tensor derived from the MCAP-replayed tracker stream and
          :func:`poll_control_events` returns the START / STOP / RESET
          edges replayed from the ``_teleop_control`` channel. The env
          steps only when ``ctrl.is_active`` is True, mirroring
          ``teleop_se3_agent.py`` and ``record_demos.py`` exactly --
          pre-START operator-setup frames render only.
        * End-of-replay terminators (any of these flips
          ``quit_requested = True`` and switches the loop into render-
          only mode until Kit drains, then maps to an exit code via
          ``replay_outcome``):
            1. Replayed STOP edge from ``_teleop_control`` -- the
               operator pressed Stop during recording. Does not
               overwrite ``replay_outcome``: if success fired earlier
               in this run, the exit code stays 0; otherwise it stays
               1 (``"incomplete"``), since stopping without reaching
               success is not a successful reproduction.
            2. Success condition met for ``--num_success_steps``
               consecutive steps -- the natural end of a
               ``record_demos.py`` single-episode capture. Sets
               ``replay_outcome = "success"`` (exit code 0). We
               intentionally skip ``_handle_reset`` here because the
               post-success MCAP tail is operator wind-down, not demo
               data.
            3. ``env.step`` ``terminated`` / ``truncated`` -- a task-
               specific failure term fired during the recorded
               trajectory. The recording did not reproduce; we have no
               operator agency to recover. Sets
               ``replay_outcome = "failure"`` (exit code 1).
            4. Wall-clock ``--max_replay_duration_s`` safety cap. Sets
               ``replay_outcome = "timeout"`` (exit code 2).

    Resource cleanup is wrapped in a ``try/finally`` so that ``env.close()``
    always runs, even when device construction or any subsequent setup
    raises -- otherwise the USD stage would leak across CI runs.

    Returns:
        The host process exit code for CI: ``0`` if the recording's
        ``success_term`` fired, ``1`` if the env terminated or truncated
        mid-trajectory (or the loop exited without any explicit
        terminator), ``2`` if ``--max_replay_duration_s`` was hit.
    """
    env: gym.Env | None = None
    replay_outcome = "incomplete"
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
            cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env),
            auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
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
            teleop_active = False
            teleop_was_active = False  # only post_quit on STOP after a real START
            quit_requested = False
            success_step_count = 0
            replay_start_s = time.monotonic()

            while simulation_app.is_running():
                try:
                    with torch.inference_mode():
                        # Wall-clock safety cap. Only hit when the recording
                        # never reaches a natural terminator -- e.g. it omits
                        # an operator STOP AND the env's success/failure
                        # terms never fire within the configured window. A
                        # clean ``record_demos.py``-style capture exits on
                        # the success edge well before this triggers.
                        elapsed_s = time.monotonic() - replay_start_s
                        if not quit_requested and elapsed_s >= args_cli.max_replay_duration_s:
                            print(
                                f"Replay reached max_replay_duration_s={args_cli.max_replay_duration_s:.1f};"
                                " asking Kit to quit."
                            )
                            replay_outcome = "timeout"
                            _request_kit_quit()
                            quit_requested = True

                        # Once we have asked Kit to quit, do not touch the
                        # teleop session or the env again. ``post_quit`` runs
                        # asynchronously on Kit's event loop, so we just spin
                        # ``env.sim.render()`` to keep the loop alive until
                        # ``simulation_app.is_running()`` flips to False.
                        # Skipping ``advance()`` and ``_handle_reset`` here
                        # prevents any late MCAP events (e.g. a recorded
                        # RESET emitted after the success edge) from
                        # re-triggering env work that would race
                        # ``post_quit`` -- which is the actual mechanism
                        # behind the "arms snap back to initial pose"
                        # end-of-replay regression we previously hit.
                        if quit_requested:
                            env.sim.render()
                            continue

                        action = teleop_interface.advance()
                        ctrl = poll_control_events(teleop_interface)

                        # Track active state from the replayed _teleop_control
                        # channel. ``ctrl.is_active`` follows the same shape
                        # that ``record_demos.py`` and ``teleop_se3_agent.py``
                        # consume; None means "no transition this frame."
                        prev_active = teleop_active
                        if ctrl.is_active is not None:
                            teleop_active = ctrl.is_active
                        if teleop_active:
                            teleop_was_active = True

                        # Clean exit on the first STOP edge after a real
                        # START -- the operator pressed Stop during
                        # recording, and ``ReplayMessageChannelTrackerImpl``
                        # surfaces that payload at the same recording-frame
                        # index it was captured at. Per-frame tracker EOF on
                        # its own does NOT trigger this branch:
                        # :class:`TeleopMessageProcessor` keeps emitting
                        # valid False booleans for KILL / RUN_TOGGLE / RESET
                        # after the message-channel MCAP exhausts, so the
                        # state manager stays in its last state and
                        # ``ctrl.is_active`` does not flip. Recordings
                        # without an operator STOP are terminated instead by
                        # the success / failure / wall-clock terminators
                        # below.
                        if prev_active and not teleop_active and teleop_was_active:
                            print("Replay end observed (STOP edge); asking Kit to quit.")
                            _request_kit_quit()
                            quit_requested = True
                            env.sim.render()
                            continue

                        if ctrl.should_reset:
                            _handle_reset(env, teleop_interface)
                            success_step_count = 0
                            continue

                        # Gate stepping on the active state (mirrors
                        # teleop_se3_agent.py:309-328). Pre-START operator
                        # setup frames render only; the recorded START flips
                        # us into the stepping branch.
                        if action is None or not teleop_active:
                            env.sim.render()
                            continue

                        actions = action.repeat(env.num_envs, 1)
                        _, _, terminated, truncated, _ = env.step(actions)

                        # Failure path: ``env.step`` already invoked
                        # ``_reset_idx`` for any env whose task-specific
                        # failure term fired (``time_out`` was cleared by
                        # ``_prepare_env_cfg``; ``success`` is handled
                        # below).
                        #
                        # Replay-specific behavior: a failure mid-trajectory
                        # means the recorded demo did not reproduce -- the
                        # operator has no agency to recover here, so the
                        # rest of the MCAP would just feed retargeted
                        # actions to a freshly-reset env, which is not
                        # meaningful replay. Treat it as a failed
                        # end-of-replay (same shape as the success branch
                        # below) and surface a non-zero exit code so CI
                        # can fail the job.
                        if bool(terminated.any().item()) or bool(truncated.any().item()):
                            print("Replay failure: env terminated/truncated mid-trajectory; asking Kit to quit.")
                            replay_outcome = "failure"
                            _request_kit_quit()
                            quit_requested = True
                            env.sim.render()
                            continue

                        # Success path: ``success_term`` was cleared from the
                        # env cfg so ``env.step`` does not auto-reset on it.
                        # ``_process_success_condition`` consults the original
                        # success term and reports when it has held for
                        # ``--num_success_steps`` consecutive steps.
                        #
                        # Replay-specific behavior: success is the natural
                        # end-of-replay for ``record_demos.py``-style single
                        # episode captures, so treat it as a terminator (ask
                        # Kit to quit, short-circuit) instead of doing a
                        # ``_handle_reset`` like the live agent does. The
                        # alternative -- resetting and continuing into the
                        # post-success MCAP tail -- would just replay
                        # operator wind-down frames (controller releases,
                        # idle motion before they hit Stop on recording),
                        # which is not meaningful demo data and quickly
                        # exhausts the per-frame tracker streams anyway.
                        success_step_count, reset_on_success = _process_success_condition(
                            env, success_term, success_step_count, args_cli.num_success_steps
                        )
                        if reset_on_success:
                            print("Recorded demo succeeded; asking Kit to quit.")
                            replay_outcome = "success"
                            _request_kit_quit()
                            quit_requested = True
                            env.sim.render()
                            continue
                except Exception:
                    # ``logger.exception`` preserves the full traceback; bare
                    # ``logger.error`` would only log the message.
                    logger.exception("Error during simulation step")
                    break
    finally:
        if env is not None:
            env.close()
            print("Environment closed")

    # Map the terminal outcome to a CI-friendly exit code.
    print(f"Replay outcome: {replay_outcome}")
    if replay_outcome == "success":
        return 0
    if replay_outcome == "timeout":
        return 2
    return 1  # "failure" or "incomplete"


if __name__ == "__main__":
    exit_code = main()
    simulation_app.update()
    simulation_app.close()
    sys.exit(exit_code)
