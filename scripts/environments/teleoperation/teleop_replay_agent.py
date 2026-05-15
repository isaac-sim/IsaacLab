# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI/automation entry point for replaying captured teleop sessions.

This is the non-interactive counterpart to ``teleop_se3_agent.py``. It builds
a teleop environment, attaches a teleop device, schedules a replay driver,
and pumps the simulation loop until the replay completes and the application
exits. The user-journey teleop script remains ``teleop_se3_agent.py``.

The current implementation drives playback through Kit's OpenXR XCR backend
and the legacy native XR ``handtracking`` device. The script is structured so
that the replay-driver call site and device selection are the only pieces
that need to change when migrating to a different replay backend in the
future (e.g. an Isaac Teleop ``TeleopSession`` running in replay mode).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Replay a captured teleop session against an Isaac Lab environment. "
        "CI/automation entry point; for interactive teleoperation see teleop_se3_agent.py."
    )
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--replay_file",
    type=str,
    required=True,
    help="Absolute path to the recorded teleop session to replay.",
)
parser.add_argument(
    "--replay_start_delay_s",
    type=float,
    default=0.0,
    help="Seconds to wait after the environment is up before starting replay (default: 120.0).",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=1,
    help=(
        "Number of consecutive steps the task success term must hold before declaring success and"
        " resetting the env. Mirrors the equivalent flag in record_demos.py. (default: 10)"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import asyncio
import logging
import time
from collections.abc import Callable

import gymnasium as gym
import torch

from isaaclab.devices import DeviceBase
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

_LEGACY_DEVICE_NAME = "handtracking"

# Module-level set of pending replay-driver tasks. The asyncio event loop only
# keeps weak references to tasks, so a task that is not referenced elsewhere
# may be garbage-collected before it completes. The completion callback below
# discards the task again once it finishes.
_PENDING_REPLAY_TASKS: set[asyncio.Future] = set()


_RENDERER_SETTLE_FRAMES: int = 30
"""Number of extra render frames pumped after the USD stage finishes loading.

Kit's stage-load status flips to ``count_loading == 0`` as soon as every referenced
asset has been resolved, but the renderer pipeline (shader compilation,
articulation-view binding, material warm-up) typically needs a few more event-loop
ticks to converge. Thirty frames at the default Kit render cadence is ~0.5 s on
most machines and is deterministic per-machine -- unlike a wall-clock delay it
does not have to be tuned for hardware.
"""

_DEFAULT_MAX_STAGE_LOAD_WAIT_S: float = 300.0
"""Safety cap on the deterministic stage-load wait.

Hit only when something is misconfigured (missing asset, slow Nucleus, etc.); a
warning is logged and the loop continues so CI does not hang silently on a
broken capture.
"""


def _wait_for_stage_load(max_wait_s: float = _DEFAULT_MAX_STAGE_LOAD_WAIT_S) -> None:
    """Block until the USD stage finishes resolving every referenced asset.

    Polls :meth:`omni.usd.UsdContext.get_stage_loading_status`. The third element of
    the returned tuple is the count of assets Kit still has pending; when it
    reaches zero the stage is fully streamed in and the renderer pipeline is ready
    to draw against it. After the count reaches zero this function pumps an
    additional :data:`_RENDERER_SETTLE_FRAMES` ``simulation_app.update()`` calls so
    shaders, materials, and articulation views finish warming up before the caller
    begins consuming replay data or stepping the env.

    Unlike :attr:`args_cli.replay_start_delay_s`, which is wall-clock and has to be
    tuned per-host, this wait is deterministic and self-adapting: it returns
    immediately on a warm asset cache and waits exactly long enough on a cold one.

    Args:
        max_wait_s: Upper bound on how long to spin on a non-zero loading count
            before warning and returning. Acts as a safety net for misconfigured
            scenes (missing assets, slow Nucleus); a successful run typically
            completes well within this bound.
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


def _create_replay_teleop_device(
    env_cfg: ManagerBasedRLEnvCfg, task: str, callbacks: dict[str, Callable[[], None]]
) -> DeviceBase:
    """Instantiate the teleop device used during replay.

    Today this returns the legacy native XR ``handtracking`` device because the
    XCR backend replays through Kit's OpenXR runtime, which is the surface
    that device consumes. When migrating to a ``TeleopSession``-driven replay
    backend, swap this for an ``IsaacTeleopDevice`` configured in replay mode.

    Args:
        env_cfg: The environment configuration.
        task: Task identifier, used for diagnostic messages.
        callbacks: Teleop-command callbacks (typically just ``"START"`` for
            replay; see :func:`main`) registered on the device. The XCR
            replay dispatches the recorded user's start gesture through
            Kit's OpenXR message bus, which the legacy
            :class:`~isaaclab.devices.openxr.OpenXRDevice` translates into
            calls into this dictionary.
    """
    if not hasattr(env_cfg, "teleop_devices") or _LEGACY_DEVICE_NAME not in env_cfg.teleop_devices.devices:
        raise ValueError(
            f"Task '{task}' does not expose a teleop device named '{_LEGACY_DEVICE_NAME}'. "
            "Use a task whose env config defines that legacy device, "
            "or update _create_replay_teleop_device to use a different backend."
        )
    teleop_interface = create_teleop_device(_LEGACY_DEVICE_NAME, env_cfg.teleop_devices.devices, callbacks)
    if teleop_interface is None:
        raise RuntimeError(f"Failed to create '{_LEGACY_DEVICE_NAME}' teleop device for task '{task}'.")
    return teleop_interface


def _on_replay_driver_done(future: asyncio.Future) -> None:
    """Surface replay-driver failures so the CI process does not hang.

    When :func:`start_xcr_replay` raises before reaching ``post_quit`` (e.g.
    :class:`FileNotFoundError`, an ``omni.kit`` import failure, or a Kit
    runtime error) the exception sits silently on the discarded future and
    Python only emits a ``Future exception was never retrieved`` warning on
    GC. The main loop would then keep spinning forever because nothing ever
    flips ``simulation_app.is_running()`` to ``False``.

    This callback retrieves the exception, logs it with traceback, and asks
    Kit to quit so the host process exits cleanly. It also drops the task
    from :data:`_PENDING_REPLAY_TASKS` now that it is done.
    """
    _PENDING_REPLAY_TASKS.discard(future)
    if future.cancelled():
        return
    exc = future.exception()
    if exc is None:
        return
    logger.error("XCR replay driver failed", exc_info=exc)
    try:
        import omni.kit.app

        omni.kit.app.get_app().post_quit()
    except Exception:
        logger.exception("Failed to post_quit after replay driver failure")


def _handle_reset(env: gym.Env, teleop_interface: DeviceBase) -> None:
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


def _schedule_replay_driver(replay_file: str, start_delay_s: float) -> None:
    """Schedule the replay driver coroutine on the running asyncio loop.

    Today this drives Kit's OpenXR XCR backend. To migrate to a different
    replay backend (e.g. ``TeleopSession`` running in replay mode), replace
    this call with the equivalent driver hook -- this is the only XCR-specific
    site outside the device-creation helper above.
    """
    from isaaclab_teleop.automation import XcrReplayConfig, start_xcr_replay

    future = asyncio.ensure_future(
        start_xcr_replay(XcrReplayConfig(replay_file=replay_file, start_delay_s=start_delay_s))
    )
    _PENDING_REPLAY_TASKS.add(future)
    future.add_done_callback(_on_replay_driver_done)


def main() -> None:
    """Replay a captured teleop session against an Isaac Lab environment.

    Builds the env, attaches a replay teleop device, schedules the replay
    driver as a background task, and runs the standard teleop step loop
    until the application is closed (driver-issued ``post_quit``, Kit
    shutdown, or operator interrupt).

    The loop deliberately does not call ``env.step()`` until the legacy
    :class:`OpenXRDevice` dispatches a ``"START"`` callback. The XCR replay
    restores the recorded user's start gesture through Kit's OpenXR message
    bus, and the device routes that into the callback registered here --
    exactly the path ``record_demos.py`` uses to know when to start
    recording. Until that ``"START"`` arrives, the OpenXR runtime is silent
    and the device's :meth:`advance` would otherwise return a default zero
    pose for both wrists, which stepping the env with would drive Pink IK
    toward the world origin.

    Unlike :file:`record_demos.py`, the replay agent does **not** subscribe
    to the ``"STOP"`` callback: Kit's ``teleop_command`` bus drains queued
    events as a batch when the AR profile is enabled, so a recorded STOP
    gesture fires within milliseconds of START and would gate the env-step
    loop off again before Pink IK had time to converge.

    Resource cleanup is wrapped in a ``try/finally`` so that ``env.close()``
    always runs, even when device construction or any subsequent setup
    raises -- otherwise the USD stage would leak across CI runs.
    """
    env: gym.Env | None = None
    try:
        env_cfg, success_term = _prepare_env_cfg(args_cli.task, args_cli.num_envs, args_cli.device)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        # Single-element list so the closure can mutate it without ``nonlocal``.
        teleop_active = [False]

        def _on_start() -> None:
            if not teleop_active[0]:
                teleop_active[0] = True
                print("Teleop START received from XCR replay; forwarding actions to env.step().")

        # Intentionally only subscribe to START, not STOP. The XCR replay
        # restores both the recorded user's start and stop gestures from the
        # capture file, and Kit's ``teleop_command`` message bus appears to
        # drain queued events as a batch when the AR profile is enabled --
        # so a STOP fires within milliseconds of START and would shut the env
        # step loop off before Pink IK has had a chance to converge. For the
        # replay agent's one-shot CI use case the only valid termination is
        # the driver's ``post_quit`` (or a real exception in the loop).
        callbacks: dict[str, Callable[[], None]] = {"START": _on_start}

        teleop_interface = _create_replay_teleop_device(env_cfg, args_cli.task, callbacks)
        print(f"Using teleop device: {teleop_interface}")

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
        # assets, then pump a fixed number of renderer-settle frames. This
        # is independent of ``--replay_start_delay_s``; the wall-clock delay
        # below covers the XCR-side OpenXR profile warm-up, while this wait
        # ensures the stage is fully streamed in before the XCR replay
        # injects its first recorded pose.
        _wait_for_stage_load()

        print(f"Replay agent started; replay will begin in {args_cli.replay_start_delay_s:.1f} seconds.")
        _schedule_replay_driver(args_cli.replay_file, args_cli.replay_start_delay_s)

        success_step_count = 0
        while simulation_app.is_running():
            try:
                with torch.inference_mode():
                    action = teleop_interface.advance()
                    if action is None or not teleop_active[0]:
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

                    # Success path: success_term was cleared from the env cfg
                    # so ``env.step`` does not auto-reset on it. Mirror
                    # record_demos.py and trigger a reset only after the
                    # success condition has held for ``num_success_steps``
                    # consecutive steps.
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
