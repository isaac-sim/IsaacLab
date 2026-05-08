# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal XCR replay driver used by ``teleop_replay_agent.py``.

Schedules a Kit ``omni.kit.xr.core`` XR Capture Replay against an already
running Kit application. This is a transitional implementation; the intended
long-term replacement drives playback through an Isaac Teleop
``TeleopSession`` rather than through Kit's OpenXR XCR backend.

All Kit imports are deferred to :func:`start_xcr_replay` so importing this
module outside of a running Kit application is safe.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class XcrReplayConfig:
    """Configuration for an XCR replay automation run.

    Args:
        replay_file: Absolute path to the ``.bin`` XCR capture to replay.
        profile_name: Name of the Kit XR profile to enable for replay. The
            CI pipelines use ``"ar"``.
        start_delay_s: Seconds to wait after the environment is up before
            starting replay. Gives the simulation time to settle so initial
            warm-up frames do not skew metrics.
        quit_on_complete: When ``True``, call
            :meth:`omni.kit.app.IApp.post_quit` once replay finishes so the
            host CI process exits cleanly.
        max_replay_duration_s: Upper bound on how long the coroutine will
            wait for ``xcr_player`` to clear its playback subscription. If
            replay never finishes (e.g. Kit-side bug, captured session
            never emits a stop event), the coroutine returns after this
            many seconds so CI does not hang indefinitely.
    """

    replay_file: str
    profile_name: str = "ar"
    start_delay_s: float = 120.0
    quit_on_complete: bool = True
    max_replay_duration_s: float = 3600.0


async def start_xcr_replay(cfg: XcrReplayConfig) -> None:
    """Drive an XCR replay against the currently running Kit application.

    This coroutine is intended to be scheduled (e.g. via
    :func:`asyncio.ensure_future`) from a host CI script after the teleop
    environment has been created. It mirrors the original
    ``xcr_perf_automation.run_xcr_replay`` flow used by the ``teleop-cicd``
    pipeline so captured CI metrics remain comparable across the patch
    migration.

    Args:
        cfg: Replay configuration. The replay file must exist on disk.

    Raises:
        FileNotFoundError: If :attr:`XcrReplayConfig.replay_file` does not
            exist when the coroutine starts.
    """
    if not os.path.exists(cfg.replay_file):
        raise FileNotFoundError(f"XCR replay file not found: {cfg.replay_file}")

    import carb.settings
    import omni.kit.app
    import omni.kit.xr.core.test_utils as test_utils
    from omni.kit.xr.core import XRCore
    from omni.kit.xr.core.recorder._xr_xcr import XCRReplayAPI
    from omni.kit.xr.core.recorder.scripts import xcr_player
    from omni.kit.xr.core.recorder.scripts.xcr_player import start_replay_if_enabled

    settings = carb.settings.get_settings()

    await omni.kit.app.get_app().next_update_async()

    settings.set("/xr/system/openxr/xcr/capture/enabled", False)
    settings.set("/xr/system/openxr/xcr/replay/enabled", True)
    settings.set("/xr/system/openxr/xcr/replay/replayFile", cfg.replay_file)
    settings.set(f"/xr/profile/{cfg.profile_name}/system/display", "OpenXR")

    XRCore.get_singleton().get_profile(cfg.profile_name)

    # Construct the replay API so the runtime registers the replay backend
    # before start_replay_if_enabled() is called. Bind to a local so the
    # object stays alive for the lifetime of the coroutine in case any
    # internal subscription is tied to the instance lifetime.
    _replay_api = XCRReplayAPI()  # noqa: F841

    logger.info("XCR replay: waiting %.1f seconds before starting replay", cfg.start_delay_s)
    await asyncio.sleep(cfg.start_delay_s)
    logger.info("XCR replay: starting replay from %s", cfg.replay_file)

    start_replay_if_enabled()

    # Pump a couple of frames so the replay service is fully initialized
    # before the AR profile is enabled.
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    logger.info("XCR replay: enabling XR profile %s", cfg.profile_name)
    async with test_utils.EnabledXRProfile(cfg.profile_name, 0):
        logger.info("XCR replay: XR profile enabled, replay should be playing")

        # The xcr_player module clears its playback subscription when replay
        # finishes; that is the public-ish signal we have for completion.
        # Polling a private attribute is fragile (it may be renamed or
        # removed in future Kit versions); the bounded wait below keeps a
        # stuck poll from hanging the CI job if that ever happens.
        poll_interval_s = 5.0
        elapsed_s = 0.0
        while xcr_player._xcr_playback_subscription is not None:
            if elapsed_s >= cfg.max_replay_duration_s:
                logger.warning(
                    "XCR replay: timed out after %.1fs waiting for playback to complete; aborting wait.",
                    cfg.max_replay_duration_s,
                )
                break
            logger.debug("XCR replay: waiting for playback subscription to clear")
            await asyncio.sleep(poll_interval_s)
            elapsed_s += poll_interval_s

    await omni.kit.app.get_app().next_update_async()

    if cfg.quit_on_complete:
        omni.kit.app.get_app().post_quit()

    logger.info("XCR replay: finished")
