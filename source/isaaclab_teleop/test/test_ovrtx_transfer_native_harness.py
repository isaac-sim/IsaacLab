# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native scene publication and renderer ownership transfer."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER") != "1",
        reason="native OVRTX transfer harness is explicitly opt-in",
    ),
]


@dataclass
class _CaseResult:
    offer_thread: int
    claim_thread: int | None = None
    renderer_id: int | None = None
    stage_id: int | None = None
    ordinal: int | None = None
    error: BaseException | None = None


def _make_env_and_bootstrap():
    import gymnasium as gym
    from isaaclab_ov.ovxr_session import OvxrSessionBootstrap
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_teleop import remove_camera_configs

    import isaaclab_tasks.contrib.pick_place  # noqa: F401 -- registers the exact GR1T2 task
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg("IsaacContrib-PickPlace-GR1T2-Abs", device="cpu", num_envs=1)
    cfg.sim.physics = OvPhysxCfg()
    cfg.stabilize_for_kitless_ovphysx()
    cfg = remove_camera_configs(cfg)
    bootstrap = OvxrSessionBootstrap(cfg.sim, OVRTXRendererCfg())
    try:
        return bootstrap, gym.make("IsaacContrib-PickPlace-GR1T2-Abs", cfg=cfg).unwrapped
    except BaseException:
        bootstrap.abort()
        raise


def _run_case(*, transfer: bool) -> _CaseResult:
    from isaaclab_teleop.ov_libraries_xr import OvLibrariesXrConfig, _IsaacLabOvxrApplicationInterface

    bootstrap, env = _make_env_and_bootstrap()
    result = _CaseResult(offer_thread=threading.get_native_id())
    host = _IsaacLabOvxrApplicationInterface(bootstrap, env, OvLibrariesXrConfig(auto_launch_cloudxr=False), ())
    capability = host.offer_render_owner_transfer()
    ready, published = threading.Event(), threading.Event()

    def attach():
        binding = host.attach_on_render_thread(capability)
        result.renderer_id, result.stage_id = id(binding.renderer), id(binding.stage)
        return binding

    def update_stage(binding):
        with host.publication_guard(), host.owner_lane():
            binding.renderer.update_from_stage(result.ordinal)

    def close():
        host.close_publication()
        try:
            host.close_on_render_thread()
        finally:
            host.close_environment_on_render_thread()

    if not transfer:
        try:
            binding = attach()
            with host.publication_guard():
                result.ordinal = host.publish_complete_scene(("native-control", 0))
            update_stage(binding)
        finally:
            close()
        return result

    with pytest.raises(RuntimeError, match="transfer is pending"):
        _ = bootstrap._lease.ovstage

    def worker():
        try:
            result.claim_thread = threading.get_native_id()
            binding = attach()
            ready.set()
            if not published.wait(60):
                raise TimeoutError("application did not complete its scene publication")
            if result.error is None:
                update_stage(binding)
            with pytest.raises(RuntimeError, match="capability is invalid"):
                bootstrap.claim_render_owner_transfer(capability)
        except BaseException as exc:
            if result.error is None:
                result.error = exc
        finally:
            ready.set()
            try:
                close()
            except BaseException as exc:
                if result.error is None:
                    result.error = exc

    thread = threading.Thread(target=worker, name="ovrtx-native-transfer-worker")
    thread.start()
    try:
        if not ready.wait(90):
            raise TimeoutError("renderer claimant did not finish binding")
        if result.error is not None:
            raise result.error
        with host.publication_guard():
            result.ordinal = host.publish_complete_scene(("native-transfer", 0))
    except BaseException as exc:
        result.error = exc
    finally:
        published.set()
        thread.join(timeout=90)
    assert not thread.is_alive(), "native transfer worker timed out"
    if result.error is not None:
        raise result.error
    assert result.claim_thread is not None and result.claim_thread != result.offer_thread
    with pytest.raises(RuntimeError, match="(owner thread|released)"):
        _ = bootstrap._lease.ovstage
    return result


def _run_native_subprocess(transfer: bool) -> None:
    # A native non-daemon thread cannot be bounded by Thread.join alone.
    # Keep the native lifetime in a child that the standard subprocess timeout owns.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "transfer" if transfer else "same-thread"],
        env={**os.environ, "ISAAC_LAB_OVRTX_USE_OVSTAGE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stderr


def test_gr1t2_same_thread_ovxr_session_native_control():
    """One-thread production scene publication remains viable."""
    _run_native_subprocess(False)


def test_gr1t2_cross_thread_preuse_owner_transfer_native():
    """The application publishes while the render thread binds, renders and closes the pair."""
    _run_native_subprocess(True)


if __name__ == "__main__":
    if os.environ.get("ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER") != "1":
        raise RuntimeError("native transfer harness requires its explicit opt-in")
    result = _run_case(transfer=sys.argv[1] == "transfer")
    assert result.renderer_id is not None and result.stage_id is not None
    assert result.ordinal is not None
    print(
        f"offer={result.offer_thread} claim={result.claim_thread} "
        f"renderer={result.renderer_id} stage={result.stage_id} ordinal={result.ordinal}"
    )
