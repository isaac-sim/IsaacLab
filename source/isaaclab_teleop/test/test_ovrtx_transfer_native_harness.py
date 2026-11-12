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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER") != "1",
        reason="native OVRTX transfer harness is explicitly opt-in",
    ),
]

_TASK = "Isaac-Cartpole"
"""Manager-based task with an ``ovphysx`` physics preset and no camera sensors."""

_TRANSFER_DEVICES = ("cuda:0", "cpu")
"""Simulation devices for the transfer case; CPU physics publishes host scene tensors."""

_GR1T2_TASK = "IsaacContrib-PickPlace-GR1T2-Abs"
"""Humanoid task whose registered entry point prepares it for kitless OvPhysX."""


@dataclass
class _CaseResult:
    offer_thread: int
    claim_thread: int | None = None
    renderer_id: int | None = None
    stage_id: int | None = None
    ordinal: int | None = None
    observed_positions: list[list[float]] = field(default_factory=list)
    error: BaseException | None = None


def _make_env_and_bootstrap(task: str, device: str):
    import gymnasium as gym
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_teleop import remove_camera_configs
    from isaaclab_teleop.ovxr_session import OvxrSessionBootstrap

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    import isaaclab_tasks  # noqa: F401 -- registers the task
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # parse_env_cfg resolves every preset to its default, so select OvPhysX while resolving.
    cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("ovphysx",))
    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    cfg.sim.device = device
    cfg.scene.num_envs = 1
    # Like an --ovxr launch, keep task cameras off the leased renderer.
    cfg = remove_camera_configs(cfg)
    cfg.scene.publication_probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PublicationProbe",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )
    bootstrap = OvxrSessionBootstrap(cfg.sim, OVRTXRendererCfg())
    try:
        return bootstrap, gym.make(task, cfg=cfg).unwrapped
    except BaseException:
        bootstrap.abort()
        raise


def _run_case(task: str, device: str, *, transfer: bool) -> _CaseResult:
    from isaaclab_teleop.ov_libraries_xr import _IsaacLabOvxrApplicationInterface
    from isaaclab_teleop.ovxr_cfg import OvLibrariesXrCfg

    bootstrap, env = _make_env_and_bootstrap(task, device)
    result = _CaseResult(offer_thread=threading.get_native_id())
    host = _IsaacLabOvxrApplicationInterface(bootstrap, env, OvLibrariesXrCfg(auto_launch_cloudxr=False))
    capability = host.offer_render_owner_transfer()
    ready, published, consumed = threading.Event(), threading.Event(), threading.Event()
    positions = ((0.0, 0.0, 3.0), (1.25, -0.5, 3.5))

    def publish(index):
        import torch

        pose = torch.tensor([[*positions[index], 0.0, 0.0, 0.0, 1.0]], device=env.device)
        env.scene["publication_probe"].write_root_pose_to_sim_index(root_pose=pose)
        with host.publication_guard():
            result.ordinal = host.publish_complete_scene(("native-value", index))
        assert result.ordinal is not None

    def attach():
        binding = host.attach_on_render_thread(capability)
        result.renderer_id, result.stage_id = id(binding.renderer), id(binding.stage)
        return binding

    def update_stage(binding):
        with host.publication_guard(), host.owner_lane():
            import ovstage

            binding.renderer.update_from_stage(result.ordinal)
            # Read the native Stage through a fresh query, independently of the publisher's buffers.
            scene = bootstrap._lease._controller._renderer.scene
            path = env.scene.env_prim_paths[0] + "/PublicationProbe"
            with scene.query([path]) as query:
                with binding.stage.read_attributes(
                    query, [scene.paths.intern_token("omni:xform")], ovstage.OrdinalRange.latest(result.ordinal)
                ) as read:
                    read.wait()
                    group = read.fetch_next()
                    assert group is not None, "published probe transform is missing"
                    with group:
                        column = group.columns[0] if hasattr(group, "columns") else group
                        matrix = np.array(column.array(0)).reshape(-1, 4, 4)[0]
                        result.observed_positions.append(matrix[3, :3].tolist())
            np.testing.assert_allclose(result.observed_positions[-1], positions[len(result.observed_positions) - 1])

    def close():
        host.close_publication()
        try:
            host.close_on_render_thread()
        finally:
            host.close_environment_on_render_thread()

    if not transfer:
        try:
            binding = attach()
            for index in range(2):
                publish(index)
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
            for _ in range(2):
                if not published.wait(60):
                    raise TimeoutError("application did not complete its scene publication")
                published.clear()
                if result.error is not None:
                    break
                update_stage(binding)
                consumed.set()
            with pytest.raises(RuntimeError, match="capability is invalid"):
                bootstrap.claim_render_owner_transfer(capability)
        except BaseException as exc:
            if result.error is None:
                result.error = exc
        finally:
            ready.set()
            consumed.set()
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
        for index in range(2):
            publish(index)
            published.set()
            if not consumed.wait(60):
                raise TimeoutError("render owner did not consume the published scene")
            consumed.clear()
            if result.error is not None:
                raise result.error
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


def _run_native_subprocess(task: str, device: str, transfer: bool) -> None:
    # A native non-daemon thread cannot be bounded by Thread.join alone.
    # Keep the native lifetime in a child that the standard subprocess timeout owns.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), task, device, "transfer" if transfer else "same-thread"],
        env={**os.environ, "ISAAC_LAB_OVRTX_USE_OVSTAGE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stderr


def test_same_thread_ovxr_session_native_control():
    """One-thread production scene publication remains viable."""
    _run_native_subprocess(_TASK, "cuda:0", False)


@pytest.mark.parametrize("device", _TRANSFER_DEVICES)
def test_cross_thread_preuse_owner_transfer_native(device):
    """The application publishes while the render thread binds, renders and closes the pair."""
    _run_native_subprocess(_TASK, device, True)


@pytest.mark.parametrize("transfer", [False, True], ids=["same-thread", "transfer"])
def test_gr1t2_cpu_ovphysx_native(transfer):
    """The prepared GR1T2 scene publishes from CPU OvPhysX, with and without the owner transfer."""
    _run_native_subprocess(_GR1T2_TASK, "cpu", transfer)


if __name__ == "__main__":
    import faulthandler

    # Preserve the native wait location if the enclosing subprocess reaches its timeout.
    faulthandler.dump_traceback_later(60)
    if os.environ.get("ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER") != "1":
        raise RuntimeError("native transfer harness requires its explicit opt-in")
    result = _run_case(sys.argv[1], sys.argv[2], transfer=sys.argv[3] == "transfer")
    assert result.renderer_id is not None and result.stage_id is not None
    assert result.ordinal is not None
    assert len(result.observed_positions) == 2
    assert result.observed_positions[0] != result.observed_positions[1]
    faulthandler.cancel_dump_traceback_later()
    print(
        f"offer={result.offer_thread} claim={result.claim_thread} "
        f"renderer={result.renderer_id} stage={result.stage_id} ordinal={result.ordinal}"
    )
