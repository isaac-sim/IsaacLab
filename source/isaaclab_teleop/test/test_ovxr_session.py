# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVXR bootstrap retirement across the teleoperation/renderer boundary."""

import builtins
from types import SimpleNamespace

import pytest

from isaaclab.sim import SimulationCfg, SimulationContext

pytest.importorskip("ovrtx")
from isaaclab_ov.renderers import OVRTXRenderer, OVRTXRendererCfg


@pytest.mark.parametrize("operation", ["close", "abort"])
def test_bootstrap_failed_transfer_cancellation_keeps_owner_retryable(monkeypatch, operation):
    """A wrong-thread close cannot destroy a pending lease or prevent its owner retiring it."""
    import threading

    from isaaclab_teleop.ovxr_session import OvxrSessionBootstrap

    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.delenv("ISAAC_LAB_SHARE_OVSTAGE", raising=False)
    sim = SimpleNamespace(device="cpu", get_scene_data_provider=lambda: None)
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    renderer = OVRTXRenderer(OVRTXRendererCfg(), sim=sim)
    lease = renderer.acquire_ov_libraries_render_lease()
    bootstrap = OvxrSessionBootstrap.__new__(OvxrSessionBootstrap)
    bootstrap._lease = lease
    bootstrap._owner_transfer_capability = lease.offer_owner_transfer()
    bootstrap._closed = bootstrap._bound = False
    bootstrap._installed_terminal_mode_sentinel = False
    bootstrap._target_frame_readers = {}
    bootstrap._sim = sim = SimulationContext.instance()
    cleared = []
    monkeypatch.setattr(sim, "clear_instance", lambda: cleared.append(True), raising=False)
    errors = []

    def wrong_owner():
        try:
            getattr(bootstrap, operation)()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=wrong_owner)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert len(errors) == 1 and "owner thread must cancel" in errors[0]
    assert not cleared
    # The real owner must still be able to retire the real renderer lease.
    getattr(bootstrap, operation)()
    assert renderer._render_lease.active is None
    assert cleared == ([True] if operation == "abort" else [])


@pytest.fixture
def bootstrap(monkeypatch):
    """Run the real bootstrap constructor and renderer lease; only the physics-backed context is replaced."""
    from isaaclab_teleop import ovxr_session

    from isaaclab.renderers import RenderContext

    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.delenv("ISAAC_LAB_SHARE_OVSTAGE", raising=False)
    monkeypatch.delattr(builtins, ovxr_session._TERMINAL_MODE_SENTINEL, raising=False)
    renderer = OVRTXRenderer(
        OVRTXRendererCfg(), sim=SimpleNamespace(device="cpu", get_scene_data_provider=lambda: None)
    )

    class Context:
        current = None

        def __init__(self, cfg):
            self.cfg = cfg
            self.render_context = RenderContext([(renderer.cfg, renderer)])
            type(self).current = self

        @classmethod
        def instance(cls):
            return cls.current

        @classmethod
        def clear_instance(cls):
            cls.current = None

        def get_or_create_backend(self, _cfg):
            return renderer

        def get_setting(self, path):
            # SimulationContext resolves a bare "cuda" device from this setting.
            return 1 if path == "/physics/cudaDevice" else None

    monkeypatch.setattr(ovxr_session, "SimulationContext", Context)
    bootstrap = ovxr_session.OvxrSessionBootstrap(SimulationCfg(device="cuda:1", dt=0.01))
    yield bootstrap
    bootstrap.abort()
    assert renderer._render_lease.active is None


def _environment(bootstrap, *, num_envs=1, device="cuda:1", dt=0.01):
    return SimpleNamespace(
        sim=bootstrap.simulation_context,
        scene=SimpleNamespace(stage=object()),
        num_envs=num_envs,
        cfg=SimpleNamespace(sim=SimulationCfg(device=device, dt=dt)),
    )


@pytest.mark.parametrize(
    ("num_envs", "device", "dt", "error"),
    [
        (1, "cuda", 0.01, None),
        (2, "cuda:1", 0.01, "exactly one environment"),
        (1, "cuda:0", 0.01, r"sim\.device='cuda:0'"),
        (1, "cuda:1", 0.02, r"sim\.dt=0\.02"),
    ],
)
def test_environment_validation_resolves_devices_and_requires_one_environment(bootstrap, num_envs, device, dt, error):
    """The caller-side check resolves a bare CUDA device like SimulationContext and rejects late worker failures."""
    env = _environment(bootstrap, num_envs=num_envs, device=device, dt=dt)
    if error is None:
        bootstrap.validate_environment(env)
    else:
        with pytest.raises(RuntimeError, match=error):
            bootstrap.validate_environment(env)


def test_binding_requires_the_claimed_owner_transfer(bootstrap):
    """Binding before the claim names the missing claim, even when no camera deferred the Stage yet."""
    with pytest.raises(RuntimeError, match="requires a claimed owner transfer"):
        bootstrap.bind_environment(_environment(bootstrap))
