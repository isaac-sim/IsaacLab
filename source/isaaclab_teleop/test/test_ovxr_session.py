# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVXR bootstrap retirement across the teleoperation/renderer boundary."""

from types import SimpleNamespace

import pytest

from isaaclab.sim import SimulationContext

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
