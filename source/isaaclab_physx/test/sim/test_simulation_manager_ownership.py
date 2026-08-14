# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify PhysX lifecycle ownership when its package is imported before Kit starts."""

import sys
from unittest.mock import MagicMock

from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device

# Launch Kit only after importing the PhysX config to reproduce normal entry-point config resolution.
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

import pytest
import torch
from isaaclab_physx.physics import IsaacEvents, PhysxManager

import carb

from isaaclab.cloner import ClonePlan
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.utils import enable_extension

enable_extension("isaacsim.core.simulation_manager")
import isaacsim.core.simulation_manager as simulation_manager_module

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_teardown():
    """Create a fresh stage and simulation context for each test."""
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    yield
    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_initialize_claims_simulation_manager_lifecycle():
    """PhysxManager disables conflicting callbacks without patching supported Isaac Sim versions."""
    original_manager = simulation_manager_module.SimulationManager
    assert original_manager is not PhysxManager
    implementation_module = sys.modules["isaacsim.core.simulation_manager.impl.simulation_manager"]
    supports_startup_setting = hasattr(implementation_module, "_SETTING_ENABLE_DEFAULT_CALLBACKS")

    SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))

    setting = "/exts/isaacsim.core.simulation_manager/enable_default_callbacks"
    assert not carb.settings.get_settings().get_as_bool(setting)
    assert not any(original_manager.get_default_callback_status().values())
    if supports_startup_setting:
        assert simulation_manager_module.SimulationManager is original_manager
    else:
        assert simulation_manager_module.SimulationManager is PhysxManager


@pytest.mark.isaacsim_ci
def test_model_init_precedes_physx_stage_attach_model_load_and_legacy_warmup(monkeypatch):
    """MODEL_INIT finishes stage authoring before PhysX attaches or loads the model."""
    SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))
    calls: list[str] = []
    subscriptions = {}
    event_bus = MagicMock()

    def observe_event(*, event_name, on_event, **_kwargs):
        subscriptions.setdefault(event_name, []).append(on_event)
        return MagicMock()

    def dispatch_event(event_name, *, payload):
        for callback in subscriptions.get(event_name, ()):
            callback(payload)

    event_bus.observe_event.side_effect = observe_event
    event_bus.dispatch_event.side_effect = dispatch_event
    physx = MagicMock()
    physx.force_load_physics_from_usd.side_effect = lambda: calls.append("model_load")
    physx.update_simulation.side_effect = lambda *_: calls.append("model_update")
    physx_sim = MagicMock()
    physx_sim.attach_stage.side_effect = lambda *_: calls.append("attach_stage")
    physx_sim.fetch_results.side_effect = lambda: calls.append("fetch_results")

    with monkeypatch.context() as patch:
        patch.setattr(PhysxManager, "_event_bus", event_bus)
        PhysxManager.register_callback(
            lambda _event: calls.append("model_init"),
            PhysicsEvent.MODEL_INIT,
            order=0,
            wrap_weak_ref=False,
        )
        event_bus.dispatch_event(IsaacEvents.PHYSICS_WARMUP.value, payload={})
        assert calls == []
        PhysxManager.register_callback(
            lambda _event: calls.append("legacy_warmup"),
            IsaacEvents.PHYSICS_WARMUP,
            order=0,
            wrap_weak_ref=False,
        )

        patch.setattr("omni.physx.get_physx_interface", lambda: physx)
        patch.setattr("omni.physx.get_physx_simulation_interface", lambda: physx_sim)
        patch.setattr(PhysxManager, "_warmup_needed", True)
        patch.setattr(PhysxManager, "_view_created", True)
        PhysxManager._warmup_and_create_views()

    assert calls == ["model_init", "attach_stage", "model_load", "model_update", "fetch_results", "legacy_warmup"]


@pytest.mark.isaacsim_ci
def test_model_init_failure_prevents_physx_model_load(monkeypatch):
    """A pre-load lifecycle failure propagates before PhysX can consume an incomplete stage."""
    SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))
    physx = MagicMock()

    def fail_model_init(_event):
        raise RuntimeError("model init failed")

    PhysxManager.register_callback(
        fail_model_init,
        PhysicsEvent.MODEL_INIT,
        order=0,
        wrap_weak_ref=False,
    )

    physx_sim = MagicMock()
    with monkeypatch.context() as patch:
        patch.setattr("omni.physx.get_physx_interface", lambda: physx)
        patch.setattr("omni.physx.get_physx_simulation_interface", lambda: physx_sim)
        patch.setattr(PhysxManager, "_warmup_needed", True)
        patch.setattr(PhysxManager, "_view_created", True)
        with pytest.raises(RuntimeError, match="model init failed"):
            PhysxManager._warmup_and_create_views()

    physx_sim.attach_stage.assert_not_called()
    physx.force_load_physics_from_usd.assert_not_called()


@pytest.mark.isaacsim_ci
def test_model_init_materializes_empty_clone_plan_env_roots():
    """PhysX reset authors every positioned root even when the plan has no asset rows."""
    sim = SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))
    stage = sim.stage
    env_template = "/World/scenes/scene_{}"
    stage.DefinePrim(env_template.format(0), "Xform")
    stage.DefinePrim(f"{env_template.format(0)}/PrototypeMarker", "Xform")

    positions = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]])
    sim.set_clone_plan(
        ClonePlan(
            sources=(),
            destinations=(),
            clone_mask=torch.zeros((0, 3), dtype=torch.bool),
            env_ids=torch.arange(3),
            positions=positions,
            env_template=env_template,
        )
    )

    root_states: list[bool] = []
    PhysxManager.register_callback(
        lambda _event: root_states.append(stage.GetPrimAtPath(env_template.format(1)).IsValid()),
        PhysicsEvent.MODEL_INIT,
        order=2,
        wrap_weak_ref=False,
    )
    PhysxManager.register_callback(
        lambda _event: root_states.append(stage.GetPrimAtPath(env_template.format(1)).IsValid()),
        PhysicsEvent.MODEL_INIT,
        order=4,
        wrap_weak_ref=False,
    )

    assert not stage.GetPrimAtPath(env_template.format(1)).IsValid()
    sim.reset()

    assert root_states == [False, True]
    for env_id, expected_position in enumerate(positions):
        root = stage.GetPrimAtPath(env_template.format(env_id))
        assert root.IsValid() and root.GetTypeName() == "Xform"
        assert tuple(root.GetAttribute("xformOp:translate").Get()) == pytest.approx(expected_position.tolist())
        assert not root.HasAttribute("xformOp:orient")
        assert not stage.GetPrimAtPath(f"{env_template.format(env_id)}/Robot").IsValid()
    assert stage.GetPrimAtPath(f"{env_template.format(0)}/PrototypeMarker").IsValid()
